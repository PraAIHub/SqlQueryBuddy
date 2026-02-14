"""Gradio web interface for SQL Query Buddy"""
import gradio as gr
import csv
import io
import logging
import math
import os
import tempfile
import time
from typing import Tuple, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.figure
from src.config import settings
from src.components.executor import DatabaseConnection, QueryExecutor, SQLiteDatabase
from src.components.nlp_processor import ContextManager
from src.components.rag_system import (
    RAGSystem,
    InMemoryVectorDB,
    FAISSVectorDB,
    SimpleEmbeddingProvider,
    FAISS_AVAILABLE,
)
from src.components.sql_generator import SQLGenerator, SQLGeneratorMock
from src.components.insights import InsightGenerator, LocalInsightGenerator
from src.components.optimizer import QueryOptimizer

logger = logging.getLogger(__name__)


class QueryBuddyApp:
    """Main application class for SQL Query Buddy"""

    def __init__(self):
        # Initialize database
        self.db_url = settings.database_url
        self.db_connection = DatabaseConnection(self.db_url)
        self.query_executor = QueryExecutor(
            self.db_connection,
            timeout_seconds=settings.query_timeout_seconds,
            max_rows=settings.max_rows_return,
        )

        # Initialize NLP and context management
        self.context_manager = ContextManager()

        # Track LLM mode for status display
        self.using_real_llm = bool(
            settings.openai_api_key and settings.openai_api_key != ""
        )

        # Initialize SQL generator (with mock fallback for API errors)
        self.mock_generator = SQLGeneratorMock()
        if self.using_real_llm:
            self.sql_generator = SQLGenerator(
                openai_api_key=settings.openai_api_key, model=settings.openai_model
            )
        else:
            self.sql_generator = self.mock_generator

        # Initialize RAG system with schema embeddings
        schema = self.db_connection.get_schema()
        self.context_manager.initialize_with_schema(schema)

        embedding_provider = SimpleEmbeddingProvider()
        # Build vocabulary from schema descriptions
        schema_texts = []
        for table_name, table_info in schema.items():
            schema_texts.append(f"Table {table_name}")
            for col_name, col_info in table_info.get("columns", {}).items():
                schema_texts.append(
                    f"Column {col_name} in {table_name} {col_info.get('type', '')}"
                )
        embedding_provider.build_vocabulary(schema_texts)

        vector_db = FAISSVectorDB() if FAISS_AVAILABLE else InMemoryVectorDB()
        self.rag_system = RAGSystem(embedding_provider, vector_db)
        self.rag_system.initialize_schema(schema)

        # Initialize insights generator (local fallback if no API key)
        if self.using_real_llm:
            self.insight_generator = InsightGenerator(
                openai_api_key=settings.openai_api_key, model=settings.openai_model
            )
        else:
            self.insight_generator = LocalInsightGenerator()

        # Initialize optimizer
        self.optimizer = QueryOptimizer()

        # Store last results for export
        self._last_results = []
        self._last_sql = ""

        # Query history: list of {"query": ..., "sql": ..., "rows": int}
        self._query_history = []

    # Column name hints that indicate monetary values
    CURRENCY_HINTS = {
        "price", "revenue", "amount", "spent", "subtotal",
        "total_spent", "total_sales", "total_revenue", "total_amount",
        "avg_order_value", "monthly_revenue",
    }

    # Column name hints that are NOT monetary (counts, IDs, quantities)
    CURRENCY_EXCLUDE = {
        "count", "orders", "customers", "products", "items", "quantity",
        "unique", "number", "num", "id",
    }

    @staticmethod
    def _format_cell(column_name: str, value) -> str:
        """Format a cell value; apply $X,XXX.XX for currency columns."""
        col_lower = column_name.lower()
        # Exclude count-like columns even if they contain a currency hint word
        if any(ex in col_lower for ex in QueryBuddyApp.CURRENCY_EXCLUDE):
            return str(value)
        if any(hint in col_lower for hint in QueryBuddyApp.CURRENCY_HINTS):
            try:
                fval = float(value)
                if math.isnan(fval):
                    return "N/A"
                return f"${fval:,.2f}"
            except (ValueError, TypeError):
                pass
        return str(value)

    def _generate_chart(self, data: list) -> Optional[matplotlib.figure.Figure]:
        """Auto-detect chartable data and return a matplotlib Figure or None."""
        if not data or len(data) < 2:
            return None

        headers = list(data[0].keys())
        if len(headers) < 2:
            return None

        date_col = None
        numeric_col = None
        categorical_col = None

        for h in headers:
            h_lower = h.lower()
            sample_val = data[0].get(h)
            if any(kw in h_lower for kw in [
                "month", "date", "year", "quarter", "week", "period",
            ]):
                date_col = h
            elif isinstance(sample_val, (int, float)):
                if numeric_col is None:
                    numeric_col = h
            elif isinstance(sample_val, str):
                try:
                    float(sample_val)
                    if numeric_col is None:
                        numeric_col = h
                except (ValueError, TypeError):
                    if categorical_col is None:
                        categorical_col = h

        if not date_col and not categorical_col and len(headers) >= 2:
            if headers[0] != numeric_col:
                categorical_col = headers[0]

        if numeric_col is None:
            return None

        fig, ax = plt.subplots(figsize=(8, 4))
        rows = data[:30]
        values = []
        for row in rows:
            try:
                values.append(float(row.get(numeric_col, 0)))
            except (ValueError, TypeError):
                values.append(0)

        if date_col:
            labels = [str(row.get(date_col, "")) for row in rows]
            ax.plot(range(len(labels)), values, marker="o", linewidth=2, color="#2563eb")
            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
            ax.set_title(f"{numeric_col} over {date_col}", fontsize=12, fontweight="bold")
            ax.set_ylabel(numeric_col)
        elif categorical_col:
            labels = [str(row.get(categorical_col, ""))[:20] for row in rows[:20]]
            vals = values[:20]
            ax.barh(range(len(labels)), vals, color="#2563eb")
            ax.set_yticks(range(len(labels)))
            ax.set_yticklabels(labels, fontsize=9)
            ax.set_title(f"{numeric_col} by {categorical_col}", fontsize=12, fontweight="bold")
            ax.set_xlabel(numeric_col)
            ax.invert_yaxis()
        else:
            plt.close(fig)
            return None

        ax.grid(axis="x", alpha=0.3)
        fig.tight_layout()
        # Note: caller should plt.close(fig) after Gradio consumes it
        return fig

    def _format_history(self) -> str:
        """Format query history as markdown."""
        if not self._query_history:
            return "*No queries yet.*"
        lines = []
        for i, entry in enumerate(reversed(self._query_history), 1):
            lines.append(
                f"**{i}.** {entry['query']}\n"
                f"   `{entry['sql'][:80]}{'...' if len(entry['sql']) > 80 else ''}`"
                f" — {entry['rows']} rows"
            )
        return "\n\n".join(lines)

    def process_query(
        self, user_message: str, chat_history: list
    ) -> Tuple[str, list, Optional[matplotlib.figure.Figure], str, str, str, str]:
        """Process user query and return response, chart, insights, history, RAG context, and SQL"""
        # Validate empty input
        if not user_message or not user_message.strip():
            return "", chat_history, None, "", self._format_history(), "", ""

        user_message = user_message.strip()

        # Validate input length (reject instead of silent truncation)
        MAX_QUERY_LENGTH = 500
        if len(user_message) > MAX_QUERY_LENGTH:
            error_response = (
                f"❌ **Query Too Long**\n\n"
                f"Your query is {len(user_message)} characters, but the maximum allowed is {MAX_QUERY_LENGTH} characters.\n\n"
                f"**Tip:** Try breaking your question into smaller, focused queries.\n\n"
                f"**Example:** Instead of a long complex question, ask:\n"
                f"- \"Show me the top 5 customers by total purchase amount\"\n"
                f"- Then: \"From those, filter to California only\""
            )
            chat_history.append({"role": "user", "content": user_message[:100] + "..."})
            chat_history.append({"role": "assistant", "content": error_response})
            return "", chat_history, None, "", self._format_history(), "", ""

        try:
            # Parse user input with NLP
            parsed = self.context_manager.process_input(user_message)
            parsed_query = parsed["parsed_query"]
            intent = parsed_query["intent"]
            entities = parsed_query["entities"]

            # Get schema context via RAG (semantic retrieval of relevant tables/columns)
            schema = self.db_connection.get_schema()
            rag_context = self.rag_system.get_schema_context_string(
                user_message,
                similarity_threshold=settings.similarity_threshold,
            )

            # Only append full schema when RAG finds nothing relevant (fallback)
            entities_str = ", ".join(entities) if entities else "none"
            if "No relevant schema found" in rag_context:
                full_schema_str = self._format_schema(schema)
                schema_str = (
                    f"{rag_context}\n\n"
                    f"Detected intent: {intent}\n"
                    f"Referenced entities: {entities_str}\n\n"
                    f"Full Schema:\n{full_schema_str}"
                )
            else:
                schema_str = (
                    f"{rag_context}\n\n"
                    f"Detected intent: {intent}\n"
                    f"Referenced entities: {entities_str}"
                )

            # Generate SQL (with mock fallback on API errors like 429)
            conversation_ctx = self.context_manager.get_full_context()
            result = self.sql_generator.generate(
                user_query=user_message,
                schema_context=schema_str,
                conversation_history=conversation_ctx,
            )

            # Fallback to mock generator on API errors (quota, rate limit)
            if (
                not result.get("success", False)
                and self.using_real_llm
                and any(
                    hint in result.get("error", "").lower()
                    for hint in ["429", "quota", "rate limit", "rate_limit"]
                )
            ):
                result = self.mock_generator.generate(
                    user_query=user_message,
                    schema_context=schema_str,
                    conversation_history=conversation_ctx,
                )

            if not result.get("success", False):
                response = (
                    "**Could not generate SQL.** Please try rephrasing your question.\n\n"
                    "Example: *'Show me the top 5 customers by total purchase amount'*"
                )
                chat_history.append({"role": "user", "content": user_message})
                chat_history.append({"role": "assistant", "content": response})
                return "", chat_history, None, "", self._format_history(), "", ""

            generated_sql = result.get("generated_sql", "")

            # Execute query with timing
            t0 = time.time()
            exec_result = self.query_executor.execute(generated_sql)
            exec_ms = (time.time() - t0) * 1000

            if not exec_result.get("success", False):
                response = (
                    f"**SQL Generated:**\n```sql\n{generated_sql}\n```\n\n"
                    "**Execution Error:** The query could not be executed. "
                    "Try rephrasing your question or check column names."
                )
                chat_history.append({"role": "user", "content": user_message})
                chat_history.append({"role": "assistant", "content": response})
                return "", chat_history, None, "", self._format_history(), rag_context, generated_sql

            # Store results for export and history
            data = exec_result.get("data", [])
            self._last_results = data
            self._last_sql = generated_sql
            self._query_history.append({
                "query": user_message,
                "sql": generated_sql,
                "rows": exec_result.get("row_count", 0),
            })
            # Cap history to last 50 entries
            if len(self._query_history) > 50:
                self._query_history = self._query_history[-50:]

            # Format response
            response_lines = [
                "**Generated SQL:**",
                f"```sql\n{generated_sql}\n```",
                "",
                f"**Explanation:** {result.get('explanation', 'N/A')}",
                "",
            ]

            # Add results with execution metadata
            row_count = exec_result.get("row_count", 0)
            truncated = exec_result.get("warning", "")
            timing_str = f"*(executed in {exec_ms:.0f}ms)*"
            limit_note = " (LIMIT applied)" if truncated else ""
            response_lines.append(f"**Results:** {row_count} rows found{limit_note} {timing_str}")

            # Show first few rows
            if data:
                response_lines.append("\n**Data Preview:**")
                headers = list(data[0].keys())
                response_lines.append("|" + "|".join(headers) + "|")
                response_lines.append("|" + "|".join(["---"] * len(headers)) + "|")
                for row in data[:10]:
                    values = [self._format_cell(h, row.get(h, "")) for h in headers]
                    response_lines.append("|" + "|".join(values) + "|")

                if len(data) > 10:
                    response_lines.append(f"\n*(Showing 10 of {len(data)} rows)*")

            # Heavy query warning (cost heuristics)
            cost = self.optimizer.estimate_query_cost(generated_sql)
            if cost["is_heavy"]:
                response_lines.append("\n**Warning: Heavy Query**")
                for w in cost["warnings"]:
                    response_lines.append(f"- {w}")
                response_lines.append(
                    "- *Consider adding date filters or LIMIT to reduce scan scope*"
                )

            # Add categorized optimization suggestions
            opt_result = self.optimizer.analyze(generated_sql, user_message)
            categorized = opt_result.get("categorized", {})

            if categorized.get("assumptions"):
                response_lines.append("\n**Assumptions:**")
                for s in categorized["assumptions"]:
                    response_lines.append(f"- {s['suggestion']}")

            if categorized.get("performance"):
                response_lines.append("\n**Performance:**")
                for s in categorized["performance"]:
                    response_lines.append(
                        f"- {s['suggestion']} *(severity: {s.get('severity', 'low')})*"
                    )

            if categorized.get("next_steps"):
                response_lines.append("\n**Next Steps:**")
                for s in categorized["next_steps"]:
                    response_lines.append(f"- {s['suggestion']}")

            # Generate AI insights (displayed in dedicated panel)
            insights_md = self.insight_generator.generate_insights(data, user_message)

            # Update context and structured query plan
            response_text = "\n".join(response_lines)
            self.context_manager.add_response(
                user_input=user_message,
                assistant_response=response_text,
                generated_sql=generated_sql,
            )
            self.context_manager.update_query_plan(
                intent=intent,
                entities=entities,
                generated_sql=generated_sql,
                user_query=user_message,
            )

            # Generate chart from results (close previous figures to avoid leaks)
            chart = None
            if data:
                chart = self._generate_chart(data)

            # Build enriched RAG display with query plan state
            plan_str = self.context_manager.query_plan.to_context_string()
            rag_display = rag_context

            if plan_str:
                # Format query plan nicely (extract replace outside f-string to avoid backslash issue)
                formatted_plan = plan_str.replace('Active Query State: ', '').replace(' | ', '\n- ')
                rag_display = (
                    f"{rag_context}\n\n"
                    f"---\n\n"
                    f"### 🔄 Active Query State:\n"
                    f"{formatted_plan}\n"
                )

            chat_history.append({"role": "user", "content": user_message})
            chat_history.append({"role": "assistant", "content": response_text})
            return "", chat_history, chart, insights_md, self._format_history(), rag_display, generated_sql

        except Exception as e:
            logger.exception("Unexpected error in process_query")
            error_response = (
                "**Something went wrong.** Please try again or rephrase your question."
            )
            chat_history.append({"role": "user", "content": user_message})
            chat_history.append({"role": "assistant", "content": error_response})
            return "", chat_history, None, "", self._format_history(), "", ""

    @staticmethod
    def _format_schema(schema: dict) -> str:
        """Format schema for LLM context including relationships"""
        lines = ["Database Schema:"]

        for table_name, table_info in schema.items():
            lines.append(f"\nTable: {table_name}")
            columns = table_info.get("columns", {})
            for col_name, col_info in columns.items():
                col_type = col_info.get("type", "unknown")
                lines.append(f"  - {col_name} ({col_type})")

            # Include foreign key relationships
            for fk in table_info.get("foreign_keys", []):
                cols = ", ".join(fk["column"])
                ref_cols = ", ".join(fk["references_column"])
                lines.append(
                    f"  FK: {cols} -> {fk['references_table']}({ref_cols})"
                )

        return "\n".join(lines)

    def _build_schema_explorer_text(self) -> str:
        """Build schema overview text for the schema explorer tab"""
        schema = self.db_connection.get_schema()
        lines = []
        for table_name, table_info in schema.items():
            lines.append(f"### {table_name}")
            columns = table_info.get("columns", {})
            lines.append("| Column | Type |")
            lines.append("|--------|------|")
            for col_name, col_info in columns.items():
                col_type = col_info.get("type", "unknown")
                lines.append(f"| {col_name} | {col_type} |")
            for fk in table_info.get("foreign_keys", []):
                cols = ", ".join(fk["column"])
                ref_cols = ", ".join(fk["references_column"])
                lines.append(
                    f"\n*Foreign Key:* `{cols}` -> "
                    f"`{fk['references_table']}({ref_cols})`"
                )
            lines.append("")
        return "\n".join(lines)

    def _build_sample_data_text(self) -> str:
        """Build sample data preview for the schema explorer tab"""
        schema = self.db_connection.get_schema()
        lines = []
        for table_name in schema:
            lines.append(f"### {table_name} (first 3 rows)")
            rows = self.db_connection.get_sample_data(table_name, limit=3)
            if rows:
                headers = list(rows[0].keys())
                lines.append("|" + "|".join(headers) + "|")
                lines.append("|" + "|".join(["---"] * len(headers)) + "|")
                for row in rows:
                    values = [self._format_cell(h, row.get(h, "")) for h in headers]
                    lines.append("|" + "|".join(values) + "|")
            else:
                lines.append("*No data available*")
            lines.append("")
        return "\n".join(lines)

    def export_csv(self):
        """Export last query results as a CSV file."""
        if not self._last_results:
            return None
        output = io.StringIO()
        headers = list(self._last_results[0].keys())
        writer = csv.DictWriter(output, fieldnames=headers)
        writer.writeheader()
        writer.writerows(self._last_results)
        # Write to a temp file for Gradio download
        tmp = tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False, prefix="query_results_"
        )
        tmp.write(output.getvalue())
        tmp.close()
        return tmp.name

    def _build_status_text(self) -> str:
        """Build system status text"""
        llm_status = (
            f"GPT-4 ({settings.openai_model})"
            if self.using_real_llm
            else "Mock (demo mode - set OPENAI_API_KEY for full LLM)"
        )
        vector_db = "FAISS" if FAISS_AVAILABLE else "In-Memory"
        return (
            f"| Component | Status |\n"
            f"|-----------|--------|\n"
            f"| Database | Connected ({settings.database_type}) |\n"
            f"| LLM Engine | {llm_status} |\n"
            f"| Vector DB | {vector_db} |\n"
            f"| RAG System | Active |\n"
        )

    def create_interface(self) -> gr.Blocks:
        """Create Gradio interface"""
        with gr.Blocks(title="SQL Query Buddy", theme=gr.themes.Soft()) as demo:
            gr.Markdown(
                "# SQL Query Buddy\n"
                "**Conversational AI for Smart Data Insights** | "
                "RAG + LangChain + FAISS\n\n"
                "Ask questions about your database in plain English. "
                "Get optimized SQL, instant results, visualizations, and AI-driven business insights."
            )

            with gr.Tabs():
                # Tab 1: Chat Interface
                with gr.Tab("Chat"):
                    # Mode banner: visual color-coded badge showing LLM mode + DB type
                    if self.using_real_llm:
                        mode_html = f"""
                        <div style='background: linear-gradient(90deg, #10b981 0%, #059669 100%);
                                    color: white;
                                    padding: 12px 20px;
                                    border-radius: 8px;
                                    text-align: center;
                                    font-weight: bold;
                                    margin-bottom: 16px;
                                    box-shadow: 0 2px 4px rgba(16, 185, 129, 0.2);'>
                            ✅ Live LLM ({settings.openai_model}) | 🗄️ DB: {settings.database_type}
                        </div>
                        """
                    else:
                        mode_html = f"""
                        <div style='background: linear-gradient(90deg, #f59e0b 0%, #d97706 100%);
                                    color: white;
                                    padding: 12px 20px;
                                    border-radius: 8px;
                                    text-align: center;
                                    font-weight: bold;
                                    margin-bottom: 16px;
                                    box-shadow: 0 2px 4px rgba(245, 158, 11, 0.2);'>
                            ⚠️ Demo Mode (Mock SQL Generator) | 🗄️ DB: {settings.database_type}
                        </div>
                        """
                    gr.HTML(mode_html)

                    # PRIMARY ACTION FIRST: Question input at the top
                    with gr.Row():
                        msg = gr.Textbox(
                            label="Your question",
                            placeholder="Type your question and press Enter or click Send...",
                            lines=1,
                            scale=4,
                        )
                        submit_btn = gr.Button(
                            "Send", variant="primary", scale=1, interactive=False  # Disabled when empty
                        )
                        export_btn = gr.Button("Export CSV", scale=1)
                        clear = gr.Button("Clear Chat", scale=1)
                    export_file = gr.File(
                        label="Download Results", visible=False
                    )

                    # EXAMPLE QUERIES: Immediate guidance on what to ask
                    gr.Markdown("**💡 Try these example queries:**")
                    with gr.Row():
                        ex1 = gr.Button(
                            "Top 5 customers by spending", size="sm"
                        )
                        ex2 = gr.Button(
                            "Revenue by product category", size="sm"
                        )
                        ex3 = gr.Button(
                            "Total sales per region", size="sm"
                        )
                        ex4 = gr.Button(
                            "Monthly revenue trend", size="sm"
                        )
                    with gr.Row():
                        ex5 = gr.Button(
                            "Avg order value for returning customers",
                            size="sm",
                        )
                        ex6 = gr.Button(
                            "Unique products sold in January", size="sm"
                        )
                        ex7 = gr.Button(
                            "Orders with more than 3 items", size="sm"
                        )
                        ex8 = gr.Button(
                            "Customers inactive 3+ months", size="sm"
                        )

                    # CONVERSATION: Shows query/response history
                    chatbot = gr.Chatbot(
                        label="Conversation",
                        height=350,
                        show_label=False,
                    )

                    # RESULTS: Visualization and AI Insights
                    with gr.Row():
                        with gr.Column(scale=3):
                            gr.Markdown("### Visualization")
                            chart_output = gr.Plot(
                                label="Chart", show_label=False,
                            )
                        with gr.Column(scale=2):
                            gr.Markdown("### AI Insights")
                            insights_output = gr.Markdown(
                                value="*Run a query to see AI-powered insights here.*",
                            )

                    # TECHNICAL DETAILS: Collapsible accordions
                    with gr.Accordion("Generated SQL (click to copy)", open=False):
                        sql_output = gr.Code(
                            label="SQL",
                            language="sql",
                            value="-- Run a query to see generated SQL here",
                        )

                    with gr.Accordion("Query History", open=False):
                        history_output = gr.Markdown(
                            value="*No queries yet.*",
                        )

                    with gr.Accordion("RAG Context (schema retrieval)", open=False):
                        rag_output = gr.Markdown(
                            value="*RAG context will appear here after a query.*",
                        )

                # Tab 2: Schema Explorer
                with gr.Tab("Schema & Sample Data"):
                    gr.Markdown("## Database Schema")
                    gr.Markdown(self._build_schema_explorer_text())
                    gr.Markdown("## Sample Data")
                    gr.Markdown(self._build_sample_data_text())

                # Tab 3: System Status
                with gr.Tab("System Status"):
                    gr.Markdown("## System Status")
                    gr.Markdown(self._build_status_text())
                    gr.Markdown("## About")
                    gr.Markdown(
                        "**SQL Query Buddy** converts natural language questions "
                        "into SQL queries using:\n"
                        "- **LangChain + GPT-4** for SQL generation\n"
                        "- **FAISS** vector database for RAG-powered schema retrieval\n"
                        "- **NLP processing** for intent detection and entity extraction\n"
                        "- **Query optimization** with performance suggestions\n"
                        "- **AI insights** with pattern detection and trend analysis\n"
                        "- **Context retention** for multi-turn conversations\n"
                    )

            # Wrapper function to handle loading states
            def process_with_loading(user_message, chat_history):
                """Process query and manage button states during execution"""
                # Call the actual process_query function
                results = self.process_query(user_message, chat_history)

                # Return results + re-enable all interactive components
                return list(results) + [
                    gr.update(interactive=True),   # msg textbox
                    gr.update(interactive=True),   # submit_btn
                    gr.update(interactive=True),   # ex1
                    gr.update(interactive=True),   # ex2
                    gr.update(interactive=True),   # ex3
                    gr.update(interactive=True),   # ex4
                    gr.update(interactive=True),   # ex5
                    gr.update(interactive=True),   # ex6
                    gr.update(interactive=True),   # ex7
                    gr.update(interactive=True),   # ex8
                ]

            # All outputs including button states
            query_outputs = [
                msg, chatbot, chart_output, insights_output,
                history_output, rag_output, sql_output,
                msg, submit_btn, ex1, ex2, ex3, ex4, ex5, ex6, ex7, ex8
            ]

            # Event handlers with loading state management
            msg.submit(
                lambda: [
                    gr.update(interactive=False),  # msg
                    gr.update(interactive=False),  # submit_btn
                    gr.update(interactive=False),  # ex1-ex8 (all buttons)
                ] * 3,  # Disable msg, submit, and all 8 example buttons
                outputs=[msg, submit_btn, ex1, ex2, ex3, ex4, ex5, ex6, ex7, ex8],
                queue=False
            ).then(
                process_with_loading, [msg, chatbot], query_outputs
            )

            submit_btn.click(
                lambda: [
                    gr.update(interactive=False),  # msg
                    gr.update(interactive=False),  # submit_btn
                    gr.update(interactive=False),  # ex1-ex8
                ] * 3,
                outputs=[msg, submit_btn, ex1, ex2, ex3, ex4, ex5, ex6, ex7, ex8],
                queue=False
            ).then(
                process_with_loading, [msg, chatbot], query_outputs
            )

            def clear_chat():
                self.context_manager.reset()
                self._query_history.clear()
                return (
                    [], "", None,
                    "*Run a query to see AI-powered insights here.*",
                    "*No queries yet.*",
                    "*RAG context will appear here after a query.*",
                    "-- Run a query to see generated SQL here",
                )

            clear.click(clear_chat, outputs=[chatbot, msg, chart_output, insights_output, history_output, rag_output, sql_output])

            def handle_export():
                path = self.export_csv()
                if path:
                    return gr.File(value=path, visible=True)
                gr.Info("No results to export. Run a query first.")
                return gr.File(visible=False)

            export_btn.click(handle_export, outputs=[export_file])

            # Enable/disable Send button based on textbox content
            def update_send_button(text):
                """Enable Send button only when textbox has content"""
                return gr.update(interactive=bool(text and text.strip()))

            msg.change(update_send_button, inputs=[msg], outputs=[submit_btn])

            # Example query buttons: fill textbox, disable buttons, auto-submit, re-enable
            example_queries = {
                ex1: "Show me the top 5 customers by total purchase amount",
                ex2: "Which product category made the most revenue?",
                ex3: "Show total sales per region",
                ex4: "Show the trend of monthly revenue over time",
                ex5: "Find the average order value for returning customers",
                ex6: "How many unique products were sold in January?",
                ex7: "How many orders contained more than 3 items?",
                ex8: "List customers who haven't ordered anything in the last 3 months",
            }
            for btn, query in example_queries.items():
                btn.click(
                    lambda q=query: q, outputs=[msg]
                ).then(
                    lambda: [gr.update(interactive=False)] * 10,  # Disable all buttons
                    outputs=[msg, submit_btn, ex1, ex2, ex3, ex4, ex5, ex6, ex7, ex8],
                    queue=False
                ).then(
                    process_with_loading, [msg, chatbot], query_outputs
                )

        return demo


def create_sample_db():
    """Create sample database if it doesn't exist"""
    db_path = settings.database_url.replace("sqlite:///", "")
    if not os.path.exists(db_path):
        SQLiteDatabase.create_sample_database(db_path)


def main():
    """Main entry point"""
    # Create sample database
    if settings.database_type == "sqlite":
        create_sample_db()

    # Create and launch app
    app = QueryBuddyApp()
    demo = app.create_interface()

    demo.queue(default_concurrency_limit=1)
    demo.launch(
        server_name=settings.server_host,
        server_port=settings.gradio_server_port,
        share=settings.gradio_share,
    )


if __name__ == "__main__":
    main()
