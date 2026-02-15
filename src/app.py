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
            logger.info(f"🤖 Using OpenAI {settings.openai_model} for AI insights")
        else:
            self.insight_generator = LocalInsightGenerator()
            logger.info("🔧 Using LocalInsightGenerator (demo mode - no OpenAI API key)")

        # Initialize optimizer
        self.optimizer = QueryOptimizer()

        # Store last results for export
        self._last_results = []
        self._last_sql = ""

        # Query history: list of {"query": ..., "sql": ..., "rows": int}
        self._query_history = []

        # Stats tracking for dashboard
        self._stats = {
            "total_queries": 0,
            "successful_queries": 0,
            "failed_queries": 0,
            "total_response_time_ms": 0,
            "query_times": [],  # Last 50 query times
        }

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
        if not data:
            return None

        headers = list(data[0].keys())

        # Special case: Single row with single numeric value (COUNT, SUM, etc.)
        if len(data) == 1 and len(headers) == 1:
            col_name = headers[0]
            value = data[0].get(col_name)
            if isinstance(value, (int, float)):
                return self._generate_single_value_card(col_name, value)
            return None

        # Need at least 2 rows for regular charts
        if len(data) < 2:
            return None

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
        is_truncated = len(data) > 30
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
            title = f"{numeric_col} over {date_col}"
            if is_truncated:
                title += f" (showing first 30 of {len(data)} points)"
            ax.set_title(title, fontsize=12, fontweight="bold")
            ax.set_ylabel(numeric_col)
        elif categorical_col:
            labels = [str(row.get(categorical_col, ""))[:20] for row in rows[:20]]
            vals = values[:20]
            ax.barh(range(len(labels)), vals, color="#2563eb")
            ax.set_yticks(range(len(labels)))
            ax.set_yticklabels(labels, fontsize=9)
            title = f"{numeric_col} by {categorical_col}"
            if is_truncated:
                title += f" (first 20 of {len(data)})"
            ax.set_title(title, fontsize=12, fontweight="bold")
            ax.set_xlabel(numeric_col)
            ax.invert_yaxis()
        else:
            plt.close(fig)
            return None

        ax.grid(axis="x", alpha=0.3)
        fig.tight_layout()
        # Note: caller should plt.close(fig) after Gradio consumes it
        return fig

    def _generate_single_value_card(self, label: str, value: float) -> matplotlib.figure.Figure:
        """Generate a large number card for single-value results (COUNT, SUM, etc.)."""
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.axis('off')  # Hide axes

        # Format the value with proper separators
        if abs(value) >= 1000000:
            display_value = f"{value/1000000:.1f}M"
        elif abs(value) >= 1000:
            display_value = f"{value/1000:.1f}K"
        elif value == int(value):
            display_value = f"{int(value):,}"
        else:
            display_value = f"{value:,.2f}"

        # Check if it's a currency column
        label_lower = label.lower()
        is_currency = any(hint in label_lower for hint in self.CURRENCY_HINTS)
        if is_currency and not any(ex in label_lower for ex in self.CURRENCY_EXCLUDE):
            if abs(value) < 1000:
                display_value = f"${value:,.2f}"
            else:
                display_value = "$" + display_value

        # Draw large number in center
        ax.text(
            0.5, 0.55, display_value,
            ha='center', va='center',
            fontsize=60, fontweight='bold',
            color='#2563eb',
            transform=ax.transAxes
        )

        # Draw label below
        formatted_label = label.replace('_', ' ').title()
        ax.text(
            0.5, 0.25, formatted_label,
            ha='center', va='center',
            fontsize=16, color='#64748b',
            transform=ax.transAxes
        )

        # Add subtle background
        rect = plt.Rectangle(
            (0.15, 0.15), 0.7, 0.7,
            fill=True, facecolor='#f8fafc',
            edgecolor='#e2e8f0', linewidth=2,
            transform=ax.transAxes, zorder=-1
        )
        ax.add_patch(rect)

        fig.tight_layout()
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
    ) -> Tuple[str, list, Optional[matplotlib.figure.Figure], str, str, str, str, str]:
        """Process user query and return response, chart, insights, history, RAG context, SQL, and filters"""
        # Validate empty input
        if not user_message or not user_message.strip():
            return "", chat_history, None, "", self._format_history(), "", "", ""

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
            return "", chat_history, None, "", self._format_history(), "", "", ""

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
                return "", chat_history, None, "", self._format_history(), "", "", ""

            generated_sql = result.get("generated_sql", "")

            # Execute query with timing
            t0 = time.time()
            exec_result = self.query_executor.execute(generated_sql)
            exec_ms = (time.time() - t0) * 1000

            # Track stats
            self._stats["total_queries"] += 1
            self._stats["total_response_time_ms"] += exec_ms
            self._stats["query_times"].append(exec_ms)
            if len(self._stats["query_times"]) > 50:
                self._stats["query_times"] = self._stats["query_times"][-50:]

            if not exec_result.get("success", False):
                self._stats["failed_queries"] += 1
                response = (
                    f"**SQL Generated:**\n```sql\n{generated_sql}\n```\n\n"
                    "**Execution Error:** The query could not be executed. "
                    "Try rephrasing your question or check column names."
                )
                chat_history.append({"role": "user", "content": user_message})
                chat_history.append({"role": "assistant", "content": response})
                return "", chat_history, None, "", self._format_history(), rag_context, generated_sql, ""

            # Track successful query
            self._stats["successful_queries"] += 1

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
            # Try LLM first, fall back to local generator on error
            try:
                insights_md = self.insight_generator.generate_insights(data, user_message)
                # If insights indicate API failure, try local fallback
                if "AI Insights unavailable" in insights_md and self.using_real_llm:
                    logger.warning("⚠️ LLM insights failed (likely rate limit or quota), falling back to local generator")
                    local_gen = LocalInsightGenerator()
                    insights_md = local_gen.generate_insights(data, user_message)
                    logger.info("✅ Local insights generated successfully as fallback")
            except Exception as e:
                logger.warning(f"⚠️ Insight generation failed: {e}, using local fallback")
                local_gen = LocalInsightGenerator()
                insights_md = local_gen.generate_insights(data, user_message)
                logger.info("✅ Local insights generated successfully as fallback")

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

            # Generate quick filter options
            filter_opts = self._detect_filter_options(data)
            filter_md = ""
            if filter_opts:
                filter_md = "**🎛️ Quick Filters:** "
                for col_name, values in filter_opts.items():
                    filter_md += f"\n\n*{col_name}:* "
                    filter_md += " • ".join([f"`{v}`" for v in values[:5]])

            chat_history.append({"role": "user", "content": user_message})
            chat_history.append({"role": "assistant", "content": response_text})
            return "", chat_history, chart, insights_md, self._format_history(), rag_display, generated_sql, filter_md

        except Exception as e:
            logger.exception("Unexpected error in process_query")
            error_type = type(e).__name__

            # Provide more specific error messages based on error type
            if "timeout" in str(e).lower() or "TimeoutError" in error_type:
                error_response = (
                    "⏱️ **Query Timeout** - The operation took too long. "
                    "Try simplifying your question or adding more specific filters."
                )
            elif "connection" in str(e).lower() or "ConnectionError" in error_type:
                error_response = (
                    "🔌 **Connection Error** - Unable to connect to the service. "
                    "Please check your network connection and try again."
                )
            elif "rate" in str(e).lower() or "429" in str(e):
                error_response = (
                    "⚠️ **Rate Limit** - Too many requests. "
                    "The system has automatically switched to demo mode. Please try your query again."
                )
            elif "OpenAIError" in error_type or "APIError" in error_type:
                error_response = (
                    "🤖 **LLM Service Error** - The AI service is temporarily unavailable. "
                    "Try again in a moment, or check if your API key is configured correctly."
                )
            else:
                error_response = (
                    f"❌ **Unexpected Error ({error_type})** - Something went wrong while processing your query. "
                    "Please try rephrasing your question or contact support if this persists."
                )

            chat_history.append({"role": "user", "content": user_message})
            chat_history.append({"role": "assistant", "content": error_response})
            return "", chat_history, None, "", self._format_history(), "", "", ""

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

    def _build_dashboard_overview(self) -> str:
        """Build dashboard overview with stats and recent queries"""
        total = self._stats["total_queries"]
        success = self._stats["successful_queries"]
        failed = self._stats["failed_queries"]

        # Calculate metrics
        success_rate = (success / total * 100) if total > 0 else 0
        avg_time = (
            self._stats["total_response_time_ms"] / total
            if total > 0
            else 0
        )

        # Stats cards
        lines = [
            "## 📊 Analytics Dashboard\n",
            "### Today's Performance\n",
            '<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 16px; margin: 20px 0;">',
        ]

        # Card 1: Total Queries - Purple gradient
        lines.append(f'''
<div style="background: linear-gradient(135deg, #7c3aed 0%, #a855f7 100%);
            padding: 20px; border-radius: 12px; color: white; box-shadow: 0 4px 12px rgba(124,58,237,0.15);">
    <div style="font-size: 14px; opacity: 0.9; margin-bottom: 8px;">🔍 Total Queries</div>
    <div style="font-size: 36px; font-weight: bold;">{total}</div>
    <div style="font-size: 12px; opacity: 0.8; margin-top: 4px;">This session</div>
</div>
''')

        # Card 2: Success Rate - Blue gradient (consistent with hero)
        lines.append(f'''
<div style="background: linear-gradient(135deg, #6366f1 0%, #4f46e5 100%);
            padding: 20px; border-radius: 12px; color: white; box-shadow: 0 4px 12px rgba(99,102,241,0.15);">
    <div style="font-size: 14px; opacity: 0.9; margin-bottom: 8px;">✅ Success Rate</div>
    <div style="font-size: 36px; font-weight: bold;">{success_rate:.0f}%</div>
    <div style="font-size: 12px; opacity: 0.8; margin-top: 4px;">{success} of {total} queries</div>
</div>
''')

        # Card 3: Avg Response Time - Purple-blue gradient
        lines.append(f'''
<div style="background: linear-gradient(135deg, #8b5cf6 0%, #6366f1 100%);
            padding: 20px; border-radius: 12px; color: white; box-shadow: 0 4px 12px rgba(139,92,246,0.15);">
    <div style="font-size: 14px; opacity: 0.9; margin-bottom: 8px;">⚡ Avg Response</div>
    <div style="font-size: 36px; font-weight: bold;">{avg_time:.0f}ms</div>
    <div style="font-size: 12px; opacity: 0.8; margin-top: 4px;">Query execution time</div>
</div>
''')

        # Card 4: Results with Data - Blue gradient
        chart_count = sum(1 for entry in self._query_history if entry.get("rows", 0) > 0)
        lines.append(f'''
<div style="background: linear-gradient(135deg, #4f46e5 0%, #3b82f6 100%);
            padding: 20px; border-radius: 12px; color: white; box-shadow: 0 4px 12px rgba(79,70,229,0.15);">
    <div style="font-size: 14px; opacity: 0.9; margin-bottom: 8px;">📈 Results with Data</div>
    <div style="font-size: 36px; font-weight: bold;">{chart_count}</div>
    <div style="font-size: 12px; opacity: 0.8; margin-top: 4px;">Queries with results</div>
</div>
''')

        lines.append('</div>\n')

        # Recent queries section
        lines.append("\n### 🔥 Recent Queries\n")

        if not self._query_history:
            lines.append("*No queries yet. Start asking questions in the Chat tab!*\n")
        else:
            lines.append("| # | Query | Rows | SQL |\n")
            lines.append("|---|-------|------|-----|\n")

            recent = list(reversed(self._query_history[-5:]))  # Last 5, newest first
            for i, entry in enumerate(recent, 1):
                query = entry['query'][:50] + "..." if len(entry['query']) > 50 else entry['query']
                sql = entry['sql'][:40] + "..." if len(entry['sql']) > 40 else entry['sql']
                rows = entry['rows']
                lines.append(f"| {i} | {query} | {rows} | `{sql}` |\n")

        # Quick tips
        lines.append("\n### 💡 Quick Tips\n")
        if total == 0:
            lines.append("- Click the **💬 Chat** tab to start querying your database\n")
            lines.append("- Try example queries like 'Top 5 customers by spending'\n")
            lines.append("- Get AI-powered insights and visualizations automatically\n")
        elif success_rate < 70:
            lines.append("- Try rephrasing queries if they fail\n")
            lines.append("- Use example queries as templates\n")
            lines.append("- Check the RAG Context tab to see what schema was retrieved\n")
        else:
            lines.append("- Great success rate! Keep exploring your data\n")
            lines.append("- Try follow-up queries to drill down into results\n")
            lines.append("- Export results as CSV for further analysis\n")

        return "".join(lines)

    def _detect_filter_options(self, data: list) -> dict:
        """Detect columns that can be used for filtering (categorical with 2-10 unique values)"""
        if not data or len(data) < 2:
            return {}

        filter_options = {}

        # Analyze each column
        for col_name in data[0].keys():
            values = [row.get(col_name) for row in data if row.get(col_name) is not None]

            # Skip numeric columns with high cardinality
            if all(isinstance(v, (int, float)) for v in values):
                continue

            # Get unique values
            unique_values = list(set(str(v) for v in values))

            # Only create filters for columns with 2-10 unique values
            if 2 <= len(unique_values) <= 10:
                filter_options[col_name] = sorted(unique_values)[:10]  # Limit to 10 options

        return filter_options

    def create_interface(self) -> gr.Blocks:
        """Create Gradio interface"""
        with gr.Blocks(title="SQL Query Buddy", theme=gr.themes.Soft()) as demo:
            # Custom CSS for modern styling
            gr.HTML("""
            <style>
                /* Chart container max-height with scroll */
                .chart-container {
                    max-height: 500px;
                    overflow-y: auto;
                    overflow-x: hidden;
                }

                /* Secondary button styling - lighter */
                button.secondary {
                    background: transparent !important;
                    border: 1px solid #d1d5db !important;
                    color: #6b7280 !important;
                }
                button.secondary:hover {
                    background: #f9fafb !important;
                    border-color: #9ca3af !important;
                }
            </style>
            <script>
                // Force chatbot to scroll to bottom - improved for Gradio 6.x
                function scrollChatbotToBottom() {
                    // Try multiple selectors for Gradio 6.x compatibility
                    const selectors = [
                        '.chatbot .overflow-y-auto',
                        '.chatbot [class*="overflow"]',
                        'gradio-chatbot .overflow-y-auto',
                        '[data-testid="chatbot"] .overflow-y-auto',
                        '.chatbot > div > div'
                    ];

                    let scrolled = false;
                    for (const selector of selectors) {
                        const elements = document.querySelectorAll(selector);
                        elements.forEach(element => {
                            if (element && element.scrollHeight > element.clientHeight) {
                                element.scrollTop = element.scrollHeight;
                                scrolled = true;
                                console.log('Scrolled chatbot using selector:', selector);
                            }
                        });
                        if (scrolled) break;
                    }

                    if (!scrolled) {
                        console.log('No scrollable chatbot element found');
                    }
                }

                // Use MutationObserver to detect when chatbot content changes
                function setupChatbotAutoScroll() {
                    // Try to find chatbot container with multiple approaches
                    const chatbotSelectors = [
                        '.chatbot',
                        'gradio-chatbot',
                        '[data-testid="chatbot"]'
                    ];

                    let chatbotContainer = null;
                    for (const selector of chatbotSelectors) {
                        chatbotContainer = document.querySelector(selector);
                        if (chatbotContainer) {
                            console.log('Found chatbot container using:', selector);
                            break;
                        }
                    }

                    if (chatbotContainer) {
                        const observer = new MutationObserver((mutations) => {
                            // Scroll to bottom whenever chatbot content changes
                            requestAnimationFrame(scrollChatbotToBottom);
                        });

                        observer.observe(chatbotContainer, {
                            childList: true,
                            subtree: true,
                            characterData: true
                        });

                        console.log('Chatbot auto-scroll observer set up successfully');
                    } else {
                        console.log('Could not find chatbot container for auto-scroll');
                    }
                }

                // Initialize when page loads
                if (document.readyState === 'loading') {
                    document.addEventListener('DOMContentLoaded', setupChatbotAutoScroll);
                } else {
                    setupChatbotAutoScroll();
                }

                // Also try multiple times to ensure Gradio is ready
                setTimeout(setupChatbotAutoScroll, 1000);
                setTimeout(setupChatbotAutoScroll, 2000);
                setTimeout(setupChatbotAutoScroll, 3000);
            </script>
            """)

            gr.Markdown(
                "# 🤖 SQL Query Buddy\n"
                "**Conversational AI for Smart Data Insights** — Powered by RAG + LangChain + FAISS"
            )

            # Hero Banner - Value Proposition (Brightened & High Contrast)
            gr.HTML("""
<div style='background: linear-gradient(135deg, #7c3aed 0%, #a855f7 50%, #6366f1 100%);
            padding: 15px 24px;
            border-radius: 12px;
            margin: 12px 0 16px 0;
            box-shadow: 0 8px 24px rgba(124, 58, 237, 0.25);
            position: relative;'>
    <!-- Brightness overlay -->
    <div style='position: absolute; top: 0; left: 0; right: 0; bottom: 0;
                background: linear-gradient(135deg, rgba(255,255,255,0.12) 0%, rgba(255,255,255,0.05) 100%);
                border-radius: 12px; pointer-events: none;'></div>

    <!-- Content -->
    <div style='position: relative; z-index: 1;'>
        <div style='font-size: 18px; font-weight: 700; margin-bottom: 10px; text-align: center;
                    color: #ffffff; text-shadow: 0 2px 4px rgba(0,0,0,0.15);'>
            💬 Ask Questions in Plain English, Get SQL-Powered Insights
        </div>

        <div style='display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 10px; margin-top: 10px;'>
            <div style='background: rgba(255,255,255,0.95); padding: 12px; border-radius: 10px;
                        border: 1px solid rgba(255,255,255,0.3);
                        box-shadow: 0 4px 12px rgba(0,0,0,0.1);'>
                <div style='font-weight: 700; margin-bottom: 5px; color: #7c3aed; font-size: 13px;'>
                    🎯 No SQL Knowledge Needed
                </div>
                <div style='font-size: 13px; color: #4b5563; line-height: 1.4;'>
                    Ask questions like "Show me top customers" — we handle the rest
                </div>
            </div>

            <div style='background: rgba(255,255,255,0.95); padding: 12px; border-radius: 10px;
                        border: 1px solid rgba(255,255,255,0.3);
                        box-shadow: 0 4px 12px rgba(0,0,0,0.1);'>
                <div style='font-weight: 700; margin-bottom: 5px; color: #7c3aed; font-size: 13px;'>
                    ⚡ AI-Powered Insights
                </div>
                <div style='font-size: 13px; color: #4b5563; line-height: 1.4;'>
                    Get charts, trends, and business recommendations automatically
                </div>
            </div>

            <div style='background: rgba(255,255,255,0.95); padding: 12px; border-radius: 10px;
                        border: 1px solid rgba(255,255,255,0.3);
                        box-shadow: 0 4px 12px rgba(0,0,0,0.1);'>
                <div style='font-weight: 700; margin-bottom: 5px; color: #7c3aed; font-size: 13px;'>
                    🔍 RAG + Query Optimizer
                </div>
                <div style='font-size: 13px; color: #4b5563; line-height: 1.4;'>
                    Semantic schema search + performance optimization built-in
                </div>
            </div>
        </div>

        <div style='margin-top: 10px; padding: 8px 14px;
                    background: rgba(255,255,255,0.25);
                    border: 1px solid rgba(255,255,255,0.4);
                    border-radius: 8px; text-align: center; font-size: 12px;
                    color: #ffffff; font-weight: 500;'>
            💡 <b>Get Started:</b> Try the example buttons below or type any question about your data
        </div>
    </div>
</div>
""")

            # Compact status indicator - small, non-interactive chips
            if self.using_real_llm:
                status_html = f"""
                <div style='text-align: center; font-size: 10px; color: #9ca3af; margin: 6px 0 12px 0; letter-spacing: 0.3px;'>
                    <span style='background: #f3f4f6; padding: 3px 8px; border-radius: 4px; margin: 0 3px; border: 1px solid #e5e7eb;'>
                        ✅ {settings.openai_model}
                    </span>
                    <span style='background: #f3f4f6; padding: 3px 8px; border-radius: 4px; margin: 0 3px; border: 1px solid #e5e7eb;'>
                        🗄️ {settings.database_type.upper()}
                    </span>
                    <span style='background: #f3f4f6; padding: 3px 8px; border-radius: 4px; margin: 0 3px; border: 1px solid #e5e7eb;'>
                        ⚡ FAISS
                    </span>
                </div>
                """
            else:
                status_html = f"""
                <div style='text-align: center; font-size: 10px; color: #9ca3af; margin: 6px 0 12px 0; letter-spacing: 0.3px;'>
                    <span style='background: #fef3c7; padding: 3px 8px; border-radius: 4px; margin: 0 3px; border: 1px solid #fde68a;'>
                        🎮 Demo
                    </span>
                    <span style='background: #f3f4f6; padding: 3px 8px; border-radius: 4px; margin: 0 3px; border: 1px solid #e5e7eb;'>
                        🗄️ {settings.database_type.upper()}
                    </span>
                    <span style='opacity: 0.6; font-style: italic; margin-left: 6px; font-size: 9px;'>
                        (Set OPENAI_API_KEY for full LLM)
                    </span>
                </div>
                """
            gr.HTML(status_html)

            with gr.Tabs():
                # Tab 1: Chat Interface with 2-pane layout (MAIN TAB)
                with gr.Tab("💬 Chat"):
                    with gr.Row():
                        # LEFT PANE: Chat interface
                        with gr.Column(scale=5):
                            # Input controls
                            with gr.Row():
                                msg = gr.Textbox(
                                    label="Ask a question about your data",
                                    placeholder="e.g., Show me the top 5 customers by spending...",
                                    lines=2,
                                    scale=5,
                                )
                            with gr.Row():
                                submit_btn = gr.Button(
                                    "▶️ Run Query", variant="primary", scale=2, interactive=False, size="lg"
                                )
                                export_btn = gr.Button("📥 Export", variant="secondary", scale=1, size="sm")
                                clear = gr.Button("🗑️ Clear", variant="secondary", scale=1, size="sm")

                            export_file = gr.File(label="Download Results", visible=False)

                            # Example query chips
                            gr.Markdown("**💡 Quick Start:**")
                            with gr.Row():
                                ex1 = gr.Button("Top customers", size="sm")
                                ex2 = gr.Button("Revenue by category", size="sm")
                                ex3 = gr.Button("Sales per region", size="sm")
                                ex4 = gr.Button("Monthly trend", size="sm")
                            with gr.Row():
                                ex5 = gr.Button("Returning customers", size="sm")
                                ex6 = gr.Button("January products", size="sm")
                                ex7 = gr.Button("Large orders", size="sm")
                                ex8 = gr.Button("Inactive customers", size="sm")

                            # Conversation history
                            chatbot = gr.Chatbot(
                                label="Conversation",
                                height=500,
                                show_label=True,
                                autoscroll=True,  # Auto-scroll to latest message
                            )

                        # RIGHT PANE: Tabbed results
                        with gr.Column(scale=5):
                            with gr.Tabs():
                                with gr.Tab("📊 Results"):
                                    gr.Markdown("""
### Query Results

**👉 Run a query to see results here**

Charts appear automatically when your query returns:
- 📈 **Time series data** (revenue over time, monthly trends)
- 📊 **Category comparisons** (sales by region, products by category)
- 🔢 **Single metrics** (total count, sum, average)
                                    """)

                                    # Quick filters
                                    filter_section = gr.Markdown(
                                        value="",
                                        visible=False,
                                    )

                                    chart_output = gr.Plot(
                                        label="Visualization",
                                        show_label=False,
                                        container=True,
                                        elem_classes="chart-container",
                                    )

                                with gr.Tab("🔍 SQL"):
                                    gr.Markdown("""
### Generated SQL Query

**👉 Your optimized SQL will appear here**

What you'll see:
- ✅ **Syntax-validated SQL** optimized for SQLite
- 📝 **Explanation** of what the query does
- 🎯 **Performance suggestions** (if applicable)
- 💡 **Assumptions** made by the AI
                                    """)
                                    sql_output = gr.Code(
                                        label="SQL Code (click to copy)",
                                        language="sql",
                                        value="",
                                        lines=12,
                                    )

                                with gr.Tab("💡 Insights"):
                                    gr.Markdown("### AI-Powered Business Insights")
                                    insights_output = gr.Markdown(
                                        value="""
**No insights yet** — Run a query to get AI-generated analysis.

**What you'll see:**
- 📈 Trend detection and patterns
- 🎯 Key findings and anomalies
- 💼 Business recommendations
- 📊 Statistical summaries
                                        """,
                                    )

                                with gr.Tab("🗂️ History"):
                                    gr.Markdown("### Query History")
                                    history_output = gr.Markdown(
                                        value="*No queries yet. Start asking questions!*",
                                    )

                                with gr.Tab("🎯 Context"):
                                    gr.Markdown("### RAG Schema Retrieval")
                                    rag_output = gr.Markdown(
                                        value="""
**RAG context will appear here after a query.**

**How it works:**
- 🔍 Semantic search finds relevant tables/columns
- 🎯 FAISS vector database matches your question to schema elements
- ⚡ Only relevant schema is sent to LLM (faster, more accurate)
                                        """,
                                    )

                # Tab 2: Dashboard Overview
                with gr.Tab("📊 Dashboard"):
                    dashboard_view = gr.Markdown(
                        value=self._build_dashboard_overview(),
                        label="Dashboard",
                    )
                    refresh_dashboard = gr.Button("🔄 Refresh Stats", variant="secondary")

                # Tab 3: Schema Explorer
                with gr.Tab("📋 Schema & Data"):
                    gr.Markdown("## 🗄️ Database Schema")
                    gr.Markdown(self._build_schema_explorer_text())
                    gr.Markdown("## 📊 Sample Data Preview")
                    gr.Markdown(self._build_sample_data_text())

                # Tab 4: System Status
                with gr.Tab("⚙️ System Status"):
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

                # Return results + re-enable all interactive components + dashboard update
                return list(results) + [
                    gr.update(interactive=True),   # submit_btn
                    gr.update(interactive=True),   # ex1
                    gr.update(interactive=True),   # ex2
                    gr.update(interactive=True),   # ex3
                    gr.update(interactive=True),   # ex4
                    gr.update(interactive=True),   # ex5
                    gr.update(interactive=True),   # ex6
                    gr.update(interactive=True),   # ex7
                    gr.update(interactive=True),   # ex8
                    self._build_dashboard_overview(),  # dashboard_view
                ]

            # All outputs including button states, dashboard, and filters
            query_outputs = [
                msg, chatbot, chart_output, insights_output,
                history_output, rag_output, sql_output, filter_section,
                submit_btn, ex1, ex2, ex3, ex4, ex5, ex6, ex7, ex8, dashboard_view
            ]

            # Event handlers with loading state management
            msg.submit(
                lambda: [gr.update(interactive=False)] * 9,  # Disable submit + 8 example buttons
                outputs=[submit_btn, ex1, ex2, ex3, ex4, ex5, ex6, ex7, ex8],
                queue=False
            ).then(
                process_with_loading, [msg, chatbot], query_outputs
            )

            submit_btn.click(
                lambda: [gr.update(interactive=False)] * 9,  # Disable submit + 8 example buttons
                outputs=[submit_btn, ex1, ex2, ex3, ex4, ex5, ex6, ex7, ex8],
                queue=False
            ).then(
                process_with_loading, [msg, chatbot], query_outputs
            )

            def clear_chat():
                self.context_manager.reset()
                self._query_history.clear()
                return (
                    [], "", None,
                    """
**No insights yet** — Run a query to get AI-generated analysis.

**What you'll see:**
- 📈 Trend detection and patterns
- 🎯 Key findings and anomalies
- 💼 Business recommendations
- 📊 Statistical summaries
                    """,
                    "*No queries yet. Start asking questions!*",
                    """
**RAG context will appear here after a query.**

**How it works:**
- 🔍 Semantic search finds relevant tables/columns
- 🎯 FAISS vector database matches your question to schema elements
- ⚡ Only relevant schema is sent to LLM (faster, more accurate)
                    """,
                    "-- Run a query to see the generated SQL here\n-- The query will be optimized for your database",
                    "",  # filter_section
                )

            clear.click(clear_chat, outputs=[chatbot, msg, chart_output, insights_output, history_output, rag_output, sql_output, filter_section])

            # Dashboard refresh button
            refresh_dashboard.click(
                lambda: self._build_dashboard_overview(),
                outputs=[dashboard_view]
            )

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

            # Example query buttons: single handler to prevent race conditions
            def handle_example_click(query_text, chat_history):
                """Handle example button click: fill textbox and process query in one go"""
                # Process the query
                results = self.process_query(query_text, chat_history)

                # Return results + re-enable all buttons + dashboard
                # Results: msg(""), chatbot, chart, insights, history, rag, sql, filter_section (8 items)
                # But we want query_text for msg, so skip results[0]
                return [query_text] + list(results[1:]) + [
                    gr.update(interactive=True),   # submit_btn
                    gr.update(interactive=True),   # ex1
                    gr.update(interactive=True),   # ex2
                    gr.update(interactive=True),   # ex3
                    gr.update(interactive=True),   # ex4
                    gr.update(interactive=True),   # ex5
                    gr.update(interactive=True),   # ex6
                    gr.update(interactive=True),   # ex7
                    gr.update(interactive=True),   # ex8
                    self._build_dashboard_overview(),  # dashboard_view
                ]

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

            # Outputs: textbox first, then all query outputs EXCEPT msg (which is already first)
            # query_outputs[0] is msg, so skip it to avoid duplicate
            example_outputs = [msg] + query_outputs[1:]

            for btn, query in example_queries.items():
                # First: Immediately disable ALL buttons when ANY example is clicked
                btn.click(
                    fn=lambda: [
                        gr.update(interactive=False),  # msg
                        gr.update(interactive=False),  # submit_btn
                        gr.update(interactive=False),  # ex1
                        gr.update(interactive=False),  # ex2
                        gr.update(interactive=False),  # ex3
                        gr.update(interactive=False),  # ex4
                        gr.update(interactive=False),  # ex5
                        gr.update(interactive=False),  # ex6
                        gr.update(interactive=False),  # ex7
                        gr.update(interactive=False),  # ex8
                    ],
                    outputs=[msg, submit_btn, ex1, ex2, ex3, ex4, ex5, ex6, ex7, ex8],
                    queue=False  # Instant disable
                ).then(
                    # Second: Process query and re-enable all buttons
                    fn=lambda ch, q=query: handle_example_click(q, ch),
                    inputs=[chatbot],
                    outputs=example_outputs,
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
