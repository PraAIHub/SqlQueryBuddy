"""Gradio web interface for SQL Query Buddy"""
import gradio as gr
import os
from typing import Optional, Tuple
from src.config import settings
from src.components.executor import DatabaseConnection, QueryExecutor, SQLiteDatabase
from src.components.nlp_processor import ContextManager
from src.components.rag_system import RAGSystem, InMemoryVectorDB
from src.components.sql_generator import SQLGenerator, SQLGeneratorMock
from src.components.insights import InsightGenerator
from src.components.optimizer import QueryOptimizer


class QueryBuddyApp:
    """Main application class for SQL Query Buddy"""

    def __init__(self):
        # Initialize database
        self.db_url = settings.database_url
        self.db_connection = DatabaseConnection(self.db_url)
        self.query_executor = QueryExecutor(self.db_connection)

        # Initialize NLP and context management
        self.context_manager = ContextManager()

        # Initialize SQL generator
        if settings.openai_api_key and settings.openai_api_key != "":
            self.sql_generator = SQLGenerator(
                openai_api_key=settings.openai_api_key, model=settings.openai_model
            )
        else:
            self.sql_generator = SQLGeneratorMock()

        # Initialize RAG system
        schema = self.db_connection.get_schema()
        self.context_manager.initialize_with_schema(schema)

        # Initialize insights generator (mock if no API key)
        if settings.openai_api_key and settings.openai_api_key != "":
            self.insight_generator = InsightGenerator(
                openai_api_key=settings.openai_api_key, model=settings.openai_model
            )
        else:
            self.insight_generator = None

        # Initialize optimizer
        self.optimizer = QueryOptimizer()

    def process_query(
        self, user_message: str, chat_history: list
    ) -> Tuple[str, list]:
        """Process user query and return response"""
        try:
            # Parse user input
            parsed = self.context_manager.process_input(user_message)

            # Get schema context
            schema = self.db_connection.get_schema()
            schema_str = self._format_schema(schema)

            # Generate SQL
            result = self.sql_generator.generate(
                user_query=user_message,
                schema_context=schema_str,
                conversation_history=self.context_manager.get_full_context(),
            )

            if not result.get("success", False):
                error_msg = result.get("error", "Unknown error")
                response = f"❌ Error generating SQL: {error_msg}"
                chat_history.append((user_message, response))
                return "", chat_history

            generated_sql = result.get("generated_sql", "")

            # Execute query
            exec_result = self.query_executor.execute(generated_sql)

            if not exec_result.get("success", False):
                error_msg = exec_result.get("error", "Query execution failed")
                response = f"❌ Error executing query: {error_msg}"
                chat_history.append((user_message, response))
                return "", chat_history

            # Format response
            response_lines = [
                "✅ Query executed successfully!",
                "",
                f"**Generated SQL:**",
                f"```sql\n{generated_sql}\n```",
                "",
                f"**Explanation:** {result.get('explanation', 'N/A')}",
                "",
            ]

            # Add results
            row_count = exec_result.get("row_count", 0)
            response_lines.append(f"**Results:** {row_count} rows found")

            # Show first few rows
            data = exec_result.get("data", [])
            if data:
                response_lines.append("\n**Data Preview:**")
                # Simple table formatting
                if data:
                    headers = list(data[0].keys())
                    response_lines.append("|" + "|".join(headers) + "|")
                    response_lines.append("|" + "|".join(["---"] * len(headers)) + "|")
                    for row in data[:5]:  # Show first 5 rows
                        values = [str(row.get(h, "")) for h in headers]
                        response_lines.append("|" + "|".join(values) + "|")

                    if len(data) > 5:
                        response_lines.append(f"\n*(Showing 5 of {len(data)} rows)*")

            # Add optimization suggestions
            opt_result = self.optimizer.analyze(generated_sql)
            if opt_result.get("suggestions"):
                response_lines.append("\n**Optimization Suggestions:**")
                for i, suggestion in enumerate(opt_result["suggestions"], 1):
                    response_lines.append(
                        f"- {suggestion.get('suggestion', '')} *(severity: {suggestion.get('severity', 'low')})*"
                    )

            # Add insights if available
            if self.insight_generator and data:
                insights = self.insight_generator.generate_insights(data, user_message)
                response_lines.append(f"\n**AI Insights:** {insights}")

            # Update context
            response_text = "\n".join(response_lines)
            self.context_manager.add_response(
                user_input=user_message,
                assistant_response=response_text,
                generated_sql=generated_sql,
            )

            chat_history.append((user_message, response_text))
            return "", chat_history

        except Exception as e:
            error_response = f"❌ Unexpected error: {str(e)}"
            chat_history.append((user_message, error_response))
            return "", chat_history

    @staticmethod
    def _format_schema(schema: dict) -> str:
        """Format schema for LLM context"""
        lines = ["Database Schema:"]

        for table_name, table_info in schema.items():
            lines.append(f"\nTable: {table_name}")
            columns = table_info.get("columns", {})
            for col_name, col_info in columns.items():
                col_type = col_info.get("type", "unknown")
                lines.append(f"  - {col_name} ({col_type})")

        return "\n".join(lines)

    def create_interface(self) -> gr.Blocks:
        """Create Gradio interface"""
        with gr.Blocks(title="SQL Query Buddy", theme=gr.themes.Soft()) as demo:
            gr.Markdown("# 🤖 SQL Query Buddy")
            gr.Markdown(
                "Ask questions about your database in natural language and get SQL queries with results!"
            )

            with gr.Row():
                with gr.Column():
                    gr.Markdown("### How to use:")
                    gr.Markdown(
                        """
                    1. Ask a question about your data
                    2. The AI will generate and execute an SQL query
                    3. View results, explanations, and optimization tips
                    4. Continue the conversation for multi-turn queries
                    """
                    )

            with gr.Row():
                chatbot = gr.Chatbot(
                    label="Conversation",
                    height=500,
                    show_label=True,
                )

            with gr.Row():
                msg = gr.Textbox(
                    label="Your question",
                    placeholder="e.g., 'Show me the top 10 customers by spending'",
                    lines=2,
                )
                clear = gr.Button("Clear Chat")

            # Set up event handlers
            msg.submit(self.process_query, [msg, chatbot], [msg, chatbot])
            clear.click(lambda: ([], []), outputs=[chatbot, msg])

            gr.Examples(
                [
                    ["Show me all users"],
                    ["How many products are in stock?"],
                    ["What are the top 5 products by price?"],
                    ["Show me orders from the last month"],
                ],
                msg,
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

    demo.launch(
        server_name=settings.fastapi_host,
        server_port=settings.gradio_server_port,
        share=settings.gradio_share,
    )


if __name__ == "__main__":
    main()
