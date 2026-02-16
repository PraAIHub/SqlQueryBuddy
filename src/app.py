"""Gradio web interface for SQL Query Buddy"""
import gradio as gr
import csv
import io
import logging
import math
import os
import re
import tempfile
import time
import uuid
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
from src.components.sql_generator import SQLGenerator, SQLGeneratorMock, SQLValidator
from src.components.insights import InsightGenerator, LocalInsightGenerator
from src.components.optimizer import QueryOptimizer
from src.components.conversation_state import ConversationState, resolve_references
from src.components.sql_validator import (
    build_schema_whitelist,
    validate_sql_identifiers,
    build_fix_message,
)

logger = logging.getLogger(__name__)


def _get_git_commit() -> str:
    """Return short git commit hash, or 'unknown' if not in a repo."""
    import subprocess
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL, text=True,
        ).strip()
    except Exception:
        return "unknown"


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

        # Determine LLM mode from APP_MODE setting
        _placeholder_keys = {"your-api-key-here", "sk-xxx", "your_api_key", ""}
        _has_valid_key = bool(
            settings.openai_api_key
            and settings.openai_api_key.strip().lower() not in _placeholder_keys
            and settings.openai_api_key.startswith("sk-")
        )

        if settings.app_mode == "local":
            self.using_real_llm = False
            self._fallback_reason = "local mode (APP_MODE=local)"
        elif settings.app_mode == "openai":
            self.using_real_llm = _has_valid_key
            self._fallback_reason = "" if _has_valid_key else "no valid API key (APP_MODE=openai)"
        else:  # auto
            self.using_real_llm = _has_valid_key
            self._fallback_reason = "" if _has_valid_key else "no API key"

        # Structured startup logging
        _key_present = bool(
            settings.openai_api_key
            and settings.openai_api_key.strip().lower() not in _placeholder_keys
        )
        logger.info(
            "Startup config: app_mode=%s model=%s llm_enabled=%s timeout=%s key_present=%s debug_panel=%s",
            settings.app_mode,
            settings.openai_model,
            self.using_real_llm,
            settings.query_timeout_seconds,
            _key_present,
            settings.show_debug_panel,
        )

        # Initialize SQL generator (with mock fallback for API errors)
        self.mock_generator = SQLGeneratorMock()
        if self.using_real_llm:
            self.sql_generator = SQLGenerator(
                openai_api_key=settings.openai_api_key,
                model=settings.openai_model,
                timeout=settings.openai_timeout,
                max_retries=settings.openai_max_retries,
                base_url=settings.openai_base_url,
                max_tokens=settings.openai_max_tokens,
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
                openai_api_key=settings.openai_api_key,
                model=settings.openai_model,
                timeout=settings.openai_timeout,
                base_url=settings.openai_base_url,
            )
            logger.info(f"🤖 Using OpenAI {settings.openai_model} for AI insights")
        else:
            self.insight_generator = LocalInsightGenerator()
            logger.info("🔧 Using LocalInsightGenerator (demo mode - no OpenAI API key)")

        # Initialize optimizer
        self.optimizer = QueryOptimizer()

        # Cache schema to avoid re-fetching on every query
        self._cached_schema = self.db_connection.get_schema()

        # Build schema whitelist for SQL identifier validation
        self._all_tables, self._table_columns = build_schema_whitelist(self._cached_schema)

        # Conversation state for multi-turn context retention
        self.conv_state = ConversationState()

        # Store last results for export
        self._last_results = []
        self._last_sql = ""
        self._auto_fallback_active = False

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

    # Column name hints that indicate percentage values (checked BEFORE currency)
    PERCENT_HINTS = {
        "percent", "pct", "ratio", "share", "rate", "proportion", "percentage",
    }

    @staticmethod
    def _format_cell(column_name: str, value) -> str:
        """Format a cell value; apply $X,XXX.XX for currency columns."""
        col_lower = column_name.lower()
        # Exclude count-like columns even if they contain a currency hint word
        if any(ex in col_lower for ex in QueryBuddyApp.CURRENCY_EXCLUDE):
            return str(value)
        # Check percent hints BEFORE currency hints
        if any(hint in col_lower for hint in QueryBuddyApp.PERCENT_HINTS):
            try:
                fval = float(value)
                if math.isnan(fval):
                    return "N/A"
                return f"{fval:.2f}%"
            except (ValueError, TypeError):
                pass
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

        # Column names that indicate IDs / non-chartable data
        _id_hints = {"id", "uuid", "pk", "key", "email", "phone", "address", "password", "hash"}

        def _is_id_column(name: str) -> bool:
            nl = name.lower()
            return nl.endswith("_id") or nl in _id_hints or nl.startswith("id_")

        # Column names that indicate a meaningful metric (revenue, count, etc.)
        _metric_hints = {
            "revenue", "sales", "amount", "total", "count", "sum", "avg",
            "quantity", "profit", "price", "spent", "orders", "customers",
        }

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
                # Prefer metric columns; skip IDs
                if not _is_id_column(h) and numeric_col is None:
                    numeric_col = h
            elif isinstance(sample_val, str):
                try:
                    float(sample_val)
                    if not _is_id_column(h) and numeric_col is None:
                        numeric_col = h
                except (ValueError, TypeError):
                    if not _is_id_column(h) and categorical_col is None:
                        categorical_col = h

        # Fallback: use first non-numeric, non-ID column as categorical
        if not date_col and not categorical_col and len(headers) >= 2:
            for h in headers:
                if h != numeric_col and not _is_id_column(h):
                    categorical_col = h
                    break

        if numeric_col is None:
            return None

        # Gate: for bar charts, only chart if the metric looks meaningful
        # (skip raw row listings like "customer_id over signup_date")
        if not date_col and categorical_col:
            n_lower = numeric_col.lower()
            # If more than 15 unique categories, it's probably raw rows, not aggregates
            unique_cats = len(set(str(row.get(categorical_col, "")) for row in data))
            if unique_cats > 15 and not any(m in n_lower for m in _metric_hints):
                return None

        # Modern chart style
        fig, ax = plt.subplots(figsize=(8, 4))
        fig.patch.set_facecolor('#ffffff')
        ax.set_facecolor('#fafafa')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#e5e7eb')
        ax.spines['bottom'].set_color('#e5e7eb')
        ax.tick_params(colors='#6b7280', labelsize=9)

        # Friendly column labels for titles
        friendly_metric = numeric_col.replace('_', ' ').title()
        friendly_date = date_col.replace('_', ' ').title() if date_col else ""
        friendly_cat = categorical_col.replace('_', ' ').title() if categorical_col else ""

        if date_col:
            # Aggregate time-series if too many raw data points
            agg_data = self._aggregate_time_series(data, date_col, numeric_col)
            labels = [str(row.get(date_col, "")) for row in agg_data]
            values = []
            for row in agg_data:
                try:
                    values.append(float(row.get(numeric_col, 0)))
                except (ValueError, TypeError):
                    values.append(0)

            x = range(len(labels))
            ax.plot(x, values, marker="o", linewidth=2.5, color="#7c3aed",
                    markersize=4 if len(labels) > 15 else 5,
                    markerfacecolor="#ffffff", markeredgewidth=1.5, markeredgecolor="#7c3aed")
            ax.fill_between(x, values, alpha=0.08, color="#7c3aed")
            # Reduce x-axis tick clutter
            max_ticks = 10
            if len(labels) > max_ticks:
                step = math.ceil(len(labels) / max_ticks)
                tick_positions = list(range(0, len(labels), step))
                ax.set_xticks(tick_positions)
                ax.set_xticklabels([labels[i] for i in tick_positions], rotation=30, ha="right")
            else:
                ax.set_xticks(range(len(labels)))
                ax.set_xticklabels(labels, rotation=30, ha="right")
            ax.set_title(f"{friendly_metric} over {friendly_date}", fontsize=13, fontweight="bold", color="#1f2937", pad=12)
            ax.set_ylabel(friendly_metric, fontsize=10, color="#6b7280")
            ax.grid(axis="y", alpha=0.15, color="#d1d5db")
        elif categorical_col:
            rows = data[:20]
            values = []
            for row in rows:
                try:
                    values.append(float(row.get(numeric_col, 0)))
                except (ValueError, TypeError):
                    values.append(0)
            labels = [str(row.get(categorical_col, ""))[:20] for row in rows]
            bars = ax.barh(range(len(labels)), values, color="#7c3aed", height=0.6, edgecolor="none")
            ax.set_yticks(range(len(labels)))
            ax.set_yticklabels(labels, fontsize=10)
            ax.set_title(f"{friendly_metric} by {friendly_cat}", fontsize=13, fontweight="bold", color="#1f2937", pad=12)
            ax.set_xlabel(friendly_metric, fontsize=10, color="#6b7280")
            ax.invert_yaxis()
            ax.grid(axis="x", alpha=0.15, color="#d1d5db")
            # Value labels on bars
            for bar, val in zip(bars, values):
                ax.text(bar.get_width() + max(values) * 0.01, bar.get_y() + bar.get_height() / 2,
                        f'{val:,.0f}', va='center', fontsize=8, color='#6b7280')
        else:
            plt.close(fig)
            return None

        fig.tight_layout()
        return fig

    @staticmethod
    def _aggregate_time_series(data: list, date_col: str, numeric_col: str, max_points: int = 30) -> list:
        """Aggregate time-series data if too many raw points.

        Groups by the date column value and sums the numeric column,
        returning at most max_points rows for clean charting.
        """
        if len(data) <= max_points:
            return data

        # Group by date value (handles daily, monthly, etc.)
        from collections import OrderedDict
        groups: OrderedDict = OrderedDict()
        for row in data:
            key = str(row.get(date_col, ""))
            if key not in groups:
                groups[key] = 0.0
            try:
                groups[key] += float(row.get(numeric_col, 0))
            except (ValueError, TypeError):
                pass

        # If grouping reduced points enough, return grouped data
        if len(groups) <= max_points:
            return [{date_col: k, numeric_col: v} for k, v in groups.items()]

        # Still too many — sample evenly
        keys = list(groups.keys())
        step = math.ceil(len(keys) / max_points)
        sampled = keys[::step]
        return [{date_col: k, numeric_col: groups[k]} for k in sampled]

    def _generate_single_value_card(self, label: str, value: float) -> matplotlib.figure.Figure:
        """Generate a large number card for single-value results (COUNT, SUM, etc.)."""
        fig, ax = plt.subplots(figsize=(8, 4))
        fig.patch.set_facecolor('#ffffff')
        ax.axis('off')

        # Format the value with proper separators
        if abs(value) >= 1000000:
            display_value = f"{value/1000000:.1f}M"
        elif abs(value) >= 1000:
            display_value = f"{value/1000:.1f}K"
        elif value == int(value):
            display_value = f"{int(value):,}"
        else:
            display_value = f"{value:,.2f}"

        # Check if it's a percent or currency column
        label_lower = label.lower()
        is_excluded = any(ex in label_lower for ex in self.CURRENCY_EXCLUDE)
        is_percent = any(hint in label_lower for hint in self.PERCENT_HINTS)
        is_currency = any(hint in label_lower for hint in self.CURRENCY_HINTS)
        if not is_excluded and is_percent:
            display_value = f"{value:.2f}%"
        elif not is_excluded and is_currency:
            if abs(value) < 1000:
                display_value = f"${value:,.2f}"
            else:
                display_value = "$" + display_value

        # Rounded card background
        rect = plt.Rectangle(
            (0.1, 0.1), 0.8, 0.8,
            fill=True, facecolor='#f5f3ff',
            edgecolor='#e9d5ff', linewidth=1.5,
            transform=ax.transAxes, zorder=-1,
            clip_on=False,
        )
        rect.set_joinstyle('round')
        ax.add_patch(rect)

        # Draw large number in center
        ax.text(
            0.5, 0.58, display_value,
            ha='center', va='center',
            fontsize=56, fontweight='bold',
            color='#7c3aed',
            transform=ax.transAxes
        )

        # Draw label below
        formatted_label = label.replace('_', ' ').title()
        ax.text(
            0.5, 0.28, formatted_label,
            ha='center', va='center',
            fontsize=14, color='#6b7280',
            transform=ax.transAxes
        )

        fig.tight_layout()
        return fig

    def _format_history(self) -> str:
        """Format query history as markdown with timestamp, status, rows, latency."""
        if not self._query_history:
            return "*No queries yet.*"
        lines = []
        for i, entry in enumerate(reversed(self._query_history), 1):
            ts = entry.get("timestamp", "")
            status = entry.get("status", "success")
            latency = entry.get("latency_ms", 0)
            rows = entry.get("rows", 0)
            status_icon = {"success": "✅", "fallback": "⚠️", "error": "❌"}.get(status, "●")
            lines.append(
                f"**{i}.** {status_icon} {entry['query']}\n"
                f"   `{entry['sql'][:80]}{'...' if len(entry['sql']) > 80 else ''}`\n"
                f"   {rows} rows · {latency}ms · {ts}"
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

        # Block multi-statement injection attempts in user input
        # (semicolon immediately followed by a dangerous SQL keyword)
        _MULTI_STMT_INJECTION = re.compile(
            r";\s*(?:DROP|DELETE|TRUNCATE|ALTER|CREATE|INSERT|UPDATE|GRANT|REVOKE)\b",
            re.IGNORECASE,
        )
        if _MULTI_STMT_INJECTION.search(user_message):
            error_response = (
                "**Blocked:** Your query contains a potentially dangerous SQL pattern. "
                "Please rephrase your question in plain English."
            )
            chat_history.append({"role": "user", "content": user_message})
            chat_history.append({"role": "assistant", "content": error_response})
            return "", chat_history, None, "", self._format_history(), "", "", ""

        try:
            # Resolve pronoun/reference expressions using conversation state
            # e.g. "that region" → "the West region", "them" → actual customer names
            resolved_message = resolve_references(user_message, self.conv_state)

            # Parse user input with NLP (use resolved version for better intent detection)
            parsed = self.context_manager.process_input(resolved_message)
            parsed_query = parsed["parsed_query"]
            intent = parsed_query["intent"]
            entities = parsed_query["entities"]

            # Get schema context via RAG (semantic retrieval of relevant tables/columns)
            schema = self._cached_schema
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

            # Append active filters context so LLM carries forward constraints
            _filters_ctx = ""
            if self.conv_state.filters_applied:
                _filters_ctx = "\n\nActive filters from previous turns: " + ", ".join(
                    f"{k}={v}" for k, v in self.conv_state.filters_applied.items()
                )

            # Generate SQL (use resolved_message so references are concrete)
            conversation_ctx = self.context_manager.get_full_context()
            result = self.sql_generator.generate(
                user_query=resolved_message,
                schema_context=schema_str + _filters_ctx,
                conversation_history=conversation_ctx,
            )

            # Fallback to mock generator on API errors (quota, rate limit, auth)
            _error_category = result.get("error_category", "")
            if (
                not result.get("success", False)
                and self.using_real_llm
                and (
                    _error_category in (
                        "quota_exceeded", "rate_limited", "invalid_api_key",
                        "model_not_found", "timeout", "network_error", "unknown",
                    )
                    or any(
                        hint in result.get("error", "").lower()
                        for hint in ["429", "quota", "rate limit", "rate_limit", "401", "403", "authentication", "unauthorized", "invalid api key", "invalid_api_key"]
                    )
                )
            ):
                # Update fallback reason for UI status display
                _category_to_reason = {
                    "quota_exceeded": "quota exceeded",
                    "rate_limited": "rate limited",
                    "invalid_api_key": "invalid API key",
                    "model_not_found": "model not found",
                    "timeout": "timeout",
                    "network_error": "network error",
                }
                self._fallback_reason = _category_to_reason.get(
                    _error_category, _error_category or "API error"
                )
                logger.warning(
                    "SQL generation failed: reason=%s app_mode=%s",
                    self._fallback_reason,
                    settings.app_mode,
                )
                # Build actionable fix message per error category
                _fix_advice = {
                    "quota_exceeded": "Your OpenAI account has no remaining quota/credits. Add credit at platform.openai.com/account/billing or use a different key.",
                    "rate_limited": "Too many requests per minute. Wait a moment and try again.",
                    "invalid_api_key": "Your API key is invalid or revoked. Check the key in HuggingFace Settings > Secrets.",
                    "model_not_found": f"Model `{settings.openai_model}` is not available for this key. Check OPENAI_MODEL in HuggingFace Settings > Variables.",
                    "timeout": f"Request timed out after {settings.openai_timeout}s. Try again or increase OPENAI_TIMEOUT.",
                    "network_error": "Cannot reach OpenAI API. Check network connectivity.",
                }
                _advice = _fix_advice.get(_error_category, "Check your API key and billing in HuggingFace Settings.")

                # APP_MODE=openai: show error, no silent fallback
                if settings.app_mode == "openai":
                    error_msg = (
                        f"**OpenAI Error: {self._fallback_reason}**\n\n"
                        f"{_advice}\n\n"
                        "`APP_MODE=openai` — silent fallback is disabled."
                    )
                    chat_history.append({"role": "user", "content": user_message})
                    chat_history.append({"role": "assistant", "content": error_msg})
                    return "", chat_history, None, "", self._format_history(), "", "", ""

                # APP_MODE=auto: fallback to mock generator with visible banner
                self._auto_fallback_active = True
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

            # Schema-aware validation: catch invented columns BEFORE execution
            unknown_ids = validate_sql_identifiers(
                generated_sql, self._all_tables, self._table_columns
            )
            if unknown_ids:
                logger.info("Auto-correcting SQL: unknown identifiers=%s", unknown_ids)
                fix_msg = build_fix_message(unknown_ids, self._table_columns)
                fix_ctx = (
                    f"{schema_str}\n\n"
                    f"INVALID SQL (has unknown columns):\n{generated_sql}\n"
                    f"{fix_msg}\n"
                    f"Rewrite the SQL using ONLY valid columns from the schema above."
                )
                try:
                    fix_result = self.sql_generator.generate(
                        user_query=resolved_message,
                        schema_context=fix_ctx,
                        conversation_history=conversation_ctx,
                    )
                    if fix_result.get("success") and fix_result.get("generated_sql"):
                        generated_sql = fix_result["generated_sql"]
                        result["explanation"] = fix_result.get("explanation", result.get("explanation", ""))
                except Exception:
                    pass  # Fall through with original SQL

            # Pre-execution SQL safety gate: validate the final SQL
            # (auto-fix loop may have produced SQL that bypasses initial validation)
            is_safe, safety_err = SQLValidator.validate(generated_sql)
            if not is_safe:
                response = (
                    f"**SQL Blocked:** {safety_err}\n\n"
                    "The generated SQL did not pass safety validation. "
                    "Please rephrase your question."
                )
                chat_history.append({"role": "user", "content": user_message})
                chat_history.append({"role": "assistant", "content": response})
                return "", chat_history, None, "", self._format_history(), "", "", ""

            # Auto-LIMIT unbounded SELECT queries to prevent full table scans
            generated_sql = self.optimizer.auto_limit_sql(generated_sql)

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
                error_detail = exec_result.get("detail", "")

                # Attempt auto-fix: regenerate SQL using the error feedback
                fix_result = None
                if error_detail and hasattr(self, 'sql_generator'):
                    fix_ctx = (
                        f"{schema_str}\n\n"
                        f"PREVIOUS FAILED SQL:\n{generated_sql}\n"
                        f"ERROR: {error_detail}\n"
                        f"Fix the SQL to avoid this error."
                    )
                    try:
                        fix_result = self.sql_generator.generate(
                            user_query=user_message,
                            schema_context=fix_ctx,
                            conversation_history=conversation_ctx,
                        )
                        if fix_result.get("success") and fix_result.get("generated_sql", "") != generated_sql:
                            fixed_sql = fix_result["generated_sql"]
                            retry_exec = self.query_executor.execute(fixed_sql)
                            if retry_exec.get("success"):
                                # Auto-fix succeeded — continue with fixed results
                                generated_sql = fixed_sql
                                exec_result = retry_exec
                                logger.info("✅ Auto-fix succeeded on retry")
                            else:
                                fix_result = None  # Retry also failed
                        else:
                            fix_result = None
                    except Exception:
                        fix_result = None

                # If auto-fix didn't work, show error with collapsible details
                if not exec_result.get("success", False):
                    detail_block = ""
                    if error_detail:
                        detail_block = (
                            f"\n\n<details><summary>Show error details</summary>\n\n"
                            f"```\n{error_detail}\n```\n\n</details>"
                        )
                    response = (
                        f"**SQL Generated:**\n```sql\n{generated_sql}\n```\n\n"
                        f"**Execution Error:** The query could not be executed. "
                        f"Try rephrasing your question or check column names."
                        f"{detail_block}"
                    )
                    chat_history.append({"role": "user", "content": user_message})
                    chat_history.append({"role": "assistant", "content": response})
                    logger.info(
                        "RUN_SUMMARY mode=%s provider=%s status=exec_error exec_ms=%d",
                        settings.app_mode,
                        "openai" if self.using_real_llm else "local",
                        round(exec_ms),
                    )
                    return "", chat_history, None, "", self._format_history(), rag_context, generated_sql, ""

            # Track successful query
            self._stats["successful_queries"] += 1

            # Store results for export and history
            data = exec_result.get("data", [])
            self._last_results = data
            self._last_sql = generated_sql

            # Update conversation state with computed entities / filters
            if data:
                self.conv_state.update_from_results(user_message, generated_sql, data)
            _run_status = "fallback" if self._auto_fallback_active else "success"
            self._query_history.append({
                "query": user_message,
                "sql": generated_sql,
                "rows": exec_result.get("row_count", 0),
                "status": _run_status,
                "latency_ms": round(exec_ms),
                "timestamp": time.strftime("%H:%M:%S"),
            })
            # Cap history to last 50 entries
            if len(self._query_history) > 50:
                self._query_history = self._query_history[-50:]

            # Format response
            response_lines = []
            # Show fallback banner if auto-mode fallback occurred
            if self._auto_fallback_active:
                response_lines.append(
                    f"> **Fallback mode active** — OpenAI unavailable ({self._fallback_reason}). "
                    "SQL and insights are approximate. Set `APP_MODE=openai` to disable fallback.\n"
                )
            # Show reference resolution note if query was rewritten
            if resolved_message != user_message:
                response_lines.append(
                    f"> *Interpreted as:* {resolved_message}\n"
                )
            response_lines.extend([
                "**Generated SQL:**",
                f"```sql\n{generated_sql}\n```",
                "",
                f"**Explanation:** {result.get('explanation', 'N/A')}",
                "",
            ])

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

            # Sensitive column warning (PII / privacy)
            sensitive_warning = self.optimizer.check_sensitive_columns(generated_sql)
            if sensitive_warning:
                response_lines.append(
                    f"\n**Privacy Notice:** {sensitive_warning}"
                )

            # Add categorized optimization suggestions as colored callouts
            opt_result = self.optimizer.analyze(generated_sql, user_message)
            categorized = opt_result.get("categorized", {})

            if categorized.get("assumptions"):
                items = "".join(f"<li>{s['suggestion']}</li>" for s in categorized["assumptions"])
                response_lines.append(
                    f"\n<div style='border-left: 3px solid #9ca3af; padding: 6px 12px; margin: 8px 0; "
                    f"background: #f9fafb; border-radius: 0 6px 6px 0; font-size: 13px;'>"
                    f"<strong>Assumptions</strong><ul style='margin: 4px 0 0 0; padding-left: 18px;'>{items}</ul></div>"
                )

            if categorized.get("performance"):
                items = "".join(
                    f"<li>{s['suggestion']} <em style='color:#92400e;'>(severity: {s.get('severity', 'low')})</em></li>"
                    for s in categorized["performance"]
                )
                response_lines.append(
                    f"\n<div style='border-left: 3px solid #f59e0b; padding: 6px 12px; margin: 8px 0; "
                    f"background: #fffbeb; border-radius: 0 6px 6px 0; font-size: 13px;'>"
                    f"<strong>Performance</strong><ul style='margin: 4px 0 0 0; padding-left: 18px;'>{items}</ul></div>"
                )

            if categorized.get("next_steps"):
                items = "".join(f"<li>{s['suggestion']}</li>" for s in categorized["next_steps"])
                response_lines.append(
                    f"\n<div style='border-left: 3px solid #3b82f6; padding: 6px 12px; margin: 8px 0; "
                    f"background: #eff6ff; border-radius: 0 6px 6px 0; font-size: 13px;'>"
                    f"<strong>Next Steps</strong><ul style='margin: 4px 0 0 0; padding-left: 18px;'>{items}</ul></div>"
                )

            # Generate AI insights (displayed in dedicated panel)
            # Try LLM first, fall back to local generator on error
            try:
                insights_md = self.insight_generator.generate_insights(data, user_message)
                # If insights indicate API failure, try local fallback
                if "AI Insights unavailable" in insights_md and self.using_real_llm:
                    req_id = uuid.uuid4().hex[:8]
                    logger.warning(
                        "LLM insights failed req_id=%s -> falling back to local generator",
                        req_id,
                    )
                    self._fallback_reason = "LLM error"
                    local_gen = LocalInsightGenerator()
                    insights_md = local_gen.generate_insights(data, user_message)
                    logger.info("Local insights generated successfully as fallback req_id=%s", req_id)
            except Exception as e:
                req_id = uuid.uuid4().hex[:8]
                logger.warning(
                    "Insight generation failed req_id=%s error=%s, using local fallback",
                    req_id,
                    str(e)[:200],
                )
                self._fallback_reason = "insight error"
                local_gen = LocalInsightGenerator()
                insights_md = local_gen.generate_insights(data, user_message)
                logger.info("Local insights generated successfully as fallback req_id=%s", req_id)

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

            # Generate chart from results
            # Close stale figures first to prevent memory leaks
            plt.close('all')
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

            # Active context filters (judge-visible pills)
            active_filters_html = self.conv_state.get_active_filters_html()

            # Generate quick filter options
            filter_opts = self._detect_filter_options(data)
            filter_md = active_filters_html  # Start with active context pills
            if filter_opts:
                filter_md += "**🎛️ Quick Filters:** "
                for col_name, values in filter_opts.items():
                    filter_md += f"\n\n*{col_name}:* "
                    filter_md += " • ".join([f"`{v}`" for v in values[:5]])

            chat_history.append({"role": "user", "content": user_message})
            chat_history.append({"role": "assistant", "content": response_text})

            # Structured run summary (single log line for Space logs readability)
            _provider = "mock" if self._auto_fallback_active else ("openai" if self.using_real_llm else "local")
            logger.info(
                "RUN_SUMMARY mode=%s provider=%s model=%s fallback_used=%s "
                "status=success rows=%d exec_ms=%d chart=%s",
                settings.app_mode,
                _provider,
                settings.openai_model,
                self._auto_fallback_active,
                exec_result.get("row_count", 0),
                round(exec_ms),
                "yes" if chart else "no",
            )

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
        if self.using_real_llm:
            if self._fallback_reason:
                llm_status = f"Fallback: {self._fallback_reason}"
            else:
                llm_status = f"Connected ({settings.openai_model})"
        else:
            reason = self._fallback_reason or "no API key"
            llm_status = f"Mock (Fallback: {reason})"
        vector_db = "FAISS" if FAISS_AVAILABLE else "In-Memory"
        status = (
            f"| Component | Status |\n"
            f"|-----------|--------|\n"
            f"| Database | Connected ({settings.database_type}) |\n"
            f"| LLM Engine | {llm_status} |\n"
            f"| Vector DB | {vector_db} |\n"
            f"| RAG System | Active |\n"
        )
        if settings.show_debug_panel:
            _masked_key = "set" if settings.openai_api_key and settings.openai_api_key.startswith("sk-") else "not set"
            _commit = _get_git_commit()
            _base_url = settings.openai_base_url or "default (api.openai.com)"
            status += (
                f"\n### Debug Panel\n"
                f"| Setting | Value |\n"
                f"|---------|-------|\n"
                f"| APP_MODE | `{settings.app_mode}` |\n"
                f"| OPENAI_MODEL | `{settings.openai_model}` |\n"
                f"| OPENAI_BASE_URL | `{_base_url}` |\n"
                f"| OPENAI_TIMEOUT | `{settings.openai_timeout}s` |\n"
                f"| OPENAI_MAX_RETRIES | `{settings.openai_max_retries}` |\n"
                f"| LOG_LEVEL | `{settings.log_level}` |\n"
                f"| API Key | `{_masked_key}` |\n"
                f"| Query Timeout | `{settings.query_timeout_seconds}s` |\n"
                f"| Similarity Threshold | `{settings.similarity_threshold}` |\n"
                f"| Max Rows | `{settings.max_rows_return}` |\n"
                f"| Fallback Reason | `{self._fallback_reason or 'none'}` |\n"
                f"| Fallback Enabled | `{settings.app_mode != 'openai'}` |\n"
                f"| Build | `{_commit}` |\n"
            )
        return status

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
        with gr.Blocks(title="SQL Query Buddy") as demo:
            # Custom CSS for modern styling
            gr.HTML("""
            <style>
                /* Chart container max-height with scroll */
                .chart-container {
                    max-height: 500px;
                    overflow-y: auto;
                    overflow-x: hidden;
                }

                /* Secondary button styling - ghost/outline (Export, Clear only) */
                button.secondary:not(.quick-start-btn) {
                    background: transparent !important;
                    border: 1px solid #e5e7eb !important;
                    color: #9ca3af !important;
                    font-size: 12px !important;
                    min-height: 32px !important;
                    padding: 4px 12px !important;
                }
                button.secondary:not(.quick-start-btn):hover {
                    background: #f9fafb !important;
                    border-color: #9ca3af !important;
                    color: #6b7280 !important;
                }

                /* Quick Start chip buttons - override all Gradio variants */
                .quick-start-btn,
                .quick-start-btn button,
                .quick-start-btn button.secondary,
                .gradio-container .quick-start-btn button,
                button.quick-start-btn {
                    background: #f5f3ff !important;
                    border: 1px solid #c4b5fd !important;
                    color: #6d28d9 !important;
                    font-size: 12px !important;
                    font-weight: 500 !important;
                    min-height: 30px !important;
                    padding: 4px 12px !important;
                    border-radius: 16px !important;
                }
                .quick-start-btn:hover,
                .quick-start-btn button:hover,
                .quick-start-btn button.secondary:hover,
                button.quick-start-btn:hover {
                    background: #ede9fe !important;
                    border-color: #8b5cf6 !important;
                    color: #5b21b6 !important;
                }

                /* Consistent tab panel typography */
                .gradio-container .prose h3,
                .gradio-container .prose h4 {
                    color: #1f2937;
                    letter-spacing: -0.01em;
                }
                .gradio-container .prose p,
                .gradio-container .prose li {
                    color: #4b5563;
                    font-size: 14px;
                    line-height: 1.6;
                }
                .gradio-container .prose table {
                    font-size: 13px;
                }
            </style>
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
                    🔍 RAG + Query Advisor
                </div>
                <div style='font-size: 13px; color: #4b5563; line-height: 1.4;'>
                    FAISS semantic schema search + performance suggestions
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

            # Status chips HTML - rendered inside right panel header below
            if self.using_real_llm:
                status_html = f"""
                <div style='display: flex; align-items: center; gap: 6px; font-size: 10px; color: #9ca3af; letter-spacing: 0.3px; padding: 4px 0;'>
                    <span style='background: #f3f4f6; padding: 2px 7px; border-radius: 4px; border: 1px solid #e5e7eb; white-space: nowrap;'>
                        ✅ {settings.openai_model}
                    </span>
                    <span style='background: #f3f4f6; padding: 2px 7px; border-radius: 4px; border: 1px solid #e5e7eb; white-space: nowrap;'>
                        🗄️ {settings.database_type.upper()}
                    </span>
                    <span style='background: #f3f4f6; padding: 2px 7px; border-radius: 4px; border: 1px solid #e5e7eb; white-space: nowrap;'>
                        ⚡ FAISS
                    </span>
                </div>
                """
            else:
                _reason = self._fallback_reason or "no API key"
                status_html = f"""
                <div style='display: flex; align-items: center; gap: 6px; font-size: 10px; color: #9ca3af; letter-spacing: 0.3px; padding: 4px 0;'>
                    <span style='background: #fef3c7; padding: 2px 7px; border-radius: 4px; border: 1px solid #fde68a; white-space: nowrap;'>
                        Fallback: {_reason}
                    </span>
                    <span style='background: #f3f4f6; padding: 2px 7px; border-radius: 4px; border: 1px solid #e5e7eb; white-space: nowrap;'>
                        {settings.database_type.upper()}
                    </span>
                    <span style='opacity: 0.6; font-style: italic; font-size: 9px; white-space: nowrap;'>
                        (Set OPENAI_API_KEY for full LLM)
                    </span>
                </div>
                """

            with gr.Tabs():
                # Tab 1: Chat Interface with 2-pane layout (MAIN TAB)
                with gr.Tab("💬 Chat"):
                    with gr.Row():
                        # LEFT PANE: Chat interface (input at bottom, ChatGPT style)
                        with gr.Column(scale=5):
                            # Conversation history (at top)
                            chatbot = gr.Chatbot(
                                label="Conversation",
                                height=500,
                                show_label=True,
                                autoscroll=True,  # Auto-scroll to latest message
                            )

                            # Input controls (at bottom - ChatGPT style)
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

                            # Example query chips (at bottom)
                            gr.Markdown("**💡 Quick Start:**")
                            with gr.Row():
                                ex1 = gr.Button("Top customers", size="sm", elem_classes="quick-start-btn")
                                ex2 = gr.Button("Revenue by category", size="sm", elem_classes="quick-start-btn")
                                ex3 = gr.Button("Sales per region", size="sm", elem_classes="quick-start-btn")
                                ex4 = gr.Button("Monthly trend", size="sm", elem_classes="quick-start-btn")
                            with gr.Row():
                                ex5 = gr.Button("Returning customers", size="sm", elem_classes="quick-start-btn")
                                ex6 = gr.Button("January products", size="sm", elem_classes="quick-start-btn")
                                ex7 = gr.Button("Large orders", size="sm", elem_classes="quick-start-btn")
                                ex8 = gr.Button("Inactive customers", size="sm", elem_classes="quick-start-btn")

                        # RIGHT PANE: Tabbed results
                        with gr.Column(scale=5):
                            gr.HTML(status_html)

                            # Empty-state HTML (dynamic — hidden once data arrives)
                            RESULTS_EMPTY_HTML = """
<div style='text-align: center; padding: 48px 24px; color: #9ca3af;'>
    <div style='font-size: 40px; margin-bottom: 8px; opacity: 0.5;'>📊</div>
    <div style='font-size: 15px; font-weight: 600; color: #6b7280; margin-bottom: 6px;'>No results yet</div>
    <div style='font-size: 13px; line-height: 1.6; max-width: 320px; margin: 0 auto;'>
        Run a query to see charts here.<br>
        Time series, bar charts, and single-value cards render automatically.
    </div>
</div>"""
                            SQL_EMPTY_HTML = """
<div style='text-align: center; padding: 32px 24px; color: #9ca3af;'>
    <div style='font-size: 40px; margin-bottom: 8px; opacity: 0.5;'>🔍</div>
    <div style='font-size: 15px; font-weight: 600; color: #6b7280; margin-bottom: 6px;'>No SQL generated yet</div>
    <div style='font-size: 13px; max-width: 300px; margin: 0 auto;'>
        Your optimized SQL query will appear here after you ask a question.
    </div>
</div>"""

                            with gr.Tabs():
                                with gr.Tab("📊 Results"):
                                    results_empty = gr.HTML(value=RESULTS_EMPTY_HTML)

                                    # Quick filters
                                    filter_section = gr.Markdown(value="")

                                    chart_output = gr.Plot(
                                        label="Visualization",
                                        show_label=False,
                                        container=True,
                                        elem_classes="chart-container",
                                    )

                                with gr.Tab("🔍 SQL"):
                                    sql_empty = gr.HTML(value=SQL_EMPTY_HTML)
                                    sql_output = gr.Code(
                                        label="SQL Code",
                                        language="sql",
                                        value="",
                                        lines=12,
                                    )
                                    copy_sql_btn = gr.Button("📋 Copy SQL", size="sm", variant="secondary")
                                    copy_sql_btn.click(
                                        None, [sql_output], None,
                                        js="(sql) => { navigator.clipboard.writeText(sql || ''); }"
                                    )

                                with gr.Tab("💡 Insights"):
                                    insights_output = gr.HTML(
                                        value="""
<div style='text-align: center; padding: 48px 24px; color: #9ca3af;'>
    <div style='font-size: 40px; margin-bottom: 8px; opacity: 0.5;'>💡</div>
    <div style='font-size: 15px; font-weight: 600; color: #6b7280; margin-bottom: 6px;'>No insights yet</div>
    <div style='font-size: 13px; line-height: 1.6; max-width: 320px; margin: 0 auto;'>
        Run a query to get AI-generated analysis.<br>
        Trends, anomalies, key metrics, and recommendations.
    </div>
</div>
                                        """,
                                    )

                                with gr.Tab("🗂️ History"):
                                    history_output = gr.Markdown(
                                        value="""
<div style='text-align: center; padding: 48px 24px; color: #9ca3af;'>
    <div style='font-size: 40px; margin-bottom: 8px; opacity: 0.5;'>🗂️</div>
    <div style='font-size: 15px; font-weight: 600; color: #6b7280; margin-bottom: 6px;'>No history yet</div>
    <div style='font-size: 13px;'>Your query history will appear here as you explore.</div>
</div>
                                        """,
                                    )

                                with gr.Tab("🎯 Context"):
                                    rag_output = gr.Markdown(
                                        value="""
<div style='text-align: center; padding: 48px 24px; color: #9ca3af;'>
    <div style='font-size: 40px; margin-bottom: 8px; opacity: 0.5;'>🎯</div>
    <div style='font-size: 15px; font-weight: 600; color: #6b7280; margin-bottom: 6px;'>No context retrieved yet</div>
    <div style='font-size: 13px; line-height: 1.6; max-width: 320px; margin: 0 auto;'>
        RAG schema retrieval results will appear here.<br>
        Shows which tables and columns were matched to your query.
    </div>
</div>
                                        """,
                                    )

                # Tab 2: Dashboard Overview
                with gr.Tab("📊 Dashboard"):
                    dashboard_view = gr.Markdown(
                        value=self._build_dashboard_overview(),
                        label="Dashboard",
                    )
                    refresh_dashboard = gr.Button("🔄 Refresh Stats", variant="secondary")

                # Tab 3: Schema Explorer (2-column layout)
                with gr.Tab("📋 Schema & Data"):
                    with gr.Row():
                        with gr.Column(scale=1):
                            gr.Markdown("## 🗄️ Database Schema")
                            gr.Markdown(self._build_schema_explorer_text())
                        with gr.Column(scale=1):
                            gr.Markdown("## 📊 Sample Data Preview")
                            gr.Markdown(self._build_sample_data_text())

                # Tab 4: About & Status
                with gr.Tab("ℹ️ About"):
                    gr.HTML("""
<div style='max-width: 720px; margin: 0 auto;'>
    <h2 style='margin-bottom: 4px;'>SQL Query Buddy</h2>
    <p style='color: #6b7280; margin-top: 0; font-size: 14px;'>
        Conversational AI for Smart Data Insights &mdash; Built for the Codecademy GenAI Bootcamp Contest
    </p>

    <h3 style='margin-bottom: 8px;'>The Problem</h3>
    <div style='background: #fef2f2; border-left: 3px solid #ef4444; padding: 12px 14px; border-radius: 0 8px 8px 0; margin-bottom: 14px; font-size: 13px; color: #374151; line-height: 1.6;'>
        Data analysts spend <strong>60% of their time</strong> writing SQL queries instead of
        analyzing insights. Non-technical stakeholders can&rsquo;t access data without engineering
        support, creating bottlenecks that delay decisions by days.
    </div>

    <h3 style='margin-bottom: 8px;'>Our Solution</h3>
    <div style='background: #f0fdf4; border-left: 3px solid #22c55e; padding: 12px 14px; border-radius: 0 8px 8px 0; margin-bottom: 14px; font-size: 13px; color: #374151; line-height: 1.6;'>
        SQL Query Buddy lets <strong>anyone</strong> ask questions in plain English and get
        optimized SQL, visual charts, and AI-generated business insights &mdash; all in
        under 5 seconds. <strong>Zero SQL knowledge required.</strong>
    </div>

    <h3 style='margin-bottom: 8px;'>How It Works (RAG Pipeline)</h3>
    <div style='background: #f9fafb; border: 1px solid #e5e7eb; border-radius: 8px; padding: 14px; margin-bottom: 6px; font-size: 12px; font-family: monospace; color: #374151; line-height: 1.8;'>
        <strong style='color: #7c3aed;'>User Query</strong> &rarr; NLP (intent + entities)<br/>
        &nbsp;&nbsp;&darr;<br/>
        <strong style='color: #7c3aed;'>TF-IDF Embedding</strong> &rarr; FAISS Cosine Similarity Search<br/>
        &nbsp;&nbsp;&darr;<br/>
        <strong style='color: #7c3aed;'>Retrieved Schema</strong>: top-5 relevant tables &amp; columns (with confidence scores)<br/>
        &nbsp;&nbsp;&darr;<br/>
        <strong style='color: #7c3aed;'>LangChain + GPT-4o-mini</strong> &rarr; SQL generation from schema context + conversation history<br/>
        &nbsp;&nbsp;&darr;<br/>
        <strong style='color: #7c3aed;'>Validate &rarr; Execute &rarr; Chart &rarr; AI Insights</strong>
    </div>
    <p style='font-size: 12px; color: #6b7280; margin-top: 0; margin-bottom: 14px;'>
        <strong>Why RAG matters:</strong> Instead of sending the entire database schema to the LLM,
        FAISS vector search retrieves only the relevant tables and columns. This reduces hallucinated
        table names, cuts token costs, and improves SQL accuracy.
    </p>

    <div style='display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-bottom: 16px;'>
        <div style='background: #f9fafb; border-radius: 8px; padding: 12px; border: 1px solid #e5e7eb;'>
            <div style='font-weight: 600; font-size: 13px; color: #7c3aed; margin-bottom: 4px;'>1. Ask</div>
            <div style='font-size: 12px; color: #4b5563;'>Type a question in plain English. NLP extracts intent &amp; entities.</div>
        </div>
        <div style='background: #f9fafb; border-radius: 8px; padding: 12px; border: 1px solid #e5e7eb;'>
            <div style='font-weight: 600; font-size: 13px; color: #7c3aed; margin-bottom: 4px;'>2. Retrieve</div>
            <div style='font-size: 12px; color: #4b5563;'>FAISS vector search finds relevant tables &amp; columns (RAG).</div>
        </div>
        <div style='background: #f9fafb; border-radius: 8px; padding: 12px; border: 1px solid #e5e7eb;'>
            <div style='font-weight: 600; font-size: 13px; color: #7c3aed; margin-bottom: 4px;'>3. Generate</div>
            <div style='font-size: 12px; color: #4b5563;'>LangChain + LLM writes optimized SQL from the schema context.</div>
        </div>
        <div style='background: #f9fafb; border-radius: 8px; padding: 12px; border: 1px solid #e5e7eb;'>
            <div style='font-weight: 600; font-size: 13px; color: #7c3aed; margin-bottom: 4px;'>4. Analyze</div>
            <div style='font-size: 12px; color: #4b5563;'>Results are charted, insights generated, and optimizations suggested.</div>
        </div>
    </div>

    <h3 style='margin-bottom: 8px;'>Safety &amp; Guardrails</h3>
    <div style='background: #fefce8; border: 1px solid #fde68a; border-radius: 8px; padding: 12px; margin-bottom: 14px; font-size: 12px; color: #374151;'>
        <div style='display: grid; grid-template-columns: 1fr 1fr; gap: 6px;'>
            <div>&#x1f6e1;&#xfe0f; <strong>Read-only DB</strong> &mdash; query_only=ON</div>
            <div>&#x1f6ab; <strong>SQL Validation</strong> &mdash; DROP/DELETE/ALTER blocked</div>
            <div>&#x1f50d; <strong>Comment Stripping</strong> &mdash; Prevents bypass via SQL comments</div>
            <div>&#x1f9f9; <strong>Prompt Sanitization</strong> &mdash; Injection markers stripped</div>
            <div>&#x23f1;&#xfe0f; <strong>Timeout Protection</strong> &mdash; 30s query limit</div>
            <div>&#x1f4ca; <strong>Row Limits</strong> &mdash; Max 1,000 rows per result</div>
        </div>
    </div>

    <h3 style='margin-bottom: 8px;'>Key Features</h3>
    <table style='width: 100%; border-collapse: collapse; font-size: 13px; margin-bottom: 16px;'>
        <tr style='border-bottom: 1px solid #e5e7eb;'>
            <td style='padding: 6px 8px; font-weight: 600; color: #374151;'>SQL Generation</td>
            <td style='padding: 6px 8px; color: #6b7280;'>LangChain + LLM with auto-fix retry on errors</td>
        </tr>
        <tr style='border-bottom: 1px solid #e5e7eb;'>
            <td style='padding: 6px 8px; font-weight: 600; color: #374151;'>RAG Retrieval</td>
            <td style='padding: 6px 8px; color: #6b7280;'>FAISS vector DB with TF-IDF embeddings for semantic schema matching</td>
        </tr>
        <tr style='border-bottom: 1px solid #e5e7eb;'>
            <td style='padding: 6px 8px; font-weight: 600; color: #374151;'>AI Insights</td>
            <td style='padding: 6px 8px; color: #6b7280;'>Trends, anomalies, top performers, concentration risk (LLM + local fallback)</td>
        </tr>
        <tr style='border-bottom: 1px solid #e5e7eb;'>
            <td style='padding: 6px 8px; font-weight: 600; color: #374151;'>Query Advisor</td>
            <td style='padding: 6px 8px; color: #6b7280;'>Performance suggestions, assumptions, and next-step recommendations</td>
        </tr>
        <tr style='border-bottom: 1px solid #e5e7eb;'>
            <td style='padding: 6px 8px; font-weight: 600; color: #374151;'>Context Memory</td>
            <td style='padding: 6px 8px; color: #6b7280;'>Multi-turn conversations with structured QueryPlan tracking</td>
        </tr>
        <tr>
            <td style='padding: 6px 8px; font-weight: 600; color: #374151;'>Visualization</td>
            <td style='padding: 6px 8px; color: #6b7280;'>Auto-charts: time series, bar charts, single-value metric cards</td>
        </tr>
    </table>

    <h3 style='margin-bottom: 8px;'>Demo Database</h3>
    <p style='font-size: 12px; color: #6b7280; margin-bottom: 14px;'>
        Pre-loaded with a retail commerce dataset: <strong>150 customers</strong>,
        <strong>25 products</strong> (5 categories), <strong>2,500 orders</strong>,
        and <strong>~6,500 order items</strong> spanning 2023&ndash;2025.
        All Quick Start queries work against this data.
    </p>

    <h3 style='margin-bottom: 8px;'>Tech Stack</h3>
    <div style='display: flex; flex-wrap: wrap; gap: 6px; margin-bottom: 16px;'>
        <span style='background: #ede9fe; color: #5b21b6; padding: 3px 10px; border-radius: 12px; font-size: 11px; font-weight: 500;'>Python</span>
        <span style='background: #dbeafe; color: #1e40af; padding: 3px 10px; border-radius: 12px; font-size: 11px; font-weight: 500;'>Gradio</span>
        <span style='background: #dcfce7; color: #166534; padding: 3px 10px; border-radius: 12px; font-size: 11px; font-weight: 500;'>LangChain</span>
        <span style='background: #fef3c7; color: #92400e; padding: 3px 10px; border-radius: 12px; font-size: 11px; font-weight: 500;'>GPT-4o-mini</span>
        <span style='background: #e0e7ff; color: #3730a3; padding: 3px 10px; border-radius: 12px; font-size: 11px; font-weight: 500;'>FAISS</span>
        <span style='background: #f3f4f6; color: #374151; padding: 3px 10px; border-radius: 12px; font-size: 11px; font-weight: 500;'>SQLite</span>
        <span style='background: #fce7f3; color: #9d174d; padding: 3px 10px; border-radius: 12px; font-size: 11px; font-weight: 500;'>Matplotlib</span>
        <span style='background: #ccfbf1; color: #115e59; padding: 3px 10px; border-radius: 12px; font-size: 11px; font-weight: 500;'>Docker</span>
    </div>
</div>
                    """)
                    gr.Markdown("### System Status")
                    gr.Markdown(self._build_status_text())
                    gr.Markdown(
                        "### Deploy Notes\n"
                        "- After changing HF **Variables/Secrets**, use **Settings > Restart this Space**.\n"
                        "- Use **Factory reboot** only if `requirements.txt` changed.\n"
                        "- Set `APP_MODE=openai` to see real errors (no silent fallback).\n"
                        "- Set `SHOW_DEBUG_PANEL=true` to see full config above.\n"
                    )

            # Hidden textbox to trigger scroll via JavaScript
            scroll_trigger = gr.Textbox(visible=False, value="0")

            # Loading card HTML for right panel during processing
            LOADING_CARD_HTML = """
<div style='display: flex; flex-direction: column; align-items: center; justify-content: center;
            padding: 48px 24px;'>
    <div style='width: 48px; height: 48px; border: 3px solid #e9d5ff; border-top-color: #7c3aed;
                border-radius: 50%; animation: spin 0.8s linear infinite; margin-bottom: 16px;'></div>
    <div style='font-weight: 600; font-size: 15px; color: #374151; margin-bottom: 10px;'>Analyzing your query</div>
    <div style='display: flex; gap: 6px; align-items: center; font-size: 12px; color: #9ca3af;'>
        <span style='background: #f5f3ff; padding: 3px 8px; border-radius: 10px; border: 1px solid #e9d5ff;'>SQL</span>
        <span>&rarr;</span>
        <span style='background: #f5f3ff; padding: 3px 8px; border-radius: 10px; border: 1px solid #e9d5ff;'>Execute</span>
        <span>&rarr;</span>
        <span style='background: #f5f3ff; padding: 3px 8px; border-radius: 10px; border: 1px solid #e9d5ff;'>Chart</span>
        <span>&rarr;</span>
        <span style='background: #f5f3ff; padding: 3px 8px; border-radius: 10px; border: 1px solid #e9d5ff;'>Insights</span>
    </div>
</div>
<style>@keyframes spin { to { transform: rotate(360deg); } }</style>
"""

            # Wrapper function to handle loading states
            def process_with_loading(user_message, chat_history):
                """Process query and manage button states during execution"""
                # Call the actual process_query function
                results = self.process_query(user_message, chat_history)

                # Generate timestamp for scroll trigger
                import time
                scroll_timestamp = str(time.time())

                # Hide empty-state placeholders when data is present
                # results[2] = chart, results[6] = generated_sql
                _has_chart = results[2] is not None
                _has_sql = bool(results[6])

                # Return results + empty states + re-enable buttons + dashboard + scroll
                return list(results) + [
                    "" if _has_chart else RESULTS_EMPTY_HTML,  # results_empty
                    "" if _has_sql else SQL_EMPTY_HTML,        # sql_empty
                    gr.update(interactive=True),   # submit_btn
                    gr.update(interactive=True),   # export_btn
                    gr.update(interactive=True),   # clear
                    gr.update(interactive=True),   # ex1
                    gr.update(interactive=True),   # ex2
                    gr.update(interactive=True),   # ex3
                    gr.update(interactive=True),   # ex4
                    gr.update(interactive=True),   # ex5
                    gr.update(interactive=True),   # ex6
                    gr.update(interactive=True),   # ex7
                    gr.update(interactive=True),   # ex8
                    self._build_dashboard_overview(),  # dashboard_view
                    scroll_timestamp,  # scroll_trigger - triggers JS on change
                ]

            # All outputs including empty states, button states, dashboard, filters, and scroll trigger
            query_outputs = [
                msg, chatbot, chart_output, insights_output,
                history_output, rag_output, sql_output, filter_section,
                results_empty, sql_empty,
                submit_btn, export_btn, clear, ex1, ex2, ex3, ex4, ex5, ex6, ex7, ex8, dashboard_view,
                scroll_trigger
            ]

            # JavaScript to scroll chatbot to bottom (triggered by scroll_trigger change)
            scroll_js = """
            function() {
                setTimeout(function() {
                    console.log('🔄 Scroll trigger activated');
                    const selectors = [
                        'div[aria-label="chatbot conversation"]',
                        '.bubble-wrap',
                        '.message-wrap',
                        'div[role="log"]'
                    ];
                    let found = false;
                    for (let selector of selectors) {
                        const elements = document.querySelectorAll(selector);
                        elements.forEach(el => {
                            if (el) {
                                el.scrollTop = el.scrollHeight;
                                console.log('✅ Scrolled to bottom using:', selector, 'scrollTop:', el.scrollTop);
                                found = true;
                            }
                        });
                        if (found) break;
                    }
                    if (!found) {
                        console.log('❌ No scrollable chatbot element found');
                    }
                }, 150);
                return null;
            }
            """

            # Attach scroll JavaScript to scroll_trigger changes
            scroll_trigger.change(None, None, None, js=scroll_js)

            # Event handlers with loading state management
            # Disable all interactive buttons (submit + export + clear + 8 examples = 11)
            all_buttons = [submit_btn, export_btn, clear, ex1, ex2, ex3, ex4, ex5, ex6, ex7, ex8]

            def show_loading():
                """Show loading card in right panel immediately."""
                return LOADING_CARD_HTML, LOADING_CARD_HTML

            msg.submit(
                lambda: [gr.update(interactive=False)] * 11,
                outputs=all_buttons,
                queue=False
            ).then(
                show_loading, outputs=[insights_output, filter_section]
            ).then(
                process_with_loading, [msg, chatbot], query_outputs
            )

            submit_btn.click(
                lambda: [gr.update(interactive=False)] * 11,
                outputs=all_buttons,
                queue=False
            ).then(
                show_loading, outputs=[insights_output, filter_section]
            ).then(
                process_with_loading, [msg, chatbot], query_outputs
            )

            # Empty state HTML cards (reused in clear_chat)
            EMPTY_INSIGHTS = """
<div style='text-align: center; padding: 48px 24px; color: #9ca3af;'>
    <div style='font-size: 40px; margin-bottom: 8px; opacity: 0.5;'>💡</div>
    <div style='font-size: 15px; font-weight: 600; color: #6b7280; margin-bottom: 6px;'>No insights yet</div>
    <div style='font-size: 13px; line-height: 1.6; max-width: 320px; margin: 0 auto;'>
        Run a query to get AI-generated analysis.<br>
        Trends, anomalies, key metrics, and recommendations.
    </div>
</div>
"""
            EMPTY_HISTORY = """
<div style='text-align: center; padding: 48px 24px; color: #9ca3af;'>
    <div style='font-size: 40px; margin-bottom: 8px; opacity: 0.5;'>🗂️</div>
    <div style='font-size: 15px; font-weight: 600; color: #6b7280; margin-bottom: 6px;'>No history yet</div>
    <div style='font-size: 13px;'>Your query history will appear here as you explore.</div>
</div>
"""
            EMPTY_CONTEXT = """
<div style='text-align: center; padding: 48px 24px; color: #9ca3af;'>
    <div style='font-size: 40px; margin-bottom: 8px; opacity: 0.5;'>🎯</div>
    <div style='font-size: 15px; font-weight: 600; color: #6b7280; margin-bottom: 6px;'>No context retrieved yet</div>
    <div style='font-size: 13px; line-height: 1.6; max-width: 320px; margin: 0 auto;'>
        RAG schema retrieval results will appear here.<br>
        Shows which tables and columns were matched to your query.
    </div>
</div>
"""

            def clear_chat():
                self.context_manager.reset()
                self.conv_state.reset()
                self._query_history.clear()
                return (
                    [], "", None,
                    EMPTY_INSIGHTS,
                    EMPTY_HISTORY,
                    EMPTY_CONTEXT,
                    "",
                    "",  # filter_section
                    RESULTS_EMPTY_HTML,  # results_empty restored
                    SQL_EMPTY_HTML,      # sql_empty restored
                )

            clear.click(clear_chat, outputs=[chatbot, msg, chart_output, insights_output, history_output, rag_output, sql_output, filter_section, results_empty, sql_empty])

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

                # Generate timestamp for scroll trigger
                import time
                scroll_timestamp = str(time.time())

                # Hide empty-state placeholders when data is present
                _has_chart = results[2] is not None
                _has_sql = bool(results[6])

                # Return results + empty states + re-enable all buttons + dashboard + scroll trigger
                # results[0]=msg, [1]=chatbot, [2]=chart, [3]=insights, [4]=history, [5]=rag, [6]=sql, [7]=filter
                # Clear textbox (query already visible in chat history)
                return [gr.update(value="", interactive=True)] + list(results[1:]) + [
                    "" if _has_chart else RESULTS_EMPTY_HTML,  # results_empty
                    "" if _has_sql else SQL_EMPTY_HTML,        # sql_empty
                    gr.update(interactive=True),   # submit_btn
                    gr.update(interactive=True),   # export_btn
                    gr.update(interactive=True),   # clear
                    gr.update(interactive=True),   # ex1
                    gr.update(interactive=True),   # ex2
                    gr.update(interactive=True),   # ex3
                    gr.update(interactive=True),   # ex4
                    gr.update(interactive=True),   # ex5
                    gr.update(interactive=True),   # ex6
                    gr.update(interactive=True),   # ex7
                    gr.update(interactive=True),   # ex8
                    self._build_dashboard_overview(),  # dashboard_view
                    scroll_timestamp,  # scroll_trigger
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
                    fn=lambda: [gr.update(interactive=False)] * 12,
                    outputs=[msg] + all_buttons,
                    queue=False  # Instant disable
                ).then(
                    show_loading, outputs=[insights_output, filter_section]
                ).then(
                    # Process query and re-enable all buttons (scroll_trigger updated here)
                    fn=lambda ch, q=query: handle_example_click(q, ch),
                    inputs=[chatbot],
                    outputs=example_outputs,
                )

        return demo


def create_sample_db():
    """Create sample database if it doesn't exist or is empty"""
    db_path = settings.database_url.replace("sqlite:///", "")
    if not os.path.exists(db_path) or os.path.getsize(db_path) == 0:
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
    # Prefer passing theme to launch() (Gradio 6+); fall back to Blocks if unsupported
    try:
        demo.launch(
            server_name=settings.server_host,
            server_port=settings.gradio_server_port,
            share=settings.gradio_share,
            theme=gr.themes.Soft(),
        )
    except TypeError:
        # Older Gradio versions don't accept theme in launch()
        demo.launch(
            server_name=settings.server_host,
            server_port=settings.gradio_server_port,
            share=settings.gradio_share,
        )


if __name__ == "__main__":
    main()
