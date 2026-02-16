"""Schema-aware SQL identifier validation.

Checks generated SQL against the actual database schema to catch
invented columns/tables BEFORE execution.  Returns a list of unknown
identifiers so the auto-fix loop can correct them.
"""

import re
import logging
from typing import Dict, List, Set, Tuple, Any

logger = logging.getLogger(__name__)

# Common SQL aliases for aggregate expressions — not real columns
_AGGREGATE_ALIASES = {
    "total_spent", "total_sales", "total_revenue", "total_orders",
    "avg_order_value", "monthly_revenue", "order_count", "item_count",
    "unique_products_sold", "unique_products", "customer_count",
    "avg_price", "min_price", "max_price", "total_amount_sum",
    "total_quantity", "prev_result", "sub", "t", "t1", "t2",
    "month", "year", "quarter",
}

# Common mistaken column names → correct mapping
COLUMN_ALIASES: Dict[str, str] = {
    "revenue": "total_amount",          # orders.total_amount
    "sales": "total_amount",            # orders.total_amount
    "amount": "total_amount",           # orders.total_amount
    "product_name": "name",             # products.name
    "customer_name": "name",            # customers.name
    "order_total": "total_amount",      # orders.total_amount
    "item_total": "subtotal",           # order_items.subtotal
    "unit_price": "price",              # products.price
    "date": "order_date",              # orders.order_date
    "purchase_date": "order_date",     # orders.order_date
}


def build_schema_whitelist(schema: Dict[str, Any]) -> Tuple[Set[str], Dict[str, Set[str]]]:
    """Build table and column whitelists from the runtime schema.

    Returns:
        (all_tables, table_columns) where table_columns maps
        table_name → set of column names.
    """
    all_tables: Set[str] = set()
    table_columns: Dict[str, Set[str]] = {}

    for table_name, table_info in schema.items():
        t_lower = table_name.lower()
        all_tables.add(t_lower)
        cols = set()
        for col_name in table_info.get("columns", {}):
            cols.add(col_name.lower())
        table_columns[t_lower] = cols

    return all_tables, table_columns


def validate_sql_identifiers(
    sql: str,
    all_tables: Set[str],
    table_columns: Dict[str, Set[str]],
) -> List[str]:
    """Check SQL for unknown table or column identifiers.

    Returns a list of unknown identifiers (empty = valid).
    """
    # Flatten all known columns across all tables
    all_columns: Set[str] = set()
    for cols in table_columns.values():
        all_columns.update(cols)

    # Also allow common aggregate aliases
    all_known = all_columns | all_tables | _AGGREGATE_ALIASES

    unknown: List[str] = []

    # Extract identifiers: table.column patterns and bare identifiers
    sql_clean = _strip_strings_and_comments(sql)

    # Resolve table aliases (e.g. "customers c" → c→customers, "orders o" → o→orders)
    alias_map: Dict[str, str] = {}
    for m in re.finditer(
        r"\b(?:FROM|JOIN)\s+(\w+)\s+(?:AS\s+)?(\w+)\b", sql_clean, re.IGNORECASE
    ):
        table_name = m.group(1).lower()
        alias = m.group(2).lower()
        if table_name in all_tables and alias not in _SQL_KEYWORDS:
            alias_map[alias] = table_name

    # Find table.column references and validate using alias resolution
    for match in re.finditer(r"(\w+)\.(\w+)", sql_clean):
        table_ref = match.group(1).lower()
        col_ref = match.group(2).lower()

        # Resolve alias to actual table name
        actual_table = alias_map.get(table_ref, table_ref)

        if actual_table in all_tables:
            if col_ref not in table_columns.get(actual_table, set()):
                unknown.append(f"{match.group(1)}.{match.group(2)}")

    # Find columns in WHERE clauses that aren't table-qualified
    # Look for patterns like: WHERE column_name = ... AND column_name ...
    where_match = re.search(r"\bWHERE\b(.+?)(?:\bGROUP\b|\bORDER\b|\bLIMIT\b|\bHAVING\b|$)",
                            sql_clean, re.IGNORECASE | re.DOTALL)
    if where_match:
        where_clause = where_match.group(1)
        for ident in re.findall(r"\b([a-zA-Z_]\w*)\b", where_clause):
            ident_lower = ident.lower()
            if (ident_lower not in all_known
                    and ident_lower not in _SQL_KEYWORDS
                    and not ident_lower.isdigit()
                    and len(ident_lower) > 1):
                unknown.append(ident)

    # Deduplicate while preserving order
    seen = set()
    result = []
    for u in unknown:
        u_lower = u.lower()
        if u_lower not in seen:
            seen.add(u_lower)
            result.append(u)

    if result:
        logger.info("SQL validation: unknown identifiers=%s", result)

    return result


def suggest_column_fix(unknown_col: str) -> str:
    """Suggest a correct column name for a common mistake."""
    return COLUMN_ALIASES.get(unknown_col.lower(), "")


def build_fix_message(unknown_ids: List[str], table_columns: Dict[str, Set[str]]) -> str:
    """Build a concise message for the auto-fix prompt."""
    lines = [f"Unknown columns/tables found: {', '.join(unknown_ids)}"]
    lines.append("Valid schema:")
    for table, cols in sorted(table_columns.items()):
        lines.append(f"  {table}: {', '.join(sorted(cols))}")

    # Suggest fixes for known aliases
    suggestions = []
    for uid in unknown_ids:
        col_part = uid.split(".")[-1]  # handle table.column
        fix = suggest_column_fix(col_part)
        if fix:
            suggestions.append(f"  {col_part} → {fix}")
    if suggestions:
        lines.append("Suggested fixes:")
        lines.extend(suggestions)

    return "\n".join(lines)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _strip_strings_and_comments(sql: str) -> str:
    """Remove string literals and comments to avoid false positives."""
    # Remove single-quoted strings
    sql = re.sub(r"'[^']*'", "''", sql)
    # Remove double-quoted identifiers (keep the identifier name)
    sql = re.sub(r'"([^"]*)"', r"\1", sql)
    # Remove block comments
    sql = re.sub(r"/\*.*?\*/", " ", sql, flags=re.DOTALL)
    # Remove line comments
    sql = re.sub(r"--[^\n]*", " ", sql)
    return sql


_SQL_KEYWORDS = {
    "select", "from", "where", "and", "or", "not", "in", "is", "null",
    "as", "on", "join", "left", "right", "inner", "outer", "cross",
    "group", "by", "order", "having", "limit", "offset", "asc", "desc",
    "distinct", "count", "sum", "avg", "min", "max", "case", "when",
    "then", "else", "end", "between", "like", "exists", "union", "all",
    "with", "recursive", "cast", "coalesce", "ifnull", "nullif",
    "strftime", "date", "now", "true", "false",
}


# ------------------------------------------------------------------
# Nested aggregate detection
# ------------------------------------------------------------------

_AGG_RE = re.compile(r"\b(SUM|AVG|MIN|MAX|COUNT)\s*\(", re.IGNORECASE)


def detect_nested_aggregates(sql: str) -> Tuple[bool, str, str]:
    """Detect aggregate functions nested inside other aggregates.

    SQLite disallows ``SUM((x - AVG(x)) * (x - AVG(x)))`` because the
    inner AVG runs at the same GROUP BY level as the outer SUM.

    Returns (detected, message, suggestion).  All falsy when clean.
    """
    cleaned = _strip_strings_and_comments(sql)

    for m in _AGG_RE.finditer(cleaned):
        outer_func = m.group(1).upper()
        start = m.end() - 1          # the '('
        depth, i = 1, start + 1
        while i < len(cleaned) and depth > 0:
            if cleaned[i] == "(":
                depth += 1
            elif cleaned[i] == ")":
                depth -= 1
            i += 1
        if depth == 0:
            inner = cleaned[start + 1 : i - 1]
            inner_match = _AGG_RE.search(inner)
            if inner_match:
                inner_func = inner_match.group(1).upper()
                return (
                    True,
                    f"Nested aggregate: {inner_func}() inside {outer_func}(). "
                    f"SQLite does not allow aggregates inside aggregates. "
                    f"Use the variance identity or a CTE to precompute.",
                    "Replace SUM((col - AVG(col)) * (col - AVG(col))) with "
                    "AVG(col * col) - AVG(col) * AVG(col)",
                )
    return (False, "", "")
