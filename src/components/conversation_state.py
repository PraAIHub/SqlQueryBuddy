"""Session-scoped conversation state for multi-turn context retention.

Tracks filters, computed entities, and result signatures across turns
so that references like "them", "that region", "#1/#2" resolve to
deterministic values.
"""

import re
import logging
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)


class ConversationState:
    """Tracks conversation context across query turns."""

    def __init__(self):
        self.reset()

    def reset(self):
        """Clear all state."""
        self.last_time_range: Optional[str] = None  # e.g. "2024"
        self.filters_applied: Dict[str, str] = {}   # e.g. {"region": "West", "year": "2024"}
        self.last_sql: str = ""
        self.last_limit: Optional[int] = None       # LIMIT N from the previous query
        self.last_result_signature: Dict[str, Any] = {}  # {"columns": [...], "row_count": int}
        self.computed_entities: Dict[str, Any] = {
            # Only populated when actually computed
            # "top_region": "West",
            # "top_customers": ["Alice", "Bob"],
            # "top_category": "Electronics",
            # "rank_1_value": ..., "rank_2_value": ...,
        }

    # ------------------------------------------------------------------
    # Update state from query results
    # ------------------------------------------------------------------

    def update_from_results(
        self,
        user_query: str,
        sql: str,
        results: List[Dict[str, Any]],
    ):
        """Extract and store state from a successful query run."""
        self.last_sql = sql
        # Extract LIMIT value from the SQL for follow-up preservation
        _limit_match = re.search(r"\bLIMIT\s+(\d+)", sql, re.IGNORECASE)
        self.last_limit = int(_limit_match.group(1)) if _limit_match else None
        columns = list(results[0].keys()) if results else []
        self.last_result_signature = {
            "columns": columns,
            "row_count": len(results),
        }

        query_lower = user_query.lower()

        # Detect time range from query
        year_match = re.search(r"(?:for|in|of|during)\s+(20\d{2})", query_lower)
        if year_match:
            year = year_match.group(1)
            self.last_time_range = year
            self.filters_applied["year"] = year

        # Detect region filter
        for col in columns:
            if col.lower() == "region" and results:
                # If query asks about "per region" / ranking, store top region
                if _is_ranking_query(query_lower) or "per region" in query_lower:
                    top_row = results[0]
                    top_region = top_row.get("region", "")
                    self.computed_entities["top_region"] = top_region
                    # Keep filters_applied in sync so follow-ups use the #1 region
                    if top_region:
                        self.filters_applied["region"] = top_region
                    if len(results) >= 2:
                        self.computed_entities["rank_1_value"] = top_region
                        self.computed_entities["rank_2_value"] = results[1].get("region", "")
                break

        # Detect category
        for col in columns:
            if col.lower() == "category" and results:
                if _is_ranking_query(query_lower) or "per category" in query_lower:
                    self.computed_entities["top_category"] = results[0].get("category", "")
                break

        # Detect customer names and IDs in results
        name_col = None
        id_col = None
        for col in columns:
            if col.lower() in ("name", "customer_name", "customer"):
                name_col = col
            if col.lower() in ("customer_id",):
                id_col = col

        if name_col and results:
            if _is_ranking_query(query_lower) or "top" in query_lower or "customer" in query_lower:
                customer_names = [r.get(name_col, "") for r in results if r.get(name_col)]
                if customer_names:
                    self.computed_entities["top_customers"] = customer_names[:10]
                # Also store customer IDs for precise SQL filtering
                if id_col:
                    customer_ids = [r.get(id_col) for r in results if r.get(id_col) is not None]
                    if customer_ids:
                        self.computed_entities["top_customer_ids"] = customer_ids[:10]

        # Detect if a specific region/category was applied as a filter
        if "region" in query_lower:
            for region_val in _extract_quoted_or_known_values(query_lower, _KNOWN_REGIONS):
                self.filters_applied["region"] = region_val
        if "category" in query_lower:
            for cat_val in _extract_quoted_or_known_values(query_lower, _KNOWN_CATEGORIES):
                self.filters_applied["category"] = cat_val

        logger.debug(
            "ConversationState updated: filters=%s entities=%s",
            self.filters_applied,
            {k: (v[:3] if isinstance(v, list) else v) for k, v in self.computed_entities.items()},
        )

    # ------------------------------------------------------------------
    # Active filters display
    # ------------------------------------------------------------------

    def get_active_filters_html(self) -> str:
        """Generate Active Filters pill HTML for the UI."""
        if not self.filters_applied and not self.computed_entities:
            return ""

        pills = []
        for key, val in self.filters_applied.items():
            label = key.replace("_", " ").title()
            pills.append(
                f"<span style='display:inline-block; background:#ede9fe; color:#6d28d9; "
                f"padding:2px 10px; border-radius:12px; font-size:12px; margin:2px 4px;'>"
                f"{label}: {val}</span>"
            )

        # Show computed entities as context indicators
        if self.computed_entities.get("top_region"):
            pills.append(
                f"<span style='display:inline-block; background:#dbeafe; color:#1d4ed8; "
                f"padding:2px 10px; border-radius:12px; font-size:12px; margin:2px 4px;'>"
                f"Top Region: {self.computed_entities['top_region']}</span>"
            )
        if self.computed_entities.get("top_category"):
            pills.append(
                f"<span style='display:inline-block; background:#dbeafe; color:#1d4ed8; "
                f"padding:2px 10px; border-radius:12px; font-size:12px; margin:2px 4px;'>"
                f"Top Category: {self.computed_entities['top_category']}</span>"
            )

        if not pills:
            return ""

        return (
            "<div style='margin:4px 0 8px 0;'>"
            "<span style='font-size:12px; color:#6b7280; margin-right:4px;'>Active Context:</span>"
            + "".join(pills)
            + "</div>"
        )


# ------------------------------------------------------------------
# Reference resolver
# ------------------------------------------------------------------

# Reference patterns → state field mapping
_REGION_REFS = re.compile(
    r"\b(?:of that region|of the region|that region|the region|this region|"
    r"the top region|the #1 region|in (?:the )?(?:top|#1|first) region)\b",
    re.IGNORECASE,
)
_CUSTOMER_REFS = re.compile(
    r"\b(?:them|those customers|these customers|the customers|from them|of them|"
    r"the top customers)\b",
    re.IGNORECASE,
)
_CATEGORY_REFS = re.compile(
    r"\b(?:that category|the category|this category|the top category)\b",
    re.IGNORECASE,
)
_RANK_1_REF = re.compile(r"(?<!\w)(?:#1|the (?:top|first|#1))(?!\w)", re.IGNORECASE)
_RANK_2_REF = re.compile(r"(?<!\w)(?:#2|the second|the (?:#2|runner.?up))(?!\w)", re.IGNORECASE)
_THIS_YEAR_REF = re.compile(r"\b(?:this year|the year|same year)\b", re.IGNORECASE)


def resolve_references(
    user_query: str,
    state: ConversationState,
) -> str:
    """Resolve pronoun/reference expressions to concrete values.

    Returns the rewritten query with references substituted, or the
    original query if no references are found / resolvable.
    """
    resolved = user_query
    changes = []

    # "that region" / "the top region" → state.computed_entities.top_region
    if _REGION_REFS.search(resolved):
        top_region = state.computed_entities.get("top_region")
        if top_region:
            resolved = _REGION_REFS.sub(f"the {top_region} region", resolved)
            changes.append(f"region→{top_region}")

    # "them" / "those customers" → prefer customer IDs for precise SQL filtering
    if _CUSTOMER_REFS.search(resolved):
        customer_ids = state.computed_entities.get("top_customer_ids")
        customers = state.computed_entities.get("top_customers")
        if customer_ids:
            ids_str = ", ".join(str(i) for i in customer_ids[:10])
            resolved = _CUSTOMER_REFS.sub(
                f"the customers with customer_id IN ({ids_str})", resolved
            )
            changes.append(f"customer_ids→{len(customer_ids)} ids")
        elif customers:
            # Fall back to names when IDs are unavailable
            names_str = ", ".join(customers[:10])
            resolved = _CUSTOMER_REFS.sub(
                f"the customers ({names_str})", resolved
            )
            changes.append(f"customers→{len(customers)} names")

    # "that category" → top_category
    if _CATEGORY_REFS.search(resolved):
        top_cat = state.computed_entities.get("top_category")
        if top_cat:
            resolved = _CATEGORY_REFS.sub(f"the {top_cat} category", resolved)
            changes.append(f"category→{top_cat}")

    # "#1" → rank_1_value, "#2" → rank_2_value
    if _RANK_2_REF.search(resolved):
        val = state.computed_entities.get("rank_2_value")
        if val:
            resolved = _RANK_2_REF.sub(str(val), resolved)
            changes.append(f"#2→{val}")

    if _RANK_1_REF.search(resolved):
        val = state.computed_entities.get("rank_1_value")
        if val:
            resolved = _RANK_1_REF.sub(str(val), resolved)
            changes.append(f"#1→{val}")

    # "this year" → last_time_range
    if _THIS_YEAR_REF.search(resolved):
        year = state.last_time_range
        if year:
            resolved = _THIS_YEAR_REF.sub(f"year {year}", resolved)
            changes.append(f"year→{year}")

    if changes:
        logger.info("Reference resolved: %s → %r", changes, resolved)

    return resolved


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

_KNOWN_REGIONS = {
    "california", "new york", "texas", "florida", "illinois",
    "washington", "georgia", "ohio", "pennsylvania", "colorado",
    "west", "east", "north", "south", "midwest",
}
_KNOWN_CATEGORIES = {
    "electronics", "furniture", "accessories", "office supplies",
}


def _is_ranking_query(query_lower: str) -> bool:
    """Detect if a query asks for ranking / top / most / best."""
    return bool(re.search(
        r"\b(?:top|most|best|highest|largest|#1|rank|leading|greatest)\b",
        query_lower,
    ))


def _extract_quoted_or_known_values(text: str, known_set: set) -> List[str]:
    """Extract known dimension values from the query text."""
    found = []
    text_lower = text.lower()
    for val in known_set:
        if val in text_lower:
            found.append(val.title())
    return found
