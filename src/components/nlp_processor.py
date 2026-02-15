"""Natural Language Processing layer for query understanding"""
import re
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class ConversationTurn:
    """Represents a single turn in the conversation"""

    user_input: str
    assistant_response: str
    generated_sql: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: dict = field(default_factory=dict)


@dataclass
class QueryContext:
    """Maintains context across multiple conversation turns"""

    conversation_history: List[ConversationTurn] = field(default_factory=list)
    schema_context: dict = field(default_factory=dict)
    user_preferences: dict = field(default_factory=dict)

    def add_turn(
        self,
        user_input: str,
        assistant_response: str,
        generated_sql: Optional[str] = None,
        metadata: Optional[dict] = None,
    ) -> None:
        """Add a new turn to conversation history"""
        turn = ConversationTurn(
            user_input=user_input,
            assistant_response=assistant_response,
            generated_sql=generated_sql,
            metadata=metadata or {},
        )
        self.conversation_history.append(turn)

    def get_last_turns(self, n: int = 5) -> List[ConversationTurn]:
        """Get the last N turns from conversation history"""
        return self.conversation_history[-n:]

    def get_context_str(self) -> str:
        """Get conversation history as formatted string for LLM context"""
        context_lines = []
        for i, turn in enumerate(self.get_last_turns(5), 1):
            context_lines.append(f"Turn {i}:")
            context_lines.append(f"User: {turn.user_input}")
            context_lines.append(f"Assistant: {turn.assistant_response}")
            if turn.generated_sql:
                context_lines.append(f"SQL: {turn.generated_sql}")
            context_lines.append("")
        return "\n".join(context_lines)

    def clear_history(self) -> None:
        """Clear conversation history"""
        self.conversation_history = []


class QueryParser:
    """Parses and analyzes natural language queries"""

    def parse(self, user_input: str) -> dict:
        """Parse user input to extract intent and entities"""
        return {
            "original_text": user_input,
            "intent": self._extract_intent(user_input),
            "entities": self._extract_entities(user_input),
            "modifiers": self._extract_modifiers(user_input),
        }

    @staticmethod
    def _extract_intent(text: str) -> str:
        """Extract the main intent from query"""
        text_lower = text.lower()

        if any(word in text_lower for word in ["show", "list", "get", "fetch"]):
            return "retrieve"
        elif any(word in text_lower for word in ["count", "how many", "total"]):
            return "aggregate"
        elif any(word in text_lower for word in ["top", "bottom", "highest", "lowest"]):
            return "ranking"
        elif any(word in text_lower for word in ["compare", "difference"]):
            return "comparison"
        elif any(word in text_lower for word in ["trend", "over time", "change"]):
            return "trend"
        else:
            return "general"

    # Known schema entities for keyword matching
    KNOWN_TABLES = ["customers", "products", "orders", "order_items"]
    KNOWN_COLUMNS = [
        "customer_id", "name", "email", "region", "signup_date",
        "product_id", "category", "price",
        "order_id", "order_date", "total_amount",
        "item_id", "quantity", "subtotal",
    ]

    @staticmethod
    def _extract_entities(text: str) -> List[str]:
        """Extract potential table/column names referenced in the query."""
        text_lower = text.lower()
        entities = []
        for table in QueryParser.KNOWN_TABLES:
            # Match singular or plural form
            singular = table.rstrip("s")
            if table in text_lower or singular in text_lower:
                entities.append(table)
        for col in QueryParser.KNOWN_COLUMNS:
            if col.replace("_", " ") in text_lower or col in text_lower:
                entities.append(col)
        return entities

    @staticmethod
    def _extract_modifiers(text: str) -> dict:
        """Extract query modifiers like LIMIT, ORDER BY preferences"""
        modifiers = {
            "limit": None,
            "order_by": None,
            "filter": [],
        }

        text_lower = text.lower()

        # Extract LIMIT
        if "top" in text_lower or "first" in text_lower:
            modifiers["limit"] = 10
        if "last" in text_lower:
            modifiers["order_by"] = "DESC"

        return modifiers


@dataclass
class QueryPlan:
    """Structured representation of the current query state.

    Tracks active filters, entities, time range, and last SQL so the
    system can build richer context without raw chat-history stuffing.
    """

    active_tables: List[str] = field(default_factory=list)
    active_filters: List[str] = field(default_factory=list)
    entities: List[str] = field(default_factory=list)
    time_range: str = "all-time"
    last_sql: str = ""
    last_intent: str = "general"
    turn_count: int = 0

    def update(
        self,
        intent: str,
        entities: List[str],
        generated_sql: str,
        user_query: str = "",
    ) -> None:
        """Update the query plan after a successful query."""
        self.last_intent = intent
        self.last_sql = generated_sql
        self.turn_count += 1

        # Merge entities (deduplicated)
        for e in entities:
            if e not in self.entities:
                self.entities.append(e)

        # Detect active tables from SQL
        sql_lower = generated_sql.lower()
        known_tables = ["customers", "products", "orders", "order_items"]
        self.active_tables = [t for t in known_tables if t in sql_lower]

        # Detect time range from SQL
        if "date('now'" in sql_lower or "strftime" in sql_lower:
            query_lower = user_query.lower()
            if "this year" in query_lower:
                self.time_range = "this year"
            elif "last year" in query_lower:
                self.time_range = "last year"
            elif "last 3 months" in query_lower or "past 3 months" in query_lower:
                self.time_range = "last 3 months"
            elif "last 6 months" in query_lower:
                self.time_range = "last 6 months"
            elif "this month" in query_lower:
                self.time_range = "this month"
            elif "this quarter" in query_lower:
                self.time_range = "this quarter"
            else:
                self.time_range = "filtered"
        else:
            self.time_range = "all-time"

        # Detect active filters from WHERE clause
        where_match = re.search(
            r"WHERE\s+(.+?)(?:GROUP|ORDER|LIMIT|HAVING|$)",
            generated_sql,
            re.IGNORECASE | re.DOTALL,
        )
        if where_match:
            self.active_filters = [
                f.strip() for f in where_match.group(1).split("AND")
            ]
        else:
            self.active_filters = []

    def to_context_string(self) -> str:
        """Serialize query plan as a compact context string for the LLM."""
        if self.turn_count == 0:
            return ""
        parts = [
            f"Tables: {', '.join(self.active_tables) if self.active_tables else 'none'}",
            f"Filters: {'; '.join(self.active_filters) if self.active_filters else 'none'}",
            f"Time range: {self.time_range}",
            f"Intent: {self.last_intent}",
        ]
        return "Active Query State: " + " | ".join(parts)

    def reset(self) -> None:
        """Reset query plan."""
        self.active_tables.clear()
        self.active_filters.clear()
        self.entities.clear()
        self.time_range = "all-time"
        self.last_sql = ""
        self.last_intent = "general"
        self.turn_count = 0


class ContextManager:
    """Manages conversation context and session state"""

    def __init__(self):
        self.current_context = QueryContext()
        self.parser = QueryParser()
        self.query_plan = QueryPlan()

    def initialize_with_schema(self, schema: dict) -> None:
        """Initialize context with database schema"""
        self.current_context.schema_context = schema

    def process_input(self, user_input: str) -> dict:
        """Process user input and update context"""
        parsed = self.parser.parse(user_input)
        return {
            "parsed_query": parsed,
            "context": self.current_context,
            "history": self.current_context.get_context_str(),
        }

    def add_response(
        self,
        user_input: str,
        assistant_response: str,
        generated_sql: Optional[str] = None,
    ) -> None:
        """Add a response to the conversation context"""
        self.current_context.add_turn(
            user_input=user_input,
            assistant_response=assistant_response,
            generated_sql=generated_sql,
        )

    def update_query_plan(
        self,
        intent: str,
        entities: List[str],
        generated_sql: str,
        user_query: str = "",
    ) -> None:
        """Update structured query plan after a successful query."""
        self.query_plan.update(intent, entities, generated_sql, user_query)

    def reset(self) -> None:
        """Reset conversation context"""
        self.current_context.clear_history()
        self.query_plan.reset()

    def get_full_context(self) -> str:
        """Get full conversation context as string, enriched with query plan."""
        history = self.current_context.get_context_str()
        plan_str = self.query_plan.to_context_string()
        if plan_str:
            return f"{plan_str}\n\n{history}"
        return history
