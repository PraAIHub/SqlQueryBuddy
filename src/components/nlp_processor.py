"""Natural Language Processing layer for query understanding"""
from typing import Optional, List
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


class ContextManager:
    """Manages conversation context and session state"""

    def __init__(self):
        self.current_context = QueryContext()
        self.parser = QueryParser()

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

    def reset(self) -> None:
        """Reset conversation context"""
        self.current_context.clear_history()

    def get_full_context(self) -> str:
        """Get full conversation context as string"""
        return self.current_context.get_context_str()
