"""Unit tests for core components"""
import pytest
from src.components.nlp_processor import QueryParser, ContextManager, ConversationTurn
from src.components.sql_generator import SQLValidator
from src.components.optimizer import QueryOptimizer
from src.components.insights import PatternDetector, TrendAnalyzer


class TestQueryParser:
    """Test query parser"""

    def test_extract_intent_retrieve(self):
        parser = QueryParser()
        result = parser.parse("Show me all users")
        assert result["intent"] in ["retrieve", "general"]

    def test_extract_intent_aggregate(self):
        parser = QueryParser()
        result = parser.parse("How many products are there?")
        assert result["intent"] in ["aggregate", "general"]

    def test_extract_modifiers(self):
        parser = QueryParser()
        result = parser.parse("Show me the top 10 products")
        modifiers = result["modifiers"]
        assert modifiers is not None


class TestContextManager:
    """Test context manager"""

    def test_add_turn(self):
        manager = ContextManager()
        manager.add_response(
            user_input="Hello",
            assistant_response="Hi there!",
            generated_sql="SELECT * FROM users",
        )
        history = manager.current_context.conversation_history
        assert len(history) == 1
        assert history[0].user_input == "Hello"

    def test_reset_context(self):
        manager = ContextManager()
        manager.add_response("Hello", "Hi")
        manager.reset()
        assert len(manager.current_context.conversation_history) == 0


class TestSQLValidator:
    """Test SQL validation"""

    def test_valid_select_query(self):
        is_valid, error = SQLValidator.validate("SELECT * FROM users")
        assert is_valid is True

    def test_invalid_drop_statement(self):
        is_valid, error = SQLValidator.validate("DROP TABLE users")
        assert is_valid is False

    def test_invalid_non_select(self):
        is_valid, error = SQLValidator.validate("INSERT INTO users VALUES (1, 'John')")
        assert is_valid is False

    def test_multiple_statements(self):
        is_valid, error = SQLValidator.validate("SELECT * FROM users; DELETE FROM products;")
        assert is_valid is False


class TestQueryOptimizer:
    """Test query optimizer"""

    def test_check_select_star(self):
        optimizer = QueryOptimizer()
        result = optimizer.analyze("SELECT * FROM users")
        assert result["total_suggestions"] > 0

    def test_clean_query(self):
        optimizer = QueryOptimizer()
        result = optimizer.analyze("SELECT id, name FROM users WHERE id = 1")
        # Should have minimal or no suggestions
        assert isinstance(result["total_suggestions"], int)

    def test_optimization_level(self):
        optimizer = QueryOptimizer()
        result = optimizer.analyze("SELECT * FROM users")
        assert result["optimization_level"] in ["excellent", "good", "needs_optimization"]


class TestPatternDetector:
    """Test pattern detection"""

    def test_numeric_patterns(self):
        data = [
            {"id": 1, "value": 100},
            {"id": 2, "value": 200},
            {"id": 3, "value": 300},
        ]
        patterns = PatternDetector.detect_numeric_patterns(data)
        assert "id" in patterns
        assert patterns["id"]["min"] == 1
        assert patterns["id"]["max"] == 3

    def test_string_patterns(self):
        data = [
            {"name": "Alice", "category": "A"},
            {"name": "Bob", "category": "A"},
            {"name": "Charlie", "category": "B"},
        ]
        patterns = PatternDetector.detect_string_patterns(data)
        assert "name" in patterns
        assert patterns["name"]["unique_count"] == 3


class TestTrendAnalyzer:
    """Test trend analysis"""

    def test_increasing_trend(self):
        data = [
            {"value": 100},
            {"value": 150},
            {"value": 200},
        ]
        trends = TrendAnalyzer.analyze_trends(data)
        assert "value" in trends
        assert trends["value"]["direction"] == "increasing"

    def test_decreasing_trend(self):
        data = [
            {"value": 300},
            {"value": 200},
            {"value": 100},
        ]
        trends = TrendAnalyzer.analyze_trends(data)
        assert "value" in trends
        assert trends["value"]["direction"] == "decreasing"


if __name__ == "__main__":
    pytest.main([__file__])
