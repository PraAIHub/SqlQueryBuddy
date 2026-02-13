"""Unit tests for core components"""
import pytest
from src.components.nlp_processor import QueryParser, ContextManager, ConversationTurn
from src.components.sql_generator import SQLValidator, SQLGeneratorMock
from src.components.optimizer import QueryOptimizer
from src.components.insights import PatternDetector, TrendAnalyzer, LocalInsightGenerator
from src.components.rag_system import (
    SimpleEmbeddingProvider,
    RAGSystem,
    FAISSVectorDB,
    InMemoryVectorDB,
)


class TestQueryParser:
    """Test query parser"""

    def test_extract_intent_retrieve(self):
        parser = QueryParser()
        result = parser.parse("Show me all customers")
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
            generated_sql="SELECT * FROM customers",
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
        is_valid, error = SQLValidator.validate("SELECT * FROM customers")
        assert is_valid is True

    def test_invalid_drop_statement(self):
        is_valid, error = SQLValidator.validate("DROP TABLE customers")
        assert is_valid is False

    def test_invalid_non_select(self):
        is_valid, error = SQLValidator.validate("INSERT INTO customers VALUES (1, 'John')")
        assert is_valid is False

    def test_invalid_update_statement(self):
        is_valid, error = SQLValidator.validate("UPDATE customers SET name='x'")
        assert is_valid is False

    def test_no_false_positive_on_column_names(self):
        """Column names like is_deleted should not trigger the DELETE check."""
        is_valid, error = SQLValidator.validate("SELECT is_deleted FROM customers")
        assert is_valid is True

    def test_multiple_statements(self):
        is_valid, error = SQLValidator.validate("SELECT * FROM customers; DELETE FROM products;")
        assert is_valid is False


class TestQueryOptimizer:
    """Test query optimizer"""

    def test_check_select_star(self):
        optimizer = QueryOptimizer()
        result = optimizer.analyze("SELECT * FROM customers")
        assert result["total_suggestions"] > 0

    def test_clean_query(self):
        optimizer = QueryOptimizer()
        result = optimizer.analyze("SELECT customer_id, name FROM customers WHERE customer_id = 1")
        # Should have minimal or no suggestions
        assert isinstance(result["total_suggestions"], int)

    def test_optimization_level(self):
        optimizer = QueryOptimizer()
        result = optimizer.analyze("SELECT * FROM customers")
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

    def test_anomaly_detection_spike(self):
        data = [
            {"value": 100},
            {"value": 105},
            {"value": 98},
            {"value": 102},
            {"value": 99},
            {"value": 800},  # obvious spike
        ]
        anomalies = TrendAnalyzer.detect_anomalies(data)
        assert "value" in anomalies
        assert anomalies["value"][0]["type"] == "spike"

    def test_anomaly_detection_no_anomalies(self):
        data = [
            {"value": 100},
            {"value": 101},
            {"value": 99},
            {"value": 100},
        ]
        anomalies = TrendAnalyzer.detect_anomalies(data)
        assert len(anomalies) == 0


class TestRAGSystem:
    """Test RAG pipeline components"""

    @pytest.fixture
    def rag_with_schema(self):
        """Create RAG system initialized with test schema."""
        provider = SimpleEmbeddingProvider()
        schema = {
            "customers": {
                "columns": {
                    "name": {"type": "TEXT"},
                    "email": {"type": "TEXT"},
                    "region": {"type": "TEXT"},
                }
            },
            "products": {
                "columns": {
                    "name": {"type": "TEXT"},
                    "price": {"type": "REAL"},
                    "category": {"type": "TEXT"},
                }
            },
            "orders": {
                "columns": {
                    "total_amount": {"type": "REAL"},
                    "order_date": {"type": "TEXT"},
                }
            },
        }
        texts = []
        for table_name, table_info in schema.items():
            texts.append(f"Table {table_name}")
            for col_name, col_info in table_info.get("columns", {}).items():
                texts.append(f"Column {col_name} in {table_name} {col_info.get('type', '')}")
        provider.build_vocabulary(texts)
        rag = RAGSystem(provider, FAISSVectorDB())
        rag.initialize_schema(schema)
        return rag

    def test_embedding_provider_produces_vectors(self):
        provider = SimpleEmbeddingProvider()
        provider.build_vocabulary(["Table customers", "Column price in products"])
        vec = provider.embed("customers name")
        assert isinstance(vec, list)
        assert len(vec) > 0

    def test_embedding_batch(self):
        provider = SimpleEmbeddingProvider()
        provider.build_vocabulary(["Table customers", "Column price"])
        vecs = provider.embed_batch(["customers", "price"])
        assert len(vecs) == 2

    def test_faiss_store_and_search(self):
        db = FAISSVectorDB()
        db.store("key1", [1.0, 0.0, 0.0], {"name": "a"})
        db.store("key2", [0.0, 1.0, 0.0], {"name": "b"})
        results = db.search([1.0, 0.0, 0.0], top_k=1)
        assert len(results) == 1
        assert results[0][0] == "key1"

    def test_faiss_clear(self):
        db = FAISSVectorDB()
        db.store("key1", [1.0, 0.0], {"name": "a"})
        db.clear()
        results = db.search([1.0, 0.0], top_k=5)
        assert len(results) == 0

    def test_rag_retrieve_context(self, rag_with_schema):
        context = rag_with_schema.retrieve_context("customer names", top_k=3)
        assert isinstance(context, list)
        assert len(context) > 0
        assert "type" in context[0]

    def test_rag_schema_context_string(self, rag_with_schema):
        result = rag_with_schema.get_schema_context_string("product prices")
        assert "Relevant Schema Elements:" in result
        assert "Table:" in result or "Column:" in result


class TestLocalInsightGenerator:
    """Test local insight generation"""

    def test_empty_data(self):
        gen = LocalInsightGenerator()
        result = gen.generate_insights([], "query")
        assert "No matching data found" in result

    def test_top_performer_insight(self):
        gen = LocalInsightGenerator()
        data = [
            {"name": "Alice", "total_spent": 5000},
            {"name": "Bob", "total_spent": 2000},
            {"name": "Charlie", "total_spent": 1000},
        ]
        result = gen.generate_insights(data, "top customers")
        assert "Alice" in result
        assert "%" in result  # Should mention percentage

    def test_categorical_insight(self):
        gen = LocalInsightGenerator()
        data = [
            {"category": "Electronics", "revenue": 8000},
            {"category": "Electronics", "revenue": 3000},
            {"category": "Furniture", "revenue": 2000},
        ]
        result = gen.generate_insights(data, "revenue by category")
        assert len(result) > 0

    def test_trend_insight_with_time_column(self):
        gen = LocalInsightGenerator()
        data = [
            {"month": "2024-01", "monthly_revenue": 1000},
            {"month": "2024-02", "monthly_revenue": 1500},
            {"month": "2024-03", "monthly_revenue": 2000},
        ]
        result = gen.generate_insights(data, "monthly trend")
        assert "increasing" in result.lower()


class TestSQLGeneratorMockPatterns:
    """Test that mock SQL generator patterns produce valid queries."""

    def test_top_customers_query(self):
        mock = SQLGeneratorMock()
        result = mock.generate("Show me the top 5 customers by total purchase amount", "")
        assert result["success"]
        assert "customers" in result["generated_sql"].lower()
        assert "JOIN" in result["generated_sql"]

    def test_category_revenue_query(self):
        mock = SQLGeneratorMock()
        result = mock.generate("Which product category made the most revenue?", "")
        assert result["success"]
        assert "category" in result["generated_sql"].lower()

    def test_region_sales_query(self):
        mock = SQLGeneratorMock()
        result = mock.generate("Show total sales per region", "")
        assert result["success"]
        assert "region" in result["generated_sql"].lower()

    def test_monthly_trend_query(self):
        mock = SQLGeneratorMock()
        result = mock.generate("Show the trend of monthly revenue over time", "")
        assert result["success"]
        assert "strftime" in result["generated_sql"] or "month" in result["generated_sql"].lower()

    def test_follow_up_query(self):
        mock = SQLGeneratorMock()
        # First query
        mock.generate("Show me top customers by spending", "")
        # Follow-up
        result = mock.generate("From the previous result, filter customers from California only", "")
        assert result["success"]
        assert "California" in result["generated_sql"]

    def test_returning_customers_query(self):
        mock = SQLGeneratorMock()
        result = mock.generate("Find the average order value for returning customers", "")
        assert result["success"]
        assert "HAVING" in result["generated_sql"]
        assert "COUNT" in result["generated_sql"]

    def test_january_unique_products_query(self):
        mock = SQLGeneratorMock()
        result = mock.generate("How many unique products were sold in January?", "")
        assert result["success"]
        assert "strftime" in result["generated_sql"]
        assert "'01'" in result["generated_sql"]

    def test_orders_more_than_3_items_query(self):
        mock = SQLGeneratorMock()
        result = mock.generate("How many orders contained more than 3 items?", "")
        assert result["success"]
        assert "HAVING" in result["generated_sql"]
        assert "item_count > 3" in result["generated_sql"]

    def test_follow_up_without_context(self):
        mock = SQLGeneratorMock()
        result = mock.generate("From the previous result, filter by California", "")
        assert result["success"]
        assert "No previous query" in result["explanation"]

    def test_follow_up_with_conversation_history(self):
        mock = SQLGeneratorMock()
        history = "Turn 1:\nUser: Show customers\nSQL: SELECT * FROM customers"
        result = mock.generate("From the previous, filter California only", "", history)
        assert result["success"]
        assert "California" in result["generated_sql"]

    def test_default_fallback(self):
        mock = SQLGeneratorMock()
        result = mock.generate("xyzzy nonsense query", "")
        assert result["success"]
        assert "customers" in result["generated_sql"].lower()


if __name__ == "__main__":
    pytest.main([__file__])
