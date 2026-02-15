"""End-to-end integration tests"""
import pytest
import tempfile
import os
from src.components.executor import DatabaseConnection, QueryExecutor, SQLiteDatabase
from src.components.nlp_processor import ContextManager
from src.components.sql_generator import SQLGeneratorMock
from src.components.optimizer import QueryOptimizer


class TestEndToEnd:
    """End-to-end tests"""

    @pytest.fixture
    def temp_db(self):
        """Create temporary database for testing"""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            SQLiteDatabase.create_sample_database(db_path)
            db_url = f"sqlite:///{db_path}"
            yield db_url

    def test_query_generation_and_execution(self, temp_db):
        """Test end-to-end query generation and execution"""
        # Setup
        db = DatabaseConnection(temp_db)
        executor = QueryExecutor(db)
        generator = SQLGeneratorMock()

        # Generate SQL
        schema = db.get_schema()
        schema_str = "\n".join(
            [f"{t}: {', '.join(c.keys())}" for t, c in schema.items()]
        )

        result = generator.generate(
            user_query="Show me all customers",
            schema_context=schema_str,
        )

        assert result["success"]
        assert "generated_sql" in result

        # Execute SQL
        sql = result["generated_sql"]
        exec_result = executor.execute(sql)

        assert exec_result["success"]
        assert exec_result["row_count"] >= 0

    def test_context_management(self, temp_db):
        """Test context management across turns"""
        manager = ContextManager()

        # First turn
        manager.add_response(
            user_input="Show customers",
            assistant_response="Here are customers",
            generated_sql="SELECT * FROM customers",
        )

        # Second turn
        manager.add_response(
            user_input="How many products?",
            assistant_response="5 products",
            generated_sql="SELECT COUNT(*) FROM products",
        )

        # Verify history
        history = manager.current_context.conversation_history
        assert len(history) == 2
        assert "SELECT * FROM customers" in history[0].generated_sql

    def test_query_optimization(self):
        """Test query optimization"""
        optimizer = QueryOptimizer()

        # Test query with suggestions
        query = "SELECT * FROM customers"
        result = optimizer.analyze(query)

        assert "suggestions" in result
        assert "total_suggestions" in result

    def test_database_schema_extraction(self, temp_db):
        """Test database schema extraction"""
        db = DatabaseConnection(temp_db)
        schema = db.get_schema()

        assert "customers" in schema
        assert "products" in schema
        assert "orders" in schema
        assert "order_items" in schema

        # Check customers table structure
        customers_cols = schema["customers"]["columns"]
        assert "customer_id" in customers_cols
        assert "name" in customers_cols
        assert "email" in customers_cols
        assert "region" in customers_cols

        # Check order_items table structure
        order_items_cols = schema["order_items"]["columns"]
        assert "item_id" in order_items_cols
        assert "order_id" in order_items_cols
        assert "product_id" in order_items_cols
        assert "quantity" in order_items_cols
        assert "subtotal" in order_items_cols

    def test_sql_injection_prevention(self):
        """Test SQL injection prevention"""
        from src.components.sql_generator import SQLValidator

        # Test dangerous queries
        dangerous_queries = [
            "DROP TABLE customers;",
            "DELETE FROM customers;",
            "INSERT INTO customers VALUES (1, 'hacker')",
        ]

        for query in dangerous_queries:
            is_valid, error = SQLValidator.validate(query)
            assert not is_valid, f"Query should be invalid: {query}"

    def test_sample_data_retrieval(self, temp_db):
        """Test retrieving sample data from tables"""
        db = DatabaseConnection(temp_db)

        # Get sample data from customers table
        sample = db.get_sample_data("customers", limit=5)

        assert isinstance(sample, list)
        assert len(sample) == 5
        assert "customer_id" in sample[0]
        assert "name" in sample[0]
        assert "region" in sample[0]

        # Verify specific sample data from PDF
        names = [row["name"] for row in sample]
        assert "Alice Chen" in names
        assert "John Patel" in names

    def test_order_items_data(self, temp_db):
        """Test order_items table has correct data"""
        db = DatabaseConnection(temp_db)

        result = db.execute_query("SELECT COUNT(*) as cnt FROM order_items")
        assert result["success"]
        assert result["rows"][0]["cnt"] >= 1000

    def test_multi_table_query(self, temp_db):
        """Test multi-table JOIN query execution"""
        db = DatabaseConnection(temp_db)
        executor = QueryExecutor(db)

        # Query that joins customers, orders, and order_items
        sql = """
            SELECT c.name, SUM(o.total_amount) as total_spent
            FROM customers c
            JOIN orders o ON c.customer_id = o.customer_id
            GROUP BY c.name
            ORDER BY total_spent DESC
            LIMIT 5
        """
        result = executor.execute(sql)
        assert result["success"]
        assert result["row_count"] > 0


    def test_all_mock_patterns_execute_successfully(self, temp_db):
        """Test that every mock SQL pattern executes against the sample DB."""
        db = DatabaseConnection(temp_db)
        executor = QueryExecutor(db)
        mock = SQLGeneratorMock()

        test_queries = [
            "Show me the top 5 customers by total purchase amount",
            "Which product category made the most revenue?",
            "Show total sales per region for 2024",
            "Find the average order value",
            "Find the average order value for returning customers",
            "Show the trend of monthly revenue over time",
            "Show me all products",
            "How many orders contained more than 3 items?",
            "How many unique products were sold?",
            "How many unique products were sold in January?",
            "Show me all orders",
            "How many customers are there?",
        ]

        for query in test_queries:
            result = mock.generate(query, "")
            assert result["success"], f"Mock failed for: {query}"
            exec_result = executor.execute(result["generated_sql"])
            assert exec_result["success"], (
                f"Execution failed for '{query}': {exec_result.get('error')}\n"
                f"SQL: {result['generated_sql']}"
            )

    def test_follow_up_query_execution(self, temp_db):
        """Test that follow-up queries produce executable SQL."""
        db = DatabaseConnection(temp_db)
        executor = QueryExecutor(db)
        mock = SQLGeneratorMock()

        # First query
        mock.generate("Show me top customers by spending", "")
        # Follow-up
        result = mock.generate(
            "From the previous result, filter customers from California only", ""
        )
        assert result["success"]
        exec_result = executor.execute(result["generated_sql"])
        assert exec_result["success"], f"Follow-up SQL failed: {exec_result.get('error')}"

    def test_local_insights_with_real_data(self, temp_db):
        """Test local insight generation on actual query results."""
        from src.components.insights import LocalInsightGenerator

        db = DatabaseConnection(temp_db)
        executor = QueryExecutor(db)

        result = executor.execute(
            "SELECT c.name, SUM(o.total_amount) AS total_spent "
            "FROM customers c JOIN orders o ON c.customer_id = o.customer_id "
            "GROUP BY c.name ORDER BY total_spent DESC"
        )
        assert result["success"]

        gen = LocalInsightGenerator()
        insights = gen.generate_insights(result["data"], "top customers")
        assert len(insights) > 0
        # Should mention a customer name from the data
        top_name = result["data"][0].get("name", "")
        assert top_name in insights or len(insights) > 20

    def test_rag_with_real_schema(self, temp_db):
        """Test RAG pipeline with actual database schema."""
        from src.components.rag_system import RAGSystem, FAISSVectorDB, SimpleEmbeddingProvider

        db = DatabaseConnection(temp_db)
        schema = db.get_schema()

        provider = SimpleEmbeddingProvider()
        texts = []
        for table_name, table_info in schema.items():
            texts.append(f"Table {table_name}")
            for col_name, col_info in table_info.get("columns", {}).items():
                texts.append(f"Column {col_name} in {table_name} {col_info.get('type', '')}")
        provider.build_vocabulary(texts)

        rag = RAGSystem(provider, FAISSVectorDB())
        rag.initialize_schema(schema)

        context = rag.get_schema_context_string("customer orders total spending")
        assert "Candidate Schema Elements" in context
        # Should find relevant tables
        assert "Table:" in context or "Tables:" in context


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
