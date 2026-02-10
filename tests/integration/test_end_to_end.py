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
            user_query="Show me all users",
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
            user_input="Show users",
            assistant_response="Here are users",
            generated_sql="SELECT * FROM users",
        )

        # Second turn
        manager.add_response(
            user_input="How many products?",
            assistant_response="10 products",
            generated_sql="SELECT COUNT(*) FROM products",
        )

        # Verify history
        history = manager.current_context.conversation_history
        assert len(history) == 2
        assert "SELECT * FROM users" in history[0].generated_sql

    def test_query_optimization(self):
        """Test query optimization"""
        optimizer = QueryOptimizer()

        # Test query with suggestions
        query = "SELECT * FROM users"
        result = optimizer.analyze(query)

        assert "suggestions" in result
        assert "total_suggestions" in result

    def test_database_schema_extraction(self, temp_db):
        """Test database schema extraction"""
        db = DatabaseConnection(temp_db)
        schema = db.get_schema()

        assert "users" in schema
        assert "products" in schema
        assert "orders" in schema

        # Check users table structure
        users_cols = schema["users"]["columns"]
        assert "id" in users_cols
        assert "name" in users_cols
        assert "email" in users_cols

    def test_sql_injection_prevention(self):
        """Test SQL injection prevention"""
        from src.components.sql_generator import SQLValidator

        # Test dangerous queries
        dangerous_queries = [
            "DROP TABLE users;",
            "DELETE FROM users;",
            "INSERT INTO users VALUES (1, 'hacker')",
        ]

        for query in dangerous_queries:
            is_valid, error = SQLValidator.validate(query)
            assert not is_valid, f"Query should be invalid: {query}"

    def test_sample_data_retrieval(self, temp_db):
        """Test retrieving sample data from tables"""
        db = DatabaseConnection(temp_db)

        # Get sample data from users table
        sample = db.get_sample_data("users", limit=5)

        assert isinstance(sample, list)
        if sample:
            assert "id" in sample[0]
            assert "name" in sample[0]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
