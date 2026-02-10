"""Query execution and database management"""
from typing import List, Dict, Any, Optional
from contextlib import contextmanager
import sqlite3
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.exc import SQLAlchemyError


class DatabaseConnection:
    """Manages database connections and queries"""

    def __init__(self, database_url: str):
        self.database_url = database_url
        self.engine = create_engine(database_url)

    @contextmanager
    def get_connection(self):
        """Context manager for database connections"""
        connection = self.engine.connect()
        try:
            yield connection
        finally:
            connection.close()

    def execute_query(
        self, query: str, timeout_seconds: int = 30
    ) -> Dict[str, Any]:
        """Execute a SQL query safely"""
        try:
            with self.get_connection() as conn:
                result = conn.execute(text(query))
                rows = result.fetchall()
                columns = list(result.keys())

                return {
                    "success": True,
                    "columns": columns,
                    "rows": [dict(zip(columns, row)) for row in rows],
                    "row_count": len(rows),
                }
        except SQLAlchemyError as e:
            return {
                "success": False,
                "error": f"Query execution failed: {str(e)}",
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"Unexpected error: {str(e)}",
            }

    def get_schema(self) -> Dict[str, Dict[str, Any]]:
        """Get database schema information"""
        inspector = inspect(self.engine)
        schema = {}

        for table_name in inspector.get_table_names():
            columns = {}
            for column in inspector.get_columns(table_name):
                columns[column["name"]] = {
                    "type": str(column["type"]),
                    "nullable": column.get("nullable", True),
                    "description": "",
                }

            schema[table_name] = {
                "columns": columns,
                "description": f"Table {table_name}",
            }

        return schema

    def get_sample_data(self, table_name: str, limit: int = 5) -> List[Dict]:
        """Get sample data from a table"""
        try:
            query = f"SELECT * FROM {table_name} LIMIT {limit}"
            result = self.execute_query(query)
            return result.get("rows", []) if result["success"] else []
        except Exception:
            return []


class QueryExecutor:
    """Executes queries with safety checks and formatting"""

    def __init__(self, db_connection: DatabaseConnection):
        self.db = db_connection

    def execute(self, sql_query: str) -> Dict[str, Any]:
        """Execute a SQL query and return formatted results"""
        result = self.db.execute_query(sql_query)

        if not result["success"]:
            return result

        # Format results
        formatted_result = {
            "success": True,
            "query": sql_query,
            "row_count": result["row_count"],
            "columns": result["columns"],
            "data": result["rows"],
            "summary": self._generate_summary(result),
        }

        return formatted_result

    @staticmethod
    def _generate_summary(result: Dict[str, Any]) -> str:
        """Generate a summary of query results"""
        row_count = result.get("row_count", 0)
        if row_count == 0:
            return "No results found."
        elif row_count == 1:
            return "1 result found."
        else:
            return f"{row_count} results found."


class SQLiteDatabase:
    """Utility for creating and managing SQLite databases"""

    @staticmethod
    def create_sample_database(db_path: str) -> None:
        """Create a sample SQLite database for testing"""
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Create users table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                email TEXT UNIQUE NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Create products table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS products (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                price REAL NOT NULL,
                category TEXT,
                stock INTEGER DEFAULT 0
            )
        """)

        # Create orders table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS orders (
                id INTEGER PRIMARY KEY,
                user_id INTEGER NOT NULL,
                product_id INTEGER NOT NULL,
                quantity INTEGER NOT NULL,
                order_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id),
                FOREIGN KEY (product_id) REFERENCES products(id)
            )
        """)

        # Insert sample data
        cursor.executemany(
            "INSERT INTO users (name, email) VALUES (?, ?)",
            [
                ("Alice Johnson", "alice@example.com"),
                ("Bob Smith", "bob@example.com"),
                ("Charlie Brown", "charlie@example.com"),
            ],
        )

        cursor.executemany(
            "INSERT INTO products (name, price, category, stock) VALUES (?, ?, ?, ?)",
            [
                ("Laptop", 999.99, "Electronics", 50),
                ("Mouse", 29.99, "Electronics", 200),
                ("Desk Chair", 199.99, "Furniture", 75),
                ("Monitor", 299.99, "Electronics", 100),
            ],
        )

        cursor.executemany(
            "INSERT INTO orders (user_id, product_id, quantity) VALUES (?, ?, ?)",
            [
                (1, 1, 1),
                (1, 2, 2),
                (2, 3, 1),
                (3, 4, 1),
                (2, 1, 1),
            ],
        )

        conn.commit()
        conn.close()
