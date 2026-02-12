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
        self, query: str, timeout_seconds: int = 30, max_rows: int = 1000
    ) -> Dict[str, Any]:
        """Execute a SQL query safely with row limit enforcement"""
        try:
            with self.get_connection() as conn:
                result = conn.execute(text(query))
                rows = result.fetchall()
                columns = list(result.keys())

                # Enforce max_rows limit
                truncated = len(rows) > max_rows
                if truncated:
                    rows = rows[:max_rows]

                response = {
                    "success": True,
                    "columns": columns,
                    "rows": [dict(zip(columns, row)) for row in rows],
                    "row_count": len(rows),
                }
                if truncated:
                    response["warning"] = (
                        f"Results truncated to {max_rows} rows"
                    )
                return response
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
        """Get database schema information including foreign keys"""
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

            # Extract foreign key relationships
            foreign_keys = []
            for fk in inspector.get_foreign_keys(table_name):
                foreign_keys.append({
                    "column": fk["constrained_columns"],
                    "references_table": fk["referred_table"],
                    "references_column": fk["referred_columns"],
                })

            schema[table_name] = {
                "columns": columns,
                "foreign_keys": foreign_keys,
                "description": f"Table {table_name}",
            }

        return schema

    def get_sample_data(self, table_name: str, limit: int = 5) -> List[Dict]:
        """Get sample data from a table"""
        try:
            # Validate table_name against schema to prevent injection
            inspector = inspect(self.engine)
            if table_name not in inspector.get_table_names():
                return []

            safe_limit = int(limit)
            query = f"SELECT * FROM {table_name} LIMIT {safe_limit}"
            result = self.execute_query(query)
            return result.get("rows", []) if result["success"] else []
        except Exception:
            return []


class QueryExecutor:
    """Executes queries with safety checks and formatting"""

    def __init__(
        self,
        db_connection: DatabaseConnection,
        timeout_seconds: int = 30,
        max_rows: int = 1000,
    ):
        self.db = db_connection
        self.timeout_seconds = timeout_seconds
        self.max_rows = max_rows

    def execute(self, sql_query: str) -> Dict[str, Any]:
        """Execute a SQL query and return formatted results"""
        result = self.db.execute_query(
            sql_query,
            timeout_seconds=self.timeout_seconds,
            max_rows=self.max_rows,
        )

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
        """Create a sample SQLite database with retail commerce schema"""
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Customers table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS customers (
                customer_id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                email TEXT UNIQUE,
                region TEXT,
                signup_date DATE
            )
        """)

        # Products table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS products (
                product_id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                category TEXT,
                price DECIMAL(10,2)
            )
        """)

        # Orders table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS orders (
                order_id INTEGER PRIMARY KEY,
                customer_id INTEGER,
                order_date DATE,
                total_amount DECIMAL(10,2),
                FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
            )
        """)

        # Order Items table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS order_items (
                item_id INTEGER PRIMARY KEY,
                order_id INTEGER,
                product_id INTEGER,
                quantity INTEGER,
                subtotal DECIMAL(10,2),
                FOREIGN KEY (order_id) REFERENCES orders(order_id),
                FOREIGN KEY (product_id) REFERENCES products(product_id)
            )
        """)

        # Sample Customers
        cursor.executemany(
            "INSERT INTO customers (customer_id, name, email, region, signup_date) VALUES (?, ?, ?, ?, ?)",
            [
                (1, 'Alice Chen', 'alice.chen@example.com', 'California', '2023-02-01'),
                (2, 'John Patel', 'john.patel@example.com', 'New York', '2023-05-15'),
                (3, 'Maria Lopez', 'maria.lopez@example.com', 'Texas', '2022-11-30'),
                (4, 'David Johnson', 'david.johnson@example.com', 'Florida', '2023-07-22'),
                (5, 'Sofia Khan', 'sofia.khan@example.com', 'Illinois', '2023-04-10'),
            ],
        )

        # Sample Products
        cursor.executemany(
            "INSERT INTO products (product_id, name, category, price) VALUES (?, ?, ?, ?)",
            [
                (1, 'Laptop Pro 15', 'Electronics', 1200.00),
                (2, 'Wireless Mouse', 'Accessories', 40.00),
                (3, 'Standing Desk', 'Furniture', 300.00),
                (4, 'Noise Cancelling Headphones', 'Electronics', 150.00),
                (5, 'Office Chair Deluxe', 'Furniture', 180.00),
            ],
        )

        # Sample Orders (total_amount matches SUM of order_items.subtotal)
        cursor.executemany(
            "INSERT INTO orders (order_id, customer_id, order_date, total_amount) VALUES (?, ?, ?, ?)",
            [
                (101, 1, '2024-01-12', 1240.00),
                (102, 2, '2024-03-05', 230.00),
                (103, 3, '2024-02-20', 1910.00),
                (104, 1, '2024-04-02', 300.00),
                (105, 4, '2024-05-15', 450.00),
                (106, 5, '2024-06-10', 180.00),
            ],
        )

        # Sample Order Items (order 103 has 4 items for "more than 3" demo)
        cursor.executemany(
            "INSERT INTO order_items (item_id, order_id, product_id, quantity, subtotal) VALUES (?, ?, ?, ?, ?)",
            [
                (1, 101, 1, 1, 1200.00),
                (2, 101, 2, 1, 40.00),
                (3, 102, 2, 2, 80.00),
                (4, 102, 4, 1, 150.00),
                (5, 103, 3, 5, 1500.00),
                (6, 103, 2, 2, 80.00),
                (7, 103, 4, 1, 150.00),
                (8, 103, 5, 1, 180.00),
                (9, 104, 5, 1, 180.00),
                (10, 104, 2, 3, 120.00),
                (11, 105, 4, 3, 450.00),
                (12, 106, 5, 1, 180.00),
            ],
        )

        conn.commit()
        conn.close()
