"""Query execution and database management"""
from typing import List, Dict, Any, Optional
from contextlib import contextmanager
import random
import sqlite3
from datetime import date, timedelta
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.exc import SQLAlchemyError


class DatabaseConnection:
    """Manages database connections and queries"""

    def __init__(self, database_url: str, read_only: bool = True):
        self.database_url = database_url
        self._read_only = read_only

        if read_only and database_url.startswith("sqlite"):
            from sqlalchemy import event

            self.engine = create_engine(database_url)

            @event.listens_for(self.engine, "connect")
            def _set_sqlite_read_only(dbapi_conn, connection_record):
                cursor = dbapi_conn.cursor()
                cursor.execute("PRAGMA query_only = ON")
                cursor.close()
        else:
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

        # --- Programmatic data generation for realistic dataset ---
        rng = random.Random(42)

        first_names = [
            "Alice", "John", "Maria", "David", "Sofia", "James", "Emma",
            "Liam", "Olivia", "Noah", "Ava", "William", "Isabella", "Lucas",
            "Mia", "Henry", "Charlotte", "Alexander", "Amelia", "Benjamin",
            "Harper", "Daniel", "Evelyn", "Matthew", "Abigail", "Samuel",
            "Emily", "Joseph", "Ella", "Jackson",
        ]
        last_names = [
            "Chen", "Patel", "Lopez", "Johnson", "Khan", "Smith", "Williams",
            "Brown", "Jones", "Garcia", "Miller", "Davis", "Rodriguez",
            "Martinez", "Anderson", "Taylor", "Thomas", "Moore", "Martin",
            "Lee", "Thompson", "White", "Harris", "Clark", "Lewis", "Young",
            "Walker", "Hall", "Allen", "King",
        ]
        regions = [
            "California", "New York", "Texas", "Florida", "Illinois",
            "Washington", "Georgia", "Ohio", "Pennsylvania", "Colorado",
        ]
        product_catalog = [
            ("Laptop Pro 15", "Electronics", 1200.00),
            ("Wireless Mouse", "Accessories", 40.00),
            ("Standing Desk", "Furniture", 300.00),
            ("Noise Cancelling Headphones", "Electronics", 150.00),
            ("Office Chair Deluxe", "Furniture", 180.00),
            ("USB-C Hub", "Accessories", 45.00),
            ("Monitor 27 inch", "Electronics", 350.00),
            ("Desk Lamp LED", "Office Supplies", 35.00),
            ("Mechanical Keyboard", "Accessories", 120.00),
            ("Webcam HD", "Electronics", 80.00),
            ("Filing Cabinet", "Furniture", 200.00),
            ("Notebook Pack", "Office Supplies", 15.00),
            ("Printer Inkjet", "Electronics", 250.00),
            ("Mouse Pad XL", "Accessories", 25.00),
            ("Whiteboard Large", "Office Supplies", 90.00),
            ("Ergonomic Footrest", "Furniture", 65.00),
            ("Cable Management Kit", "Office Supplies", 20.00),
            ("External SSD 1TB", "Electronics", 110.00),
            ("Laptop Stand", "Accessories", 55.00),
            ("Desk Organizer", "Office Supplies", 30.00),
            ("Bluetooth Speaker", "Electronics", 75.00),
            ("Wireless Charger", "Accessories", 35.00),
            ("Bookshelf Compact", "Furniture", 150.00),
            ("Sticky Notes Bulk", "Office Supplies", 12.00),
            ("Tablet Pro 11", "Electronics", 650.00),
        ]

        # Customers: keep original 5 at IDs 1-5, generate 145 more (150 total)
        customers = [
            (1, "Alice Chen", "alice.chen@example.com", "California", "2023-02-01"),
            (2, "John Patel", "john.patel@example.com", "New York", "2023-05-15"),
            (3, "Maria Lopez", "maria.lopez@example.com", "Texas", "2022-11-30"),
            (4, "David Johnson", "david.johnson@example.com", "Florida", "2023-07-22"),
            (5, "Sofia Khan", "sofia.khan@example.com", "Illinois", "2023-04-10"),
        ]
        used_emails = {c[2] for c in customers}
        for i in range(6, 151):
            first = rng.choice(first_names)
            last = rng.choice(last_names)
            email = f"{first.lower()}.{last.lower()}{i}@example.com"
            while email in used_emails:
                email = f"{first.lower()}.{last.lower()}{i}x@example.com"
            used_emails.add(email)
            signup = date(2022, 1, 1) + timedelta(days=rng.randint(0, 1460))
            customers.append(
                (i, f"{first} {last}", email, rng.choice(regions), signup.isoformat())
            )
        cursor.executemany(
            "INSERT INTO customers (customer_id, name, email, region, signup_date) VALUES (?, ?, ?, ?, ?)",
            customers,
        )

        # Products: 20 items from catalog
        products = [
            (i + 1, name, cat, price)
            for i, (name, cat, price) in enumerate(product_catalog)
        ]
        cursor.executemany(
            "INSERT INTO products (product_id, name, category, price) VALUES (?, ?, ?, ?)",
            products,
        )

        # Build price lookup
        price_map = {p[0]: p[3] for p in products}
        num_products = len(products)

        # Orders (2500) and Order Items (~6500) generated together for consistency
        orders = []
        order_items = []
        item_id = 1
        # Date range: Jan 2023 to Feb 2026 (1132 days)
        date_start = date(2023, 1, 1)
        date_range_days = (date(2026, 2, 12) - date_start).days
        for order_id in range(101, 2601):
            cust_id = rng.randint(1, 150)
            order_date = date_start + timedelta(days=rng.randint(0, date_range_days))
            num_items = rng.randint(1, 5)
            order_total = 0.0
            for _ in range(num_items):
                prod_id = rng.randint(1, num_products)
                qty = rng.randint(1, 4)
                subtotal = round(qty * price_map[prod_id], 2)
                order_total += subtotal
                order_items.append((item_id, order_id, prod_id, qty, subtotal))
                item_id += 1
            orders.append(
                (order_id, cust_id, order_date.isoformat(), round(order_total, 2))
            )
        cursor.executemany(
            "INSERT INTO orders (order_id, customer_id, order_date, total_amount) VALUES (?, ?, ?, ?)",
            orders,
        )
        cursor.executemany(
            "INSERT INTO order_items (item_id, order_id, product_id, quantity, subtotal) VALUES (?, ?, ?, ?, ?)",
            order_items,
        )

        conn.commit()
        conn.close()
