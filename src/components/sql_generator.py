"""SQL generation engine powered by LangChain and LLMs"""
from typing import Optional, Dict, List
import re

try:
    from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
    from langchain_core.messages import HumanMessage, SystemMessage
except ImportError:
    try:
        from langchain.prompts import PromptTemplate, ChatPromptTemplate
        from langchain.schema import HumanMessage, SystemMessage
    except ImportError:
        PromptTemplate = None
        ChatPromptTemplate = None
        HumanMessage = None
        SystemMessage = None

try:
    from langchain_openai import ChatOpenAI
except ImportError:
    try:
        from langchain.chat_models import ChatOpenAI
    except ImportError:
        ChatOpenAI = None


class SQLPromptBuilder:
    """Builds structured LangChain prompts for SQL generation."""

    SQL_SYSTEM_PROMPT = (
        "You are an expert SQL database assistant. Your task is to convert "
        "natural language questions into SQL queries.\n\n"
        "Instructions:\n"
        "1. Generate a valid SQL query that answers the user's question\n"
        "2. Use ONLY the tables and columns from the schema provided\n"
        "3. Never use subqueries if a simpler query works\n"
        "4. Add comments explaining complex parts\n"
        "5. Validate the query syntax before returning\n\n"
        "Return ONLY the SQL query, no explanations."
    )

    # LangChain PromptTemplate for SQL generation
    SQL_GENERATION_TEMPLATE = PromptTemplate(
        input_variables=["schema_context", "conversation_history", "user_query"],
        template=(
            "Database Schema:\n{schema_context}\n\n"
            "Conversation History:\n{conversation_history}\n\n"
            "User Question: {user_query}"
        ),
    ) if PromptTemplate else None

    # LangChain PromptTemplate for SQL explanation
    EXPLANATION_TEMPLATE = PromptTemplate(
        input_variables=["schema_context", "generated_sql"],
        template=(
            "Given the following SQL query and database schema, explain what "
            "this query does in simple English.\n\n"
            "Schema:\n{schema_context}\n\n"
            "Query:\n{generated_sql}\n\n"
            "Provide a clear, concise explanation (2-3 sentences) of what this query does."
        ),
    ) if PromptTemplate else None

    @staticmethod
    def build_sql_generation_prompt(
        schema_context: str, conversation_history: str, user_query: str
    ) -> str:
        """Build the SQL generation prompt using LangChain PromptTemplate."""
        if SQLPromptBuilder.SQL_GENERATION_TEMPLATE:
            return SQLPromptBuilder.SQL_GENERATION_TEMPLATE.format(
                schema_context=schema_context,
                conversation_history=conversation_history,
                user_query=user_query,
            )
        # Fallback if LangChain not available
        return (
            f"Database Schema:\n{schema_context}\n\n"
            f"Conversation History:\n{conversation_history}\n\n"
            f"User Question: {user_query}"
        )

    @staticmethod
    def build_explanation_prompt(schema_context: str, generated_sql: str) -> str:
        """Build the SQL explanation prompt using LangChain PromptTemplate."""
        if SQLPromptBuilder.EXPLANATION_TEMPLATE:
            return SQLPromptBuilder.EXPLANATION_TEMPLATE.format(
                schema_context=schema_context, generated_sql=generated_sql
            )
        return (
            f"Given the following SQL query and database schema, explain what "
            f"this query does in simple English.\n\n"
            f"Schema:\n{schema_context}\n\nQuery:\n{generated_sql}\n\n"
            f"Provide a clear, concise explanation (2-3 sentences)."
        )


class SQLValidator:
    """Validates generated SQL queries"""

    # Keywords that should not appear in queries for safety (matched as whole words)
    DANGEROUS_KEYWORDS = [
        "DROP", "DELETE", "TRUNCATE", "ALTER", "CREATE INDEX",
        "EXEC", "UPDATE", "INSERT", "GRANT", "REVOKE",
    ]

    @staticmethod
    def validate(sql_query: str) -> tuple[bool, Optional[str]]:
        """Validate SQL query for safety and syntax"""
        sql_upper = sql_query.upper().strip()

        # Check for dangerous keywords using word boundaries
        for keyword in SQLValidator.DANGEROUS_KEYWORDS:
            if re.search(rf"\b{keyword}\b", sql_upper):
                return False, f"Dangerous operation detected: {keyword}"

        # Check for basic syntax
        if not sql_upper.startswith(("SELECT", "WITH")):
            return False, "Query must be a SELECT statement"

        # Check for multiple statements (reject any semicolon not at the very end)
        stripped = sql_query.strip().rstrip(";")
        if ";" in stripped:
            return False, "Multiple statements detected"

        return True, None


class SQLGenerator:
    """Generates SQL queries from natural language using LangChain.

    Uses LangChain ChatOpenAI with structured message types
    (SystemMessage + HumanMessage) for proper prompt engineering.
    """

    def __init__(self, openai_api_key: str, model: str = "gpt-4"):
        if ChatOpenAI is None:
            raise ImportError("LangChain OpenAI integration not available")
        self.llm = ChatOpenAI(
            api_key=openai_api_key, model=model, temperature=0.2
        )
        self.validator = SQLValidator()
        self.prompt_builder = SQLPromptBuilder()

    def generate(
        self,
        user_query: str,
        schema_context: str,
        conversation_history: str = "",
    ) -> Dict[str, str]:
        """Generate SQL from natural language query using LangChain messages."""
        try:
            # Build structured prompt with LangChain message types
            user_content = self.prompt_builder.build_sql_generation_prompt(
                schema_context=schema_context,
                conversation_history=conversation_history,
                user_query=user_query,
            )

            messages = []
            if SystemMessage:
                messages.append(SystemMessage(content=self.prompt_builder.SQL_SYSTEM_PROMPT))
                messages.append(HumanMessage(content=user_content))
            else:
                messages = [user_content]

            # Generate SQL via LangChain LLM
            response = self.llm.invoke(messages)
            generated_sql = response.content.strip()

            # Remove markdown code blocks if present
            if generated_sql.startswith("```"):
                generated_sql = generated_sql.split("```")[1]
                if generated_sql.startswith("sql"):
                    generated_sql = generated_sql[3:]
                generated_sql = generated_sql.strip()

            # Validate
            is_valid, error_msg = self.validator.validate(generated_sql)
            if not is_valid:
                return {
                    "success": False,
                    "error": error_msg,
                    "generated_sql": generated_sql,
                }

            # Generate explanation
            explanation = self._generate_explanation(schema_context, generated_sql)

            return {
                "success": True,
                "generated_sql": generated_sql,
                "explanation": explanation,
                "original_query": user_query,
            }

        except Exception as e:
            return {
                "success": False,
                "error": f"SQL generation failed: {str(e)}",
            }

    def _generate_explanation(self, schema_context: str, generated_sql: str) -> str:
        """Generate a natural language explanation of the SQL"""
        try:
            prompt = self.prompt_builder.build_explanation_prompt(
                schema_context=schema_context, generated_sql=generated_sql
            )
            response = self.llm.invoke(prompt)
            return response.content.strip()
        except Exception:
            return "Unable to generate explanation"

    def validate_query(self, sql_query: str) -> tuple[bool, Optional[str]]:
        """Validate a SQL query"""
        return self.validator.validate(sql_query)


class SQLGeneratorMock:
    """Context-aware mock SQL generator for demo without API key.

    Parses user query keywords to generate relevant SQL across
    the retail commerce schema (customers, products, orders, order_items).
    """

    # Mapping of keyword patterns to SQL queries and explanations
    QUERY_PATTERNS = [
        # Aggregation / top customers
        {
            "keywords": ["top", "customer", "purchase", "spending", "spent"],
            "sql": (
                "SELECT c.name, SUM(o.total_amount) AS total_spent "
                "FROM customers c "
                "JOIN orders o ON c.customer_id = o.customer_id "
                "GROUP BY c.customer_id, c.name "
                "ORDER BY total_spent DESC LIMIT 5;"
            ),
            "explanation": (
                "This query joins customers with their orders, calculates total spending "
                "per customer, and returns the top 5 by total purchase amount."
            ),
        },
        # Revenue by category
        {
            "keywords": ["category", "revenue", "product category"],
            "sql": (
                "SELECT p.category, SUM(oi.subtotal) AS total_revenue "
                "FROM products p "
                "JOIN order_items oi ON p.product_id = oi.product_id "
                "GROUP BY p.category "
                "ORDER BY total_revenue DESC;"
            ),
            "explanation": (
                "This query calculates total revenue per product category by joining "
                "products with order items and summing subtotals."
            ),
        },
        # Sales per region
        {
            "keywords": ["region", "sales", "per region"],
            "sql": (
                "SELECT c.region, SUM(o.total_amount) AS total_sales "
                "FROM customers c "
                "JOIN orders o ON c.customer_id = o.customer_id "
                "GROUP BY c.region "
                "ORDER BY total_sales DESC;"
            ),
            "explanation": (
                "This query groups orders by customer region and calculates total sales "
                "for each region."
            ),
        },
        # Average order value (general)
        {
            "keywords": ["average", "order value", "avg"],
            "sql": (
                "SELECT AVG(total_amount) AS avg_order_value, "
                "COUNT(*) AS total_orders "
                "FROM orders;"
            ),
            "explanation": (
                "This query calculates the average order value and total number of orders."
            ),
        },
        # Average order value for returning customers
        {
            "keywords": ["average", "returning", "repeat", "order value"],
            "sql": (
                "SELECT AVG(total_amount) AS avg_order_value, "
                "COUNT(*) AS total_orders "
                "FROM orders "
                "WHERE customer_id IN ("
                "SELECT customer_id FROM orders "
                "GROUP BY customer_id HAVING COUNT(*) >= 2);"
            ),
            "explanation": (
                "This query calculates the average order value only for returning "
                "customers (those who have placed at least 2 orders)."
            ),
        },
        # Monthly revenue / trend
        {
            "keywords": ["monthly", "revenue", "trend", "over time"],
            "sql": (
                "SELECT strftime('%Y-%m', order_date) AS month, "
                "SUM(total_amount) AS monthly_revenue, "
                "COUNT(*) AS order_count "
                "FROM orders "
                "GROUP BY month "
                "ORDER BY month;"
            ),
            "explanation": (
                "This query groups orders by month and calculates monthly revenue and "
                "order count to show trends over time."
            ),
        },
        # Products / product list
        {
            "keywords": ["product", "products", "catalog", "inventory"],
            "sql": (
                "SELECT product_id, name, category, price "
                "FROM products "
                "ORDER BY price DESC;"
            ),
            "explanation": (
                "This query retrieves all products from the catalog, ordered by price "
                "from highest to lowest."
            ),
        },
        # Orders with more than 3 items
        {
            "keywords": ["order", "items", "more than", "contain"],
            "sql": (
                "SELECT o.order_id, o.order_date, o.total_amount, "
                "COUNT(oi.item_id) AS item_count "
                "FROM orders o "
                "JOIN order_items oi ON o.order_id = oi.order_id "
                "GROUP BY o.order_id "
                "HAVING item_count > 3 "
                "ORDER BY item_count DESC;"
            ),
            "explanation": (
                "This query finds orders containing more than 3 items by joining "
                "orders with order_items and filtering with HAVING."
            ),
        },
        # Unique products sold (general)
        {
            "keywords": ["unique", "sold", "distinct"],
            "sql": (
                "SELECT COUNT(DISTINCT oi.product_id) AS unique_products_sold "
                "FROM order_items oi;"
            ),
            "explanation": (
                "This query counts the number of distinct products that appear in order items."
            ),
        },
        # Unique products sold in January
        {
            "keywords": ["unique", "sold", "january", "products sold"],
            "sql": (
                "SELECT COUNT(DISTINCT oi.product_id) AS unique_products_sold "
                "FROM order_items oi "
                "JOIN orders o ON oi.order_id = o.order_id "
                "WHERE strftime('%m', o.order_date) = '01';"
            ),
            "explanation": (
                "This query counts distinct products sold in January by joining "
                "order items with orders and filtering by month."
            ),
        },
        # Orders listing
        {
            "keywords": ["order", "orders", "purchase"],
            "sql": (
                "SELECT o.order_id, c.name AS customer_name, o.order_date, o.total_amount "
                "FROM orders o "
                "JOIN customers c ON o.customer_id = c.customer_id "
                "ORDER BY o.order_date DESC;"
            ),
            "explanation": (
                "This query lists all orders with customer names, ordered by date (most recent first)."
            ),
        },
        # Count queries
        {
            "keywords": ["how many", "count", "total number"],
            "sql": (
                "SELECT "
                "(SELECT COUNT(*) FROM customers) AS total_customers, "
                "(SELECT COUNT(*) FROM products) AS total_products, "
                "(SELECT COUNT(*) FROM orders) AS total_orders;"
            ),
            "explanation": (
                "This query counts the total number of customers, products, and orders in the database."
            ),
        },
    ]

    # Phrases that indicate a follow-up referencing a previous result
    FOLLOW_UP_PHRASES = [
        "from the previous", "from that", "from those", "from the result",
        "filter", "narrow", "only show", "now show", "but only",
        "of those", "of them", "among them", "among those",
        "refine", "drill down", "zoom in",
    ]

    # Map of filter keywords to SQL WHERE conditions
    # Uses generic column names (no table aliases) for subquery compatibility
    FILTER_MAP = {
        # Regions
        "california": "region = 'California'",
        "new york": "region = 'New York'",
        "texas": "region = 'Texas'",
        "florida": "region = 'Florida'",
        "illinois": "region = 'Illinois'",
        # Categories
        "electronics": "category = 'Electronics'",
        "furniture": "category = 'Furniture'",
        "accessories": "category = 'Accessories'",
        # Specific customers
        "alice": "name LIKE '%Alice%'",
        "john": "name LIKE '%John%'",
        "maria": "name LIKE '%Maria%'",
        "david": "name LIKE '%David%'",
        "sofia": "name LIKE '%Sofia%'",
    }

    def __init__(self):
        self._last_sql: Optional[str] = None
        self._last_explanation: Optional[str] = None

    def _is_follow_up(self, query_lower: str) -> bool:
        """Detect if the query is a follow-up referencing previous results."""
        return any(phrase in query_lower for phrase in self.FOLLOW_UP_PHRASES)

    # Region filter queries that include the region column
    REGION_QUERY_TEMPLATE = (
        "SELECT c.name, c.region, SUM(o.total_amount) AS total_spent "
        "FROM customers c "
        "JOIN orders o ON c.customer_id = o.customer_id "
        "WHERE {condition} "
        "GROUP BY c.customer_id, c.name, c.region "
        "ORDER BY total_spent DESC;"
    )

    # Category filter queries
    CATEGORY_QUERY_TEMPLATE = (
        "SELECT p.name, p.category, SUM(oi.subtotal) AS total_revenue "
        "FROM products p "
        "JOIN order_items oi ON p.product_id = oi.product_id "
        "WHERE {condition} "
        "GROUP BY p.product_id, p.name, p.category "
        "ORDER BY total_revenue DESC;"
    )

    def _build_follow_up_sql(self, user_query: str) -> Optional[Dict[str, str]]:
        """Build a follow-up query that refines the previous result."""
        if not self._last_sql:
            return None

        query_lower = user_query.lower()
        conditions = []
        filter_type = None  # track what kind of filter for template selection
        for keyword, condition in self.FILTER_MAP.items():
            if keyword in query_lower:
                conditions.append(condition)
                if "region" in condition:
                    filter_type = "region"
                elif "category" in condition:
                    filter_type = "category"
                elif "name" in condition:
                    filter_type = "name"

        # Extract numeric filters like "more than 1000", "greater than 500"
        numeric_match = re.search(
            r"(?:more than|greater than|above|over|exceeding)\s+(\d+(?:\.\d+)?)",
            query_lower,
        )
        if numeric_match:
            threshold = numeric_match.group(1)
            base_upper = self._last_sql.upper()
            if "TOTAL_SPENT" in base_upper:
                conditions.append(f"total_spent > {threshold}")
            elif "TOTAL_SALES" in base_upper:
                conditions.append(f"total_sales > {threshold}")
            elif "TOTAL_REVENUE" in base_upper:
                conditions.append(f"total_revenue > {threshold}")
            elif "TOTAL_AMOUNT" in base_upper:
                conditions.append(f"total_amount > {threshold}")
            filter_type = "numeric"

        if not conditions:
            return None

        where_clause = " AND ".join(conditions)

        # For region/name filters on a customers query, build a fresh query
        # that includes the filter column (avoids subquery column issues)
        if filter_type == "region" or filter_type == "name":
            follow_up_sql = self.REGION_QUERY_TEMPLATE.format(
                condition=" AND ".join(
                    c for c in conditions if "region" in c or "name" in c
                )
            )
        elif filter_type == "category":
            follow_up_sql = self.CATEGORY_QUERY_TEMPLATE.format(
                condition=" AND ".join(c for c in conditions if "category" in c)
            )
        elif filter_type == "numeric":
            # Numeric filters can safely wrap the previous query as subquery
            base_sql = self._last_sql.rstrip(";").strip()
            numeric_conds = [c for c in conditions if ">" in c]
            follow_up_sql = (
                f"SELECT * FROM ({base_sql}) AS prev_result "
                f"WHERE {' AND '.join(numeric_conds)};"
            )
        else:
            base_sql = self._last_sql.rstrip(";").strip()
            follow_up_sql = (
                f"SELECT * FROM ({base_sql}) AS prev_result "
                f"WHERE {where_clause};"
            )

        explanation = (
            f"This query refines the previous result by applying filters: "
            f"{where_clause}."
        )

        return {
            "success": True,
            "generated_sql": follow_up_sql,
            "explanation": explanation,
            "original_query": user_query,
        }

    def _extract_last_sql_from_history(self, conversation_history: str) -> None:
        """Extract the most recent SQL from conversation history string."""
        sql_matches = re.findall(r"SQL:\s*(.+?)(?:\n|$)", conversation_history)
        if sql_matches:
            self._last_sql = sql_matches[-1].strip()

    def generate(
        self,
        user_query: str,
        schema_context: str,
        conversation_history: str = "",
    ) -> Dict[str, str]:
        """Generate SQL by matching user query keywords to predefined patterns.

        Uses schema_context for entity hints and conversation_history for
        follow-up detection. Supports context-aware follow-up queries.
        """
        query_lower = user_query.lower()

        # Restore context from conversation_history if _last_sql is missing
        if not self._last_sql and conversation_history:
            self._extract_last_sql_from_history(conversation_history)

        # Check if this is a follow-up query
        if self._is_follow_up(query_lower):
            if self._last_sql:
                follow_up = self._build_follow_up_sql(user_query)
                if follow_up:
                    self._last_sql = follow_up["generated_sql"]
                    self._last_explanation = follow_up["explanation"]
                    return follow_up
            else:
                # No prior context available for follow-up
                return {
                    "success": True,
                    "generated_sql": "SELECT * FROM customers LIMIT 10;",
                    "explanation": (
                        "No previous query found to follow up on. "
                        "Please ask a question first, then use follow-up "
                        "queries to refine the results."
                    ),
                    "original_query": user_query,
                }

        # Extract entity hints from schema_context to boost scoring
        entity_boost = set()
        if schema_context:
            ctx_lower = schema_context.lower()
            if "referenced entities:" in ctx_lower:
                match = re.search(
                    r"referenced entities:\s*(.+?)(?:\n|$)", ctx_lower
                )
                if match and match.group(1).strip() != "none":
                    entity_boost = {
                        e.strip() for e in match.group(1).split(",")
                    }

        # Standard keyword matching with entity boosting
        best_match = None
        best_score = 0

        for pattern in self.QUERY_PATTERNS:
            score = sum(1 for kw in pattern["keywords"] if kw in query_lower)
            # Boost score if NLP entities overlap with pattern keywords
            if entity_boost:
                score += sum(
                    0.5 for kw in pattern["keywords"] if kw in entity_boost
                )
            if score > best_score:
                best_score = score
                best_match = pattern

        if best_match and best_score > 0:
            self._last_sql = best_match["sql"]
            self._last_explanation = best_match["explanation"]
            return {
                "success": True,
                "generated_sql": best_match["sql"],
                "explanation": best_match["explanation"],
                "original_query": user_query,
            }

        # Default fallback: show all customers
        fallback_sql = "SELECT * FROM customers LIMIT 10;"
        self._last_sql = fallback_sql
        self._last_explanation = (
            "This query retrieves the first 10 customers from the database. "
            "Try asking about specific topics like 'top customers by spending', "
            "'revenue by category', or 'monthly sales trend'."
        )
        return {
            "success": True,
            "generated_sql": fallback_sql,
            "explanation": self._last_explanation,
            "original_query": user_query,
        }

    def validate_query(self, sql_query: str) -> tuple[bool, Optional[str]]:
        """Validate a SQL query"""
        return SQLValidator.validate(sql_query)
