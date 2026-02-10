"""SQL generation engine powered by LangChain and LLMs"""
from typing import Optional, Dict, List
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI


class SQLPromptBuilder:
    """Builds prompts for SQL generation"""

    SQL_GENERATION_TEMPLATE = """You are an expert SQL database assistant. Your task is to convert natural language questions into SQL queries.

Database Schema:
{schema_context}

Conversation History:
{conversation_history}

User Question: {user_query}

Instructions:
1. Generate a valid SQL query that answers the user's question
2. Use ONLY the tables and columns from the schema provided
3. Never use subqueries if a simpler query works
4. Add comments explaining complex parts
5. Validate the query syntax before returning

Return ONLY the SQL query, no explanations."""

    EXPLANATION_TEMPLATE = """Given the following SQL query and database schema, explain what this query does in simple English.

Schema:
{schema_context}

Query:
{generated_sql}

Provide a clear, concise explanation (2-3 sentences) of what this query does."""

    @staticmethod
    def build_sql_generation_prompt(
        schema_context: str, conversation_history: str, user_query: str
    ) -> str:
        """Build the SQL generation prompt"""
        return SQLPromptBuilder.SQL_GENERATION_TEMPLATE.format(
            schema_context=schema_context,
            conversation_history=conversation_history,
            user_query=user_query,
        )

    @staticmethod
    def build_explanation_prompt(schema_context: str, generated_sql: str) -> str:
        """Build the SQL explanation prompt"""
        return SQLPromptBuilder.EXPLANATION_TEMPLATE.format(
            schema_context=schema_context, generated_sql=generated_sql
        )


class SQLValidator:
    """Validates generated SQL queries"""

    # Keywords that should not appear in queries for safety
    DANGEROUS_KEYWORDS = ["DROP", "DELETE", "TRUNCATE", "ALTER", "CREATE INDEX", "EXEC"]

    @staticmethod
    def validate(sql_query: str) -> tuple[bool, Optional[str]]:
        """Validate SQL query for safety and syntax"""
        sql_upper = sql_query.upper().strip()

        # Check for dangerous keywords
        for keyword in SQLValidator.DANGEROUS_KEYWORDS:
            if keyword in sql_upper:
                return False, f"Dangerous operation detected: {keyword}"

        # Check for basic syntax
        if not sql_upper.startswith(("SELECT", "WITH")):
            return False, "Query must be a SELECT statement"

        # Check for injection patterns
        if ";" in sql_query and not sql_query.rstrip().endswith(";"):
            return False, "Multiple statements detected"

        return True, None


class SQLGenerator:
    """Generates SQL queries from natural language using LangChain"""

    def __init__(self, openai_api_key: str, model: str = "gpt-4"):
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
        """Generate SQL from natural language query"""
        try:
            # Build prompt
            prompt = self.prompt_builder.build_sql_generation_prompt(
                schema_context=schema_context,
                conversation_history=conversation_history,
                user_query=user_query,
            )

            # Generate SQL
            response = self.llm.invoke(prompt)
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
    """Mock SQL generator for testing without API key"""

    def generate(
        self,
        user_query: str,
        schema_context: str,
        conversation_history: str = "",
    ) -> Dict[str, str]:
        """Mock SQL generation"""
        # Simple mock that returns a SELECT * query
        return {
            "success": True,
            "generated_sql": "SELECT * FROM users LIMIT 10;",
            "explanation": "This query retrieves the first 10 records from the users table.",
            "original_query": user_query,
        }

    def validate_query(self, sql_query: str) -> tuple[bool, Optional[str]]:
        """Validate a SQL query"""
        return SQLValidator.validate(sql_query)
