# SQL Query Buddy - API Documentation

This document describes the core components and their APIs for developers extending or integrating with SQL Query Buddy.

## Core Components

### 1. NLP Processor (`src/components/nlp_processor.py`)

Handles natural language understanding and conversation context management.

#### Classes

**QueryParser**
```python
from src.components.nlp_processor import QueryParser

parser = QueryParser()
result = parser.parse("Show me top 10 users")
# Returns:
# {
#     "original_text": "Show me top 10 users",
#     "intent": "retrieve",  # retrieve, aggregate, ranking, comparison, trend, general
#     "entities": [],
#     "modifiers": {"limit": 10, "order_by": None, "filter": []}
# }
```

**ContextManager**
```python
from src.components.nlp_processor import ContextManager

manager = ContextManager()
manager.initialize_with_schema(schema_dict)
manager.add_response(user_input="Query", assistant_response="Response", generated_sql="SELECT...")
history = manager.get_full_context()
manager.reset()
```

---

### 2. RAG System (`src/components/rag_system.py`)

Retrieval Augmented Generation for schema-aware SQL queries.

#### Classes

**RAGSystem**
```python
from src.components.rag_system import RAGSystem, InMemoryVectorDB
from langchain.embeddings import OpenAIEmbeddings

embedding_provider = OpenAIEmbeddings()
vector_db = InMemoryVectorDB()
rag = RAGSystem(embedding_provider, vector_db)

# Initialize with schema
rag.initialize_schema(schema_dict)

# Retrieve relevant schema elements
context = rag.retrieve_context("Show top products", top_k=5)
schema_string = rag.get_schema_context_string("Show top products")
```

---

### 3. SQL Generator (`src/components/sql_generator.py`)

Generates SQL from natural language using LLMs.

#### Classes

**SQLGenerator**
```python
from src.components.sql_generator import SQLGenerator

generator = SQLGenerator(openai_api_key="sk-...", model="gpt-4")

result = generator.generate(
    user_query="Top 10 products",
    schema_context="Tables: products(id, name, price)",
    conversation_history=""
)
# Returns:
# {
#     "success": True,
#     "generated_sql": "SELECT * FROM products ORDER BY price DESC LIMIT 10",
#     "explanation": "This query retrieves the 10 most expensive products...",
#     "original_query": "Top 10 products"
# }

# Validate a query
is_valid, error = generator.validate_query("SELECT * FROM users")
```

---

### 4. Query Executor (`src/components/executor.py`)

Executes SQL queries safely against databases.

#### Classes

**DatabaseConnection**
```python
from src.components.executor import DatabaseConnection

db = DatabaseConnection("postgresql://user:pass@localhost/dbname")
schema = db.get_schema()
result = db.execute_query("SELECT * FROM users LIMIT 10")
sample_data = db.get_sample_data("users", limit=5)
```

**QueryExecutor**
```python
from src.components.executor import QueryExecutor

executor = QueryExecutor(db_connection)
result = executor.execute("SELECT * FROM users")
# Returns:
# {
#     "success": True,
#     "query": "SELECT * FROM users",
#     "row_count": 100,
#     "columns": ["id", "name", "email"],
#     "data": [{"id": 1, "name": "Alice", "email": "alice@example.com"}, ...],
#     "summary": "100 results found."
# }
```

---

### 5. Insights Generator (`src/components/insights.py`)

Generates AI-driven insights from query results.

#### Classes

**InsightGenerator**
```python
from src.components.insights import InsightGenerator

generator = InsightGenerator(openai_api_key="sk-...", model="gpt-4")
insights = generator.generate_insights(
    query_results=[{"name": "Alice", "spending": 1000}, ...],
    user_query="Top spending customers"
)
# Returns a string with actionable insights
```

**PatternDetector**
```python
from src.components.insights import PatternDetector

numeric_patterns = PatternDetector.detect_numeric_patterns(data)
# Returns: {"column_name": {"min": 0, "max": 100, "avg": 50.5, "count": 20}}

string_patterns = PatternDetector.detect_string_patterns(data)
# Returns: {"column_name": {"unique_values": [...], "unique_count": 5}}
```

**TrendAnalyzer**
```python
from src.components.insights import TrendAnalyzer

trends = TrendAnalyzer.analyze_trends(time_series_data)
# Returns: {"column": {"direction": "increasing", "average_change": 5.2, "total_change": 52}}
```

---

### 6. Query Optimizer (`src/components/optimizer.py`)

Analyzes queries and suggests optimizations.

#### Classes

**QueryOptimizer**
```python
from src.components.optimizer import QueryOptimizer

optimizer = QueryOptimizer()
result = optimizer.analyze("SELECT * FROM large_table")
# Returns:
# {
#     "total_suggestions": 2,
#     "optimization_level": "needs_optimization",
#     "suggestions": [
#         {
#             "type": "efficiency",
#             "severity": "medium",
#             "suggestion": "Specify only needed columns instead of SELECT *"
#         },
#         ...
#     ]
# }
```

---

## Configuration

All configuration is managed through environment variables loaded in `src/config.py`:

```python
from src.config import settings

print(settings.openai_api_key)
print(settings.database_url)
print(settings.openai_model)
```

See `.env.example` for all available settings.

---

## Integration Examples

### Example 1: Complete Query Processing Pipeline

```python
from src.components.nlp_processor import ContextManager
from src.components.rag_system import RAGSystem, InMemoryVectorDB
from src.components.sql_generator import SQLGenerator
from src.components.executor import DatabaseConnection, QueryExecutor
from src.components.optimizer import QueryOptimizer
from src.components.insights import InsightGenerator

# Initialize components
db = DatabaseConnection("sqlite:///retail.db")
context_mgr = ContextManager()
context_mgr.initialize_with_schema(db.get_schema())

sql_gen = SQLGenerator(openai_api_key="sk-...")
optimizer = QueryOptimizer()
executor = QueryExecutor(db)
insights_gen = InsightGenerator(openai_api_key="sk-...")

# Process user query
user_query = "Show me top 5 products by sales"

# Generate SQL
sql_result = sql_gen.generate(
    user_query=user_query,
    schema_context=str(db.get_schema()),
    conversation_history=context_mgr.get_full_context()
)

if sql_result["success"]:
    # Optimize
    opt_result = optimizer.analyze(sql_result["generated_sql"])

    # Execute
    exec_result = executor.execute(sql_result["generated_sql"])

    # Generate insights
    if exec_result["success"]:
        insights = insights_gen.generate_insights(
            exec_result["data"],
            user_query
        )

    # Update context
    context_mgr.add_response(
        user_input=user_query,
        assistant_response=insights,
        generated_sql=sql_result["generated_sql"]
    )
```

### Example 2: Custom Embedding Provider

```python
from src.components.rag_system import EmbeddingProvider, RAGSystem, InMemoryVectorDB

class CustomEmbeddingProvider(EmbeddingProvider):
    def embed(self, text: str) -> list:
        # Your embedding logic
        return [0.1, 0.2, 0.3, ...]

    def embed_batch(self, texts: list) -> list:
        return [self.embed(text) for text in texts]

provider = CustomEmbeddingProvider()
rag = RAGSystem(provider, InMemoryVectorDB())
```

---

## Testing

Run the test suite:

```bash
# All tests
pytest

# Specific test file
pytest tests/unit/test_components.py

# With coverage
pytest --cov=src tests/
```

---

## Error Handling

All components return structured results with success/error indicators:

```python
result = component.some_operation()

if result.get("success"):
    print("Operation succeeded")
else:
    error_msg = result.get("error", "Unknown error")
    print(f"Failed: {error_msg}")
```

---

## Performance Considerations

- **RAG System**: Vector similarity search is O(n) in memory. For production, use proper vector DB (FAISS, Chroma, Weaviate)
- **SQL Generator**: LLM calls take 1-5 seconds. Cache results for identical queries
- **Query Execution**: Use connection pooling for databases with many users
- **Insights Generation**: Optional feature, disable for faster responses

---

## Contributing

When adding new components:

1. Inherit from base classes where applicable
2. Return consistent result dictionaries with `success`, `error` fields
3. Add unit tests in `tests/unit/`
4. Add integration tests in `tests/integration/`
5. Document public APIs in docstrings

---

For more information, see:
- [README.md](../README.md)
- [specification.md](specification.md)
- [Source code](../src/)
