# SQL Query Buddy - Technical Specification

**Document Version**: 1.0
**Last Updated**: February 10, 2026
**Status**: Active Development
**Contest**: Codecademy GenAI Bootcamp
**Deadline**: February 15, 2026

---

## 1. Executive Summary

### Purpose
SQL Query Buddy is an intelligent conversational AI agent that bridges the gap between natural language and database queries. It empowers non-technical users to extract insights from databases using natural language while maintaining explainability and safety.

### Goals
1. Convert natural language questions to accurate SQL queries with >85% semantic accuracy
2. Provide context-aware conversational experience with multi-turn support
3. Generate actionable insights beyond raw query results
4. Suggest query optimizations for performance improvement
5. Ensure security through query sandboxing and injection prevention
6. Deliver results in <3 seconds for typical queries

### Target Users
- Business analysts without SQL expertise
- Data scientists exploring new datasets
- Product managers analyzing user behavior
- Non-technical stakeholders making data-driven decisions
- Developers building data-driven applications

### Success Criteria
- ✓ Accurately translates 85%+ of conversational queries to correct SQL
- ✓ Supports multi-turn conversations with persistent context
- ✓ Provides meaningful insights on query results
- ✓ Suggests optimization improvements
- ✓ Explains generated SQL in natural language
- ✓ Zero SQL injection vulnerabilities
- ✓ <3 second query response time
- ✓ Intuitive web interface with >8/10 usability score

---

## 2. System Architecture

### 2.1 High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      User Interface Layer                        │
│                    (Gradio/React Chat UI)                        │
└────────────────────────────┬────────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────────┐
│               Natural Language Processing Layer                  │
│  ┌──────────────────┐  ┌──────────────────┐  ┌─────────────┐   │
│  │  Query Parser    │  │ Context Manager  │  │ Conversation│   │
│  │                  │  │                  │  │   History   │   │
│  └──────────────────┘  └──────────────────┘  └─────────────┘   │
└────────────────────────────┬────────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────────┐
│                    RAG System Layer                              │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ Schema Embeddings │ Vector Search │ Semantic Retrieval  │   │
│  └─────────────────────────────────────────────────────────┘   │
│              ↓                                                   │
│        Vector Database (FAISS/Chroma)                           │
└────────────────────────────┬────────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────────┐
│             SQL Generation Engine (LangChain Agent)              │
│  ┌──────────────────┐  ┌──────────────────┐  ┌─────────────┐   │
│  │ Prompt Templates │  │ Multi-table Logic│  │   Validator  │   │
│  └──────────────────┘  └──────────────────┘  └─────────────┘   │
└────────────────────────────┬────────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────────┐
│         Query Optimization & Safety Layer                        │
│  ┌──────────────────┐  ┌──────────────────┐  ┌─────────────┐   │
│  │  Optimizer       │  │   Query Sandbox  │  │ Injection   │   │
│  │                  │  │                  │  │ Prevention  │   │
│  └──────────────────┘  └──────────────────┘  └─────────────┘   │
└────────────────────────────┬────────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────────┐
│         Query Execution Layer                                    │
│  ┌──────────────────┐  ┌──────────────────┐  ┌─────────────┐   │
│  │ DB Connection Mgr│  │  Query Executor  │  │   Formatter │   │
│  └──────────────────┘  └──────────────────┘  └─────────────┘   │
└────────────────────────────┬────────────────────────────────────┘
                             │
                    ┌────────▼────────┐
                    │  Target Database │
                    │ (SQLite/PG/MySQL)│
                    └─────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────────┐
│         Insight Generation Layer                                 │
│  ┌──────────────────┐  ┌──────────────────┐  ┌─────────────┐   │
│  │ Pattern Detection│  │ Trend Analysis   │  │   NL Insights   │
│  └──────────────────┘  └──────────────────┘  └─────────────┘   │
└────────────────────────────┬────────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────────┐
│                    Response Layer                                │
│            (SQL | Results | Insights | Explanations)            │
└────────────────────────────────────────────────────────────────┘
```

### 2.2 Data Flow Architecture

```
User Query
    ↓
[Parsing] → Extract intent, entities, temporal references
    ↓
[Context Retrieval] → Load conversation history, previous queries
    ↓
[Schema Search] → Use RAG to find relevant tables/columns
    ↓
[Prompt Engineering] → Build contextualized LangChain prompt
    ↓
[SQL Generation] → LLM generates SQL with reasoning
    ↓
[Validation] → Check syntax and semantic correctness
    ↓
[Optimization] → Suggest improvements, apply rewrites
    ↓
[Safety Check] → Validate against injection patterns
    ↓
[Execution] → Run query with timeout protection
    ↓
[Result Processing] → Format results, aggregate stats
    ↓
[Insight Generation] → Analyze results, generate insights
    ↓
[Response Assembly] → Combine SQL, results, insights, explanations
    ↓
User Response
```

### 2.3 RAG Pipeline Design

**Objective**: Retrieve relevant schema information to guide SQL generation

**Process**:
1. **Schema Embedding Phase** (Initialization)
   - Extract all tables, columns, relationships
   - Create descriptions: "customers table with id, name, email, created_at columns"
   - Generate embeddings for each entity using OpenAI Embeddings

2. **Query Time Retrieval** (Per Query)
   - Embed the user's natural language query
   - Perform similarity search in vector database
   - Retrieve top-K relevant tables/columns (K=5-10)
   - Include sample values and relationships

3. **Context Building**
   - Combine retrieved schema with conversation history
   - Add query rewrite hints based on patterns
   - Include optimization suggestions from previous runs

**Example**:
```
User: "Top products by revenue last month"
    ↓
Embedding: [0.34, -0.12, 0.89, ...] (semantic vector)
    ↓
Retrieval: [
  "products table (id, name, category, price)",
  "order_items table (order_id, product_id, quantity, unit_price)",
  "orders table (id, customer_id, order_date)"
]
    ↓
Context: Use products, order_items, orders for revenue calculation
```

---

## 3. Core Components

### 3.1 Frontend (Chat Interface)

**Technology**: Gradio or React
**Purpose**: Provide intuitive conversational interface

**UI Requirements**:
- Chat message layout with user/assistant distinction
- Message timestamp and metadata
- SQL code block display with syntax highlighting
- Results table with pagination and sorting
- Insights panel with key findings
- Query explanation section
- Conversation history sidebar
- Settings/preferences panel

**Component Breakdown**:
```python
ChatInterface:
├── MessageDisplay
│   ├── UserMessage
│   ├── AssistantMessage
│   └── SystemMessage
├── SQLViewer
│   ├── CodeBlock (syntax highlighted)
│   ├── OptimizationSuggestions
│   └── ExplanationPanel
├── ResultsDisplay
│   ├── DataTable (with sorting/pagination)
│   ├── StatsSummary
│   └── VisualizationPanel
├── InsightsPanel
│   ├── KeyFindings
│   ├── Trends
│   └── Anomalies
└── HistoryPanel
    ├── PreviousQueries
    └── SavedQueries
```

**User Interactions**:
- Type natural language query
- Submit query (Enter key or button)
- Review generated SQL
- Accept, edit, or reject SQL
- View results and insights
- Ask follow-up questions
- Save/export conversations

### 3.2 Natural Language Processing Layer

**Purpose**: Parse user input and maintain conversation state

**Key Functions**:

**Query Parser**:
```python
def parse_query(user_input, conversation_history):
    """
    Extract:
    - Primary intent (SELECT, ANALYSIS, COMPARISON)
    - Entities (tables, columns, values)
    - Time references (last month, Q1, 2024)
    - Aggregations (SUM, COUNT, AVG, etc.)
    - Filters and conditions
    - Sort order and limits
    Returns: ParsedQuery object
    """
```

**Context Manager**:
```python
class ContextManager:
    def __init__(self, max_history=10):
        self.conversation_history = []
        self.last_query_context = {}
        self.previous_tables = set()

    def add_turn(self, user_input, assistant_response):
        """Add Q&A pair to history"""

    def get_context(self):
        """Return relevant context for current turn"""

    def update_last_context(self, tables, columns, filters):
        """Update context from last successful query"""
```

**Conversation State Model**:
```json
{
  "conversation_id": "uuid",
  "session_start": "timestamp",
  "turns": [
    {
      "turn_id": 1,
      "user_input": "...",
      "parsed_intent": "...",
      "entities": {...},
      "generated_sql": "...",
      "results_summary": {...},
      "insights": [...]
    }
  ],
  "context": {
    "last_tables": [...],
    "last_columns": [...],
    "implicit_filters": {...}
  }
}
```

### 3.3 RAG System

**Purpose**: Semantic retrieval of relevant schema and examples

**Vector Database Schema**:
```
Table: schema_embeddings
├── id (UUID)
├── entity_type (TABLE | COLUMN | RELATIONSHIP)
├── entity_name (string)
├── description (string)
├── embedding (vector[1536])
├── table_reference (string)
├── example_values (json)
├── created_at (timestamp)

Table: query_examples
├── id (UUID)
├── user_query (string)
├── generated_sql (string)
├── query_embedding (vector[1536])
├── success (boolean)
├── execution_time (float)
├── created_at (timestamp)
```

**Embedding Strategy**:
1. **Table Embeddings**: Table name + column names + sample descriptions
2. **Column Embeddings**: Column name + data type + description
3. **Relationship Embeddings**: Foreign key descriptions
4. **Example Embeddings**: Previous successful queries

**Semantic Search**:
```python
def retrieve_relevant_schema(query_embedding, top_k=10):
    """
    Cosine similarity search in vector database
    Returns: List[RetrievedSchema]
    """
    # Search both current schema and successful query examples
    schema_results = vector_db.search(query_embedding, top_k=5)
    example_results = vector_db.search(query_embedding,
                                       filter={"entity_type": "QUERY"},
                                       top_k=5)
    return combine_and_rank_results(schema_results, example_results)
```

### 3.4 SQL Generation Engine

**Technology**: LangChain Agent
**LLM**: GPT-4 (primary) / Sonnet 4.5 (fallback)

**Agent Configuration**:
```python
class SQLGeneratorAgent:
    def __init__(self, database_schema, llm):
        self.database_schema = database_schema
        self.llm = llm
        self.memory = ConversationBufferMemory()
        self.tools = [
            Tool("get_schema", self.get_schema_info),
            Tool("validate_sql", self.validate_sql),
            Tool("test_query", self.test_query_on_sample),
        ]
        self.agent = initialize_agent(
            self.tools,
            self.llm,
            agent="zero-shot-react-description",
            memory=self.memory
        )
```

**Prompt Engineering Template**:
```
You are a SQL expert helping translate natural language to SQL.

Database Schema:
{retrieved_schema}

Conversation Context:
{conversation_history}

User Question: {user_input}

Instructions:
1. Identify the intent and required tables
2. Determine necessary JOINs
3. Apply filters and aggregations
4. Suggest optimizations
5. Generate the SQL query

Provide your response in this format:
REASONING: [Step-by-step reasoning]
TABLES_USED: [List of tables]
JOINS: [JOIN logic]
SQL: [The generated SQL]
EXPLANATION: [Natural language explanation of the query]
```

**Multi-Table Reasoning**:
```python
def identify_joins(required_tables, schema_graph):
    """
    Build JOIN graph based on foreign keys
    Example:
    customers JOIN orders ON customers.id = orders.customer_id
    JOIN order_items ON orders.id = order_items.order_id
    """
```

**Query Validation**:
```python
def validate_query(sql, database_schema):
    """
    Checks:
    - SQL syntax correctness
    - Table existence
    - Column existence and type compatibility
    - Circular join detection
    - Aggregation correctness
    Returns: Validation result with any errors/warnings
    """
```

### 3.5 Query Optimization Module

**Purpose**: Analyze and suggest improvements to generated queries

**Performance Analysis**:
```python
def analyze_query_performance(sql, database_connection):
    """
    Using EXPLAIN PLAN:
    - Identify full table scans
    - Check join order efficiency
    - Find missing indexes
    Returns: OptimizationReport
    """
```

**Optimization Suggestions**:
1. **Index Recommendations**:
   ```sql
   CREATE INDEX idx_orders_customer_id ON orders(customer_id)
   ```

2. **Join Reordering**:
   ```
   Reorder joins for selectivity (most restrictive first)
   ```

3. **Query Rewriting**:
   ```
   Convert correlated subqueries to JOINs
   Push down WHERE conditions
   Use window functions instead of self-joins
   ```

4. **Materialization**:
   ```
   Suggest temporary tables for complex aggregations
   ```

**Query Optimization Rules**:
```python
OPTIMIZATION_RULES = {
    "push_down_predicates": lambda query: push_filters_early(query),
    "convert_to_join": lambda query: convert_subqueries_to_joins(query),
    "use_window_functions": lambda query: detect_window_function_candidates(query),
    "materialization": lambda query: identify_cte_candidates(query),
}
```

### 3.6 Query Execution Layer

**Database Connection Management**:
```python
class DatabaseExecutor:
    def __init__(self, database_url, pool_size=5):
        self.engine = create_engine(database_url, pool_size=pool_size)
        self.pool = ConnectionPool(engine)

    def execute_query(self, sql, timeout=30):
        """
        Execute with:
        - Connection pooling
        - Query timeout (prevent long-running queries)
        - Transaction management
        Returns: QueryResult object
        """
```

**Safety Mechanisms**:
1. **Query Sandboxing**: Read-only connection to prevent modifications
2. **Row Limits**: MAX_ROWS = 50000 to prevent large data transfers
3. **Execution Timeout**: Abort queries exceeding time limit
4. **Result Size Limits**: Truncate large result sets

**Result Formatting**:
```python
def format_results(raw_results, metadata):
    """
    Convert database results to:
    - JSON format for API
    - DataFrames for analysis
    - HTML tables for display
    Returns: FormattedResults
    """
```

### 3.7 Insight Generation

**Purpose**: Transform raw results into actionable insights

**Pattern Detection**:
```python
def detect_patterns(results_dataframe):
    """
    Identify:
    - Outliers (>2σ deviation)
    - Trends (linear regression)
    - Clusters (grouping patterns)
    - Anomalies (unusual values)
    Returns: List[Pattern]
    """
```

**Trend Analysis**:
```python
def analyze_trends(time_series_data):
    """
    Analyze temporal patterns:
    - Growth rate (month-over-month, year-over-year)
    - Seasonality
    - Direction (increasing/decreasing)
    - Forecast next period
    """
```

**Natural Language Insight Generation**:
```python
def generate_insights(results, query_context):
    """
    Generate insights like:
    "Revenue is up 15% this month"
    "Top 3 customers account for 45% of revenue"
    "East region showing 8% decline vs last quarter"
    Uses LLM to generate contextual natural language insights
    """
```

---

## 4. Data Models

### 4.1 Database Schema (Retail Domain Example)

```sql
CREATE TABLE customers (
    id INT PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    email VARCHAR(255) UNIQUE,
    phone VARCHAR(20),
    city VARCHAR(100),
    state VARCHAR(50),
    country VARCHAR(50),
    registration_date DATE,
    lifecycle_stage VARCHAR(20),  -- NEW, ACTIVE, AT_RISK, CHURNED
    annual_revenue DECIMAL(12,2),
    created_at TIMESTAMP,
    updated_at TIMESTAMP
);

CREATE TABLE products (
    id INT PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    category VARCHAR(100),
    subcategory VARCHAR(100),
    sku VARCHAR(50) UNIQUE,
    unit_price DECIMAL(10,2),
    cost DECIMAL(10,2),
    inventory_count INT,
    reorder_level INT,
    supplier_id INT,
    created_at TIMESTAMP,
    updated_at TIMESTAMP,
    FOREIGN KEY (supplier_id) REFERENCES suppliers(id)
);

CREATE TABLE orders (
    id INT PRIMARY KEY,
    customer_id INT NOT NULL,
    order_date DATE NOT NULL,
    ship_date DATE,
    delivery_date DATE,
    status VARCHAR(20),  -- PENDING, SHIPPED, DELIVERED, CANCELLED
    total_amount DECIMAL(12,2),
    discount DECIMAL(12,2),
    tax DECIMAL(12,2),
    shipping_cost DECIMAL(10,2),
    created_at TIMESTAMP,
    updated_at TIMESTAMP,
    FOREIGN KEY (customer_id) REFERENCES customers(id),
    INDEX idx_customer_id (customer_id),
    INDEX idx_order_date (order_date)
);

CREATE TABLE order_items (
    id INT PRIMARY KEY,
    order_id INT NOT NULL,
    product_id INT NOT NULL,
    quantity INT NOT NULL,
    unit_price DECIMAL(10,2),
    discount DECIMAL(10,2),
    total_price DECIMAL(12,2),
    created_at TIMESTAMP,
    FOREIGN KEY (order_id) REFERENCES orders(id),
    FOREIGN KEY (product_id) REFERENCES products(id),
    INDEX idx_order_id (order_id),
    INDEX idx_product_id (product_id)
);

CREATE TABLE suppliers (
    id INT PRIMARY KEY,
    name VARCHAR(255),
    contact_email VARCHAR(255),
    contact_phone VARCHAR(20),
    address VARCHAR(255),
    city VARCHAR(100),
    country VARCHAR(50),
    created_at TIMESTAMP
);
```

### 4.2 Vector Store Schema

```
Vector Database (FAISS/Chroma):

Collection: schema_entities
├── id: entity unique identifier
├── entity_type: TABLE, COLUMN, RELATIONSHIP
├── name: entity name
├── description: human-readable description
├── embedding: 1536-dimensional vector
├── metadata: {table, schema_info, example_values}

Collection: query_examples
├── id: example unique identifier
├── user_query: original natural language
├── sql_query: generated SQL
├── embedding: 1536-dimensional vector
├── success: whether query executed successfully
├── metadata: {domain, complexity, tables_used}
```

### 4.3 Conversation State Model

```json
{
  "conversation_id": "uuid-string",
  "user_id": "user-uuid",
  "session_start": "2026-02-10T10:30:00Z",
  "messages": [
    {
      "id": "msg-001",
      "type": "user",
      "content": "Show top 10 products by revenue last month",
      "timestamp": "2026-02-10T10:30:15Z"
    },
    {
      "id": "msg-002",
      "type": "assistant",
      "content": {
        "sql": "SELECT p.id, p.name, SUM(oi.quantity * oi.unit_price) as revenue ...",
        "explanation": "This query joins products with order_items and orders...",
        "results": {
          "rows": [...],
          "count": 10,
          "execution_time_ms": 245
        },
        "insights": ["Top product generated $45K in revenue", "..."],
        "optimization_suggestions": ["Add index on order_date", "..."]
      },
      "timestamp": "2026-02-10T10:30:18Z"
    }
  ],
  "context": {
    "last_tables": ["products", "order_items", "orders"],
    "filters": {"month": "2026-01"},
    "aggregation_type": "SUM"
  },
  "metadata": {
    "total_turns": 5,
    "last_activity": "2026-02-10T10:35:00Z"
  }
}
```

### 4.4 Query History Structure

```json
{
  "query_id": "qry-uuid",
  "user_id": "user-uuid",
  "natural_language": "Show top 10 products by revenue last month",
  "generated_sql": "SELECT p.id, p.name, ...",
  "sql_validated": true,
  "query_optimizations": [
    {
      "type": "index_suggestion",
      "suggestion": "CREATE INDEX idx_order_date ON orders(order_date)",
      "estimated_improvement": "15% faster"
    }
  ],
  "execution": {
    "status": "success",
    "execution_time_ms": 245,
    "rows_returned": 10,
    "query_plan": "{...}"
  },
  "results_summary": {
    "total_rows": 10,
    "column_count": 3,
    "data_types": {"id": "INT", "name": "VARCHAR", "revenue": "DECIMAL"}
  },
  "insights": [
    {
      "type": "ranking",
      "insight": "Product A leads with $45,000 revenue"
    }
  ],
  "timestamp": "2026-02-10T10:30:18Z",
  "user_feedback": {
    "helpful": true,
    "rating": 5,
    "corrections": null
  }
}
```

---

## 5. API Design

### 5.1 REST Endpoints (FastAPI)

```python
# Chat endpoint
POST /api/v1/chat
Request:
{
  "conversation_id": "uuid",
  "user_query": "Top products by revenue",
  "include_optimization": true,
  "include_insights": true
}

Response:
{
  "message_id": "uuid",
  "sql_generated": "SELECT ...",
  "sql_explanation": "This query...",
  "results": {
    "columns": ["id", "name", "revenue"],
    "rows": [[1, "Product A", 45000], ...],
    "row_count": 10
  },
  "insights": ["Top product generated...", ...],
  "optimization_suggestions": ["Add index on...", ...],
  "execution_time_ms": 245,
  "confidence_score": 0.95
}

# Get conversation history
GET /api/v1/conversations/{conversation_id}

# Save query
POST /api/v1/queries/{query_id}/save

# Get optimization suggestions
GET /api/v1/queries/{query_id}/optimizations

# Health check
GET /api/v1/health
```

### 5.2 WebSocket (for streaming)

```python
# Streaming response for long-running queries
WS /ws/chat/{conversation_id}

Message types:
- typing: Agent thinking through solution
- sql_generated: SQL has been generated
- executing: Query is executing
- results: Partial results streaming
- insights: Generated insights
- complete: Full response ready
```

---

## 6. Features Specification

### F1: Conversational Querying

**Description**: Multi-turn conversation with context retention

**User Stories**:
- As a user, I want to ask follow-up questions without repeating context
- As a user, I want the system to remember which tables I've been analyzing
- As a user, I want to refine previous results with new filters

**Acceptance Criteria**:
- ✓ System maintains conversation history across turns
- ✓ Follow-up questions understood in context of previous query
- ✓ Context automatically updated from successful queries
- ✓ Support for implicit references ("the top 10", "these customers")

**Technical Implementation**:
```python
class ConversationalAgent:
    def process_query(self, user_input, conversation_id):
        # Retrieve conversation history
        history = self.history_manager.get(conversation_id)

        # Build context-aware prompt
        context = self.build_context(user_input, history)

        # Generate response with context
        response = self.generate_with_context(user_input, context)

        # Update history
        self.history_manager.add(conversation_id, user_input, response)

        return response
```

---

### F2: RAG-Powered SQL Generation

**Description**: Semantic search over schema to guide SQL generation

**User Stories**:
- As a user, I want accurate queries for complex multi-table questions
- As a system, I want to learn from successful previous queries
- As a user, I want queries optimized for my database structure

**Acceptance Criteria**:
- ✓ Semantic matching of user queries to schema entities
- ✓ 85%+ accuracy on standard benchmark queries
- ✓ Support for complex joins (3+ tables)
- ✓ Learning from successful query patterns

**Technical Implementation**:
```python
class RAGSQLGenerator:
    def __init__(self, vector_db, llm):
        self.vector_db = vector_db
        self.llm = llm

    def generate(self, query):
        # Retrieve relevant schema
        schema_context = self.vector_db.search(query, top_k=10)

        # Retrieve similar successful queries
        similar_queries = self.vector_db.search_examples(query)

        # Build prompt with context
        prompt = self.build_prompt(query, schema_context, similar_queries)

        # Generate SQL
        sql = self.llm.generate(prompt)

        return sql
```

---

### F3: Query Optimization

**Description**: Analyze and suggest improvements to queries

**User Stories**:
- As a user, I want to know if my query is inefficient
- As a user, I want suggestions to make slow queries faster
- As a user, I want to understand what indexes would help

**Acceptance Criteria**:
- ✓ Identify 80%+ of obvious optimization opportunities
- ✓ Provide actionable suggestions with estimated impact
- ✓ Validate optimized queries produce same results
- ✓ Support for different database engines

**Technical Implementation**:
```python
class QueryOptimizer:
    def optimize(self, sql, database):
        # Parse query
        query_ast = parse_sql(sql)

        # Get execution plan
        plan = database.explain(sql)

        # Identify bottlenecks
        issues = self.identify_issues(plan, query_ast)

        # Generate suggestions
        suggestions = []
        for issue in issues:
            suggestion = self.generate_suggestion(issue)
            suggestions.append(suggestion)

        return OptimizationReport(issues, suggestions)
```

---

### F4: AI-Driven Insights

**Description**: Generate meaningful insights from query results

**User Stories**:
- As a user, I want key findings highlighted automatically
- As a user, I want to understand trends in the data
- As a user, I want anomalies pointed out

**Acceptance Criteria**:
- ✓ Automatically identify top/bottom items
- ✓ Calculate growth/decline percentages
- ✓ Detect anomalies with statistical rigor
- ✓ Provide insights in natural language

**Technical Implementation**:
```python
class InsightGenerator:
    def generate(self, results, query_context):
        insights = []

        # Statistical analysis
        outliers = self.detect_outliers(results)
        insights.extend(self.format_outliers(outliers))

        # Trend analysis
        if self.has_temporal_data(results):
            trends = self.analyze_trends(results)
            insights.extend(self.format_trends(trends))

        # Comparative analysis
        comparisons = self.identify_comparisons(results, query_context)
        insights.extend(self.format_comparisons(comparisons))

        # Generate natural language
        nl_insights = [self.llm.generate_insight(i) for i in insights]

        return nl_insights
```

---

### F5: Explainable SQL

**Description**: Provide step-by-step explanation of generated SQL

**User Stories**:
- As a user, I want to understand why this SQL was generated
- As a user, I want to learn SQL from examples
- As a user, I want to verify the logic is correct

**Acceptance Criteria**:
- ✓ Explain SELECT clause, WHERE clause, JOINs separately
- ✓ Use natural language for technical concepts
- ✓ Provide example of what each part retrieves
- ✓ Highlight assumptions made

**Technical Implementation**:
```python
class SQLExplainer:
    def explain(self, sql, query_context):
        explanation = {
            "overview": self.explain_overview(sql, query_context),
            "tables": self.explain_tables(sql),
            "joins": self.explain_joins(sql),
            "filters": self.explain_where(sql),
            "aggregations": self.explain_aggregations(sql),
            "order": self.explain_order(sql),
            "key_assumptions": self.extract_assumptions(sql, query_context)
        }
        return explanation
```

---

### F6: Context Retention

**Description**: Maintain state across conversation turns

**User Stories**:
- As a user, I want to say "Show me the same data for last year"
- As a user, I want to say "Sort by revenue instead"
- As a user, I want implicit references understood

**Acceptance Criteria**:
- ✓ Maintain last N turns of history (N=10)
- ✓ Update context from successful queries
- ✓ Support implicit references in 80%+ of cases
- ✓ Allow users to reset context explicitly

**Technical Implementation**:
```python
class ContextRetentionManager:
    def __init__(self, max_history=10):
        self.history = []
        self.max_history = max_history

    def build_context(self, new_query):
        # Build implicit context from last queries
        context = {
            "tables": self.get_recent_tables(),
            "filters": self.get_recent_filters(),
            "aggregations": self.get_recent_aggregations(),
            "implicit_references": self.extract_references(new_query)
        }
        return context

    def update_from_query(self, query, results):
        # Update context from successful query
        parsed = parse_sql(query)
        self.context.update({
            "tables": extract_tables(parsed),
            "filters": extract_conditions(parsed)
        })
```

---

### F7: Chat Interface

**Description**: User-friendly web interface for querying

**User Stories**:
- As a user, I want an intuitive chat interface
- As a user, I want to see results clearly formatted
- As a user, I want to save and revisit past queries

**Acceptance Criteria**:
- ✓ Fast, responsive interface (<2s load time)
- ✓ Mobile-friendly design
- ✓ Syntax-highlighted SQL display
- ✓ Export results (CSV, JSON)
- ✓ Save favorite queries

**Technical Implementation**:
- Frontend: Gradio or React with TypeScript
- State Management: Redux or Context API
- UI Components: Custom or Material-UI
- Real-time Updates: WebSocket for streaming results

---

## 7. Security & Safety

### 7.1 SQL Injection Prevention

**Mechanisms**:
1. **Parameterized Queries**: All user inputs passed as parameters
   ```python
   cursor.execute("SELECT * FROM users WHERE id = %s", (user_id,))
   ```

2. **Query Validation**: Parse and validate generated SQL before execution
   ```python
   def validate_query(sql):
       # Parse SQL AST
       # Check for only SELECT operations
       # Verify referenced tables/columns exist
       # Flag suspicious patterns
   ```

3. **Allowlist**: Only SELECT operations permitted
   ```python
   ALLOWED_OPERATIONS = {"SELECT", "WITH"}
   FORBIDDEN_OPERATIONS = {"INSERT", "UPDATE", "DELETE", "DROP", "TRUNCATE"}
   ```

### 7.2 Query Sandboxing

- Read-only database connection
- Connection timeout: 30 seconds
- Result size limit: 50,000 rows
- Query execution in isolated transaction

### 7.3 Rate Limiting

- Per-user rate limit: 100 queries/hour
- Per-IP rate limit: 1000 queries/hour
- Burst protection: Max 10 concurrent queries

### 7.4 Input Validation

```python
def validate_input(user_query):
    # Max length: 5000 characters
    if len(user_query) > 5000:
        raise ValueError("Query too long")

    # Forbidden words
    forbidden = {"DROP", "DELETE", "TRUNCATE", "ALTER"}
    if any(word in user_query.upper() for word in forbidden):
        raise SecurityError("Forbidden operation requested")

    # Check for SQL injection patterns
    if detect_injection_patterns(user_query):
        raise SecurityError("Potential injection detected")
```

---

## 8. Performance Requirements

### 8.1 Response Time Targets

- Simple queries: <1 second
- Complex queries (3+ joins): <3 seconds
- Query optimization analysis: <2 seconds
- Insight generation: <1 second
- **Total E2E response**: <5 seconds

### 8.2 Scalability

- Support 100+ concurrent users
- Handle databases with 1M+ rows
- Vector DB search: <500ms for top-k retrieval
- Context history: Support 50+ turn conversations

### 8.3 Caching Strategy

```
Query Cache:
├── SQL Query Hash → Results (TTL: 1 hour)
├── Conversation Context (TTL: 24 hours)
└── Schema Embeddings (TTL: 7 days)

Optimization Tips:
- Cache schema embeddings at startup
- Use result pagination (don't cache all rows)
- Implement LRU cache with memory limits
```

---

## 9. Testing Strategy

### 9.1 Unit Testing

**Coverage**: >80% of code

**Test Areas**:
- SQL parsing and validation
- Query generation with various inputs
- Context management
- Optimization suggestions
- Insight generation
- Input sanitization

**Example Test**:
```python
def test_sql_generation_simple_select():
    generator = SQLGenerator(mock_schema, mock_llm)
    result = generator.generate("Show all customers")
    assert "SELECT" in result
    assert "customers" in result
    assert result is not None
```

### 9.2 Integration Testing

**Test Scenarios**:
- End-to-end query flow
- Multi-turn conversations
- Database integration
- RAG retrieval accuracy
- Optimization application

**Example Test**:
```python
def test_full_query_flow():
    buddy = QueryBuddy(test_db_url)
    response = buddy.query("Top 10 products by revenue")
    assert response.sql is not None
    assert response.results is not None
    assert len(response.insights) > 0
```

### 9.3 Example Test Queries

```
1. Simple aggregation: "Total revenue by product category"
2. Multi-table join: "Top customers by order count"
3. Time-based: "Sales trend last 12 months"
4. Complex filtering: "Customers from California with >5 orders"
5. Window functions: "Customer ranking by lifetime value"
6. Subqueries: "Products purchased by top 10% of customers"
7. Grouping with HAVING: "Categories with >$100K revenue"
8. Date functions: "Orders placed in last quarter"
9. Comparison: "Growth rate vs. previous year"
10. Edge case: Empty result set handling
```

### 9.4 Edge Cases

- Empty result sets
- NULL values in results
- Ambiguous table/column names
- User typos and misspellings
- Contradictory queries
- Very large result sets
- Timeout scenarios

---

## 10. Deployment

### 10.1 Environment Setup

```bash
# Python version
Python 3.9+

# Environment variables
DATABASE_URL=postgresql://user:pass@localhost/db
OPENAI_API_KEY=sk-...
VECTOR_DB_PATH=./vector_store
LOG_LEVEL=INFO
```

### 10.2 Configuration Management

```python
class Config:
    # Database
    DATABASE_URL = os.getenv("DATABASE_URL")
    DB_POOL_SIZE = 5
    DB_MAX_OVERFLOW = 10

    # LLM
    LLM_MODEL = "gpt-4"
    LLM_TEMPERATURE = 0.3
    LLM_MAX_TOKENS = 1000

    # RAG
    VECTOR_DB_TYPE = "faiss"  # or "chroma", "milvus"
    EMBEDDING_MODEL = "text-embedding-3-large"

    # Performance
    QUERY_TIMEOUT = 30
    RESULT_MAX_ROWS = 50000
    CACHE_TTL = 3600
```

### 10.3 Docker Containerization (Optional)

```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

---

## 11. Dependencies

### 11.1 Python Packages

```
# Core
langchain==0.1.0
openai==1.3.0
fastapi==0.104.0
uvicorn==0.24.0

# Database
sqlalchemy==2.0.0
psycopg2-binary==2.9.0
pymysql==1.1.0

# Vector Search
faiss-cpu==1.7.4
chromadb==0.4.0

# Data Processing
pandas==2.1.0
numpy==1.24.0

# UI
gradio==4.0.0

# Utilities
python-dotenv==1.0.0
pydantic==2.0.0
pydantic-settings==2.0.0

# Monitoring & Logging
python-json-logger==2.0.0
```

### 11.2 External Services

- **LLM**: OpenAI GPT-4 API
- **Embeddings**: OpenAI Embeddings API
- **Database**: PostgreSQL / MySQL / SQLite
- **Vector Store**: FAISS / Chromadb / Milvus
- **Monitoring**: Optional - CloudWatch / DataDog

### 11.3 Model Requirements

- **LLM**: GPT-4 (recommended) or Claude 3.5 Sonnet
- **Embeddings**: text-embedding-3-large (1536 dimensions)
- **Total RAM**: 4GB minimum (8GB recommended)

---

## 12. Development Roadmap

### Phase 1: Core MVP (Week 1-2) ✓
**Deadline**: February 14, 2026

**Deliverables**:
- ✓ Basic SQL generation from natural language
- ✓ Database connection and query execution
- ✓ Simple Gradio chat interface
- ✓ Context retention across turns
- ✓ RAG system with basic schema embeddings

**Success Metrics**:
- 80%+ accuracy on basic query generation
- <3 second response time
- Working MVP deployed

### Phase 2: Optimization & Insights (Week 2-3)
**Deadline**: February 15, 2026

**Deliverables**:
- 🔄 Query optimization suggestions
- 🔄 AI-driven insights generation
- 🔄 Advanced multi-table reasoning
- 🔄 Query explanation feature
- 🔄 Performance metrics dashboard

**Success Metrics**:
- 85%+ accuracy on complex queries
- Optimization suggestions for 90%+ of queries
- Meaningful insights generated for all results

### Phase 3: Polish & Deployment (Optional)
**For Contest Showcase**:
- 🔄 Enhanced UI/UX improvements
- 🔄 Comprehensive documentation
- 🔄 Docker containerization
- 🔄 Performance optimization
- 🔄 Production deployment

---

## 13. Appendix

### A. Key Assumptions

1. Database schema is stable (doesn't change frequently)
2. Database has reasonable row counts (<100M)
3. User has appropriate database permissions
4. OpenAI API is available and has sufficient quota
5. SQLite/PostgreSQL/MySQL are supported databases

### B. Known Limitations

1. Does not support complex stored procedures
2. Limited to SELECT queries for safety
3. Requires manual schema description in vector DB
4. Performance depends on LLM API latency
5. Does not support custom domain-specific SQL functions

### C. Future Enhancements

1. Multi-language query support
2. Advanced visualization charts
3. Predictive analytics (forecasting)
4. Automated report generation
5. Mobile app
6. Integration with BI tools (Tableau, Power BI)
7. Custom LLM fine-tuning
8. Data lineage tracking

---

**Document Status**: APPROVED FOR IMPLEMENTATION
**Last Review**: February 10, 2026
**Next Review**: February 20, 2026
