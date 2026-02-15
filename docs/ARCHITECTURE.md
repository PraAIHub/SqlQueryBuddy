# SQL Query Buddy - Architecture Diagrams

## System Architecture

```mermaid
graph TB
    subgraph "User Interface Layer"
        UI[Gradio Chat Interface]
    end

    subgraph "NLP Processing Layer"
        QP[Query Parser]
        CM[Context Manager]
        CH[Conversation History]
    end

    subgraph "RAG System Layer"
        SE[Schema Embedder]
        VS[Vector Search]
        VDB[(Vector Database<br/>FAISS/Chroma)]
    end

    subgraph "SQL Generation Layer"
        PT[Prompt Templates]
        LLM[LangChain + GPT-4]
        SV[SQL Validator]
    end

    subgraph "Optimization & Safety"
        QO[Query Optimizer]
        IP[Injection Prevention]
    end

    subgraph "Execution Layer"
        DBM[DB Connection Manager]
        QE[Query Executor]
        RF[Result Formatter]
    end

    subgraph "Insight Layer"
        PD[Pattern Detector]
        TA[Trend Analyzer]
        IG[Insight Generator]
    end

    subgraph "Data Layer"
        DB[(SQLite / PostgreSQL<br/>/ MySQL)]
    end

    UI -->|User Question| QP
    QP --> CM
    CM --> CH
    CM -->|Parsed Query| SE
    SE --> VS
    VS --> VDB
    VS -->|Schema Context| PT
    CH -->|History| PT
    PT --> LLM
    LLM -->|Generated SQL| SV
    SV --> QO
    QO --> IP
    IP -->|Safe Query| QE
    QE --> DBM
    DBM --> DB
    DB -->|Results| RF
    RF -->|Formatted Data| PD
    PD --> TA
    TA --> IG
    IG -->|Insights| UI
    RF -->|Results| UI

    style UI fill:#4A90D9,stroke:#333,color:#fff
    style DB fill:#27AE60,stroke:#333,color:#fff
    style VDB fill:#8E44AD,stroke:#333,color:#fff
    style LLM fill:#E74C3C,stroke:#333,color:#fff
```

## Data Flow Pipeline

```mermaid
sequenceDiagram
    participant U as User
    participant UI as Gradio UI
    participant NLP as NLP Processor
    participant RAG as RAG System
    participant SQL as SQL Generator
    participant VAL as Validator
    participant OPT as Optimizer
    participant EXE as Executor
    participant DB as Database
    participant INS as Insight Engine

    U->>UI: Natural language question
    UI->>NLP: Parse input
    NLP->>NLP: Extract intent, entities, modifiers
    NLP->>RAG: Query + context
    RAG->>RAG: Embed query
    RAG->>RAG: Similarity search for relevant schema
    RAG-->>SQL: Schema context + conversation history
    SQL->>SQL: Build prompt with schema + history
    SQL->>SQL: LLM generates SQL
    SQL->>VAL: Validate generated SQL
    VAL-->>VAL: Check for injection, dangerous keywords
    VAL->>OPT: Valid SQL
    OPT-->>OPT: Analyze for optimization suggestions
    OPT->>EXE: Execute query
    EXE->>DB: Run SQL
    DB-->>EXE: Raw results
    EXE->>INS: Results + query context
    INS->>INS: Detect patterns & trends
    INS-->>UI: SQL + Results + Insights + Explanation
    UI-->>U: Formatted response
```

## Entity Relationship Diagram (Retail Commerce Schema)

```mermaid
erDiagram
    customers {
        INTEGER customer_id PK
        TEXT name
        TEXT email UK
        TEXT region
        DATE signup_date
    }

    products {
        INTEGER product_id PK
        TEXT name
        TEXT category
        DECIMAL price
    }

    orders {
        INTEGER order_id PK
        INTEGER customer_id FK
        DATE order_date
        DECIMAL total_amount
    }

    order_items {
        INTEGER item_id PK
        INTEGER order_id FK
        INTEGER product_id FK
        INTEGER quantity
        DECIMAL subtotal
    }

    customers ||--o{ orders : "places"
    orders ||--o{ order_items : "contains"
    products ||--o{ order_items : "included in"
```

## RAG Pipeline Detail

```mermaid
graph LR
    subgraph "Initialization (Startup)"
        S1[Load DB Schema] --> S2[Create Table Descriptions]
        S2 --> S3[Generate Embeddings]
        S3 --> S4[Store in Vector DB]
    end

    subgraph "Query Time (Per Request)"
        Q1[User Query] --> Q2[Embed Query]
        Q2 --> Q3[Cosine Similarity Search]
        Q3 --> Q4[Top-K Relevant Tables/Columns]
        Q4 --> Q5[Build Context String]
        Q5 --> Q6[Inject into LLM Prompt]
    end

    S4 -.->|Vector Store| Q3

    style S4 fill:#8E44AD,stroke:#333,color:#fff
    style Q6 fill:#E74C3C,stroke:#333,color:#fff
```

## Component Dependency Map

```mermaid
graph TD
    APP[app.py<br/>Main Application] --> NLP[nlp_processor.py<br/>Query Parser + Context]
    APP --> RAG[rag_system.py<br/>Schema Embeddings + Retrieval]
    APP --> GEN[sql_generator.py<br/>LangChain SQL Generation]
    APP --> EXE[executor.py<br/>DB Connection + Execution]
    APP --> INS[insights.py<br/>Pattern Detection + Trends]
    APP --> OPT[optimizer.py<br/>Query Analysis + Suggestions]
    APP --> CFG[config.py<br/>Settings + Environment]
    GEN --> SAN[sanitizer.py<br/>Prompt Input Sanitization]
    INS --> SAN

    GEN --> |validates| GEN
    EXE --> |schema for| RAG
    NLP --> |context for| GEN
    RAG --> |schema for| GEN
    EXE --> |results for| INS
    GEN --> |SQL for| OPT

    style APP fill:#3498DB,stroke:#333,color:#fff
    style GEN fill:#E74C3C,stroke:#333,color:#fff
    style RAG fill:#8E44AD,stroke:#333,color:#fff
    style EXE fill:#27AE60,stroke:#333,color:#fff
```

## Deployment Architecture

```mermaid
graph TB
    subgraph "Docker Container"
        subgraph "Application"
            GRADIO[Gradio Server<br/>Port 7860]
            PYTHON[Python 3.9]
            SRC[Source Code]
        end
        subgraph "Data"
            SQLITE[(SQLite DB)]
            VECTORDB[(Vector Store)]
        end
    end

    subgraph "External Services"
        OPENAI[OpenAI API<br/>GPT-4 / Embeddings]
    end

    subgraph "Optional Production DB"
        PG[(PostgreSQL)]
        MYSQL[(MySQL)]
    end

    USER[Browser] -->|HTTP :7860| GRADIO
    PYTHON -->|API Calls| OPENAI
    PYTHON --> SQLITE
    PYTHON --> VECTORDB
    PYTHON -.->|Production| PG
    PYTHON -.->|Production| MYSQL

    style GRADIO fill:#4A90D9,stroke:#333,color:#fff
    style OPENAI fill:#E74C3C,stroke:#333,color:#fff
    style SQLITE fill:#27AE60,stroke:#333,color:#fff
```
