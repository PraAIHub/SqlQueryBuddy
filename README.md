---
title: SQL Query Buddy
emoji: 🤖
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
---

# SQL Query Buddy 🤖

**Conversational AI for Smart Data Insights** — Built for the [Codecademy GenAI Bootcamp Contest](https://www.codecademy.com) #CodecademyGenAIBootcamp

An intelligent conversational AI agent that transforms natural language questions into optimized SQL queries, executes them against your database, and provides **AI-driven insights** on the results — not just raw data, but trends, anomalies, and actionable recommendations.

## What Makes This Special

| Feature | What It Does |
|---------|-------------|
| **RAG-Powered SQL** | FAISS vector search finds relevant schema before generating SQL — no hallucinated tables |
| **AI Insights** | Every query gets trend analysis, anomaly detection, and business recommendations |
| **Auto-Fix Retry** | If SQL fails, the system automatically regenerates using the error feedback |
| **Context Memory** | Multi-turn conversations with structured QueryPlan tracking |
| **Works Offline** | LocalInsightGenerator provides full insights even without an API key |
| **Smart Charts** | Auto-detects time series, categorical data, and single-value metrics |

## Quick Start

```bash
# Clone and setup
git clone <repo-url>
cd SqlQueryBuddy

# Run with script (recommended)
bash run.sh

# Or run with Docker
docker-compose up

# Or manually
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
python -m src.main
```

**Open your browser to:** `http://localhost:7860`

## Core Features

- **Conversational Querying** — Chat naturally: "Show top customers", then "Filter to California only"
- **RAG-Powered SQL Generation** — LangChain + FAISS semantic search over schema embeddings
- **Query Optimization** — 8 optimization rules with categorized suggestions (performance, assumptions, next steps)
- **AI-Driven Insights** — LLM-generated analysis with local fallback (trends, anomalies, key metrics)
- **Explainable SQL** — Every query includes a natural language explanation
- **Context Retention** — QueryPlan tracks tables, filters, time ranges across turns
- **Interactive Dashboard** — Analytics cards, query history, performance metrics
- **Auto-Fix Retry** — Automatically regenerates SQL using error feedback on failures

## Database Schema (Retail Commerce)

| Table | Records | Description |
|-------|---------|-------------|
| **customers** | 150 | Customer info (name, email, region, signup_date) |
| **products** | 25 | Product catalog (name, category, price) |
| **orders** | 2,500 | Purchase records (customer_id, order_date, total_amount) |
| **order_items** | ~6,500 | Line items linking orders to products (quantity, subtotal) |

## Demo Queries

Try these in the chat:
1. "Show me the top 5 customers by total purchase amount"
2. "Which product category made the most revenue?"
3. "List customers who haven't ordered anything in the last 3 months"
4. "Show total sales per region for 2024"
5. "Find the average order value for returning customers"
6. "How many unique products were sold in January?"
7. "Show the trend of monthly revenue over time"
8. "How many orders contained more than 3 items?"
9. "From the previous result, filter customers from New York only"
10. "Which salesperson generated the highest sales last month?"

## Tech Stack

| Layer | Technology |
|-------|-----------|
| **Frontend** | Gradio (modern 2-pane chat UI with tabs) |
| **AI Engine** | LangChain + ChatOpenAI (GPT-4o-mini default) |
| **Vector Search** | FAISS (Facebook AI Similarity Search) |
| **Embeddings** | TF-IDF with synonym expansion + stemming |
| **Backend** | Python 3.9+ |
| **Database** | SQLite (also supports PostgreSQL, MySQL) |
| **Charts** | Matplotlib (auto-detection: line, bar, value cards) |
| **Containerization** | Docker + docker-compose |

## Architecture

```
User Question
    ↓
[NLP Parser] → Extract intent, entities, modifiers
    ↓
[RAG System] → FAISS vector search for relevant schema
    ↓
[SQL Generator] → LangChain + GPT builds optimized SQL
    ↓
[Validator] → Safety checks (injection, dangerous ops)
    ↓
[Optimizer] → Performance analysis + suggestions
    ↓
[Executor] → Read-only execution with timeout
    ↓
[Insight Generator] → AI analysis (LLM or local fallback)
    ↓
[Response] → SQL + Results + Chart + Insights + Explanation
```

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for detailed Mermaid diagrams.

## Project Structure

```
SqlQueryBuddy/
├── src/
│   ├── components/
│   │   ├── nlp_processor.py      # NLP: intent extraction, context management
│   │   ├── rag_system.py         # RAG: FAISS, embeddings, schema retrieval
│   │   ├── sql_generator.py      # SQL: LangChain generation + mock fallback
│   │   ├── executor.py           # DB: connection, execution, safety
│   │   ├── insights.py           # AI: LLM insights + local fallback
│   │   ├── optimizer.py          # Perf: 8 optimization rules
│   │   └── sanitizer.py          # Security: shared prompt sanitization
│   ├── app.py                    # Gradio interface (1600 lines)
│   ├── config.py                 # Pydantic settings from .env
│   └── main.py                   # Entry point
├── tests/
│   ├── unit/test_components.py   # 37+ unit tests
│   └── integration/test_end_to_end.py  # 12+ integration tests
├── docs/                         # 7 documentation files
├── reviews/                      # 10-agent critique reports
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── run.sh
```

## Testing

```bash
# Run all tests (46+ tests, no API key needed)
pytest

# Run with coverage
pytest --cov=src tests/

# Run specific suite
pytest tests/unit/test_components.py -v
pytest tests/integration/test_end_to_end.py -v
```

## Configuration

Key environment variables (set in `.env`):

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | *(empty)* | OpenAI API key (optional - mock mode without) |
| `OPENAI_MODEL` | `gpt-4o-mini` | LLM model for SQL generation |
| `DATABASE_URL` | `sqlite:///retail.db` | Database connection string |
| `DATABASE_TYPE` | `sqlite` | Database type (sqlite/postgresql/mysql) |
| `SIMILARITY_THRESHOLD` | `0.4` | RAG similarity threshold |
| `GRADIO_SERVER_PORT` | `7860` | Web server port |
| `GRADIO_SHARE` | `false` | Create public share link |

## Security

- **Read-only DB**: SQLite PRAGMA query_only=ON enforced at connection level
- **SQL Validation**: Rejects INSERT/UPDATE/DELETE/DROP with word-boundary matching
- **Comment Stripping**: SQL comments stripped before validation (prevents bypass)
- **Prompt Sanitization**: User input sanitized against injection markers
- **Input Limits**: 500 character query limit
- **Timeout Protection**: 30-second query timeout via threading
- **Row Limits**: Maximum 1,000 rows per query result

## Documentation

| Document | Description |
|----------|-------------|
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | Mermaid architecture diagrams |
| [docs/specification.md](docs/specification.md) | Full technical specification |
| [docs/SECURITY.md](docs/SECURITY.md) | Security measures and audit |
| [docs/TESTING.md](docs/TESTING.md) | Testing strategy and guide |
| [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md) | Deployment instructions |
| [docs/API.md](docs/API.md) | API documentation |
| [reviews/CRITIQUE_REPORT.md](reviews/CRITIQUE_REPORT.md) | 10-agent code critique |

## Troubleshooting

**No OpenAI API Key?** The app works in demo mode with a context-aware mock generator. Set `OPENAI_API_KEY` for full LLM-powered SQL generation and insights.

**Database issues?** SQLite database is auto-created with 150 customers, 25 products, 2,500 orders on first run.

**Port already in use?** Set `GRADIO_SERVER_PORT=7861` in `.env`

## License

MIT License - See [LICENSE](LICENSE) file

---

**Built for the Codecademy GenAI Bootcamp Contest** | Deadline: February 15, 2026

`#CodecademyGenAIBootcamp`
