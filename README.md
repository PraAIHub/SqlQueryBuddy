# SQL Query Buddy 🤖

An intelligent conversational AI agent that transforms natural language questions into optimized SQL queries, executes them against your database, and provides AI-driven insights on the results.

**Quick Start:**
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

## Features ✨

- 🗣️ **Conversational Querying** - Chat naturally with your database
- 🧠 **RAG-Powered SQL Generation** - Semantic search over schema
- ⚡ **Query Optimization** - Automatic analysis and suggestions
- 📊 **AI-Driven Insights** - Pattern analysis and natural language summaries
- 🔍 **Explainable SQL** - Transparent generation with step-by-step reasoning
- 💾 **Context Retention** - Maintains conversation history
- 🎨 **Clean Interface** - Intuitive Gradio web interface

## Demo Queries

Try these in the chat:
- "Show me all users"
- "How many products are in stock?"
- "What are the top 5 products by price?"
- "Show me orders from the last month"

See more examples in [demo_queries.md](demo_queries.md)

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Frontend | Gradio 4.0 |
| AI Engine | LangChain + GPT-4 |
| Vector Search | FAISS |
| Backend | Python + FastAPI |
| Database | SQLite / PostgreSQL / MySQL |

## Project Structure

```
SqlQueryBuddy/
├── docs/                    # Documentation
│   ├── README.md           # Detailed guide
│   └── specification.md     # Technical specs
├── src/
│   ├── components/         # Core modules
│   │   ├── nlp_processor.py      # NLP layer
│   │   ├── rag_system.py         # RAG pipeline
│   │   ├── sql_generator.py      # SQL generation
│   │   ├── executor.py           # Query execution
│   │   ├── insights.py           # Insight generation
│   │   └── optimizer.py          # Query optimization
│   ├── app.py             # Gradio interface
│   ├── config.py          # Configuration
│   └── main.py            # Entry point
├── tests/                 # Test suite
├── requirements.txt       # Dependencies
├── docker-compose.yml     # Docker setup
└── run.sh                # Start script
```

## Configuration

Edit `.env` file (copy from `.env.example`):

```env
OPENAI_API_KEY=your_key_here
OPENAI_MODEL=gpt-4
DATABASE_URL=sqlite:///retail.db
```

## Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src tests/

# Run specific test
pytest tests/unit/test_components.py -v
```

## Environment Variables

See `.env.example` for all available options. Key ones:
- `OPENAI_API_KEY` - Required for full functionality
- `DATABASE_URL` - Database connection string
- `DATABASE_TYPE` - sqlite, postgresql, or mysql
- `GRADIO_SHARE` - Share interface publicly

## Troubleshooting

**No OpenAI API Key?** The app uses a mock generator by default. Set `OPENAI_API_KEY` to enable real SQL generation.

**Database issues?** SQLite database is created automatically. For PostgreSQL/MySQL, ensure database exists and URL is correct.

**Port already in use?** Change `GRADIO_SERVER_PORT` in `.env`

## Documentation

- Full documentation: [docs/README.md](docs/README.md)
- Technical specs: [docs/specification.md](docs/specification.md)
- Demo queries: [demo_queries.md](demo_queries.md)

## License

MIT License - See [LICENSE](LICENSE) file

---

**Made for the Codecademy GenAI Bootcamp Contest**

Deadline: February 15, 2026
