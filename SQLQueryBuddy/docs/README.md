# SQL Query Buddy

[![Codecademy GenAI Bootcamp](https://img.shields.io/badge/Codecademy-GenAI%20Bootcamp-blue)](https://www.codecademy.com/)
[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-brightgreen)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Contest Submission](https://img.shields.io/badge/Contest-GenAI%20Bootcamp-orange)](https://www.codecademy.com/)

## 🎯 Overview

**SQL Query Buddy** is an intelligent conversational AI agent that transforms natural language questions into optimized SQL queries, executes them against your database, and provides AI-driven insights on the results. Built with LangChain, RAG, and modern LLMs, it makes database querying accessible to everyone—regardless of SQL expertise.

Ask questions like *"Show me the top 10 products by revenue last quarter"* and let SQL Query Buddy handle the SQL complexity while explaining every step.

## ✨ Key Features

- **🗣️ Conversational Querying** - Chat naturally with your database, maintaining conversation context across multiple turns
- **🧠 RAG-Powered SQL Generation** - Semantic search over your schema combined with LangChain agents for intelligent query generation
- **⚡ Query Optimization** - Automatic analysis and suggestions for JOIN optimization, indexing, and query rewriting
- **📊 AI-Driven Insights** - Beyond raw results: trend detection, pattern analysis, and natural language summary insights
- **🔍 Explainable SQL** - Transparent SQL generation with step-by-step explanations of the reasoning
- **💾 Context Retention** - Maintains conversation history and query context for sophisticated multi-turn interactions
- **🎨 Clean Chat Interface** - Intuitive Gradio/React web interface for seamless user experience

## 🎬 Demo

*Screenshots and GIFs coming soon!*

## 🚀 Quick Start

### Prerequisites
- Python 3.9 or higher
- pip package manager
- Your database credentials (SQLite, PostgreSQL, or MySQL)

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd SQLQueryBuddy

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your database and API credentials
```

### Basic Usage

```python
from sql_query_buddy import QueryBuddy

# Initialize the query buddy
buddy = QueryBuddy(
    database_url="sqlite:///retail.db",
    llm_model="gpt-4",
    openai_api_key="your-api-key"
)

# Ask a question
response = buddy.query("What are the top 5 products by sales?")
print(response)
```

### Running the Web Interface

```bash
python app.py
# Visit http://localhost:7860 in your browser
```

## 🛠️ Tech Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **Frontend** | Gradio / React | Interactive chat interface |
| **AI Engine** | LangChain + GPT-4 | Query generation & insights |
| **Vector Search** | FAISS / Chroma | Schema embeddings & retrieval |
| **Backend** | Python FastAPI | REST API & agent orchestration |
| **Database** | SQLite / PostgreSQL / MySQL | Data storage & querying |
| **RAG Framework** | Custom RAG Pipeline | Context-aware retrieval |
| **Embeddings** | OpenAI Embeddings | Semantic search |

## 🏗️ Architecture

```
User Query
    ↓
[Chat Interface - Gradio/React]
    ↓
[Natural Language Processing Layer]
    ├─ Query Parser
    ├─ Context Manager
    └─ Conversation History
    ↓
[RAG System]
    ├─ Vector Database (FAISS/Chroma)
    ├─ Schema Embeddings
    └─ Semantic Retrieval
    ↓
[SQL Generation Engine - LangChain Agent]
    ├─ Prompt Engineering
    ├─ Multi-table Reasoning
    └─ Query Validation
    ↓
[Query Optimization Module]
    ├─ Performance Analysis
    ├─ Index Suggestions
    └─ Query Rewriting
    ↓
[Query Execution Layer]
    ├─ Connection Management
    ├─ Safety Checks
    └─ Result Formatting
    ↓
[Insight Generation Engine]
    ├─ Pattern Detection
    ├─ Trend Analysis
    └─ Natural Language Insights
    ↓
Response to User
```

## 📚 Example Queries

SQL Query Buddy handles a wide variety of natural language questions:

1. **"Show me the top 10 customers by total spending this year"**
2. **"What products have declining sales trends in the last quarter?"**
3. **"Calculate average order value by product category"**
4. **"Find all customers who purchased more than $1000 in the last 30 days"**
5. **"Which product categories are most popular by region?"**
6. **"Show me the customer retention rate for each month"**
7. **"What are the peak ordering times by day of week?"**
8. **"Find products with inventory below safety threshold"**
9. **"Compare revenue growth year-over-year for each product line"**
10. **"Which customer segments have the highest lifetime value?"**

## 📁 Project Structure

```
SQLQueryBuddy/
├── docs/
│   ├── README.md                    # This file
│   └── specification.md             # Technical specification
├── src/
│   ├── __init__.py
│   ├── main.py                      # Entry point
│   ├── app.py                       # Gradio web interface
│   ├── config.py                    # Configuration management
│   └── components/
│       ├── chat_interface.py        # UI components
│       ├── nlp_processor.py         # NLP layer
│       ├── rag_system.py            # RAG pipeline
│       ├── sql_generator.py         # LangChain agent
│       ├── optimizer.py             # Query optimization
│       ├── executor.py              # Query execution
│       └── insights.py              # Insight generation
├── data/
│   └── schema/                      # Database schema definitions
├── tests/
│   ├── unit/
│   ├── integration/
│   └── fixtures/
├── requirements.txt                 # Python dependencies
├── .env.example                     # Environment variables template
├── .gitignore
├── LICENSE
└── README.md
```

## 🔧 Development

### Setting Up Development Environment

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Set up pre-commit hooks
pre-commit install

# Run code formatting
black src/ tests/

# Run linting
flake8 src/ tests/
```

### Project Structure for Developers

The project is organized in functional modules:

- **components/nlp_processor.py** - Handles user input parsing and context management
- **components/rag_system.py** - Manages vector database and semantic retrieval
- **components/sql_generator.py** - LangChain agent configuration and prompt templates
- **components/optimizer.py** - Query optimization analysis and suggestions
- **components/executor.py** - Safe database connection and query execution
- **components/insights.py** - Post-execution analysis and insight generation

## 🧪 Testing

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src tests/

# Run specific test module
pytest tests/unit/test_sql_generator.py

# Run integration tests
pytest tests/integration/
```

### Test Coverage

- Unit tests for each component (target: >80% coverage)
- Integration tests for end-to-end workflows
- Example test queries covering all demo scenarios
- Edge case testing for malicious/ambiguous queries

## 🗺️ Roadmap

### Phase 1: Core MVP (Week 1-2)
- ✅ Basic SQL generation from natural language
- ✅ Database connection & execution
- ✅ Simple Gradio chat interface
- ✅ Context retention across turns
- ✅ RAG system with schema embeddings

### Phase 2: Optimization & Insights (Week 2-3)
- 🔄 Query optimization suggestions
- 🔄 AI-driven insights generation
- 🔄 Advanced multi-table reasoning
- 🔄 Query explanation feature
- 🔄 Performance metrics dashboard

### Phase 3: Polish & Deployment (Week 3+)
- 🔄 Enhanced UI/UX improvements
- 🔄 Comprehensive testing suite
- 🔄 Docker containerization
- 🔄 Production deployment setup
- 🔄 Documentation and API docs

**Timeline**: Aligned with February 15, 2026 contest deadline

## 🤝 Contributing

This is a Codecademy GenAI Bootcamp Contest submission. Contributions are welcome!

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Code Guidelines
- Follow PEP 8 style guidelines
- Add tests for new features
- Update documentation as needed
- Keep commits atomic and descriptive

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Codecademy GenAI Bootcamp** - For the contest opportunity and learning resources
- **LangChain** - Powerful framework for LLM applications
- **OpenAI** - GPT models and embeddings API
- **Vector Databases** - FAISS, Chroma, Milvus communities
- **Open Source Community** - All the amazing libraries we build upon

## 📞 Support

For questions or issues:
- 📧 Email: [Contact Information]
- 🐙 GitHub Issues: [Repository Issues]
- 💬 Discussions: [GitHub Discussions]

---

**Made with ❤️ for the Codecademy GenAI Bootcamp Contest**

Last Updated: February 10, 2026
