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
- **🎨 Clean Chat Interface** - Intuitive Gradio web interface for seamless user experience

## 🎬 Demo

**Live Demo:** [https://huggingface.co/spaces/rsprasanna/SqlQueryBuddy](https://huggingface.co/spaces/rsprasanna/SqlQueryBuddy)

## 🚀 Quick Start

### Prerequisites
- Python 3.9 or higher
- pip package manager
- SQLite database (default; PostgreSQL/MySQL experimental with LLM mode)

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
from src.app import QueryBuddyApp

# Initialize and launch
app = QueryBuddyApp()
demo = app.create_interface()
demo.launch()
# Visit http://localhost:7860 in your browser
```

### Running the Web Interface

```bash
python -m src.app
# Or use the startup script: ./run.sh
```

## 🛠️ Tech Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **Frontend** | Gradio | Interactive chat interface |
| **AI Engine** | LangChain + GPT-4 | Query generation & insights |
| **Vector Search** | FAISS | Schema embeddings & retrieval |
| **Backend** | Python + Gradio | Web server & orchestration |
| **Database** | SQLite (default) | Data storage & querying |
| **RAG Framework** | Custom RAG Pipeline | Context-aware retrieval |
| **Embeddings** | TF-IDF (local) | Semantic search (no API needed) |

## 🏗️ Architecture

See [ARCHITECTURE.md](ARCHITECTURE.md) for detailed Mermaid diagrams including:
- System Architecture Diagram
- Data Flow Sequence Diagram
- Entity Relationship Diagram
- RAG Pipeline Detail
- Component Dependency Map
- Deployment Architecture

**High-Level Flow:**
```
User Question → NLP Parser → RAG Schema Search → SQL Generation (LangChain + GPT-4)
    → Validation → Optimization → Execution → Insight Generation → Formatted Response
```

## 📚 Example Queries

SQL Query Buddy handles a wide variety of natural language questions:

1. **"Show me the top 5 customers by total purchase amount."**
2. **"Which product category made the most revenue this quarter?"**
3. **"List customers who haven't ordered anything in the last 3 months."**
4. **"Show total sales per region for 2024."**
5. **"Find the average order value for returning customers."**
6. **"How many unique products were sold in January?"**
7. **"Which customer had the highest spending last month?"**
8. **"From the previous result, filter customers from New York only."**
9. **"Show the trend of monthly revenue over time."**
10. **"How many orders contained more than 3 items?"**

## 📁 Project Structure

```
SQLQueryBuddy/
├── docs/
│   ├── README.md                    # This file
│   ├── ARCHITECTURE.md              # Architecture diagrams
│   └── specification.md             # Technical specification
├── src/
│   ├── __init__.py
│   ├── main.py                      # Entry point
│   ├── app.py                       # Gradio web interface
│   ├── config.py                    # Configuration management
│   └── components/
│       ├── nlp_processor.py         # NLP layer (query parsing, context)
│       ├── rag_system.py            # RAG pipeline (FAISS, TF-IDF)
│       ├── sql_generator.py         # LangChain SQL generation + mock
│       ├── optimizer.py             # Query optimization suggestions
│       ├── executor.py              # Database connection & execution
│       └── insights.py              # AI insight generation
├── tests/
│   ├── unit/                        # Unit tests (41 tests)
│   └── integration/                 # End-to-end tests (12 tests)
├── requirements.txt                 # Python dependencies
├── Dockerfile                       # Docker deployment
├── docker-compose.yml
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

### MVP - COMPLETE
- ✅ Natural language to SQL generation (LangChain + GPT-4)
- ✅ RAG-powered schema-aware query generation (FAISS/Chroma)
- ✅ Conversational querying with context retention
- ✅ Query optimization suggestions
- ✅ AI-driven insights generation
- ✅ Explainable SQL with reasoning
- ✅ Retail commerce schema (customers, products, orders, order_items)
- ✅ Gradio chat interface
- ✅ SQL injection prevention & security
- ✅ Comprehensive testing (unit + integration)
- ✅ Docker containerization
- ✅ Full documentation & architecture diagrams

**Submitted**: February 15, 2026 contest deadline

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

**Made with ❤️ for the Codecademy GenAI Bootcamp Contest**

Last Updated: February 13, 2026
