# 🚀 Quick Start Guide

Get SQL Query Buddy running in 5 minutes!

## Option 1: Using the Startup Script (Easiest) ⭐

```bash
# Clone repository
git clone https://github.com/PraAIHub/SqlQueryBuddy.git
cd SqlQueryBuddy

# Run the startup script
bash run.sh

# Open browser to http://localhost:7860
```

The script will:
- Create a virtual environment
- Install dependencies
- Set up the sample database
- Launch the Gradio interface

## Option 2: Using Docker 🐳

```bash
# Build and run with docker-compose
docker-compose up

# Open browser to http://localhost:7860
```

## Option 3: Manual Setup

```bash
# Clone repository
git clone https://github.com/PraAIHub/SqlQueryBuddy.git
cd SqlQueryBuddy

# Create virtual environment
python -m venv venv

# Activate it
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Create .env file
cp .env.example .env

# (Optional) Add your OpenAI API key to .env for full functionality

# Run the app
python -m src.main
```

## First Steps

1. **Open the Interface**: Go to `http://localhost:7860`

2. **Try a Sample Query**: Click on one of the examples or type:
   - "Show me all users"
   - "How many products are in stock?"

3. **See the Magic**: Observe:
   - ✅ Generated SQL
   - 📝 Query explanation
   - 📊 Results table
   - 💡 AI insights

## Configuration

Edit `.env` to customize:

```env
# Use your own OpenAI API key (optional)
OPENAI_API_KEY=sk-...

# Change the model
OPENAI_MODEL=gpt-4

# Connect to your database
DATABASE_URL=postgresql://user:pass@localhost/db

# Adjust port
GRADIO_SERVER_PORT=7860
```

## Connecting Your Database

### SQLite (Default)
Already works! Just place a `.db` file in the project root and update `DATABASE_URL`:
```
DATABASE_URL=sqlite:///your_database.db
```

### PostgreSQL
```
DATABASE_URL=postgresql://user:password@localhost:5432/database_name
```

### MySQL
```
DATABASE_URL=mysql+pymysql://user:password@localhost:3306/database_name
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Port 7860 already in use | Change `GRADIO_SERVER_PORT` in `.env` |
| No SQL generation | Add `OPENAI_API_KEY` to `.env` (will use mock otherwise) |
| Database connection error | Verify `DATABASE_URL` and ensure database exists |
| Python not found | Install Python 3.9+ from python.org |

## Next Steps

- 📚 Read [docs/README.md](docs/README.md) for detailed documentation
- 🔍 Try [demo_queries.md](demo_queries.md) for more examples
- 🧪 Run tests: `pytest`
- 📖 Check [docs/specification.md](docs/specification.md) for technical details

## Need Help?

- Check error messages in the terminal
- Review `.env.example` for all available options
- See [Troubleshooting section in docs/README.md](docs/README.md#troubleshooting)

Happy querying! 🎉
