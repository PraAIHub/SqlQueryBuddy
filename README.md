---
title: SQL Query Buddy
emoji: 🤖
colorFrom: purple
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
---

# 🤖 SQL Query Buddy — Conversational AI for Data Insights

> **Ask questions in plain English. Get optimized SQL, charts, and AI business insights in seconds.**
> Built for the [Codecademy GenAI Bootcamp Contest](https://www.codecademy.com) `#CodecademyGenAIBootcamp`

---

## 🎯 What It Does

Type any data question → instantly get SQL + chart + AI analysis:

```
You:  "Show me the top 5 customers by total purchase amount"
      → SELECT c.name, SUM(o.total_amount) AS total_spent ...
      → 📊 Bar chart renders automatically
      → 💡 "Alice Chen leads with $12,450 — 8.3% of total revenue.
            Top 5 represent 34% combined — moderate concentration risk."

You:  "Now only include California"                ← context retained!
      → Adds WHERE c.region = 'California' to previous query

You:  "What % of total revenue do they represent?"   ← pronoun resolved!
      → Builds CTE, computes cohort share: 12.4% of total revenue
```

**Click 🎬 Run Demo** in the app to see this 3-step chain execute live.

---

## 🏗️ Architecture — RAG Pipeline

```
User Question (natural language)
    ↓
[NLP] Intent + entity extraction (ContextManager)
    ↓
[RAG] FAISS cosine similarity → retrieves relevant tables & columns
      (only the 3-5 most relevant, not the full schema → fewer hallucinations)
    ↓
[LangChain + GPT-4o-mini] SQL generation using retrieved schema context
      + full conversation history for multi-turn context retention
    ↓
[Validator] Safety: blocks DROP/DELETE/ALTER, strips SQL comments
    ↓
[Optimizer] Performance tips: missing LIMIT, heavy joins, index suggestions
    ↓
[SQLite] Read-only execution (PRAGMA query_only=ON)
    ↓
[InsightGenerator] GPT-4o-mini → trend analysis, anomaly detection,
      concentration risk, business recommendations
    ↓
[Auto-Fix] On SQL error → regenerate with error context, retry once
```

---

## ✨ Feature Matrix

| Feature | Implementation |
|---------|---------------|
| **Conversational context** | `ConversationState` tracks filters, entities, last SQL, LIMIT across turns |
| **Pronoun resolution** | "them" → resolved customer IDs; "that region" → "West" |
| **RAG schema retrieval** | FAISS + TF-IDF embeddings; only relevant tables sent to LLM |
| **Auto-chart detection** | Time series → line chart · rankings → bar chart · single value → card |
| **AI insights** | GPT-4o-mini: trend, anomaly, concentration risk, business recs |
| **Local fallback** | Full functionality without API key (`LocalInsightGenerator`) |
| **Auto-fix retry** | SQL errors trigger regeneration with error context |
| **171 tests** | Unit + integration test suite, no API key required |
| **DML/injection blocked** | SQL validator + input sanitizer + read-only DB |

---

## 🗄️ Dataset — Realistic Retail Commerce

| Table | Rows | Key Columns |
|-------|------|-------------|
| `customers` | 150 | name, email, region (10 US states), signup_date |
| `products` | 25 | name, category (Electronics/Furniture/Accessories/Office Supplies), price |
| `orders` | 2,500 | customer_id, order_date (2022–2025), total_amount |
| `order_items` | ~6,500 | order_id, product_id, quantity, subtotal |

Schema join path: `customers → orders → order_items → products`

---

## 🚀 Try These Queries (in order — demonstrates context retention)

1. `Show me the top 5 customers by total purchase amount`
2. `Now only include California` ← follow-up, builds on Q1
3. `What % of total revenue do they represent?` ← pronoun resolved
4. `Monthly revenue trend last 12 months` ← time series line chart
5. `Which product category drives the most revenue?` ← pivot to products
6. `Do the top 10 customers account for more than 40% of revenue?` ← threshold check

Or click **🎬 Run Demo** to see queries 1–3 execute automatically.

---

## 🔒 Security

- Read-only SQLite (`PRAGMA query_only=ON` enforced at connection level)
- DML blocked at validation layer (DROP/DELETE/UPDATE/INSERT)
- Prompt sanitization against injection markers
- 500-char input limit · 30s query timeout · 1,000-row result cap

---

## 🛠️ Local Setup

```bash
git clone <repo-url> && cd SqlQueryBuddy
pip install -r requirements.txt
# Optional: set OPENAI_API_KEY in .env for full LLM mode
python -m src.main
# Open http://localhost:7860
```

**No API key?** The app runs in demo mode with a context-aware mock generator — all features work except LLM-powered SQL and AI insights.

---

*Built for the Codecademy GenAI Bootcamp Contest · February 2026*
