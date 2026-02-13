# SQL Query Buddy - Review & Comparison Report

**Date:** February 13, 2026 (Re-review after fixes)
**Deployed App:** https://huggingface.co/spaces/rsprasanna/SqlQueryBuddy
**Competitor App:** http://3.129.10.17:7860/ (AI Assisted MySQL Query Whisperer)
**Requirements:** GenAI Bootcamp Contest #1 (Deadline: Feb 15, 2026)

---

## Requirements Compliance

| # | Requirement | Status | Notes |
|---|-----------|--------|-------|
| 1 | Conversational Querying | PASS | Natural language to SQL works correctly |
| 2 | RAG-Powered SQL Generation | PASS | FAISS + TF-IDF embeddings + LangChain, visible in System Status tab |
| 3 | Query Optimization | PASS | Multiple rules with severity levels shown after each query |
| 4 | AI-Driven Insights | PASS | Separate AI Insights panel + inline insights in chat |
| 5 | Explainable SQL | PASS | Beginner-friendly explanation provided for every query |
| 6 | Context Retention | **FAIL** | Follow-up query did NOT apply filter - see Issue #1 below |
| 7 | Chat Interface (Gradio) | PASS | Clean tabbed UI with Chat, Schema, System Status tabs |
| 8 | SQL Execution + Results | PASS | Executes and displays results in markdown tables with $ formatting |
| 9 | SQL Injection Prevention | PASS | Blocks DROP/DELETE/ALTER with clear error message |
| 10 | Data Visualization | PASS (NEW) | matplotlib charts generated for multi-row results |

---

## New Features Since Last Review (All Verified Working)

| Feature | Status | Details |
|---------|--------|---------|
| Charts/Visualization | ADDED | matplotlib plots for bar charts (regions, categories) and line charts (trends) |
| AI Insights Panel | ADDED | Separate panel below chart showing business insights |
| Currency Formatting | ADDED | Values display as $49,315.00 instead of raw 49315 |
| Expanded Dataset | ADDED | 150 customers, 25 products, 2500 orders, 7500+ items (~10K rows) |
| 2026 Date Coverage | ADDED | "This quarter" queries now return real data |
| 10 US Regions | ADDED | Meaningful regional analysis across CA, NY, TX, FL, IL, WA, GA, OH, PA, CO |

---

## Functional Test Results (13 Tests)

All tests run against deployed HuggingFace Spaces app on Feb 13, 2026.

### Test 1: "Show me the top 5 customers by total purchase amount"
- **Result:** PASS
- Top customer: Benjamin Williams at $49,315.00
- Chart: YES (matplotlib bar chart)
- Insights: YES (separate panel with business analysis)

### Test 2: "From the previous result, filter customers from New York only"
- **Result:** FAIL - Context retention NOT working
- **Bug:** Returned the SAME top 5 results as Test 1 without applying New York filter
- SQL generated was identical to Test 1 - no WHERE region = 'New York' clause
- **This is a regression** - it worked in the previous local version

### Test 3: "Which product category made the most revenue this quarter?"
- **Result:** PASS - Returns Electronics at $154,550.00
- Chart: None (single row - acceptable)
- Insights: YES

### Test 4: "Show total sales per region for 2024"
- **Result:** PASS - All 10 regions returned with realistic values ($61K-$192K)
- Chart: YES (matplotlib)
- New York leads at $192,341.00

### Test 5: "Show the trend of monthly revenue over time"
- **Result:** PASS - 38 months of data (Jan 2023 - Feb 2026)
- Chart: YES (matplotlib line chart)
- Insights: YES (seasonal fluctuation analysis)

### Test 6: SQL Injection ("DROP TABLE customers; SELECT * FROM orders")
- **Result:** PASS - Blocked with "Query must be a SELECT statement"

### Test 7: Export CSV
- **Result:** PARTIAL - Returns `{'visible': False}` when no prior query data
- Should show a user-friendly message like "Run a query first to export"

### Test 8: Example Query Buttons
- **Result:** PASS - Buttons trigger correct queries with charts

### Test 9: Clear Chat
- **Result:** PASS - Resets conversation, input, chart, and insights

### Test 10: "Find the average order value for returning customers"
- **Result:** FAIL - OpenAI API quota exceeded (429 error)
- This query is NOT covered by mock patterns, so it requires the real LLM
- Falls through to error instead of graceful fallback

### Test 11: "How many unique products were sold in January?"
- **Result:** PASS - Returns 25 unique products
- Correct SQL using COUNT(DISTINCT) + strftime

### Test 12: "How many orders contained more than 3 items?"
- **Result:** FAIL - OpenAI API quota exceeded (429 error)
- Same issue as Test 10 - no mock pattern fallback

### Test 13: "List customers who haven't ordered in last 3 months"
- **Result:** PARTIAL - SQL has logic bug
- Returns 1000 rows with duplicate customer names (Alice Chen appears 10+ times)
- Bug: LEFT JOIN matches ALL old orders per customer instead of checking if LATEST order is old
- Correct approach: subquery with MAX(order_date) per customer

---

## Issues to Fix (Priority Order)

### CRITICAL

#### 1. Context Retention BROKEN on Deployed App
- **Problem:** "From the previous result, filter customers from New York only" returns the SAME results as the original query without applying any filter
- **Impact:** This is a core contest requirement (Requirement #6) and is currently FAILING
- **Evidence:** SQL generated is identical to Test 1 - no WHERE clause added
- **Note:** This worked correctly in the previous local version. Likely a regression in how chat_history is passed to the context manager in the new Gradio message format
- **Fix:** Check that the Gradio chatbot message format (role/content dicts) is being correctly parsed by ContextManager for follow-up detection

#### 2. OpenAI API Quota Exhausted - No Graceful Fallback
- **Problem:** Queries not covered by mock patterns (Tests 10, 12) fail with 429 error instead of falling back to mock generator
- **Impact:** 2 out of 10 contest example queries fail completely with an API error
- **Fix:** Catch the 429/quota error in the SQL generator and fall back to SQLGeneratorMock. OR add mock patterns for "average order value for returning customers" and "orders with more than 3 items"

### HIGH PRIORITY

#### 3. Inactive Customers Query Returns Duplicates
- **Problem:** Test 13 returns 1000 rows with duplicate names. The LEFT JOIN produces one row per old order instead of one row per customer
- **Impact:** Results are misleading and look broken
- **Fix:** The correct SQL should use a subquery: `WHERE customer_id NOT IN (SELECT customer_id FROM orders WHERE order_date >= date('now', '-3 months'))`

#### 4. ~~No Data Visualization / Charts~~ FIXED
- Charts now generated using matplotlib for multi-row results
- Bar charts for categorical data, line charts for trends
- Single-row results correctly skip chart generation

#### 5. ~~Small Sample Dataset~~ FIXED
- Expanded to ~10,000 total rows across 4 tables
- Date range Jan 2023 - Feb 2026

#### 6. ~~Revenue by Category Returns 0 Rows~~ FIXED
- Now returns Electronics at $154,550.00

#### 7. ~~No Currency Formatting~~ FIXED
- Values now display as $49,315.00

### MEDIUM PRIORITY

#### 8. Export CSV Doesn't Work Without Prior Query
- **Problem:** Returns `{'visible': False}` instead of a user-friendly message
- **Fix:** Show a toast/message like "Run a query first before exporting"

#### 9. Optimization Suggestions Sometimes Suggest Indexing Function Names
- **Problem:** For Test 11, suggests "Consider adding indexes on: strftime, m" - these are function names not columns
- **Fix:** Filter out function names (strftime, date, etc.) from index suggestions

### LOW PRIORITY

#### 10. No Query History / Rerun Feature
- Competitor has this but it's not a core requirement

---

## Comparison: Our App vs Competitor (Updated)

### Features We Have That Competitor Doesn't
- Data visualization (matplotlib charts) - NOW WORKING
- Currency formatting ($49,315.00)
- Export to CSV
- System Status dashboard (shows DB, LLM, VectorDB, RAG status)
- 8 clickable example query buttons
- Visible RAG/FAISS/LangChain integration (key contest requirement)
- Works without API key (mock fallback mode)
- Multi-database support (SQLite/PostgreSQL/MySQL)
- Separate AI Insights panel
- Docker deployment ready
- ~10,000 rows of realistic data

### Features Competitor Has That We Don't
- **Query history with rerun** - nice to have
- **Larger dataset** (they have more but ours is now substantial at 10K rows)
- Separate panels layout (dashboard style vs our tabbed chat style)

### Competitor Bugs Still Present
- "Which product category made the most revenue this quarter?" returns **blank**
- Context retention uses "New York" but DB uses "NY" abbreviations - returns 0 results
- Charts panel exists but often returns None
- No visible RAG/VectorDB implementation (key contest requirement missing)
- No query optimization suggestions

---

## Tech Stack Comparison

| Layer | Our App | Competitor | Required |
|---|---|---|---|
| Frontend | Gradio 6.5.1 | Gradio 6.5.1 | Gradio/React |
| AI Layer | LangChain + GPT-4 (mock fallback) | OpenAI (direct?) | LangChain + LLM |
| Vector Search | FAISS + InMemory fallback | Not evident | FAISS/Pinecone/Chroma |
| Backend | Python + Gradio | Python + uvicorn | Python/FastAPI |
| Database | SQLite (10K rows) | MySQL (large) | SQLite/PostgreSQL/MySQL |
| RAG | TF-IDF embeddings + schema retrieval | Not evident | Schema Embeddings + Retrieval |
| Visualization | matplotlib (bar + line charts) | Plotly (often broken) | Not required but impressive |

---

## Summary

**Our app now covers all core contest requirements EXCEPT context retention is broken on the deployed version.** This is the #1 priority fix.

The app has been significantly improved with charts, expanded data, currency formatting, and a polished UI. The remaining critical issues are:

1. **FIX CONTEXT RETENTION** - follow-up queries don't apply filters (regression)
2. **FIX API QUOTA FALLBACK** - catch 429 errors and fall back to mock generator
3. **FIX INACTIVE CUSTOMERS SQL** - duplicate rows due to JOIN logic

Once these 3 issues are fixed, the app will be **stronger than the competitor** on every requirement dimension.
