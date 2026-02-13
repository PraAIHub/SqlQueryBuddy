# SQL Query Buddy - TODO

## Completed
- [x] Fix 3 broken mock SQL patterns (Q4 returning customers, Q5 January, Q8 more than 3 items)
- [x] Wire parsed NLP output and conversation_history into generator
- [x] Handle follow-up query gracefully when no prior context
- [x] Clean config: remove dead refs, enforce timeout and max_rows
- [x] Reconcile orders.total_amount with SUM(order_items.subtotal)
- [x] SQL validator word-boundary regex (no false positives)
- [x] Add UPDATE/INSERT/GRANT/REVOKE to dangerous keywords
- [x] Fix semicolon injection bypass
- [x] Add foreign key extraction to schema
- [x] Fix LangChain import order + use messages in InsightGenerator
- [x] Clean requirements.txt (remove dead deps, use >= constraints)
- [x] All reviews addressed (3 rounds, 9 agent reviews)
- [x] 51 tests passing
- [x] Fix missing `Tuple` import in app.py (review finding)

## Completed - Feb 12
- [x] Configure and test with OpenAI API key
- [x] Test all 8 demo queries with real LLM (8/8 passing)
- [x] Test multi-turn conversation with real LLM (3/3 passing)
- [x] Fix SQLite syntax in LLM prompt (no MySQL DATE_SUB/EXTRACT)
- [x] Fix SQL comment stripping before validation
- [x] Deploy to Hugging Face Spaces (Docker)
- [x] Verify deployed demo end-to-end

## Future Enhancements
- [ ] OpenAI API integration for production
- [ ] Production database connection (PostgreSQL/MySQL)
- [ ] Cloud deployment (HF Spaces / Railway)
- [ ] Test coverage expansion
- [ ] Query caching layer
