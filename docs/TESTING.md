# Testing Strategy & Guide

Complete testing approach for development, staging, and production environments.

## Testing Pyramid 🔺

```
                    ▲
                   /E\  E2E / Manual Tests (5%)
                  /   \     (Full workflow in UI)
                 /─────\
                /       \   Integration Tests (15%)
               /   I     \  (Components interact)
              /───────────\
             /             \ Unit Tests (80%)
            /       U       \(Individual functions)
           /─────────────────\
          └───────────────────┘
```

---

## 1. LOCAL TESTING (Development) 💻

### Setup

```bash
# Install dev dependencies
pip install -r requirements-dev.txt

# Setup pre-commit hooks
pre-commit install
```

### 1.1 Unit Tests

Test individual components in isolation.

```bash
# Run all unit tests
pytest tests/unit/ -v

# Run specific test
pytest tests/unit/test_components.py::TestQueryParser -v

# Run with coverage
pytest tests/unit/ --cov=src --cov-report=html
```

**Coverage Report**:
```bash
pytest tests/unit/ --cov=src tests/
# View: htmlcov/index.html
```

**What's Tested**:
- ✅ Query Parser (intent extraction, modifiers)
- ✅ Context Manager (conversation history)
- ✅ SQL Validator (injection prevention)
- ✅ Query Optimizer (suggestions)
- ✅ Pattern Detector (numeric/string patterns)
- ✅ Trend Analyzer (trend detection)

**Example**:
```python
# tests/unit/test_components.py

def test_sql_injection_prevention():
    """Verify dangerous queries are blocked"""
    is_valid, error = SQLValidator.validate("DROP TABLE users")
    assert is_valid is False
    assert "DROP" in error
```

### 1.2 Integration Tests

Test component interactions end-to-end.

```bash
# Run integration tests
pytest tests/integration/ -v

# Run with detailed output
pytest tests/integration/ -v -s

# Run specific test
pytest tests/integration/test_end_to_end.py::TestEndToEnd::test_query_generation_and_execution -v
```

**What's Tested**:
- ✅ SQL generation + execution pipeline
- ✅ Context management across turns
- ✅ Database schema extraction
- ✅ SQL injection prevention
- ✅ Sample data retrieval

**Example**:
```python
# tests/integration/test_end_to_end.py

def test_query_generation_and_execution(self, temp_db):
    """Test complete pipeline"""
    db = DatabaseConnection(temp_db)
    executor = QueryExecutor(db)
    generator = SQLGeneratorMock()

    result = generator.generate(
        user_query="Show me all users",
        schema_context=str(db.get_schema()),
    )

    assert result["success"]
    exec_result = executor.execute(result["generated_sql"])
    assert exec_result["success"]
```

### 1.3 Manual / UI Testing

Test the Gradio interface interactively.

```bash
# Start the app
python -m src.main

# Open browser to http://localhost:7860
```

**Test Checklist**:
```
[ ] Start Chat
  [ ] Type "Show me all users"
  [ ] Verify SQL is generated
  [ ] Check results display
  [ ] Verify explanation shown

[ ] Multi-turn Conversation
  [ ] Ask first query
  [ ] Ask second query (should use context)
  [ ] Clear chat button works

[ ] Error Handling
  [ ] Intentionally type bad query
  [ ] Verify error message shown

[ ] UI Elements
  [ ] Chat loads quickly (<2s)
  [ ] Results table is readable
  [ ] SQL syntax highlighting works
  [ ] Example queries are clickable
```

### 1.4 Code Quality Checks

```bash
# Format code
black src/ tests/

# Sort imports
isort src/ tests/

# Lint
flake8 src/ tests/

# Type checking (optional)
mypy src/

# Run all checks
pre-commit run --all-files
```

---

## 2. PRE-DEPLOYMENT TESTING (Staging) 🔍

### 2.1 Full Test Suite

```bash
# Run everything
pytest

# With coverage report
pytest --cov=src --cov-report=html tests/

# Generate coverage badge
coverage-badge -o coverage.svg
```

**Target**: >80% code coverage

### 2.2 Database Compatibility Tests

```bash
# Test with different databases

# SQLite (default)
DATABASE_URL=sqlite:///test.db pytest tests/

# PostgreSQL
DATABASE_URL=postgresql://user:pass@localhost/test_db pytest tests/

# MySQL
DATABASE_URL=mysql+pymysql://user:pass@localhost/test_db pytest tests/
```

### 2.3 Performance Testing

```bash
# Create performance test file: tests/performance/test_speed.py

import time

def test_query_execution_speed():
    """Verify queries complete in <3 seconds"""
    db = DatabaseConnection(database_url)

    start = time.time()
    result = executor.execute("SELECT * FROM users")
    elapsed = time.time() - start

    assert elapsed < 3.0, f"Query took {elapsed}s (expected <3s)"
```

### 2.4 Load Testing (Optional)

```bash
# Install locust
pip install locust

# Create locustfile.py
from locust import HttpUser, task

class UserBehavior(HttpUser):
    @task
    def query_endpoint(self):
        self.client.post("/api/query", json={"query": "Show users"})

# Run load test
locust -f locustfile.py --host=http://localhost:7860
```

### 2.5 Security Testing

**SQL Injection Tests**:
```python
# tests/security/test_sql_injection.py

def test_sql_injection_attempts():
    """Verify protection against SQL injection"""
    dangerous_queries = [
        "'; DROP TABLE users; --",
        "1' OR '1'='1",
        "admin' --",
        "' UNION SELECT * FROM passwords --"
    ]

    for query in dangerous_queries:
        is_valid, error = SQLValidator.validate(query)
        assert not is_valid, f"Injection not caught: {query}"
```

**API Security**:
```python
def test_api_rate_limiting():
    """Verify rate limiting is active"""
    for _ in range(100):
        response = client.post("/api/query", json={...})

    # Should get 429 Too Many Requests
    assert response.status_code == 429
```

---

## 3. DEPLOYMENT TESTING (Production) 🚀

### 3.1 Pre-Deployment Checklist

```bash
# Create deployment_checklist.sh
#!/bin/bash

echo "🔍 Pre-deployment checks..."

# 1. Tests pass
pytest --tb=short && echo "✅ Tests passed" || exit 1

# 2. No hardcoded secrets
! grep -r "sk-" src/ --include="*.py" && echo "✅ No secrets found" || exit 1

# 3. Requirements.txt is up to date
pip check && echo "✅ Dependencies OK" || exit 1

# 4. Code style is good
black --check src/ && echo "✅ Code style OK" || exit 1

# 5. Docker builds successfully
docker build -t sql-query-buddy:latest . && echo "✅ Docker build OK" || exit 1

echo "✅ All checks passed! Ready to deploy."
```

### 3.2 Docker Testing

```bash
# Build image
docker build -t sql-query-buddy:test .

# Test image with sample DB
docker run -p 7860:7860 \
  -e DATABASE_URL="sqlite:///test.db" \
  sql-query-buddy:test

# Test with docker-compose
docker-compose up --build

# Run tests in container
docker run sql-query-buddy:test pytest tests/
```

### 3.3 Smoke Tests (Post-Deployment)

```bash
# Quick tests after deployment to verify everything works

# Test 1: Service is up
curl -f http://localhost:7860/health || exit 1

# Test 2: Database connects
curl -X POST http://localhost:7860/api/schema || exit 1

# Test 3: Simple query works
curl -X POST http://localhost:7860/api/query \
  -H "Content-Type: application/json" \
  -d '{"query": "Show me all users"}' || exit 1

echo "✅ All smoke tests passed!"
```

### 3.4 Production Monitoring

**Health Check Endpoint** (add to `src/app.py`):
```python
@app.route("/health")
def health():
    """Health check endpoint"""
    try:
        db = DatabaseConnection(settings.database_url)
        schema = db.get_schema()
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "tables": len(schema)
        }
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}, 500
```

**Monitor in Production**:
```bash
# Check health every 60 seconds
while true; do
  curl -f http://your-domain/health
  if [ $? -ne 0 ]; then
    # Send alert
    echo "Service down! Alert sent."
  fi
  sleep 60
done
```

---

## 4. TEST SCENARIOS 📋

### Scenario 1: Basic Query
```
Input:  "Show me all users"
Expected:
  ✅ SQL generated
  ✅ Query executes
  ✅ Results returned
  ✅ No errors
```

### Scenario 2: Multi-turn Conversation
```
Turn 1: "Show users"
Turn 2: "How many products?"
Turn 3: "Show me Alice's orders"

Expected:
  ✅ Context maintained
  ✅ Each query correct
  ✅ References resolved
```

### Scenario 3: Error Handling
```
Input:  "DROP TABLE users"
Expected:
  ✅ Blocked with clear error
  ✅ No data modified
  ✅ User guided to try different query
```

### Scenario 4: Edge Cases
```
- Empty database
- Very large result sets (>1000 rows)
- Special characters in data
- Null values in columns
- Concurrent requests
```

---

## 5. TESTING COMMANDS SUMMARY 📝

### Development
```bash
# Quick test (unit only)
pytest tests/unit/ -v

# Full test (unit + integration)
pytest tests/ -v

# With coverage
pytest --cov=src tests/

# Specific component
pytest tests/unit/test_components.py::TestQueryParser -v
```

### Pre-Deployment
```bash
# All tests + quality checks
pytest && black . && flake8 src/

# Docker test
docker build -t sql-query-buddy:test .
docker run sql-query-buddy:test pytest

# Security scan
bandit -r src/
```

### Post-Deployment
```bash
# Health check
curl http://your-domain/health

# Smoke tests
bash smoke_tests.sh

# View logs
docker logs -f sql-query-buddy
```

---

## 6. CI/CD INTEGRATION (GitHub Actions) 🔄

**Create `.github/workflows/test.yml`**:
```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - uses: actions/setup-python@v2
        with:
          python-version: 3.9

      - name: Install dependencies
        run: pip install -r requirements-dev.txt

      - name: Run tests
        run: pytest --cov=src

      - name: Code quality
        run: |
          black --check src/
          flake8 src/

      - name: Upload coverage
        uses: codecov/codecov-action@v2
```

**Run automatically on**:
- ✅ Every push to main
- ✅ Every pull request
- ✅ Can manually trigger

---

## 7. TEST DATA MANAGEMENT 📊

### Sample Data Included
- 3 users
- 4 products
- 5 orders with foreign keys

### Add More Test Data
```python
# Create tests/fixtures/sample_data.py

EXTENDED_TEST_DATA = {
    "users": [
        {"id": 1, "name": "User 1", "email": "user1@example.com"},
        # ... more users
    ],
    "products": [
        {"id": 1, "name": "Product 1", "price": 99.99},
        # ... more products
    ]
}
```

---

## 8. TROUBLESHOOTING TESTS 🔧

| Issue | Solution |
|-------|----------|
| Tests fail locally but pass in CI | Check Python version & dependencies |
| Database tests fail | Ensure test DB is clean before run |
| Performance tests timeout | Reduce dataset size or increase timeout |
| Import errors | Reinstall: `pip install -e .` |

---

## 9. TESTING CHECKLIST FOR RELEASE ✅

Before pushing to production:
- [ ] All tests pass (`pytest`)
- [ ] Coverage >80% (`pytest --cov`)
- [ ] No linting errors (`flake8`)
- [ ] Code formatted (`black`)
- [ ] Docker builds (`docker build`)
- [ ] Security scan passed (`bandit`)
- [ ] Manual UI testing completed
- [ ] Database backups ready
- [ ] Monitoring configured
- [ ] Rollback plan documented

---

## 10. CONTINUOUS MONITORING IN PROD

```python
# Add to src/components/monitoring.py

class ProductionMonitoring:
    def __init__(self):
        self.metrics = {
            "queries_executed": 0,
            "errors": 0,
            "avg_response_time": 0,
            "db_connection_issues": 0
        }

    def log_query_execution(self, duration_ms, success):
        """Track query metrics"""
        self.metrics["queries_executed"] += 1
        if not success:
            self.metrics["errors"] += 1
        # Update avg response time
```

Set up alerts for:
- ❌ Error rate >5%
- ⏱️ Response time >3s
- 💾 Database connection failures
- 🔑 API rate limit errors
- 📊 Disk usage >80%

---

## Summary

| Stage | What | How | Pass Criteria |
|-------|------|-----|---------------|
| **Development** | Unit + Integration | `pytest` | All pass |
| **Pre-Deploy** | Full suite + Security | `pytest + bandit` | Coverage >80% |
| **Deployment** | Docker + Smoke tests | `docker build` + `curl` | Health check passes |
| **Production** | Monitoring + Logs | Continuous | <5% error rate |

**Next Step**: Run `pytest` locally to verify everything works! 🚀
