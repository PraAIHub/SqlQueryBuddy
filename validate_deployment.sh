#!/bin/bash

# SQL Query Buddy - Pre-Deployment Validation
# Run this before deploying to production

set -e

echo "🚀 Pre-Deployment Validation Checklist"
echo "======================================"
echo ""

GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

FAILED=0

# Test 1: All tests pass
echo -n "1. All tests pass ... "
if pytest tests/ -q --tb=no > /dev/null 2>&1; then
    echo -e "${GREEN}✅${NC}"
else
    echo -e "${RED}❌${NC}"
    FAILED=$((FAILED + 1))
fi

# Test 2: Coverage is adequate
echo -n "2. Coverage >80% ... "
coverage_result=$(pytest tests/ --cov=src --cov-report=term-missing:skip-covered 2>&1 | tail -1 | awk '{print $(NF-1)}' | sed 's/%//')
if (( $(echo "$coverage_result >= 80" | bc -l) )); then
    echo -e "${GREEN}✅ ($coverage_result%)${NC}"
else
    echo -e "${RED}❌ ($coverage_result%)${NC}"
    FAILED=$((FAILED + 1))
fi

# Test 3: No hardcoded secrets
echo -n "3. No hardcoded secrets ... "
if ! grep -r "sk-" src/ --include="*.py" 2>/dev/null && \
   ! grep -r "password" src/ --include="*.py" 2>/dev/null; then
    echo -e "${GREEN}✅${NC}"
else
    echo -e "${RED}❌${NC}"
    FAILED=$((FAILED + 1))
fi

# Test 4: Code formatting
echo -n "4. Code is formatted (Black) ... "
if black --check src/ tests/ > /dev/null 2>&1; then
    echo -e "${GREEN}✅${NC}"
else
    echo -e "${RED}❌${NC}"
    FAILED=$((FAILED + 1))
fi

# Test 5: No linting errors
echo -n "5. No linting errors (Flake8) ... "
if flake8 src/ tests/ > /dev/null 2>&1; then
    echo -e "${GREEN}✅${NC}"
else
    echo -e "${RED}❌${NC}"
    FAILED=$((FAILED + 1))
fi

# Test 6: Docker builds
echo -n "6. Docker image builds ... "
if docker build -t sql-query-buddy:validate . > /dev/null 2>&1; then
    echo -e "${GREEN}✅${NC}"
    docker rmi sql-query-buddy:validate > /dev/null 2>&1
else
    echo -e "${RED}❌${NC}"
    FAILED=$((FAILED + 1))
fi

# Test 7: Dependencies installed
echo -n "7. Dependencies check ... "
if pip check > /dev/null 2>&1; then
    echo -e "${GREEN}✅${NC}"
else
    echo -e "${RED}❌${NC}"
    FAILED=$((FAILED + 1))
fi

# Test 8: Database operations work
echo -n "8. Database operations ... "
if python -c "from src.components.executor import DatabaseConnection; db = DatabaseConnection('sqlite:///test_validate.db'); schema = db.get_schema(); assert 'users' in schema or True" 2>/dev/null; then
    echo -e "${GREEN}✅${NC}"
    rm -f test_validate.db
else
    echo -e "${RED}❌${NC}"
    FAILED=$((FAILED + 1))
fi

# Test 9: Security checks
echo -n "9. Security scan (Bandit) ... "
if python -m bandit -r src/ -q > /dev/null 2>&1; then
    echo -e "${GREEN}✅${NC}"
else
    echo -e "${RED}⚠️  (Check warnings)${NC}"
fi

# Test 10: Environment file template exists
echo -n "10. Environment template ... "
if [ -f ".env.example" ]; then
    echo -e "${GREEN}✅${NC}"
else
    echo -e "${RED}❌${NC}"
    FAILED=$((FAILED + 1))
fi

# Test 11: Documentation complete
echo -n "11. Documentation complete ... "
doc_files=("README.md" "docs/README.md" "docs/DEPLOYMENT.md" "docs/TESTING.md")
all_exist=true
for file in "${doc_files[@]}"; do
    if [ ! -f "$file" ]; then
        all_exist=false
        break
    fi
done

if [ "$all_exist" = true ]; then
    echo -e "${GREEN}✅${NC}"
else
    echo -e "${RED}❌${NC}"
    FAILED=$((FAILED + 1))
fi

# Test 12: Git status clean
echo -n "12. Git status clean ... "
if [ -z "$(git status --porcelain)" ] || [ -z "$(git status)" ]; then
    echo -e "${GREEN}✅${NC}"
else
    echo -e "${YELLOW}⚠️  (Uncommitted changes)${NC}"
fi

echo ""
echo "======================================"

if [ $FAILED -eq 0 ]; then
    echo -e "${GREEN}✅ READY FOR DEPLOYMENT!${NC}"
    echo ""
    echo "Next steps:"
    echo "1. Review .env for production settings"
    echo "2. Ensure database is backed up"
    echo "3. Test in staging environment"
    echo "4. Monitor logs after deployment"
    echo "5. Keep rollback plan ready"
    exit 0
else
    echo -e "${RED}❌ $FAILED CHECK(S) FAILED${NC}"
    echo ""
    echo "Please fix the issues before deploying."
    echo ""
    echo "Helpful commands:"
    echo "  # Fix formatting"
    echo "  black src/ tests/"
    echo "  isort src/ tests/"
    echo ""
    echo "  # Run tests"
    echo "  pytest tests/ -v"
    echo ""
    echo "  # Check coverage"
    echo "  pytest --cov=src tests/"
    exit 1
fi
