#!/bin/bash

# SQL Query Buddy - Testing Script

set -e

echo "🧪 SQL Query Buddy Test Suite"
echo "=============================="
echo ""

# Color codes
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo -e "${YELLOW}Virtual environment not found. Creating...${NC}"
    python -m venv venv
    source venv/bin/activate
    pip install -q -r requirements-dev.txt
else
    source venv/bin/activate
fi

# Function to run test and report
run_test() {
    local name=$1
    local command=$2

    echo -e "${YELLOW}Running: $name${NC}"
    if eval "$command"; then
        echo -e "${GREEN}✅ $name PASSED${NC}\n"
        return 0
    else
        echo -e "${RED}❌ $name FAILED${NC}\n"
        return 1
    fi
}

FAILED=0

# Test 1: Unit Tests
run_test "Unit Tests" "pytest tests/unit/ -v --tb=short" || FAILED=$((FAILED + 1))

# Test 2: Integration Tests
run_test "Integration Tests" "pytest tests/integration/ -v --tb=short" || FAILED=$((FAILED + 1))

# Test 3: Coverage Report
run_test "Coverage Report" "pytest tests/ --cov=src --cov-report=term-missing --cov-report=html" || FAILED=$((FAILED + 1))

# Test 4: Code Formatting
run_test "Code Formatting (Black)" "black --check src/ tests/" || FAILED=$((FAILED + 1))

# Test 5: Linting
run_test "Linting (Flake8)" "flake8 src/ tests/" || FAILED=$((FAILED + 1))

# Test 6: Import Sorting
run_test "Import Sorting (isort)" "isort --check-only src/ tests/" || FAILED=$((FAILED + 1))

# Summary
echo ""
echo "=============================="
if [ $FAILED -eq 0 ]; then
    echo -e "${GREEN}✅ ALL TESTS PASSED!${NC}"
    echo ""
    echo "Coverage report: open htmlcov/index.html"
    exit 0
else
    echo -e "${RED}❌ $FAILED TEST(S) FAILED${NC}"
    echo ""
    echo "To fix formatting issues, run:"
    echo "  black src/ tests/"
    echo "  isort src/ tests/"
    exit 1
fi
