#!/bin/bash

# SQL Query Buddy - Smoke Tests for Post-Deployment
# Run these tests after deploying to verify the application is working

set -e

# Configuration
HOST=${1:-"http://localhost:7860"}
TIMEOUT=10

echo "🔥 Running Smoke Tests"
echo "====================="
echo "Target: $HOST"
echo ""

# Color codes
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

PASSED=0
FAILED=0

# Test function
test_endpoint() {
    local name=$1
    local method=$2
    local endpoint=$3
    local data=$4

    echo -n "Testing: $name ... "

    if [ "$method" == "GET" ]; then
        if timeout $TIMEOUT curl -f "$HOST$endpoint" > /dev/null 2>&1; then
            echo -e "${GREEN}✅ PASSED${NC}"
            PASSED=$((PASSED + 1))
        else
            echo -e "${RED}❌ FAILED${NC}"
            FAILED=$((FAILED + 1))
        fi
    elif [ "$method" == "POST" ]; then
        if timeout $TIMEOUT curl -f -X POST "$HOST$endpoint" \
            -H "Content-Type: application/json" \
            -d "$data" > /dev/null 2>&1; then
            echo -e "${GREEN}✅ PASSED${NC}"
            PASSED=$((PASSED + 1))
        else
            echo -e "${RED}❌ FAILED${NC}"
            FAILED=$((FAILED + 1))
        fi
    fi
}

# Wait for service to be ready
echo "Waiting for service to be ready..."
for i in {1..30}; do
    if curl -f "$HOST" > /dev/null 2>&1; then
        echo -e "${GREEN}✅ Service is up${NC}\n"
        break
    fi
    if [ $i -eq 30 ]; then
        echo -e "${RED}❌ Service failed to start${NC}"
        exit 1
    fi
    echo -n "."
    sleep 1
done

# Smoke Tests

# Test 1: Service is up and responds
test_endpoint "Service Health" "GET" "/" ""

# Test 2: API endpoint available (if implemented)
echo -n "Testing: Database Connection ... "
if python -c "from src.components.executor import DatabaseConnection; DatabaseConnection('sqlite:///retail.db').get_schema()" 2>/dev/null; then
    echo -e "${GREEN}✅ PASSED${NC}"
    PASSED=$((PASSED + 1))
else
    echo -e "${RED}❌ FAILED${NC}"
    FAILED=$((FAILED + 1))
fi

# Test 3: Query generation (mock)
echo -n "Testing: SQL Generation ... "
if python -c "from src.components.sql_generator import SQLGeneratorMock; gen = SQLGeneratorMock(); result = gen.generate('test', 'schema'); assert result['success']" 2>/dev/null; then
    echo -e "${GREEN}✅ PASSED${NC}"
    PASSED=$((PASSED + 1))
else
    echo -e "${RED}❌ FAILED${NC}"
    FAILED=$((FAILED + 1))
fi

# Test 4: Query validation
echo -n "Testing: SQL Validation ... "
if python -c "from src.components.sql_generator import SQLValidator; valid, _ = SQLValidator.validate('SELECT * FROM users'); assert valid" 2>/dev/null; then
    echo -e "${GREEN}✅ PASSED${NC}"
    PASSED=$((PASSED + 1))
else
    echo -e "${RED}❌ FAILED${NC}"
    FAILED=$((FAILED + 1))
fi

# Test 5: Injection prevention
echo -n "Testing: SQL Injection Prevention ... "
if python -c "from src.components.sql_generator import SQLValidator; valid, _ = SQLValidator.validate('DROP TABLE users'); assert not valid" 2>/dev/null; then
    echo -e "${GREEN}✅ PASSED${NC}"
    PASSED=$((PASSED + 1))
else
    echo -e "${RED}❌ FAILED${NC}"
    FAILED=$((FAILED + 1))
fi

# Summary
echo ""
echo "====================="
echo "Results: $PASSED passed, $FAILED failed"
echo ""

if [ $FAILED -eq 0 ]; then
    echo -e "${GREEN}✅ ALL SMOKE TESTS PASSED!${NC}"
    echo ""
    echo "The application is ready for use."
    exit 0
else
    echo -e "${RED}❌ SOME TESTS FAILED${NC}"
    echo ""
    echo "Please check the application logs:"
    echo "  docker logs <container-name>"
    exit 1
fi
