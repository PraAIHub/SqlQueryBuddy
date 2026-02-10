# SQL Query Buddy - Demo Queries

Try these sample queries in the chat interface to explore SQL Query Buddy's capabilities.

## Basic Retrieval Queries

1. **"Show me all users"**
   - Tests basic SELECT query generation
   - Demonstrates data retrieval and table listing

2. **"List all products with their prices"**
   - Tests column selection
   - Shows how the system chooses specific columns

3. **"Get the first 5 customers"**
   - Tests LIMIT clause handling
   - Demonstrates query optimization awareness

## Aggregation Queries

4. **"How many products are in stock?"**
   - Tests COUNT aggregation
   - Shows numeric result formatting

5. **"What is the average product price?"**
   - Tests AVG calculation
   - Demonstrates numeric insight generation

6. **"Total number of orders placed"**
   - Tests COUNT on different tables
   - Shows schema understanding

## Filtering Queries

7. **"Show me orders from user 1"**
   - Tests WHERE clause generation
   - Demonstrates JOIN understanding between orders and users

8. **"Which products cost more than $100?"**
   - Tests comparison operators
   - Shows filtering with numeric values

## Multi-Table Queries

9. **"Show me all orders with customer names"**
   - Tests JOIN operations
   - Demonstrates relationship understanding

10. **"What products did Alice purchase?"**
    - Tests complex JOINs and filtering
    - Shows context-aware query generation

## Advanced Queries

11. **"List products sorted by price (highest first)"**
    - Tests ORDER BY clause
    - Demonstrates sorting logic

12. **"Group products by category and count items in each"**
    - Tests GROUP BY clause
    - Shows aggregation with grouping

## Conversational Queries (Multi-turn)

First ask: **"Show me the top customers"**
Then ask: **"How many orders did they place?"**

This demonstrates:
- Context retention across turns
- Reference resolution
- Conversation history usage

## Edge Cases

13. **"Show me all active users"**
    - Tests handling of undefined columns
    - Shows error handling and user guidance

14. **"Find duplicates in email addresses"**
    - Tests complex query requirements
    - Shows system creativity in query construction

## Performance Testing

15. **"Get all order data with customer and product information"**
    - Tests complex JOIN scenarios
    - Shows optimization suggestions

After each query, observe:
- ✅ The generated SQL
- 📝 The explanation of what the query does
- 📊 The results table
- 💡 Optimization suggestions
- 🔍 AI-driven insights
