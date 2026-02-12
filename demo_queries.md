# SQL Query Buddy - Demo Queries

Try these sample queries in the chat interface to explore SQL Query Buddy's capabilities.

## Schema Overview

| Table | Description |
|-------|-------------|
| **customers** | Customer information (name, email, region, signup_date) |
| **products** | Product catalog (name, category, price) |
| **orders** | Purchase records (customer_id, order_date, total_amount) |
| **order_items** | Line items linking orders to products (quantity, subtotal) |

---

## Contest Example Queries (from Requirements)

These are the 10 official demo queries from the contest specification:

1. **"Show me the top 5 customers by total purchase amount."**
   - Tests: Multi-table JOIN, aggregation, ORDER BY, LIMIT
   - Tables: customers, orders

2. **"Which product category made the most revenue this quarter?"**
   - Tests: Aggregation with GROUP BY, date filtering, MAX
   - Tables: products, order_items, orders

3. **"List customers who haven't ordered anything in the last 3 months."**
   - Tests: LEFT JOIN, NULL detection, date arithmetic
   - Tables: customers, orders

4. **"Show total sales per region for 2024."**
   - Tests: GROUP BY with region, date filtering, SUM
   - Tables: customers, orders

5. **"Find the average order value for returning customers."**
   - Tests: Subquery/HAVING, AVG aggregation
   - Tables: customers, orders

6. **"How many unique products were sold in January?"**
   - Tests: COUNT DISTINCT, date filtering
   - Tables: order_items, orders

7. **"Which salesperson generated the highest sales last month?"**
   - Tests: Edge case - no salesperson column (graceful handling)

8. **"From the previous result, filter customers from New York only."**
   - Tests: Context retention, follow-up queries
   - Demonstrates conversational memory

9. **"Show the trend of monthly revenue over time."**
   - Tests: Date grouping, trend analysis
   - Tables: orders

10. **"How many orders contained more than 3 items?"**
    - Tests: Subquery with HAVING, COUNT
    - Tables: orders, order_items

---

## Basic Retrieval Queries

1. **"Show me all customers"**
   - Tests basic SELECT query generation

2. **"List all products with their prices"**
   - Tests column selection from products table

3. **"Get the first 5 customers by signup date"**
   - Tests LIMIT and ORDER BY

---

## Aggregation Queries

4. **"What is the total revenue across all orders?"**
   - Tests SUM aggregation on orders.total_amount

5. **"What is the average product price?"**
   - Tests AVG on products.price

6. **"How many orders has each customer placed?"**
   - Tests COUNT with GROUP BY and JOIN

---

## Filtering Queries

7. **"Show me customers from California"**
   - Tests WHERE clause on region column

8. **"Which products cost more than $200?"**
   - Tests comparison operators on price

9. **"Show orders placed in 2024"**
   - Tests date-based WHERE filtering

---

## Multi-Table Queries

10. **"Show all orders with customer names and product details"**
    - Tests multi-table JOIN across customers, orders, order_items, products

11. **"What products did Alice Chen purchase?"**
    - Tests JOINs with name-based filtering

12. **"Show revenue by product category"**
    - Tests JOIN + GROUP BY across products and order_items

---

## Conversational Queries (Multi-turn)

First ask: **"Show me the top 5 customers by total purchase amount"**
Then ask: **"Now filter them to California only"**
Then ask: **"What's the total revenue from them this year?"**

This demonstrates:
- Context retention across turns
- Understanding "them" refers to previous results
- Understanding "this year" as a temporal reference

---

## Advanced Analytics

13. **"What percentage of total revenue comes from Electronics?"**
    - Tests percentage calculation with category filtering

14. **"Find customers who ordered more than the average order value"**
    - Tests subquery with AVG comparison

15. **"Show monthly order trends for 2024"**
    - Tests date extraction and grouping

---

After each query, observe:
- The generated SQL query
- The beginner-friendly explanation
- The raw query results
- Optimization suggestions
- AI-driven insights and patterns
