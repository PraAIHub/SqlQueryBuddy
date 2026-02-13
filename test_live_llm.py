"""Live LLM integration test - runs 8 demo queries + multi-turn conversation."""
import sys
import os

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(__file__))

from src.app import QueryBuddyApp, create_sample_db
from src.config import settings


def run_query(app, query, chat_history):
    """Run a single query and print results."""
    _, updated_history, _chart, _insights = app.process_query(query, chat_history)
    # Messages format: list of {"role": ..., "content": ...}
    last_response = "No response"
    if updated_history:
        last = updated_history[-1]
        if isinstance(last, dict):
            last_response = last.get("content", "No response")
        else:
            last_response = last[1] if isinstance(last, (list, tuple)) else str(last)
    return updated_history, last_response


def main():
    # Verify API key
    if not settings.openai_api_key or settings.openai_api_key == "your_openai_api_key_here":
        print("ERROR: No valid OpenAI API key found in .env")
        sys.exit(1)

    print(f"Model: {settings.openai_model}")
    print(f"API Key: ...{settings.openai_api_key[-4:]}")
    print("=" * 70)

    # Create sample DB if needed
    if settings.database_type == "sqlite":
        create_sample_db()

    # Initialize app
    print("Initializing app...")
    app = QueryBuddyApp()
    print(f"SQL Generator: {type(app.sql_generator).__name__}")
    print("=" * 70)

    # === PART 1: 8 Demo Queries (independent) ===
    demo_queries = [
        "Show me the top 5 customers by total purchase amount",
        "Which product category made the most revenue this quarter?",
        "List customers who haven't ordered anything in the last 3 months",
        "Show total sales per region for 2024",
        "Find the average order value for returning customers",
        "How many unique products were sold in January?",
        "Show the trend of monthly revenue over time",
        "How many orders contained more than 3 items?",
    ]

    print("\n--- PART 1: 8 Demo Queries ---\n")
    passed = 0
    failed = 0

    for i, query in enumerate(demo_queries, 1):
        print(f"\nQ{i}: {query}")
        print("-" * 50)
        chat_history = []  # fresh context per query
        try:
            history, response = run_query(app, query, chat_history)
            is_success = "Generated SQL" in response and "Error" not in response
            status = "PASS" if is_success else "FAIL"
            if is_success:
                passed += 1
            else:
                failed += 1
            print(f"  Status: {status}")
            # Print first 5 lines of response
            lines = response.split("\n")
            for line in lines[:8]:
                print(f"  {line}")
            if len(lines) > 8:
                print(f"  ... ({len(lines) - 8} more lines)")
        except Exception as e:
            failed += 1
            print(f"  Status: ERROR - {e}")

    print(f"\n{'=' * 70}")
    print(f"Demo Queries: {passed} passed, {failed} failed out of {len(demo_queries)}")
    print(f"{'=' * 70}")

    # === PART 2: Multi-turn Conversation ===
    print("\n--- PART 2: Multi-turn Conversation ---\n")
    app2 = QueryBuddyApp()  # fresh app instance
    chat_history = []

    multi_turn_queries = [
        "Show me the top 5 customers by total purchase amount",
        "Now filter them to California only",
        "What's the total revenue from them this year?",
    ]

    mt_passed = 0
    for i, query in enumerate(multi_turn_queries, 1):
        print(f"\nTurn {i}: {query}")
        print("-" * 50)
        try:
            chat_history, response = run_query(app2, query, chat_history)
            is_success = "Generated SQL" in response and "Error" not in response
            status = "PASS" if is_success else "FAIL"
            if is_success:
                mt_passed += 1
            print(f"  Status: {status}")
            lines = response.split("\n")
            for line in lines[:8]:
                print(f"  {line}")
            if len(lines) > 8:
                print(f"  ... ({len(lines) - 8} more lines)")
        except Exception as e:
            print(f"  Status: ERROR - {e}")

    print(f"\n{'=' * 70}")
    print(f"Multi-turn: {mt_passed} passed out of {len(multi_turn_queries)}")
    print(f"{'=' * 70}")

    # Final summary
    total_passed = passed + mt_passed
    total_tests = len(demo_queries) + len(multi_turn_queries)
    print(f"\n TOTAL: {total_passed}/{total_tests} passed")

    if total_passed == total_tests:
        print("All tests passed!")
    else:
        print(f"{total_tests - total_passed} tests need attention.")

    return 0 if total_passed == total_tests else 1


if __name__ == "__main__":
    sys.exit(main())
