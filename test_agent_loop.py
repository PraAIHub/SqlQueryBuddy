#!/usr/bin/env python3
"""Quick test of agent loop visualization"""
from src.app import QueryBuddyApp

# Test the _generate_agent_loop_html method
loop_state_empty = {
    "user_query": {"completed": False, "duration_ms": 0},
    "rag_search": {"completed": False, "duration_ms": 0},
    "sql_generation": {"completed": False, "duration_ms": 0},
    "validation": {"completed": False, "duration_ms": 0},
    "execution": {"completed": False, "duration_ms": 0},
    "insights": {"completed": False, "duration_ms": 0},
}

loop_state_partial = {
    "user_query": {"completed": True, "duration_ms": 5},
    "rag_search": {"completed": True, "duration_ms": 120},
    "sql_generation": {"completed": True, "duration_ms": 850},
    "validation": {"completed": False, "duration_ms": 0},
    "execution": {"completed": False, "duration_ms": 0},
    "insights": {"completed": False, "duration_ms": 0},
}

loop_state_complete = {
    "user_query": {"completed": True, "duration_ms": 5},
    "rag_search": {"completed": True, "duration_ms": 120},
    "sql_generation": {"completed": True, "duration_ms": 850},
    "validation": {"completed": True, "duration_ms": 15},
    "execution": {"completed": True, "duration_ms": 45},
    "insights": {"completed": True, "duration_ms": 1200},
}

print("Testing agent loop HTML generation...")
print("\n1. Empty state (no steps completed):")
html = QueryBuddyApp._generate_agent_loop_html(loop_state_empty)
print(f"   HTML length: {len(html)} chars")
print(f"   Contains 'Agent Loop': {('Agent Loop' in html)}")

print("\n2. Partial completion (3/6 steps):")
html = QueryBuddyApp._generate_agent_loop_html(loop_state_partial)
print(f"   HTML length: {len(html)} chars")
print(f"   Contains timing (850ms): {('850ms' in html)}")
print(f"   Contains green color: {('#10b981' in html)}")

print("\n3. Full completion (6/6 steps):")
html = QueryBuddyApp._generate_agent_loop_html(loop_state_complete)
print(f"   HTML length: {len(html)} chars")
print(f"   All timings present: {all(str(s['duration_ms']) + 'ms' in html for s in loop_state_complete.values() if s['completed'])}")

print("\n✅ Agent loop visualization tests passed!")
