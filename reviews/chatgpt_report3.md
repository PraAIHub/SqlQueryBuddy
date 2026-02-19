### 0) Align to contest requirements (do NOT downplay SQL)

Implement and demo all of these as first-class features:

* Accurate + optimized SQL generation + execution
* AI-driven insights **after execution**
* Context retention across turns (filters + “them/this year”)
* Clean chat UI that displays question, SQL, raw results, insights

---did y

## 1) Fix the “No results yet / No SQL generated yet” UI bug (this is in your code)

### What’s happening (root cause)

In `src/app.py`, the empty-state messages are **static HTML blocks** that are always rendered in the tabs, even after you populate the SQL/code/chart components. That’s why the tab can show:

* “No SQL generated yet” **and** the SQL code editor at the bottom
* “No results yet” **and** the chart region / plot area

### Required fix

Replace those static empty-state HTML blocks with **conditional components** whose `visible` flag is updated every run via a single shared `query_state`.

**Implementation approach:**

* Create a `query_state` dict that is the single source of truth:

  * `has_sql`, `has_results`, `has_insights`, `has_chart`
  * `row_count`, `col_count`, `last_run_status`, `error_category`
* In the Results tab:

  * have `results_empty_html` and `results_view` (chart/table container)
  * toggle:

    * `results_empty_html.visible = not has_results`
    * `results_view.visible = has_results`
* In the SQL tab:

  * toggle:

    * `sql_empty_html.visible = not has_sql`
    * `sql_code.visible = has_sql`
* Same pattern for Insights/Context.

**Acceptance criteria:**

* If SQL exists, **no “No SQL generated yet”** is visible.
* If a chart/table exists, **no “No results yet”** is visible.
* All tabs agree (no conflicting “empty” messages).

---

## 2) Context retention: why it’s failing + what to change (based on your code)

### What’s happening (root causes)

You do have a `ContextManager`, but retention breaks for “that region / them / this year” because:

1. **You don’t store computed entities from results**
   Example: after “Show total sales per region”, you never persist `top_region = California` from the actual result set. So “Of that region…” has nothing reliable to bind to.

2. **Your `update_query_plan()` parser is too weak**
   In `src/components/nlp_processor.py`, `update_query_plan()` tries to infer filters by splitting `WHERE` on `AND` and then matching a few strings. This often misses:

   * explicit year ranges (`>= '2024-01-01' AND < '2025-01-01'`)
   * region filters that are expressed differently
   * derived filters (CTEs/subqueries)

3. **“this year” resolution isn’t explicitly enforced**
   The spec expects the bot to understand “this year” naturally—that requires an explicit resolver step, not hoping the LLM guesses correctly.

### Required fix: implement a real ConversationState

Add a structured session state updated **after every successful execution**:

**State must include**

* `filters_applied`: `{time_range, region, category, customer_ids,...}`
* `computed_entities`: `{top_region, top_customer_ids, top_category,...}` (ONLY if actually computed)
* `last_result_summary`: `{row_count, columns, sample_hash}`
* `last_sql`

### Required fix: add a reference resolver step BEFORE SQL generation

Implement `resolve_references(user_text, state)`:

* “this year” → resolve into an explicit date range filter (current year or last known year)
* “them” → last entity set (e.g., last customer_ids)
* “that region” → `computed_entities.top_region`
* “#1 / #2” → require a prior ranked result; otherwise ask a clarification

**Clarify-or-block rule (mandatory)**
If the user references something that does not exist in state, ask a short clarification instead of guessing.

This is exactly what “remembers previous queries, filters, and results” means in the spec.

---

## 3) AI Insights must be grounded in executed data

The spec is explicit: insights happen **after executing** the query.

**Required inputs to insights generator**

* `result_schema`
* `row_count`
* `result_sample` (first N rows)
* basic stats (top values, totals, min/max)

**Required output format**

* 3–5 bullets: **Observation → Why it matters → Action**
* Footer: “Grounded in: X rows, columns used: …, filters: …”

---

## 4) UI structure (keep it contest-clean)

The spec calls for a clean interface that displays user questions, SQL, raw results, and insights.

**Mandatory cleanups**

* Ensure there is **only one** “source of truth” per artifact:

  * SQL lives in SQL tab
  * Results/charts live in Results tab
  * Insights live in Insights tab
* If you keep a preview on the left, it must not conflict with the right tabs (no duplicate “empty state” logic).

---

## 5) Demo script should match their examples

Use prompts directly from the PDF examples (judges will recognize them):

* “Show total sales per region for 2024.”
* “Now filter them to California only.”
* “What’s the total revenue from them this year?”

 
