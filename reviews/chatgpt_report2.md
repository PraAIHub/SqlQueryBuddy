# SqlQueryBuddy – Fix/Hardening Checklist for Claude (non‑SQL focus)

## What we observed (from your Space + logs)

1. **The app *is* trying to call OpenAI**  
   Logs show requests to `https://api.openai.com/v1/chat/completions` and the app printing:  
   - `🤖 Using OpenAI gpt-4o-mini for AI insights`  
   - `Startup config: app_mode=openai model=gpt-4o-mini llm_enabled=True ... key_present=True ...`  
   Then the request fails with **HTTP 429** and `insufficient_quota` / `quota_exceeded`.

2. **The 429 is not a “bug in SQL”**  
   The error is coming from OpenAI API quota/billing status for the key used in the Space env vars (or the org/project tied to that key).  
   Your UI message matches this: **“OpenAI Error: quota exceeded … Check your API key, billing, and model name.”**

3. **Model used by the code (important for transparency + config)**
   - The code uses **OpenAI via LangChain** (`ChatOpenAI`) with the model coming from env var (default is `gpt-4o-mini`).  
   - It is **not using Hugging Face hosted LLMs** for generation by default.
   - When OpenAI is unavailable (depending on mode), the app can fall back to a **local/mock generator** for demo behavior.

---

## High-priority fixes to request from Claude (non‑SQL)

### 1) Make model/provider fully configurable and visible in the UI
**Goal:** You can switch between OpenAI and fallback explicitly, and confirm what’s running.

**Ask Claude to:**
- Ensure env vars are consistently supported (single source of truth):
  - `APP_MODE` (or `LLM_PROVIDER`) with allowed values: `openai`, `mock`, `auto`.
  - `OPENAI_MODEL` (default `gpt-4o-mini`).
  - `OPENAI_BASE_URL` (optional; allow compatibility providers if you ever need them).
  - `OPENAI_TIMEOUT` and `OPENAI_MAX_RETRIES`.
- Add an **“Runtime / Config” section** on the About tab showing:
  - provider/mode, model, timeout, retry count
  - whether key is present (boolean only, never print actual key)
  - whether fallback is enabled
  - build/version info (commit hash if available)

### 2) Improve OpenAI error handling + user-facing messages
**Goal:** “Quota exceeded” should be unambiguous and actionable.

**Ask Claude to:**
- Detect and classify OpenAI failures:
  - `insufficient_quota` (billing/quota)
  - `rate_limit_exceeded` (RPM/TPM)
  - `invalid_api_key` / auth errors
  - `model_not_found`
  - network timeouts
- In UI, show a **short actionable summary**:
  - “Your OpenAI API key has no remaining quota (billing/quota). Add credit / correct project/org / use a different key.”
  - If it’s rate-limits: “Too many requests. Try again; increasing backoff…”
- In logs, include:
  - request id (if present), HTTP status, `error.type`, `error.code`
  - retry attempt number, backoff delay

### 3) Make logging actually useful for debugging (without leaking secrets)
**Goal:** Your current logs are close, but need more context for quick diagnosis.

**Ask Claude to:**
- Add a structured logger (or consistent log fields) for:
  - provider/mode chosen (and *why*—e.g., “no key found”, “app_mode=auto and OpenAI failed”)
  - OpenAI call metadata: endpoint, model, timeout, attempt #, elapsed time
  - exception class + parsed OpenAI error fields
- Add `LOG_LEVEL` env var support (INFO/DEBUG).
- Add an optional `SHOW_DEBUG_PANEL=true` to display sanitized runtime config in the UI.

### 4) Stop “silent fallback” confusion (make it explicit)
**Goal:** If fallback happens, the user should clearly know it’s fallback and what’s degraded.

**Ask Claude to:**
- When running in `auto` mode: if OpenAI fails, the UI should show:
  - “OpenAI unavailable → using fallback mode (SQL generation and insights are approximate).”
- When running in `openai` mode: **do not fallback**, fail loudly (you already did this—keep it).
- Ensure the “LLM status” badge updates appropriately:
  - ✅ OpenAI active
  - ⚠️ Fallback active
  - ❌ OpenAI error (with type)

### 5) Fix Hugging Face operational ergonomics (restart/rebuild UX)
**Goal:** Reduce confusion on how to apply env var changes.

**Ask Claude to add a short “Deploy Notes” section** on About:
- “After changing HF Variables/Secrets, use **Settings → ‘Restart this Space’**.”
- If available, “Factory reboot” is only shown on some Spaces/plans; restart is usually enough.
- Mention that a “rebuild” is only needed if `requirements.txt` changed.

### 6) UX polish issues to fix before/alongside testing
(These are non-SQL and improve contest impression.)
- **Input appending bug**: typing a new prompt should replace, not append.
- “No results yet” state: show a sample output or a better empty state message.
- Make the “Export” button robust (CSV formatting, numeric formatting).
- Mobile responsiveness: basic layout checks.

---

## What to tell Claude about the current root cause (so it doesn’t chase ghosts)

- The current failure is **OpenAI API 429 with `insufficient_quota` / quota exceeded** coming from the key’s billing/quota state.  
- The app is correctly configured to use `gpt-4o-mini` (per logs), and the request is reaching OpenAI—so this is **not** a Hugging Face model issue.

---

## Quick verification checklist once Claude implements the changes

1. **Model transparency**
   - About/Debug shows provider, model, key_present, fallback_enabled.
2. **Error clarity**
   - Enter a prompt with invalid key → UI shows “invalid_api_key”.
   - Enter a prompt with no quota → UI shows “insufficient_quota”.
3. **Logging**
   - Logs show attempt counts, backoff delays, parsed OpenAI error type/code.
4. **Fallback clarity**
   - In `auto`, OpenAI fail triggers visible “Fallback active” badge + banner.
   - In `openai`, OpenAI fail shows a hard error, no fallback.
5. **UX**
   - Input does not append previous text.
   - Results/Insights tabs update cleanly after each run.

---

## Optional: recommended Hugging Face env vars (keep minimal)

- `APP_MODE=openai` (or `auto` once you trust fallback messaging)
- `OPENAI_MODEL=gpt-4o-mini`
- `LOG_LEVEL=INFO` (switch to `DEBUG` temporarily)
- `SHOW_DEBUG_PANEL=true`

(Keep secrets only in HF “Secrets”: `OPENAI_API_KEY`.)

---

## Notes (for Claude)
- Do **not** print secrets, headers, or full exception payloads that might include them.
- If you add config display, mask sensitive values: show booleans or last 4 chars only.
