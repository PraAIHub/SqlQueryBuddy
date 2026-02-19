Understood. Also: **some of the files you uploaded earlier expired on my side**, so I can’t reference their exact code paths right now. But you *did* paste the runtime logs, and that’s enough to write a clean `.md` you can hand to Claude to “get it ready” + add the missing logs.

Here’s a **single `.md` file** (copy/paste as-is):

```md
# SqlQueryBuddy – Hugging Face Readiness + Logging Upgrade (NO SQL changes)
Date: 2026-02-15

## Objective
Make the Space production-ready for judging/testing by:
1) Improving runtime observability (logs + UI status).
2) Making configuration explicit and safe (Hugging Face Variables/Secrets).
3) Making OpenAI failures self-explanatory (quota/rate-limit/auth/model).
4) Preventing “silent fallback” confusion (always show mode/provider in UI).

Important: Do NOT change SQL generation logic or prompts in this task.

---

## Current problem (from logs)
Runtime logs show:
- "OpenAI rate limit exceeded - will use local fallback"
- "LLM insights failed ... falling back to local generator"
- "RateLimitError: 429 ... code: insufficient_quota"

Interpretation:
- This is NOT a true per-minute “rate limit”. The OpenAI response indicates `insufficient_quota` (billing/credit issue).
- The app currently logs too little context: no model name, no provider mode, no request id, no config state, and the message text is misleading (“rate limit exceeded”).

---

## 1) Add a clear “AI Status” indicator in the UI (always visible)
Add a header badge that shows:

### When OpenAI is active
✅ **OpenAI Connected**  
- Model: `<OPENAI_MODEL>`  
- Provider: OpenAI  
- Mode: `openai`

### When fallback is active
🟡 **Fallback Mode (Local Insights)**  
- Reason: `insufficient_quota` / `invalid_api_key` / `rate_limited` / `timeout` / `network_error`  
- Last error timestamp

### When misconfigured
🔴 **Misconfigured**
- Missing `OPENAI_API_KEY` (if required)
- Invalid `OPENAI_MODEL`

Also add a small “Details” expand section that shows:
- provider/mode
- model
- temperature
- timeout
- insights enabled?
- masked key present? (true/false)
- last error code + message snippet (no secrets)

---

## 2) Improve logging (more useful, minimal noise)
### 2.1 Log configuration summary at startup (safe)
At startup log **one line**:
- mode/provider
- model
- temperature
- timeout
- insights enabled
- OPENAI_API_KEY present? true/false (never print key)

Example:
`Startup config: mode=auto provider=openai model=gpt-4o-mini insights=true timeout=60 key_present=true`

### 2.2 Log every LLM call outcome with structured fields
On each insights generation attempt, log:
- model
- provider
- duration_ms
- outcome: success/fallback
- error_type + error_code if error
- correlation_id (UUID per request)

Example on failure:
`LLM call failed: req_id=... provider=openai model=gpt-4o-mini duration_ms=842 error_code=insufficient_quota http=429 -> fallback=local`

### 2.3 Fix misleading message text
If OpenAI returns:
- 429 + `insufficient_quota` → log: **OpenAI quota/billing exceeded**
- 429 + `rate_limit_exceeded` → log: **OpenAI rate limited**
- 401 → log: **OpenAI invalid API key**
- 404 → log: **model_not_found**
- timeout → log: **timeout**
- network errors → log: **network**

Do not call everything “rate limit exceeded”.

### 2.4 Capture the full exception safely
Store:
- exception class
- error.code (if present)
- http_status (if present)
- message (truncate to 200 chars)
Never store secrets, headers, or full stack traces in UI.

---

## 3) Add a “Test OpenAI Connection” button in the UI
Add a button that:
- uses the configured OpenAI client
- makes a minimal request (tiny prompt, low tokens)
- returns success/fail with a helpful message

If it fails, show exact categorized fix:
- `insufficient_quota`: “Your OpenAI account has no active quota/credits. Check Billing in OpenAI platform.”
- `invalid_api_key`: “Key is invalid or revoked.”
- `model_not_found`: “Model name not available for this key.”
- `rate_limit_exceeded`: “Too many requests; wait or lower request frequency.”
- timeout: “Increase timeout or check network.”

---

## 4) Hugging Face configuration contract (Secrets/Variables)
The Space must rely on HF Settings and NOT on committed `.env`.

### Secrets (HF Settings → Secrets)
- OPENAI_API_KEY = `sk-...` (required for OpenAI mode)

### Variables (HF Settings → Variables)
- OPENAI_MODEL = `gpt-4o-mini` (default if not set)
- APP_MODE = `auto` | `openai` | `local`
  - auto: use OpenAI if key exists else local
  - openai: require OpenAI; if fails, show error (no silent fallback)
  - local: always use local generator (for demos)
- OPENAI_TIMEOUT_SECONDS = `60`
- OPENAI_TEMPERATURE = `0`
- SHOW_DEBUG_PANEL = `true/false`

---

## 5) Behavior rules (no more silent confusion)
### APP_MODE=openai
- If OpenAI fails, do NOT silently switch.
- Show UI error badge with reason + fix.

### APP_MODE=auto
- Fallback is allowed.
- Must display fallback badge and reason clearly.
- Must log the reason with code.

### APP_MODE=local
- Never call OpenAI.
- Show “Local Mode”.

---

## 6) Acceptance criteria (what “ready” means)
1) UI always shows which mode is active (OpenAI vs Local).
2) Logs always include model + req_id + categorized error code for failures.
3) `insufficient_quota` is correctly labeled (not “rate limit exceeded”).
4) “Test OpenAI Connection” tells me exactly what is wrong and how to fix it.
5) No secrets ever printed in logs or UI.

---

## 7) Notes on the current error
Your log indicates: `429` with `code=insufficient_quota`.
This typically means:
- billing not enabled, or
- no remaining credits/quota, or
- using a key/org without quota.

Once the app has the “Test OpenAI Connection” tool + better logs, we can confirm exactly.
```

### What to do after Claude applies this

In Hugging Face Space → **Settings**:

* **Secrets**: set `OPENAI_API_KEY`
* **Variables**:

  * `OPENAI_MODEL=gpt-4o-mini`
  * `APP_MODE=openai` (so it won’t silently fallback while you test)
  * `SHOW_DEBUG_PANEL=true`

Then restart Space and click **Test OpenAI Connection**. 
