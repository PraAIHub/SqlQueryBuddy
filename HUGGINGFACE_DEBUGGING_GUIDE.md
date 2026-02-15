# HuggingFace Deployment Debugging Guide

## Overview

You have `OPENAI_API_KEY` set as a **HuggingFace Secret**, which means:
- ✅ Local development uses demo mode (placeholder in .env)
- ✅ HuggingFace deployment uses real OpenAI API (from secrets)

The intermittent insights behavior is happening on **HuggingFace**, not locally.

---

## Step 1: Check HuggingFace Logs

### How to Access Logs

1. Go to your HuggingFace Space: https://huggingface.co/spaces/YOUR-USERNAME/YOUR-SPACE
2. Click the **"Logs"** tab at the top
3. Logs update in real-time as users interact with your app

### What to Look For

#### **Startup Message (First line after restart)**

You should see one of these:

**✅ Good - API Key Loaded:**
```
🤖 Using OpenAI gpt-4 for AI insights
```
or
```
🤖 Using OpenAI gpt-4o-mini for AI insights
```

**❌ Bad - API Key Not Loaded:**
```
🔧 Using LocalInsightGenerator (demo mode - no OpenAI API key)
```

If you see the "demo mode" message, your HuggingFace Secret isn't being read correctly.

#### **When Insights Fail**

Look for these warning messages:

**Rate Limiting (Most Common):**
```
WARNING:src.components.insights:LLM insights generation failed: RateLimitError: Rate limit exceeded
WARNING:src.components.insights:OpenAI rate limit exceeded - will use local fallback
WARNING:src.app:⚠️ LLM insights failed (likely rate limit or quota), falling back to local generator
INFO:src.app:✅ Local insights generated successfully as fallback
```

**Quota/Billing Issues:**
```
WARNING:src.components.insights:LLM insights generation failed: InsufficientQuotaError: You exceeded your current quota
ERROR:src.components.insights:OpenAI quota/billing issue detected - check your account
```

**Authentication Failed:**
```
WARNING:src.components.insights:LLM insights generation failed: AuthenticationError: Invalid API key
ERROR:src.components.insights:OpenAI authentication failed - check API key
```

**Network Issues:**
```
WARNING:src.components.insights:LLM insights generation failed: APIConnectionError: Connection timeout
```

---

## Step 2: Check OpenAI Usage Dashboard

### Go to OpenAI Platform

Visit: https://platform.openai.com/usage

### What to Check

**Usage Tab:**
- Are you hitting daily/monthly token limits?
- Look for spike in usage when HF space is active
- Check for 429 rate limit errors

**Limits Tab:**
- What's your current tier? (Free, Pay-as-you-go, etc.)
- What are your rate limits?
  - **Free Tier:** 3 requests per minute (RPM)
  - **Tier 1:** 500 RPM
  - **Tier 2+:** Higher limits

**Billing Tab:**
- Is your payment method valid?
- Do you have available credits/balance?
- Any spending limits set?

---

## Step 3: Verify HuggingFace Secrets Setup

### Check Secret Configuration

1. Go to your Space settings
2. Click **"Repository secrets"** or **"Settings"** → **"Variables and secrets"**
3. Verify `OPENAI_API_KEY` secret exists and was updated recently

### Test Secret Loading

Add this temporary code to `src/app.py` (remove after testing):

```python
# At the top of __init__ in GradioInterface class
import os
api_key = os.getenv("OPENAI_API_KEY", "")
logger.info(f"🔍 DEBUG: API key length: {len(api_key)} characters")
logger.info(f"🔍 DEBUG: API key starts with: {api_key[:7] if len(api_key) > 7 else 'EMPTY'}")
```

Then check logs for:
```
🔍 DEBUG: API key length: 51 characters
🔍 DEBUG: API key starts with: sk-proj
```

If you see length 0 or "your-api-key-here", the secret isn't loading.

---

## Step 4: Common Issues and Solutions

### Issue 1: "Demo mode" showing on HuggingFace

**Symptom:** Logs show "Using LocalInsightGenerator (demo mode)"

**Cause:** HF Secret not being read

**Solutions:**
- Restart the Space (Settings → Factory reboot)
- Check secret name is exactly `OPENAI_API_KEY` (case-sensitive)
- Try deleting and re-creating the secret
- Check if secret is in correct scope (Space-level, not User-level)

### Issue 2: Rate Limiting (Intermittent Insights)

**Symptom:** Logs show "RateLimitError" or "429"

**Cause:** Hitting OpenAI rate limits (most common on free tier)

**Solutions:**

**Free Tier (3 RPM):**
- Upgrade to Pay-as-you-go for 500 RPM
- Add rate limiting in your app
- Use caching to reduce API calls

**Paid Tier:**
- Check your tier at https://platform.openai.com/account/limits
- Tier upgrades happen automatically with usage/payment history
- Takes 7-14 days to move up tiers

**App-Level Rate Limiting:**
Add to your Space:
```python
import time
from functools import wraps

def rate_limit(calls_per_minute=3):
    """Decorator to rate limit function calls"""
    min_interval = 60.0 / calls_per_minute
    last_called = [0.0]

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            elapsed = time.time() - last_called[0]
            if elapsed < min_interval:
                time.sleep(min_interval - elapsed)
            result = func(*args, **kwargs)
            last_called[0] = time.time()
            return result
        return wrapper
    return decorator
```

### Issue 3: Quota Exhausted

**Symptom:** Logs show "InsufficientQuotaError" or "billing"

**Cause:** Ran out of credits or hit spending limit

**Solutions:**
- Add payment method: https://platform.openai.com/account/billing
- Increase spending limits
- Check if you have any credits remaining
- For free trial: Credits expire after 3 months

### Issue 4: Invalid API Key

**Symptom:** Logs show "AuthenticationError" or "401"

**Cause:** API key is invalid, expired, or wrong

**Solutions:**
- Verify key at https://platform.openai.com/api-keys
- Check if key was deleted or rotated
- Create new key and update HF secret
- Ensure no extra spaces or characters in secret value

---

## Step 5: Optimize for Cost and Reliability

### Use Cheaper Model

Edit your `.env` (for local) or add HF secret:

```bash
OPENAI_MODEL=gpt-4o-mini
```

**Cost comparison:**
- `gpt-4`: $0.03/1K tokens (input)
- `gpt-4o-mini`: $0.00015/1K tokens (input) → **99% cheaper!**

For insights generation, `gpt-4o-mini` works just as well.

### Add Caching

The app already caches query results, but you could cache insights too:

```python
# In app.py, add simple insight caching
self._insight_cache = {}  # Add to __init__

# Before generating insights:
cache_key = f"{user_message}:{hash(str(data))}"
if cache_key in self._insight_cache:
    insights_md = self._insight_cache[cache_key]
else:
    insights_md = self.insight_generator.generate_insights(data, user_message)
    self._insight_cache[cache_key] = insights_md
```

### Monitor Usage

Set up billing alerts:
1. Go to https://platform.openai.com/account/billing/limits
2. Set monthly limit (e.g., $10)
3. Enable email notifications at 50%, 75%, 100%

---

## Step 6: Recommended Settings for HuggingFace

### Add These Secrets to Your HuggingFace Space:

```bash
OPENAI_API_KEY=sk-your-real-key-here
OPENAI_MODEL=gpt-4o-mini
LOG_LEVEL=INFO
```

### Benefits:
- ✅ Uses cheaper model (99% cost reduction)
- ✅ Still gets AI-powered insights
- ✅ Better logging visibility
- ✅ Easier debugging

---

## What Logs Should Look Like (Healthy)

### On Startup:
```
INFO:src.app:🤖 Using OpenAI gpt-4o-mini for AI insights
INFO:src.app:Gradio interface initialized
```

### On Successful Query:
```
INFO:src.app:Processing query: Show me top customers
INFO:src.database_connector:Executing query: SELECT ...
INFO:src.app:Query returned 10 rows
(no warnings about insights)
```

### On Rate Limited Query:
```
INFO:src.app:Processing query: Show revenue trends
WARNING:src.components.insights:LLM insights generation failed: RateLimitError: Rate limit exceeded
WARNING:src.components.insights:OpenAI rate limit exceeded - will use local fallback
WARNING:src.app:⚠️ LLM insights failed (likely rate limit or quota), falling back to local generator
INFO:src.app:✅ Local insights generated successfully as fallback
```

---

## Quick Diagnosis Checklist

Run through this checklist:

- [ ] Go to HF Space → Logs tab
- [ ] Check startup message shows "Using OpenAI" not "demo mode"
- [ ] Run a test query and watch logs
- [ ] Look for warning messages about insights
- [ ] Check OpenAI usage dashboard for errors
- [ ] Verify billing/quota status on OpenAI
- [ ] Check if using free tier (3 RPM) or paid tier

**If insights work sometimes:**
→ Almost certainly rate limiting (3 RPM on free tier)
→ Solution: Upgrade to paid tier or use gpt-4o-mini to reduce calls

**If insights never work:**
→ API key not loading or authentication failed
→ Solution: Check HF secret, verify key on OpenAI dashboard

**If logs show "demo mode":**
→ Secret not being read by HF
→ Solution: Factory reboot Space, re-create secret

---

## Next Steps

1. **Deploy these logging improvements:**
   ```bash
   git push
   ```
   HuggingFace will auto-rebuild your Space

2. **Check logs after deployment:**
   - Wait for Space to restart (2-3 minutes)
   - Watch logs for startup message
   - Run a test query
   - Look for insights warnings

3. **Share findings:**
   - Copy relevant log lines
   - Share OpenAI usage stats (if comfortable)
   - We can diagnose from there

4. **Consider switching to gpt-4o-mini:**
   - Add `OPENAI_MODEL=gpt-4o-mini` as HF secret
   - 99% cheaper, same quality for insights
   - Reduces rate limit pressure
