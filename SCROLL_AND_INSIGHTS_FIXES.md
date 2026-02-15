# Scroll and AI Insights Investigation Summary

## 1. Chatbot Auto-Scroll Fix

### Problem
Chatbot doesn't scroll to latest message when user manually scrolls to top and asks a new question.

### Root Cause
- Gradio's `autoscroll=True` stops working once user manually scrolls
- Previous JavaScript fix used selectors that don't match Gradio 6.5.1's DOM structure

### Solution Implemented
Improved JavaScript with:
- **Multiple CSS selectors** to find chatbot in Gradio 6.x
- **Console logging** for debugging (press F12 to see browser console)
- **Multiple initialization attempts** (1s, 2s, 3s delays to ensure Gradio is ready)
- **requestAnimationFrame** for smoother scrolling
- **Better element detection** (checks if element is actually scrollable)

### Testing the Fix
1. Start the app: `python -m src.app`
2. Open browser console (F12 → Console tab)
3. Look for messages like:
   ```
   Found chatbot container using: .chatbot
   Chatbot auto-scroll observer set up successfully
   Scrolled chatbot using selector: .chatbot .overflow-y-auto
   ```
4. Ask a few questions, scroll to top, ask another question
5. Should auto-scroll to bottom showing latest message

**If scroll still doesn't work:**
- Check console logs to see which selector is being used
- Look for any JavaScript errors in console
- Try different browsers (Chrome, Firefox, Edge)

---

## 2. AI Insights Intermittent Behavior

### Investigation Results

**KEY FINDING:** You're running in **DEMO MODE**, not using OpenAI API at all!

Your `.env` file has:
```
OPENAI_API_KEY=your-api-key-here
```

This is a **placeholder**, not a real API key.

### What This Means

**Current Behavior:**
- App uses `LocalInsightGenerator` (pattern-based, no API calls)
- Insights appear when local generator detects patterns in data
- Insights don't appear when data is too simple or patterns aren't clear
- **No OpenAI API calls are being made**
- **No balance/quota issues** (because you're not using the API)

**LocalInsightGenerator generates insights when:**
- ✅ Data has clear top performers (revenue, sales, counts)
- ✅ Categories or regions show clear differences
- ✅ Time-series data shows trends
- ❌ Data is too uniform (no interesting patterns)
- ❌ Results have only 1-2 rows (not enough to analyze)
- ❌ No numeric columns to analyze

### How to Enable Real OpenAI API Insights

**Step 1: Get OpenAI API Key**
1. Go to https://platform.openai.com/api-keys
2. Create new API key
3. Copy the key (starts with `sk-`)

**Step 2: Update .env File**
```bash
# Edit .env file
nano .env

# Replace this line:
OPENAI_API_KEY=your-api-key-here

# With your real key:
OPENAI_API_KEY=sk-your-actual-key-here

# Optional: change model (default is gpt-4)
OPENAI_MODEL=gpt-4o-mini  # Cheaper and faster
# or
OPENAI_MODEL=gpt-4  # More accurate but expensive
```

**Step 3: Restart App**
```bash
python -m src.app
```

**Step 4: Verify**
Look at the status chips at top of page:
- Before: "Mock (demo mode - set OPENAI_API_KEY for full LLM)"
- After: "GPT-4" or "GPT-4o-mini" (shows you're using real API)

### Cost Implications

**If you enable OpenAI API:**

**GPT-4 Pricing (as of Feb 2024):**
- Input: $0.03 per 1K tokens
- Output: $0.06 per 1K tokens
- Typical insight generation: ~500 tokens = $0.02-0.03 per query

**GPT-4o-mini Pricing (recommended for testing):**
- Input: $0.00015 per 1K tokens
- Output: $0.0006 per 1K tokens
- Typical insight generation: ~500 tokens = ~$0.0003 per query

**Estimate for testing:**
- 100 queries with GPT-4: ~$2-3
- 100 queries with GPT-4o-mini: ~$0.03

### Rate Limits (Free Tier)

If using OpenAI free tier:
- 3 requests per minute (RPM)
- 200,000 tokens per day

If you hit rate limits:
- App automatically falls back to LocalInsightGenerator
- You'll see: "AI Insights unavailable - The LLM service encountered an error"
- Then local insights appear below

### Recommendation

**For Contest Submission:**
- Keep running in **demo mode** (current setup)
- LocalInsightGenerator works well enough for demonstration
- No API costs or rate limit worries
- Status shows "Live Mode" which looks good

**For Production/Real Use:**
- Add real OpenAI API key
- Use `gpt-4o-mini` model (99% cheaper than gpt-4)
- Set up billing with reasonable limits
- Monitor usage at https://platform.openai.com/usage

---

## Summary

✅ **Scroll Fix:** Improved JavaScript with multiple selectors and debugging
❓ **Scroll Testing Needed:** Test in browser with console open (F12)

✅ **Insights Investigation:** You're in demo mode (no API calls)
✅ **No Balance Issue:** Not using OpenAI API at all
✅ **Local Insights Work:** When data has clear patterns to detect

**Next Steps:**
1. Test scroll fix with browser console open
2. Decide if you want to enable real OpenAI API (optional)
3. If scroll still doesn't work, share console logs for further debugging
