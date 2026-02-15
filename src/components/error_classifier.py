"""Categorized error classification for OpenAI / LLM API failures.

Provides a single function, ``classify_llm_error``, that inspects an
exception and returns a short machine-readable category string plus a
user-friendly message.  This avoids the previous behaviour of lumping
every 429 as "rate limit exceeded" when the real cause is often
``insufficient_quota`` (a billing issue).
"""

from typing import Tuple


def classify_llm_error(exc: BaseException) -> Tuple[str, str]:
    """Classify an LLM-related exception into a category and message.

    Returns
    -------
    (category, user_message) where *category* is one of:
        "quota_exceeded", "rate_limited", "invalid_api_key",
        "model_not_found", "timeout", "network_error", "unknown"
    """
    error_str = str(exc).lower()
    error_type = type(exc).__name__

    # Extract HTTP status if present (openai / httpx style)
    http_status = getattr(exc, "status_code", None) or getattr(exc, "http_status", None)
    error_code = getattr(exc, "code", None)
    if error_code:
        error_code = str(error_code).lower()

    # --------------------------------------------------
    # 429 errors: distinguish quota from true rate-limit
    # --------------------------------------------------
    if http_status == 429 or "429" in error_str:
        if (error_code and "insufficient_quota" in error_code) or "insufficient_quota" in error_str:
            return (
                "quota_exceeded",
                "OpenAI quota/billing exceeded",
            )
        if (error_code and "rate_limit" in error_code) or "rate_limit_exceeded" in error_str:
            return (
                "rate_limited",
                "OpenAI rate limited",
            )
        # Ambiguous 429 -- default to quota since that is the most common
        return (
            "quota_exceeded",
            "OpenAI quota/billing exceeded",
        )

    # --------------------------------------------------
    # 401 - authentication
    # --------------------------------------------------
    if http_status == 401 or "401" in error_str or "authentication" in error_str or "invalid api key" in error_str or "invalid_api_key" in error_str:
        return (
            "invalid_api_key",
            "OpenAI invalid API key",
        )

    # --------------------------------------------------
    # 404 - model not found
    # --------------------------------------------------
    if http_status == 404 or "model_not_found" in error_str or ("404" in error_str and "model" in error_str):
        return (
            "model_not_found",
            "Model not found",
        )

    # --------------------------------------------------
    # Timeout
    # --------------------------------------------------
    if "timeout" in error_type.lower() or "timeout" in error_str or "timed out" in error_str:
        return (
            "timeout",
            "Request timeout",
        )

    # --------------------------------------------------
    # Network / connection errors
    # --------------------------------------------------
    if any(kw in error_type.lower() for kw in ("connection", "network", "dns", "ssl")):
        return (
            "network_error",
            "Network/connection error",
        )
    if any(kw in error_str for kw in ("connection", "network", "dns", "ssl", "unreachable")):
        return (
            "network_error",
            "Network/connection error",
        )

    # --------------------------------------------------
    # Fallback
    # --------------------------------------------------
    return (
        "unknown",
        "Network/connection error",
    )
