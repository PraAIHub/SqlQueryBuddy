"""Shared input sanitization utilities for LLM prompt safety."""
import re


def sanitize_prompt_input(text: str, max_length: int = 500) -> str:
    """Sanitize user input to prevent prompt injection attacks.

    Args:
        text: User-provided input text
        max_length: Maximum allowed length

    Returns:
        Sanitized text safe for LLM prompts
    """
    if not text:
        return ""

    # Limit length
    text = str(text)[:max_length]

    # Remove common prompt-injection markers (case-insensitive substring match)
    _prompt_injection_patterns = [
        ("ignore all previous", "disregard prior"),
        ("ignore previous", "disregard prior"),
        ("forget everything", "disregard prior context"),
        ("new instructions:", "additional context:"),
        ("system:", "note:"),
        ("assistant:", "response:"),
    ]

    text_lower = text.lower()
    for pattern, replacement in _prompt_injection_patterns:
        if pattern in text_lower:
            text = re.sub(re.escape(pattern), replacement, text, flags=re.IGNORECASE)

    # Block SQL-specific dangerous patterns using word-boundary regex
    # so that benign phrases ("drop in revenue", "show deleted records") pass through
    _sql_danger_patterns = [
        (re.compile(r"\bdrop\s+table\b", re.IGNORECASE), "reference table"),
        (re.compile(r"\bdelete\s+from\b", re.IGNORECASE), "query from"),
    ]
    for pattern, replacement in _sql_danger_patterns:
        text = pattern.sub(replacement, text)

    return text.strip()
