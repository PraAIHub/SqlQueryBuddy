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

    # Remove common injection markers (case-insensitive)
    dangerous_patterns = [
        ("ignore all previous", "disregard prior"),
        ("ignore previous", "disregard prior"),
        ("forget everything", "disregard prior context"),
        ("new instructions:", "additional context:"),
        ("system:", "note:"),
        ("assistant:", "response:"),
        ("drop table", "reference table"),
        ("delete from", "query from"),
    ]

    text_lower = text.lower()
    for pattern, replacement in dangerous_patterns:
        if pattern in text_lower:
            text = re.sub(re.escape(pattern), replacement, text, flags=re.IGNORECASE)

    return text.strip()
