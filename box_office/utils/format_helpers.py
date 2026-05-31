"""Safe numeric formatting helpers for log/report strings."""

from typing import Any


def safe_format(value: Any, fmt: str, fallback: str = "N/A") -> str:
    """Return ``format(value, fmt)``; return ``fallback`` on TypeError/ValueError."""
    try:
        return format(value, fmt)
    except (TypeError, ValueError):
        return fallback
