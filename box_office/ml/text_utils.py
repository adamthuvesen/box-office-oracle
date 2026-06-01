"""Shared text-parsing utilities used by feature engineering and inference.

``MAX_LITERAL_EVAL_BYTES`` and ``LiteralEvalTooLarge`` cap the input we hand
to ``ast.literal_eval``. The parser is recursive and an attacker-controlled,
deeply-nested payload can exhaust CPU/memory; this bound stops the predict
path from being a DoS surface.
"""

from __future__ import annotations

import ast
from typing import Any, List

import numpy as np
import pandas as pd

MAX_LITERAL_EVAL_BYTES = 8192


class LiteralEvalTooLarge(ValueError):
    """Raised when a string exceeds ``MAX_LITERAL_EVAL_BYTES``.

    Inherits ``ValueError`` so existing ``except (ValueError, SyntaxError)``
    callers continue to swallow it as a parse failure; the inference layer
    catches the specific subclass and translates to a 400 response.
    """


def process_text_list(value: Any) -> List[str]:
    """Normalize a text-list field to a clean lower-cased ``list[str]``.

    Accepts None/NaN, empty strings, JSON-like strings, Python lists, and
    scalar strings. String inputs are length-bounded by
    ``MAX_LITERAL_EVAL_BYTES`` before being handed to ``ast.literal_eval``.
    """
    if value is None or (isinstance(value, float) and np.isnan(value)) or value == "[]":
        return []

    if isinstance(value, list):
        return [str(tok).strip().lower() for tok in value if tok and not pd.isna(tok)]

    if isinstance(value, str):
        if len(value.encode("utf-8")) > MAX_LITERAL_EVAL_BYTES:
            raise LiteralEvalTooLarge(
                f"Input exceeds MAX_LITERAL_EVAL_BYTES={MAX_LITERAL_EVAL_BYTES}"
            )
        try:
            parsed = ast.literal_eval(value)
            if isinstance(parsed, list):
                return [str(tok).strip().lower() for tok in parsed if tok]
        except (ValueError, SyntaxError):
            pass
        return [value.strip().lower()]

    return [str(value).strip().lower()]
