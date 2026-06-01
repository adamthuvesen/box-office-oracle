"""Runtime feature flags used across the ML pipeline."""

from __future__ import annotations

import os


def strict_features_enabled() -> bool:
    """Return True when ``ML_STRICT_FEATURES`` enables strict-column mode.

    Strict mode raises on missing core feature columns instead of silently
    filling with zero — keeps an upstream typo from masquerading as a real
    prediction.
    """
    return os.environ.get("ML_STRICT_FEATURES", "").lower() in ("1", "true", "yes")
