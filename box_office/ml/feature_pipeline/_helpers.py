"""Internal helpers shared across feature transformers."""

from __future__ import annotations

import pandas as pd


def _column(X: pd.DataFrame, col: str) -> pd.Series:
    """Take the last occurrence when duplicated — engineered numeric beats raw object."""
    result = X[col]
    if isinstance(result, pd.DataFrame):
        result = result.iloc[:, -1]
    return result
