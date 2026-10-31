"""Internal helpers shared across feature transformers."""

from __future__ import annotations

from typing import cast

import pandas as pd


def last_column(X: pd.DataFrame, col: str) -> pd.Series:
    """Take the last occurrence when duplicated — engineered numeric beats raw object."""
    # A duplicated label returns a DataFrame; the stubs only type the unique-label case.
    result = cast("pd.Series | pd.DataFrame", X[col])
    if isinstance(result, pd.DataFrame):
        return result.iloc[:, -1]
    return result
