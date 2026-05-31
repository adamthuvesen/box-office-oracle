"""Train/validation split and target transforms for the ML pipeline."""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import pandas as pd


class DataSplitter:
    """Temporal train/val split — sorted by date, no shuffling."""

    @staticmethod
    def split_data(
        df: pd.DataFrame,
        target_column: str,
        test_size: float = 0.2,
        date_column: Optional[str] = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Sort by date and take the bottom ``1 - test_size`` quantile as train.

        Every validation row's date is greater than or equal to every training
        row's date to avoid look-ahead bias.
        """
        if date_column is None:
            if "RELEASE_DATE" in df.columns:
                date_column = "RELEASE_DATE"
            elif "RELEASE_YEAR" in df.columns:
                date_column = "RELEASE_YEAR"
            else:
                raise ValueError(
                    "DataSplitter.split_data requires a date column "
                    "(RELEASE_DATE or RELEASE_YEAR) to perform a temporal split."
                )

        if date_column not in df.columns:
            raise ValueError(
                f"date_column '{date_column}' not found in DataFrame columns"
            )

        if not 0.0 < test_size < 1.0:
            raise ValueError(f"test_size must be in (0, 1); got {test_size}")

        df_sorted = df.sort_values(by=date_column, kind="mergesort").reset_index(
            drop=True
        )
        n = len(df_sorted)
        split_idx = int(round(n * (1.0 - test_size)))
        if n >= 2:
            split_idx = max(1, min(split_idx, n - 1))

        train_df = df_sorted.iloc[:split_idx]
        val_df = df_sorted.iloc[split_idx:]
        return (
            train_df.drop(columns=[target_column]),
            val_df.drop(columns=[target_column]),
            train_df[target_column],
            val_df[target_column],
        )


class TargetTransformer:
    """log1p / expm1 round-trip for the heavy-tailed revenue target."""

    @staticmethod
    def log_transform(
        y_train: pd.Series, y_val: pd.Series
    ) -> Tuple[pd.Series, pd.Series]:
        return np.log1p(y_train), np.log1p(y_val)
