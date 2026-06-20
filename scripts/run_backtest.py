"""Offline per-year backtest driver.

Reproduces the per-year expanding-window backtest the README quotes — model
dollar-space R², log-space R², and rank correlation against a log-budget
baseline, per year, with median APE — without SageMaker, Lambda, or any AWS
round-trip. It runs entirely against the local training snapshot under
``analysis/datasets_high/``.

Why this exists separately from ``box_office.ml.backtest_report``'s CLI: that
CLI expects a raw-movie parquet plus a SageMaker-produced ``cv_results.json``,
and neither exists offline. This driver loads the training snapshot, scores the
selected-feature production model (``SELECTED_FEATURES`` — the active schema
contract), runs the
repo's ``TimeSeriesCrossValidator`` with the production XGBoost wrapper, then
feeds the folds into the same report builder the CLI would call.

Data hygiene: the snapshot carries 6 rows where ``production_budget`` was filled
as a fixed 0.4 x ``worldwide_gross`` (a budget-column imputation artifact); those
are dropped before scoring so the budget feature and the baseline can't read the
target off them.

Run:  uv run python scripts/run_backtest.py

The snapshot CSVs are gitignored by policy, so this is not rerunnable by a
stranger without `make datasets` (Snowflake credentials). Only the derived table
(``results/per_year_table.{md,json}``) is committed.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from box_office.ml.backtest import render_metrics_table_markdown
from box_office.ml.backtest_report import build_report
from box_office.ml.cv import TimeSeriesCrossValidator
from box_office.ml.feature_pipeline.constants import SELECTED_FEATURES
from box_office.ml.model import BoxOfficeXGBoostModel

DATA_DIR = Path("analysis/datasets_high")
OUTPUT = Path("results/per_year_table")

TARGET_COL = "worldwide_gross"
BUDGET_COL = "production_budget"
YEAR_COL = "release_year"

# production_budget filled as 0.4 * worldwide_gross marks an imputed budget;
# drop those rows so the target can't be read off the budget column.
IMPUTE_RATIO = 0.4
IMPUTE_TOL = 1e-3


def load_snapshot() -> tuple[pd.DataFrame, pd.Series]:
    """Load the aligned ``X_train`` / ``y_train`` pair, or fail loudly."""
    x_path = DATA_DIR / "X_train.csv"
    y_path = DATA_DIR / "y_train.csv"
    missing = [str(p) for p in (x_path, y_path) if not p.exists()]
    if missing:
        raise SystemExit(
            f"needs analysis/datasets_high snapshot: missing {', '.join(missing)}.\n"
            "These CSVs are gitignored by policy; pull them with `make datasets` "
            "(requires Snowflake credentials) before running this backtest."
        )

    X = pd.read_csv(x_path)
    y = pd.read_csv(y_path).iloc[:, 0]
    if len(X) != len(y):
        raise SystemExit(
            f"snapshot misaligned: X_train has {len(X)} rows, y_train has {len(y)}; "
            "expected one aligned pair."
        )
    for col in (BUDGET_COL, YEAR_COL):
        if col not in X.columns:
            raise SystemExit(f"snapshot X_train missing required column '{col}'.")
    return X, y


def drop_budget_artifact_rows(
    X: pd.DataFrame, y_raw: pd.Series
) -> tuple[pd.DataFrame, pd.Series, int]:
    """Drop rows where ``production_budget`` was imputed as 0.4 x the target."""
    keep = (X[BUDGET_COL] / y_raw - IMPUTE_RATIO).abs() >= IMPUTE_TOL
    return (
        X[keep].reset_index(drop=True),
        y_raw[keep].reset_index(drop=True),
        int((~keep).sum()),
    )


def select_production_features(X: pd.DataFrame) -> pd.DataFrame:
    """Subset to the production contract (``SELECTED_FEATURES``).

    The contract is stored in production casing; the snapshot uses lowercase
    column names, so match case-insensitively.
    """
    by_lower = {c.lower(): c for c in X.columns}
    missing = [f for f in SELECTED_FEATURES if f.lower() not in by_lower]
    if missing:
        raise SystemExit(
            f"snapshot X_train is missing production features {missing}; "
            "cannot score the active production contract."
        )
    return X[[by_lower[f.lower()] for f in SELECTED_FEATURES]]


def main() -> None:
    X, y_raw = load_snapshot()
    n_raw = len(y_raw)
    X, y_raw, n_dropped = drop_budget_artifact_rows(X, y_raw)

    dates = X[YEAR_COL].copy()
    X_model = select_production_features(X)

    # The model trains on log1p(worldwide_gross); the CV inverts with expm1 to
    # report dollar-space R² and median APE per fold alongside log-space R².
    y_log = np.log1p(y_raw)

    cv = TimeSeriesCrossValidator(start_eval_year=2015)
    cv_results = cv.cross_validate(BoxOfficeXGBoostModel, X_model, y_log, dates)

    # The log-budget baseline needs raw dollars + budget + year, scored on the
    # same rows so the per-year comparison is apples-to-apples.
    raw_df = pd.DataFrame(
        {
            YEAR_COL: X[YEAR_COL].to_numpy(),
            BUDGET_COL: X[BUDGET_COL].to_numpy(),
            TARGET_COL: y_raw.to_numpy(),
        }
    )
    table = build_report(
        raw_df=raw_df,
        cv_results=cv_results,
        target_col=TARGET_COL,
        budget_col=BUDGET_COL,
        year_col=YEAR_COL,
    )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    # Trailing newline keeps both files stable under the end-of-file-fixer hook,
    # so a re-run reproduces the committed bytes, not just the committed numbers.
    OUTPUT.with_suffix(".json").write_text(
        table.to_json(orient="records", indent=2) + "\n"
    )
    markdown = render_metrics_table_markdown(table) + "\n"
    OUTPUT.with_suffix(".md").write_text(markdown)

    print(
        f"snapshot: {n_raw} rows; dropped {n_dropped} imputed-budget rows -> "
        f"n={len(y_raw)} movies"
    )
    print(
        f"scored the {X_model.shape[1]}-feature production model: {list(X_model.columns)}"
    )
    print(f"{len(table)} per-year folds\n")
    print(markdown)
    print(
        f"wrote {OUTPUT.with_suffix('.json')} and {OUTPUT.with_suffix('.md')}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
