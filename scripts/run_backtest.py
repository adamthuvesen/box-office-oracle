"""Offline per-year backtest driver.

Reproduces the per-year expanding-window backtest the README quotes — model
dollar-space R² against a log-budget baseline, per year, with median APE —
without SageMaker, Lambda, or any AWS round-trip. It runs entirely against the
local gitignored training snapshot under ``analysis/datasets_high/``.

Why this exists separately from ``box_office.ml.backtest_report``'s CLI: that
CLI expects a raw-movie parquet plus a SageMaker-produced ``cv_results.json``,
and neither exists offline. This driver loads the training snapshot, applies the
v2 leak controls the snapshot predates, runs the repo's ``TimeSeriesCrossValidator``
with the production XGBoost wrapper, then feeds the folds into the same report
builder the CLI would call.

Leak controls applied (the snapshot is the OLD, pre-v2 leaky feature set; see
``analysis/feature_selection_study.py`` for the codified rationale):
  1. drop the target-synthesized feature family — ``social_media_buzz`` and its
     derivatives were built from ``worldwide_gross``, so they leak the target;
  2. drop the rows carrying the ``production_budget = 0.4 * worldwide_gross``
     imputation signature, which encodes the target into the budget feature.

Committing numbers without these controls would re-introduce the exact leak this
repo fixed in v2 — the disowned "0.70–0.85 R²".

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
from box_office.ml.model import BoxOfficeXGBoostModel

DATA_DIR = Path("analysis/datasets_high")
OUTPUT = Path("results/per_year_table")

TARGET_COL = "worldwide_gross"
BUDGET_COL = "production_budget"
YEAR_COL = "release_year"

# Target-synthesized features: built from worldwide_gross pre-v2, so scientifically
# invalid as predictors. Mirrors ``feature_selection_study.LEAKED``.
LEAKED = (
    "social_media_buzz",
    "viral_potential",
    "social_buzz_to_budget",
    "buzz_to_votes_ratio",
    "marketing_efficiency",
)

# production_budget ≈ 0.4 * worldwide_gross marks imputed budgets — the target
# leaking into the feature. Same tolerance the feature-selection study uses.
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


def apply_leak_controls(
    X: pd.DataFrame, y_raw: pd.Series
) -> tuple[pd.DataFrame, pd.Series, int, list[str]]:
    """Return the de-leaked (X, y) plus a record of what was removed."""
    # Control 2: drop the budget-imputation-signature rows.
    imputed = (X[BUDGET_COL] / y_raw - IMPUTE_RATIO).abs() < IMPUTE_TOL
    X = X[~imputed].reset_index(drop=True)
    y_raw = y_raw[~imputed].reset_index(drop=True)

    # Control 1: drop the target-synthesized feature family.
    leaked_present = [c for c in X.columns if c in LEAKED]
    X = X.drop(columns=leaked_present)
    return X, y_raw, int(imputed.sum()), leaked_present


def main() -> None:
    X, y_raw = load_snapshot()
    n_raw = len(y_raw)
    X, y_raw, n_imputed, leaked_dropped = apply_leak_controls(X, y_raw)

    # The model trains on log1p(worldwide_gross); the CV inverts with expm1 to
    # report dollar-space R² and median APE per fold.
    y_log = np.log1p(y_raw)
    dates = X[YEAR_COL]

    cv = TimeSeriesCrossValidator(start_eval_year=2015)
    cv_results = cv.cross_validate(BoxOfficeXGBoostModel, X, y_log, dates)

    # The log-budget baseline needs raw dollars + budget + year, fit on the same
    # de-leaked rows so the per-year comparison is apples-to-apples.
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
        f"snapshot: {n_raw} rows; dropped {n_imputed} imputed-signature rows -> "
        f"n={len(y_raw)} movies"
    )
    print(f"dropped leaked features: {leaked_dropped}")
    print(f"trained on {X.shape[1]} features across {len(table)} per-year folds\n")
    print(markdown)
    print(
        f"wrote {OUTPUT.with_suffix('.json')} and {OUTPUT.with_suffix('.md')}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
