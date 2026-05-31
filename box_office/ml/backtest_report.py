"""Combine SageMaker per-year model metrics with a log-budget baseline.

Inside the SageMaker training container, the model emits per-fold dollar R²
and median APE — but raw ``production_budget`` lives outside that container,
so the baseline column is produced here against a raw-data snapshot. The
output is the per-year metrics table the README quotes.

Usage:
    python -m box_office.ml.backtest_report \\
        --raw-data /path/to/raw_movies.parquet \\
        --cv-results /path/to/cv_results.json \\
        --target-col worldwide_gross \\
        --budget-col production_budget \\
        --year-col RELEASE_YEAR \\
        --output /path/to/per_year_metrics
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from box_office.ml.backtest import (
    assemble_per_year_metrics_table,
    compute_baseline_per_year,
    render_metrics_table_markdown,
)


def build_report(
    *,
    raw_df: pd.DataFrame,
    cv_results: dict,
    target_col: str,
    budget_col: str,
    year_col: str,
) -> pd.DataFrame:
    fold_results = cv_results.get("fold_results", [])
    eval_years = [int(f["eval_year"]) for f in fold_results if f.get("error") is None]
    baseline_results = compute_baseline_per_year(
        raw_df,
        target_col=target_col,
        budget_col=budget_col,
        year_col=year_col,
        eval_years=eval_years,
    )
    return assemble_per_year_metrics_table(
        model_fold_results=fold_results,
        baseline_results=baseline_results,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-data", required=True, type=Path)
    parser.add_argument("--cv-results", required=True, type=Path)
    parser.add_argument("--target-col", default="worldwide_gross")
    parser.add_argument("--budget-col", default="production_budget")
    parser.add_argument("--year-col", default="RELEASE_YEAR")
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Output path *without* extension. Writes <output>.json + <output>.md",
    )
    args = parser.parse_args()

    raw_df = (
        pd.read_parquet(args.raw_data)
        if args.raw_data.suffix == ".parquet"
        else pd.read_csv(args.raw_data)
    )
    cv_results = json.loads(args.cv_results.read_text())

    table = build_report(
        raw_df=raw_df,
        cv_results=cv_results,
        target_col=args.target_col,
        budget_col=args.budget_col,
        year_col=args.year_col,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    table.to_json(args.output.with_suffix(".json"), orient="records", indent=2)
    args.output.with_suffix(".md").write_text(
        render_metrics_table_markdown(table) + "\n"
    )


if __name__ == "__main__":
    main()
