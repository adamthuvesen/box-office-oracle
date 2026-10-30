"""Phase-3 evaluation of the local 1980-2026 TMDB retrain.

Leakage-free CV (expanding window, production ``TimeSeriesCrossValidator`` +
``BoxOfficeXGBoostModel``): every fold fits a fresh ``FeaturePreprocessorHigh``
on train-years rows only, so frequency features cannot see the eval year.

Confirmation discipline: iterate against eval years <= 2023; touch 2024-2025
only for a final confirmation run.

- Default (iteration mode): eval years 2015-2023. Writes
  ``docs/internal/experiments/local_retrain/iteration_report.md`` +
  ``iteration_results.json``.
- ``--confirm``: adds 2024-2025 and writes the frozen-confirmation
  ``report.md`` / ``results.json`` / ``per_year_table.{md,json}``. For the
  v9 contract 2024-2025 are SPENT (v8's frozen confirmation stands; 2026
  actuals will confirm v9), so ``--confirm`` additionally requires
  ``--i-know-this-burns-the-holdout``.

Headline metric = the recent window (2023 onward): per-year table plus pooled
median APE and mean log-R² over those years. The full per-year table stays as
a diagnostic.

Run:  uv run python scripts/evaluate_local_retrain.py [--confirm]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from box_office.eval_windows import CONFIRMATION_END_YEAR, ITERATION_END_YEAR, START_EVAL_YEAR
from box_office.ml.backtest import render_metrics_table_markdown
from box_office.ml.backtest_report import build_report
from box_office.ml.cv import TimeSeriesCrossValidator
from box_office.ml.feature_preprocessor import FeaturePreprocessorHigh
from box_office.ml.feature_schema import CURRENT_FEATURE_SCHEMA_VERSION
from box_office.ml.metrics_models import CrossValidationResults
from box_office.ml.model import BoxOfficeXGBoostModel
from box_office.ml.target import from_log_target, to_log_target

FRAME_PATH = Path("data/generated/training/train_frame_1980_2026.parquet")
DROPPED_PATH = Path("data/generated/training/dropped_rows_1980_2026.csv")
FLAGGED_PATH = Path("data/generated/training/flagged_kept_rows_1980_2026.csv")
OUTPUT_DIR = Path("docs/internal/experiments/local_retrain")

PREPROCESSOR_INPUT_COLUMNS: tuple[str, ...] = (
    "RELEASE_YEAR",
    "RELEASE_DATE",
    "PRODUCTION_BUDGET",
    "RUNTIME",
    "MPAA",
    "GENRES",
    "DIRECTOR",
    "PRODUCTION_COMPANY",
    "ACTORS",
    "IP_TIER",
    "PRIOR_FRANCHISE_GROSS_LOG",
    "IS_FRANCHISE_FOLLOWUP",
)

RECENT_WINDOW_START = 2023

DISCIPLINE_RULE = (
    "**Confirmation discipline:** iterate against eval years <= 2023 "
    "(default mode); 2024-2025 are held out and touched only for a final "
    "`--confirm` run. Once confirmed, the numbers are frozen — further "
    "iteration against 2024-2025 would turn the confirmation set into a "
    "validation set."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the local retrain with leakage-free CV.")
    parser.add_argument(
        "--confirm",
        action="store_true",
        help=(
            "Add the held-out 2024-2025 eval years and write the frozen "
            "confirmation report.md/results.json (default: iterate on "
            "2015-2023 only)"
        ),
    )
    parser.add_argument(
        "--i-know-this-burns-the-holdout",
        action="store_true",
        dest="burn_holdout",
        help=(
            "Required together with --confirm: 2024-2025 are a spent "
            "confirmation set for the v9 contract (2026 actuals will "
            "confirm); re-evaluating them turns the confirmation set into "
            "a validation set."
        ),
    )
    args = parser.parse_args()
    if args.confirm and not args.burn_holdout:
        parser.error(
            "--confirm re-evaluates the spent 2024-2025 confirmation set; "
            "pass --i-know-this-burns-the-holdout to do it deliberately."
        )
    return args


def run_headline_cv(frame: pd.DataFrame, end_eval_year: int, *, burn_holdout: bool = False) -> dict:
    """One leakage-free CV run on the full frame (NaN budgets pass through)."""
    X_input = frame[list(PREPROCESSOR_INPUT_COLUMNS)].reset_index(drop=True)
    y_raw = frame["WORLDWIDE_GROSS"].astype(float).reset_index(drop=True)
    y_log = pd.Series(to_log_target(y_raw), name="GROSS_LOG")
    dates = frame["RELEASE_YEAR"].astype(int).reset_index(drop=True)

    cv = TimeSeriesCrossValidator(
        cv_folds=end_eval_year - START_EVAL_YEAR + 1,
        start_eval_year=START_EVAL_YEAR,
        end_year=end_eval_year,
        allow_confirmation_years=burn_holdout,
    )
    cv_results = cv.cross_validate(
        BoxOfficeXGBoostModel,
        X_input,
        y_log,
        dates,
        preprocessor_factory=FeaturePreprocessorHigh,
    )

    return {
        "cv_results": cv_results,
        "y_raw": y_raw,
        "dates": dates,
    }


def pooled_window_metrics(headline: dict, start_year: int, end_year: int) -> dict:
    """Pooled median APE + mean fold log-R² over an eval-year window."""
    cv_results = headline["cv_results"]
    y_raw = headline["y_raw"]
    dates = headline["dates"]

    fold_r2 = [
        f.model_r2_log for f in cv_results.fold_results if start_year <= f.eval_year <= end_year
    ]

    oof = pd.DataFrame([record.model_dump() for record in cv_results.oof_records])
    years = dates.iloc[oof["idx"]].to_numpy()
    in_window = (years >= start_year) & (years <= end_year)
    pred_dollars = from_log_target(oof["pred"].to_numpy()[in_window])
    true_dollars = y_raw.iloc[oof["idx"]].to_numpy()[in_window]
    pooled_median_ape = float(
        np.median(np.abs(pred_dollars - true_dollars) / np.maximum(true_dollars, 1.0))
    )

    return {
        "years": f"{start_year}-{end_year}",
        "n_movies": int(in_window.sum()),
        "mean_r2_log": float(np.mean(fold_r2)),
        "pooled_median_ape": pooled_median_ape,
    }


def build_per_year_table(frame: pd.DataFrame, cv_results: CrossValidationResults) -> pd.DataFrame:
    raw_df = pd.DataFrame(
        {
            "release_year": frame["RELEASE_YEAR"].to_numpy(),
            "production_budget": frame["PRODUCTION_BUDGET"].to_numpy(),
            "worldwide_gross": frame["WORLDWIDE_GROSS"].astype(float).to_numpy(),
        }
    )
    return build_report(
        raw_df=raw_df,
        cv_results=cv_results,
        target_col="worldwide_gross",
        budget_col="production_budget",
        year_col="release_year",
    )


def dropped_rows_summary() -> tuple[str, dict]:
    dropped = pd.read_csv(DROPPED_PATH)
    counts = dropped["DROP_REASON"].str.split(";").explode().value_counts().to_dict()
    lines = [f"Dropped rows: {len(dropped)} (reasons overlap; counts are per rule)"]
    lines += [f"- `{reason}`: {count}" for reason, count in counts.items()]
    flagged = pd.read_csv(FLAGGED_PATH)
    lines.append(
        f"\nFlagged by the spec's 50x gross/budget rule but KEPT after a hand "
        f"check ({len(flagged)} legitimate low-budget sleeper hits):"
    )
    lines += [
        f"- {r['TITLE']} ({int(r['RELEASE_YEAR'])}): budget "
        f"${r['PRODUCTION_BUDGET']:,.0f}, gross ${r['WORLDWIDE_GROSS']:,.0f}"
        for _, r in flagged.iterrows()
    ]
    return "\n".join(lines), {"drop_counts": counts, "n_dropped": len(dropped)}


def render_recent_window_markdown(recent_table: pd.DataFrame, recent_summary: dict) -> str:
    table_md = render_metrics_table_markdown(recent_table)
    summary = (
        f"Pooled over {recent_summary['years']} "
        f"({recent_summary['n_movies']} movies): "
        f"**median APE {recent_summary['pooled_median_ape']:.1%}**, "
        f"**mean log-R² {recent_summary['mean_r2_log']:.3f}**."
    )
    return "\n\n".join([summary, table_md.rstrip()])


def main() -> None:
    args = parse_args()
    end_eval_year = CONFIRMATION_END_YEAR if args.confirm else ITERATION_END_YEAR
    mode = "confirm" if args.confirm else "iteration"

    for path in (FRAME_PATH, DROPPED_PATH, FLAGGED_PATH):
        if not path.exists():
            raise SystemExit(f"missing input: {path}. Run scripts/prepare_training_frame.py first.")

    frame = pd.read_parquet(FRAME_PATH)

    headline = run_headline_cv(frame, end_eval_year, burn_holdout=args.burn_holdout)
    cv_results = headline["cv_results"]

    full_table = build_per_year_table(frame, cv_results)
    recent_table = full_table[full_table["year"] >= RECENT_WINDOW_START]
    recent_summary = pooled_window_metrics(headline, RECENT_WINDOW_START, end_eval_year)
    full_summary = pooled_window_metrics(headline, START_EVAL_YEAR, end_eval_year)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    full_md = render_metrics_table_markdown(full_table) + "\n"
    if args.confirm:
        report_path = OUTPUT_DIR / "report.md"
        results_path = OUTPUT_DIR / "results.json"
        full_table.to_json(OUTPUT_DIR / "per_year_table.json", orient="records", indent=2)
        (OUTPUT_DIR / "per_year_table.md").write_text(full_md)
        mode_label = (
            "**FROZEN CONFIRMATION RUN** (`--confirm`): eval years include the "
            "held-out 2024-2025. These numbers are final for this retrain; do "
            "not iterate against them."
        )
    else:
        report_path = OUTPUT_DIR / "iteration_report.md"
        results_path = OUTPUT_DIR / "iteration_results.json"
        mode_label = (
            "Iteration mode: eval years 2015-2023 only. 2024-2025 stay held "
            "out until a final `--confirm` run."
        )

    dropped_md, dropped_json = dropped_rows_summary()

    report = [
        "# Local retrain evaluation — 1980-2026 TMDB dataset",
        "",
        mode_label,
        "",
        DISCIPLINE_RULE,
        "",
        f"Frame: `{FRAME_PATH}` ({len(frame)} rows, "
        f"{int(frame['PRODUCTION_BUDGET'].isna().sum())} missing budgets kept as "
        f"NaN). CV: expanding window, eval years {START_EVAL_YEAR}-{end_eval_year}, "
        "production `TimeSeriesCrossValidator` + `BoxOfficeXGBoostModel`, "
        f"feature contract frozen at v{CURRENT_FEATURE_SCHEMA_VERSION}. "
        "Leakage-free: each fold fits a fresh "
        "`FeaturePreprocessorHigh` on train-years rows only.",
        "",
        f"## Headline: recent window ({RECENT_WINDOW_START}-{end_eval_year})",
        "",
        render_recent_window_markdown(recent_table, recent_summary),
        "",
        f"## Diagnostic: full per-year table ({START_EVAL_YEAR}-{end_eval_year})",
        "",
        f"Pooled over {full_summary['years']} ({full_summary['n_movies']} "
        f"movies): median APE {full_summary['pooled_median_ape']:.1%}, "
        f"mean log-R² {full_summary['mean_r2_log']:.3f}.",
        "",
        full_md.rstrip(),
        "",
        "## Dropped rows",
        "",
        dropped_md,
    ]
    report_path.write_text("\n".join(report) + "\n")

    results_json = {
        "settings": {
            "mode": mode,
            "start_eval_year": START_EVAL_YEAR,
            "end_year": end_eval_year,
            "recent_window_start": RECENT_WINDOW_START,
            "cv_folds": end_eval_year - START_EVAL_YEAR + 1,
            "feature_schema_version": CURRENT_FEATURE_SCHEMA_VERSION,
            "leakage_free_cv": True,
        },
        "recent_window": recent_summary,
        "full_window": full_summary,
        "mean_cv_mae_log": cv_results.mean_cv_mae,
        "std_cv_mae_log": cv_results.std_cv_mae,
        "mean_cv_rmsle": cv_results.mean_cv_rmsle,
        "per_year": json.loads(full_table.to_json(orient="records")),
        "dropped_rows": dropped_json,
    }
    results_path.write_text(json.dumps(results_json, indent=2) + "\n")

    print(f"mode: {mode} (eval years {START_EVAL_YEAR}-{end_eval_year})")
    print(render_recent_window_markdown(recent_table, recent_summary))
    print()
    print(full_md)
    print(f"\nwrote {report_path} and {results_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
