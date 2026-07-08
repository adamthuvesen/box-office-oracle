"""Phase-3 evaluation of the local 1980-2026 TMDB retrain.

Leakage-free CV (expanding window, production ``TimeSeriesCrossValidator`` +
``BoxOfficeXGBoostModel``): every fold fits a fresh ``FeaturePreprocessorHigh``
on train-years rows only, so frequency features cannot see the eval year.

Confirmation discipline: iterate against eval years <= 2023; touch 2024-2025
only for a final confirmation run.

- Default (iteration mode): eval years 2015-2023. Writes
  ``results/local_retrain/iteration_report.md`` + ``iteration_results.json``.
- ``--confirm``: adds 2024-2025 and writes the frozen-confirmation
  ``report.md`` / ``results.json`` / ``per_year_table.{md,json}``. For the
  v9 contract 2024-2025 are SPENT (v8's frozen confirmation stands; 2026
  actuals will confirm v9), so ``--confirm`` additionally requires
  ``--i-know-this-burns-the-holdout``.

Headline metric = the recent window (2023 onward): per-year table plus pooled
median APE and mean log-R² over those years. The full per-year table stays as
a diagnostic. The pre-leakage-fix variant comparison is kept in the report as
history (those numbers were inflated by frequency features fit on the full
frame before CV).

Run:  uv run python scripts/evaluate_local_retrain.py [--confirm]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

from box_office.ml.backtest import render_metrics_table_markdown
from box_office.ml.backtest_report import build_report
from box_office.ml.cv import TimeSeriesCrossValidator
from box_office.ml.feature_preprocessor import FeaturePreprocessorHigh
from box_office.ml.feature_schema import CURRENT_FEATURE_SCHEMA_VERSION
from box_office.ml.model import BoxOfficeXGBoostModel
from box_office.ml.regression_metrics import spearman_rank_corr

FRAME_PATH = Path("data/generated/training/train_frame_1980_2026.parquet")
DROPPED_PATH = Path("data/generated/training/dropped_rows_1980_2026.csv")
FLAGGED_PATH = Path("data/generated/training/flagged_kept_rows_1980_2026.csv")
OLD_TABLE_PATH = Path("results/per_year_table.json")
OLD_SNAPSHOT_X = Path("analysis/datasets_high/X_train.csv")
OUTPUT_DIR = Path("results/local_retrain")

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

START_EVAL_YEAR = 2015
ITERATION_END_YEAR = 2023  # iterate against <= 2023 only
CONFIRM_END_YEAR = 2025  # touched once, for the frozen confirmation
RECENT_WINDOW_START = 2023

DISCIPLINE_RULE = (
    "**Confirmation discipline:** iterate against eval years <= 2023 "
    "(default mode); 2024-2025 are held out and touched only for a final "
    "`--confirm` run. Once confirmed, the numbers are frozen — further "
    "iteration against 2024-2025 would turn the confirmation set into a "
    "validation set."
)

# History: numbers produced BEFORE the frequency-feature leakage fix, when
# FeaturePreprocessorHigh was fit once on the full frame and the engineered
# matrix was split afterwards (director/company/actor frequencies and the
# MPAA encoding saw future years inside each fold).
PRE_FIX_R2_LOG_BY_YEAR: dict[int, float] = {
    2015: 0.621,
    2016: 0.607,
    2017: 0.630,
    2018: 0.646,
    2019: 0.601,
    2020: 0.364,
    2021: 0.437,
    2022: 0.612,
    2023: 0.595,
    2024: 0.586,
    2025: 0.669,
}

PRE_FIX_VARIANT_TABLE = """\
| Variant | n | missing budgets | Mean CV MAE (log) | Per-year R² (log) range | Pooled median APE |
|---|---:|---:|---:|---:|---:|
| headline_nan_passthrough | 6080 | 323 | 0.7290 ± 0.0496 | 0.364 – 0.669 | 55.3% |
| drop_missing_budget | 5757 | 0 | 0.7431 ± 0.0489 | 0.312 – 0.635 | 54.8% |
| missing_budget_flag_11th_col | 6080 | 323 | 0.7304 ± 0.0548 | 0.367 – 0.670 | 54.4% |
| gross_50m_plus | 2640 | 18 | 0.4712 ± 0.0561 | 0.102 – 0.667 | 39.2% |
| min_year_1990 | 5185 | 254 | 0.7355 ± 0.0529 | 0.327 – 0.650 | 55.5% |"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the local retrain with leakage-free CV."
    )
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


def run_headline_cv(frame: pd.DataFrame, end_eval_year: int) -> dict:
    """One leakage-free CV run on the full frame (NaN budgets pass through)."""
    X_input = frame[list(PREPROCESSOR_INPUT_COLUMNS)].reset_index(drop=True)
    y_raw = frame["WORLDWIDE_GROSS"].astype(float).reset_index(drop=True)
    y_log = pd.Series(np.log1p(y_raw), name="GROSS_LOG")
    dates = frame["RELEASE_YEAR"].astype(int).reset_index(drop=True)

    cv = TimeSeriesCrossValidator(
        cv_folds=end_eval_year - START_EVAL_YEAR + 1,
        start_eval_year=START_EVAL_YEAR,
        end_year=end_eval_year,
    )
    cv_results = cv.cross_validate(
        BoxOfficeXGBoostModel,
        X_input,
        y_log,
        dates,
        preprocessor_factory=FeaturePreprocessorHigh,
    )

    failed = [f for f in cv_results["fold_results"] if f["error"] is not None]
    if failed:
        raise RuntimeError(f"{len(failed)} CV folds failed: {failed}")

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
        f["model_r2_log"]
        for f in cv_results["fold_results"]
        if start_year <= f["eval_year"] <= end_year
    ]

    oof = pd.DataFrame(cv_results["oof_records"])
    years = dates.iloc[oof["idx"]].to_numpy()
    in_window = (years >= start_year) & (years <= end_year)
    pred_dollars = np.expm1(oof["pred"].to_numpy()[in_window])
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


def build_per_year_table(frame: pd.DataFrame, cv_results: dict) -> pd.DataFrame:
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


def build_delta_table(new_table: pd.DataFrame, old_table: pd.DataFrame) -> pd.DataFrame:
    merged = old_table.merge(
        new_table, on="year", suffixes=("_old", "_new"), how="inner"
    )
    return pd.DataFrame(
        {
            "year": merged["year"],
            "n_old": merged["n_movies_old"],
            "n_new": merged["n_movies_new"],
            "model_r2_log_old": merged["model_r2_log_old"],
            "model_r2_log_new": merged["model_r2_log_new"],
            "delta_r2_log": merged["model_r2_log_new"] - merged["model_r2_log_old"],
            "model_spearman_old": merged["model_spearman_old"],
            "model_spearman_new": merged["model_spearman_new"],
            "delta_spearman": (
                merged["model_spearman_new"] - merged["model_spearman_old"]
            ),
            "median_ape_old": merged["model_median_ape_old"],
            "median_ape_new": merged["model_median_ape_new"],
            "delta_median_ape": (
                merged["model_median_ape_new"] - merged["model_median_ape_old"]
            ),
        }
    )


def overlap_mask_from_old_snapshot(frame: pd.DataFrame) -> pd.Series | None:
    """Approximate membership in the old snapshot via (year, runtime, budget).

    The old snapshot carries only engineered features — no imdb_id, no title —
    so an exact identity join is impossible. Returns None when the gitignored
    snapshot is absent.
    """
    if not OLD_SNAPSHOT_X.exists():
        return None
    old = pd.read_csv(
        OLD_SNAPSHOT_X, usecols=["release_year", "runtime", "production_budget"]
    )
    old_keys = set(
        zip(
            old["release_year"].astype(int),
            old["runtime"].astype(float),
            old["production_budget"].astype(float),
            strict=True,
        )
    )
    keys = zip(
        frame["RELEASE_YEAR"].astype(int),
        frame["RUNTIME"].astype(float),
        frame["PRODUCTION_BUDGET"].astype(float),
        strict=True,
    )
    return pd.Series([k in old_keys for k in keys], index=frame.index)


def per_year_metrics_on_subset(
    headline: dict, subset_positions: np.ndarray, max_year: int
) -> pd.DataFrame:
    """Per-year OOF metrics restricted to a row subset (positional indices)."""
    oof = pd.DataFrame(headline["cv_results"]["oof_records"]).set_index("idx")
    y_raw = headline["y_raw"]
    dates = headline["dates"]
    rows = []
    for year in sorted(dates.unique()):
        if not (START_EVAL_YEAR <= year <= max_year):
            continue
        year_positions = np.flatnonzero((dates == year).to_numpy())
        positions = np.intersect1d(year_positions, subset_positions)
        positions = np.array([p for p in positions if p in oof.index])
        if len(positions) < 2:
            continue
        pred_log = oof.loc[positions, "pred"].to_numpy()
        true_log = np.log1p(y_raw.iloc[positions].to_numpy())
        true_dollars = y_raw.iloc[positions].to_numpy()
        pred_dollars = np.expm1(pred_log)
        rows.append(
            {
                "year": int(year),
                "n_movies": int(len(positions)),
                "model_r2_log": float(r2_score(true_log, pred_log)),
                "model_spearman": spearman_rank_corr(true_log, pred_log),
                "model_median_ape": float(
                    np.median(
                        np.abs(pred_dollars - true_dollars)
                        / np.maximum(true_dollars, 1.0)
                    )
                ),
            }
        )
    return pd.DataFrame(rows)


def render_delta_markdown(delta: pd.DataFrame) -> str:
    header = (
        "| Year | n old | n new | R² log old | R² log new | ΔR² log "
        "| ρ old | ρ new | Δρ | APE old | APE new | ΔAPE |\n"
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
    )
    rows = [
        f"| {int(r['year'])} | {int(r['n_old'])} | {int(r['n_new'])} "
        f"| {r['model_r2_log_old']:.3f} | {r['model_r2_log_new']:.3f} "
        f"| {r['delta_r2_log']:+.3f} "
        f"| {r['model_spearman_old']:.3f} | {r['model_spearman_new']:.3f} "
        f"| {r['delta_spearman']:+.3f} "
        f"| {r['median_ape_old']:.1%} | {r['median_ape_new']:.1%} "
        f"| {r['delta_median_ape']:+.1%} |"
        for _, r in delta.iterrows()
    ]
    return "\n".join([header, *rows])


def render_overlap_markdown(overlap: pd.DataFrame, old_table: pd.DataFrame) -> str:
    merged = old_table[
        ["year", "n_movies", "model_r2_log", "model_spearman", "model_median_ape"]
    ].merge(overlap, on="year", suffixes=("_old", "_ovl"), how="inner")
    header = (
        "| Year | n old | n overlap | R² log old | R² log overlap "
        "| ρ old | ρ overlap | APE old | APE overlap |\n"
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|"
    )
    rows = [
        f"| {int(r['year'])} | {int(r['n_movies_old'])} | {int(r['n_movies_ovl'])} "
        f"| {r['model_r2_log_old']:.3f} | {r['model_r2_log_ovl']:.3f} "
        f"| {r['model_spearman_old']:.3f} | {r['model_spearman_ovl']:.3f} "
        f"| {r['model_median_ape_old']:.1%} | {r['model_median_ape_ovl']:.1%} |"
        for _, r in merged.iterrows()
    ]
    return "\n".join([header, *rows])


def pre_fix_history_section(new_table: pd.DataFrame) -> str:
    """Pre-leakage-fix variant table plus a quantified before/after note."""
    new_by_year = dict(
        zip(
            new_table["year"].astype(int),
            new_table["model_r2_log"].astype(float),
            strict=True,
        )
    )
    common_years = sorted(set(PRE_FIX_R2_LOG_BY_YEAR) & set(new_by_year))
    pre_mean = float(np.mean([PRE_FIX_R2_LOG_BY_YEAR[y] for y in common_years]))
    post_mean = float(np.mean([new_by_year[y] for y in common_years]))
    note = (
        f"Numbers below predate the frequency-feature leakage fix (preprocessor "
        f"fit on the full frame before CV) and are inflated: over "
        f"{common_years[0]}-{common_years[-1]}, mean per-year log-R² dropped "
        f"from {pre_mean:.3f} (pre-fix) to {post_mean:.3f} (leakage-free), "
        f"{post_mean - pre_mean:+.3f}."
    )
    return "\n\n".join([note, PRE_FIX_VARIANT_TABLE])


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
    return "\n".join(lines), {"drop_counts": counts, "n_dropped": int(len(dropped))}


def render_recent_window_markdown(
    recent_table: pd.DataFrame, recent_summary: dict
) -> str:
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
    end_eval_year = CONFIRM_END_YEAR if args.confirm else ITERATION_END_YEAR
    mode = "confirm" if args.confirm else "iteration"

    for path in (FRAME_PATH, DROPPED_PATH, FLAGGED_PATH):
        if not path.exists():
            raise SystemExit(
                f"missing input: {path}. Run scripts/prepare_training_frame.py first."
            )

    frame = pd.read_parquet(FRAME_PATH)

    headline = run_headline_cv(frame, end_eval_year)
    cv_results = headline["cv_results"]

    full_table = build_per_year_table(frame, cv_results)
    recent_table = full_table[full_table["year"] >= RECENT_WINDOW_START]
    recent_summary = pooled_window_metrics(headline, RECENT_WINDOW_START, end_eval_year)
    full_summary = pooled_window_metrics(headline, START_EVAL_YEAR, end_eval_year)

    old_table = pd.read_json(OLD_TABLE_PATH) if OLD_TABLE_PATH.exists() else None
    delta_table = (
        build_delta_table(full_table, old_table) if old_table is not None else None
    )

    overlap_mask = (
        overlap_mask_from_old_snapshot(frame) if old_table is not None else None
    )
    overlap_table = None
    if overlap_mask is None:
        overlap_note = (
            "Old snapshot (`analysis/datasets_high/X_train.csv`) not found; "
            "overlap view skipped."
        )
    else:
        overlap_positions = np.flatnonzero(overlap_mask.to_numpy())
        overlap_table = per_year_metrics_on_subset(
            headline, overlap_positions, max_year=int(old_table["year"].max())
        )
        overlap_note = (
            f"Overlap key: (release_year, runtime, production_budget) — the old "
            f"snapshot has no imdb_id or title, so this match is approximate. "
            f"{int(overlap_mask.sum())} of {len(frame)} new rows matched "
            f"{len(pd.read_csv(OLD_SNAPSHOT_X))} old rows."
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    full_md = render_metrics_table_markdown(full_table) + "\n"
    if args.confirm:
        report_path = OUTPUT_DIR / "report.md"
        results_path = OUTPUT_DIR / "results.json"
        full_table.to_json(
            OUTPUT_DIR / "per_year_table.json", orient="records", indent=2
        )
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

    old_model_section: list[str] = []
    if delta_table is not None:
        old_model_section = [
            "## Delta vs the committed old-model table (results/per_year_table.md)",
            "",
            "Positive ΔR²/Δρ and negative ΔAPE mean the new run is better. The eval "
            "population changed (old ~2.7k-row snapshot vs the new $5M+ frame), so "
            "this table confounds model and population — judge on the overlap view. "
            "The old table also predates the leakage fix.",
            "",
            render_delta_markdown(delta_table),
            "",
            "## Overlap view (eval restricted to movies in the old snapshot)",
            "",
            overlap_note,
            "",
            (
                render_overlap_markdown(overlap_table, old_table)
                if overlap_table is not None and not overlap_table.empty
                else "_No overlap rows with OOF predictions._"
            ),
            "",
        ]

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
        *old_model_section,
        "## Pre-leakage-fix history: variant comparison",
        "",
        pre_fix_history_section(full_table),
        "",
        "## Dropped rows",
        "",
        dropped_md,
        "",
        "## Caveats",
        "",
        "- The overlap join is approximate (composite key, no stable id).",
        "- Per-fold preprocessor refits make CV slower than the pre-fix runs; "
        "the numbers are not comparable to pre-fix reports (see the history "
        "section for the quantified gap).",
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
        "mean_cv_mae_log": float(cv_results["mean_cv_mae"]),
        "std_cv_mae_log": float(cv_results["std_cv_mae"]),
        "mean_cv_rmsle": float(cv_results["mean_cv_rmsle"]),
        "per_year": json.loads(full_table.to_json(orient="records")),
        "delta_vs_committed": (
            json.loads(delta_table.to_json(orient="records"))
            if delta_table is not None
            else None
        ),
        "overlap_per_year": (
            json.loads(overlap_table.to_json(orient="records"))
            if overlap_table is not None
            else None
        ),
        "overlap_note": overlap_note,
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
