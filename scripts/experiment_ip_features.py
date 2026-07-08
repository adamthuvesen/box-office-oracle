"""IP-feature experiment: does franchise/IP awareness improve the model?

SUPERSEDED (2026-07-08): contract v9 adopted the E-variant features
(IP_TIER, PRIOR_FRANCHISE_GROSS_LOG, IS_FRANCHISE_FOLLOWUP) into
``SELECTED_FEATURES``. Production uses collection-keyed franchise history
only (box_office/franchise_history.py); this script's umbrella-ip_name
grouping in ``build_franchise_key`` is retired and kept for the historical
record of the experiment.

Experiment only — never touched the then-current v8 feature contract
(v9 is the live contract now), saved artifacts,
or results/local_retrain/. Iterate mode only: eval years 2015-2023 (2024-2025
are a spent confirmation set per results/local_retrain/report.md).

Variants, all through the identical leakage-fixed CV path
(``TimeSeriesCrossValidator`` + per-fold ``FeaturePreprocessorHigh``):

- A ``baseline``: the current 10 v8 features, rerun fresh.
- C ``time_safe_ip``: baseline + per-movie features computed only from
  franchise history strictly before the movie's release date
  (PRIOR_FRANCHISE_GROSS_LOG, PRIOR_FRANCHISE_FILM_COUNT,
  IS_FRANCHISE_FOLLOWUP). First films of a franchise get 0/0/0.
- E ``time_safe_tier``: baseline + the restructured time-safe ``ip_tier``
  (ordinal; as-of-date brand rules + prior-franchise gross + source-work
  rules, no total-collection gross) + IS_FRANCHISE_FOLLOWUP +
  PRIOR_FRANCHISE_GROSS_LOG.

Earlier variants B (naive total-collection-gross tier, leaky) and D
(BRAND_NONFILM_TIER) were retired when the tier system was restructured;
their results remain in the first section of report.md.

Extra columns ride along in the raw CV frame and are appended AFTER the
fold-fitted preprocessor's 10 engineered features (same pattern as the old
``missing_budget_flag_11th_col`` variant).

Run:  uv run python scripts/experiment_ip_features.py
Appends a dated section to results/ip_experiment/report.md and writes
results/ip_experiment/results_time_safe_tier.json.
"""

from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from box_office.franchise_history import prior_franchise_stats
from box_office.ml.cv import TimeSeriesCrossValidator
from box_office.ml.feature_preprocessor import FeaturePreprocessorHigh
from box_office.ml.model import BoxOfficeXGBoostModel

FRAME_PATH = Path("data/generated/training/train_frame_1980_2026.parquet")
IP_PATH = Path("data/generated/ip/ip_classification_1980_2026.parquet")
RAW_JSONL_PATH = Path(
    "data/generated/tmdb/rich_backfill_1980_2026/tmdb_rich_raw_5m_1980_2026.jsonl"
)
OUTPUT_DIR = Path("results/ip_experiment")

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
)

START_EVAL_YEAR = 2015
END_EVAL_YEAR = 2023  # iterate mode only; 2024-2025 are a spent confirmation set

TIME_SAFE_COLUMNS = (
    "PRIOR_FRANCHISE_GROSS_LOG",
    "PRIOR_FRANCHISE_FILM_COUNT",
    "IS_FRANCHISE_FOLLOWUP",
)


def compute_time_safe_franchise_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Time-safe franchise features from the frame's own rows.

    ``frame`` needs columns: ``franchise_key`` (nullable str; null = no
    franchise), ``release_date`` (datetime64), ``worldwide_gross`` (float).

    For each movie, only films in the same franchise with release_date
    STRICTLY earlier count — same-day releases do not see each other, and a
    franchise's first film gets 0/0/0. Returns a DataFrame aligned to
    ``frame.index`` with PRIOR_FRANCHISE_GROSS_LOG, PRIOR_FRANCHISE_FILM_COUNT,
    IS_FRANCHISE_FOLLOWUP.
    """
    prior = prior_franchise_stats(frame)
    return pd.DataFrame(
        {
            "PRIOR_FRANCHISE_GROSS_LOG": np.log1p(prior["prior_gross"]),
            "PRIOR_FRANCHISE_FILM_COUNT": prior["prior_count"].astype(float),
            "IS_FRANCHISE_FOLLOWUP": (prior["prior_count"] > 0).astype(float),
        },
        index=frame.index,
    )


def load_collection_map(jsonl_path: Path) -> dict[int, str]:
    """tmdb_id -> collection key ("collection:<id>") from the raw TMDB JSONL."""
    mapping: dict[int, str] = {}
    with jsonl_path.open() as fh:
        for line in fh:
            row = json.loads(line)
            coll = row.get("payload", {}).get("belongs_to_collection")
            if coll and coll.get("id") is not None:
                mapping[int(row["tmdb_id"])] = f"collection:{coll['id']}"
    return mapping


def build_franchise_key(frame: pd.DataFrame, ip: pd.DataFrame) -> pd.Series:
    """Franchise grouping key per frame row: umbrella ip_name, else collection.

    ip_name groups umbrella brands (Star Wars, DC, ...) across collections;
    movies without an ip_name fall back to their TMDB collection.
    """
    collection_map = load_collection_map(RAW_JSONL_PATH)
    ip_names = frame["TMDB_ID"].map(ip.set_index("tmdb_id")["ip_name"])
    collections = frame["TMDB_ID"].map(collection_map)
    return ip_names.where(ip_names.notna(), collections)


class AugmentedPreprocessor:
    """FeaturePreprocessorHigh + experimental columns appended after it.

    The inner v8 pipeline sees only the contract input columns; the extras
    pass through untouched, appended positionally after the 10 engineered
    features. The v8 contract itself is never modified.
    """

    def __init__(self, extra_columns: tuple[str, ...]):
        self.extra_columns = extra_columns
        self.inner = FeaturePreprocessorHigh()

    def _append(self, X: pd.DataFrame, base: pd.DataFrame) -> pd.DataFrame:
        if len(base) != len(X):
            raise ValueError(f"Preprocessor changed row count: {len(X)} -> {len(base)}")
        base = base.reset_index(drop=True)
        for col in self.extra_columns:
            base[col] = X[col].to_numpy()
        return base

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        base = self.inner.fit_transform(X[list(PREPROCESSOR_INPUT_COLUMNS)])
        return self._append(X, base)

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        base = self.inner.transform(X[list(PREPROCESSOR_INPUT_COLUMNS)])
        return self._append(X, base)


def run_variant(frame: pd.DataFrame, extra_columns: tuple[str, ...]) -> dict:
    """One leakage-free CV run; extras appended after the fold preprocessor."""
    X_input = frame[list(PREPROCESSOR_INPUT_COLUMNS) + list(extra_columns)].reset_index(
        drop=True
    )
    y_raw = frame["WORLDWIDE_GROSS"].astype(float).reset_index(drop=True)
    y_log = pd.Series(np.log1p(y_raw), name="GROSS_LOG")
    dates = frame["RELEASE_YEAR"].astype(int).reset_index(drop=True)

    cv = TimeSeriesCrossValidator(
        cv_folds=END_EVAL_YEAR - START_EVAL_YEAR + 1,
        start_eval_year=START_EVAL_YEAR,
        end_year=END_EVAL_YEAR,
    )
    cv_results = cv.cross_validate(
        BoxOfficeXGBoostModel,
        X_input,
        y_log,
        dates,
        preprocessor_factory=lambda: AugmentedPreprocessor(extra_columns),
    )

    failed = [f for f in cv_results["fold_results"] if f["error"] is not None]
    if failed:
        raise RuntimeError(f"{len(failed)} CV folds failed: {failed}")

    oof = pd.DataFrame(cv_results["oof_records"])
    pred_dollars = np.expm1(oof["pred"].to_numpy())
    true_dollars = y_raw.iloc[oof["idx"]].to_numpy()
    pooled_median_ape = float(
        np.median(np.abs(pred_dollars - true_dollars) / np.maximum(true_dollars, 1.0))
    )

    per_year = [
        {
            "year": f["eval_year"],
            "n": f["val_samples"],
            "mae_log": f["mae_score"],
            "r2_log": f["model_r2_log"],
            "spearman": f["model_spearman"],
            "median_ape": f["model_median_ape"],
        }
        for f in cv_results["fold_results"]
    ]

    return {
        "n": int(len(frame)),
        "mean_cv_mae": float(cv_results["mean_cv_mae"]),
        "std_cv_mae": float(cv_results["std_cv_mae"]),
        "mean_r2_log": float(
            np.mean([f["model_r2_log"] for f in cv_results["fold_results"]])
        ),
        "pooled_median_ape": pooled_median_ape,
        "per_year": per_year,
    }


def build_frame_with_ip_features() -> tuple[pd.DataFrame, dict]:
    frame = pd.read_parquet(FRAME_PATH)
    ip = pd.read_parquet(IP_PATH)
    ip_by_id = ip.set_index("tmdb_id")

    # Variant E: the restructured time-safe ip_tier, joined as-is
    # (as-of-date brand rules + prior-franchise gross + source-work rules;
    # no total-collection gross). Movies missing from the IP table -> 5.
    frame["IP_TIER_TIME_SAFE"] = (
        frame["TMDB_ID"].map(ip_by_id["ip_tier"]).fillna(5).astype(float)
    )

    # Variant C (time-safe, from the frame's own rows)
    work = pd.DataFrame(
        {
            "franchise_key": build_franchise_key(frame, ip),
            "release_date": pd.to_datetime(frame["RELEASE_DATE"], errors="coerce"),
            "worldwide_gross": frame["WORLDWIDE_GROSS"].astype(float),
        },
        index=frame.index,
    )
    time_safe = compute_time_safe_franchise_features(work)
    for col in TIME_SAFE_COLUMNS:
        frame[col] = time_safe[col]

    diagnostics = {
        "rows": int(len(frame)),
        "rows_with_franchise_key": int(work["franchise_key"].notna().sum()),
        "rows_franchise_followup": int(frame["IS_FRANCHISE_FOLLOWUP"].sum()),
        "ip_tier_counts": {
            int(k): int(v)
            for k, v in frame["IP_TIER_TIME_SAFE"].value_counts().sort_index().items()
        },
    }
    return frame, diagnostics


VARIANTS: dict[str, tuple[str, ...]] = {
    "baseline": (),
    "time_safe_ip": TIME_SAFE_COLUMNS,
    "time_safe_tier": (
        "IP_TIER_TIME_SAFE",
        "IS_FRANCHISE_FOLLOWUP",
        "PRIOR_FRANCHISE_GROSS_LOG",
    ),
}


def fmt_pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def render_report(results: dict[str, dict], diagnostics: dict, run_date: str) -> str:
    """Dated section appended to report.md for the restructured-tier run."""
    lines = [
        "",
        "---",
        "",
        f"# Restructured time-safe ip_tier run ({run_date}, eval years 2015-2023)",
        "",
        "Re-run after restructuring the tier system: `ip_tier` is now pre-sold",
        "magnitude at release (as-of-date `tier_by_period` brand rules,",
        "prior-franchise gross strictly before release, `source_works` rules).",
        "The total-collection-gross thresholds are abolished, so the old leaky",
        "variant B no longer exists. Same leakage-fixed CV path as above;",
        "2024-2025 were never evaluated.",
        "",
        "Variants: A `baseline` (10 v8 features), C `time_safe_ip` (baseline +",
        "PRIOR_FRANCHISE_GROSS_LOG + PRIOR_FRANCHISE_FILM_COUNT +",
        "IS_FRANCHISE_FOLLOWUP), E `time_safe_tier` (baseline + new ordinal",
        "IP_TIER_TIME_SAFE + IS_FRANCHISE_FOLLOWUP + PRIOR_FRANCHISE_GROSS_LOG).",
        "",
        "## Comparison",
        "",
        "| Variant | n | Mean CV MAE (log) | Mean per-year R² (log) | Pooled median APE |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, res in results.items():
        lines.append(
            f"| {name} | {res['n']} | "
            f"{res['mean_cv_mae']:.4f} ± {res['std_cv_mae']:.4f} | "
            f"{res['mean_r2_log']:.4f} | {fmt_pct(res['pooled_median_ape'])} |"
        )

    for name, res in results.items():
        lines += [
            "",
            f"## Per-year: {name}",
            "",
            "| Year | n | MAE (log) | R² (log) | Spearman | Median APE |",
            "|---:|---:|---:|---:|---:|---:|",
        ]
        for row in res["per_year"]:
            lines.append(
                f"| {row['year']} | {row['n']} | {row['mae_log']:.4f} | "
                f"{row['r2_log']:.4f} | {row['spearman']:.4f} | "
                f"{fmt_pct(row['median_ape'])} |"
            )

    a = results["baseline"]
    lines += [
        "",
        "## Diagnostics",
        "",
        f"- Franchise coverage: {diagnostics['rows_with_franchise_key']} of "
        f"{diagnostics['rows']} rows have a franchise key; "
        f"{diagnostics['rows_franchise_followup']} are follow-ups.",
        f"- New ip_tier counts in the training frame: {diagnostics['ip_tier_counts']}.",
        "",
        "## Recommendation",
        "",
        _recommendation("C (time_safe_ip)", a, results["time_safe_ip"]),
        _recommendation("E (time_safe_tier)", a, results["time_safe_tier"]),
    ]
    return "\n".join(lines) + "\n"


def _recommendation(label: str, baseline: dict, variant: dict) -> str:
    """Accept only if the variant beats baseline on both mean log-R² and MAE
    by more than one std of the baseline's fold MAE spread."""
    r2_gain = variant["mean_r2_log"] - baseline["mean_r2_log"]
    mae_gain = baseline["mean_cv_mae"] - variant["mean_cv_mae"]
    beats = r2_gain > 0 and mae_gain > baseline["std_cv_mae"]
    verdict = "ACCEPT" if beats else "REJECT"
    return (
        f"- **{verdict} {label}**: R² gain {r2_gain:+.4f}, MAE gain "
        f"{mae_gain:+.4f} vs baseline std {baseline['std_cv_mae']:.4f} "
        f"(accept requires beating baseline on R² AND MAE by more than one std)."
    )


def main() -> int:
    frame, diagnostics = build_frame_with_ip_features()
    print(f"Diagnostics: {diagnostics}", flush=True)

    results: dict[str, dict] = {}
    for name, extra_columns in VARIANTS.items():
        print(f"Running variant {name} (extras={list(extra_columns)})", flush=True)
        results[name] = run_variant(frame, extra_columns)
        print(
            f"  mean MAE {results[name]['mean_cv_mae']:.4f} "
            f"± {results[name]['std_cv_mae']:.4f}, "
            f"mean R2(log) {results[name]['mean_r2_log']:.4f}, "
            f"pooled median APE {fmt_pct(results[name]['pooled_median_ape'])}",
            flush=True,
        )

    run_date = date.today().isoformat()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "results_time_safe_tier.json").write_text(
        json.dumps(
            {
                "run_date": run_date,
                "eval_years": f"{START_EVAL_YEAR}-{END_EVAL_YEAR}",
                "diagnostics": diagnostics,
                "variants": results,
            },
            indent=2,
        )
        + "\n"
    )
    report_path = OUTPUT_DIR / "report.md"
    existing = report_path.read_text() if report_path.exists() else ""
    report_path.write_text(existing + render_report(results, diagnostics, run_date))
    print(
        f"Appended to {report_path}; wrote {OUTPUT_DIR}/results_time_safe_tier.json",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
