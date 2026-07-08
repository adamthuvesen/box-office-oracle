"""Per-year backtest reporting: log-budget baseline + assembled metrics table.

The model card's headline number is presented as **gain over baseline**:
"model R² X (baseline R² Y, gain +Z)". A single aggregate R² across nine OOF
folds with tiny early training sets is misleading; per-year exposes regime
shifts (COVID, strikes, streaming) that a senior reviewer will probe. This
module owns the honest reporting.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

from box_office.ml.regression_metrics import spearman_rank_corr


@dataclass(frozen=True)
class BaselineFoldResult:
    year: int
    n_train: int
    n_val: int
    baseline_r2_dollars: float
    baseline_r2_log: float
    baseline_spearman: float


class LogBudgetBaseline:
    """`log1p(worldwide_gross) ~ log1p(production_budget)` linear regression.

    Predictions are returned on the log scale; callers `expm1` to compare on
    dollar scale. Rows with non-positive or NaN budget are dropped at fit
    time and predicted as the training-mean log-target at predict time —
    keeps the baseline well-defined without leaking the target.
    """

    def __init__(self) -> None:
        self._model = LinearRegression()
        self._fallback_log_pred: float = 0.0

    def fit(self, budget: pd.Series, target_log: pd.Series) -> LogBudgetBaseline:
        budget = pd.to_numeric(budget, errors="coerce")
        target_log = pd.to_numeric(target_log, errors="coerce")

        mask = (budget > 0) & target_log.notna()
        if mask.sum() < 2:
            raise ValueError(
                "LogBudgetBaseline.fit needs at least 2 rows with positive budget; "
                f"got {int(mask.sum())}."
            )

        x_log = np.log1p(budget[mask].to_numpy()).reshape(-1, 1)
        y_log = target_log[mask].to_numpy()
        self._model.fit(x_log, y_log)
        self._fallback_log_pred = float(np.mean(y_log))
        return self

    def predict_log(self, budget: pd.Series) -> np.ndarray:
        budget = pd.to_numeric(budget, errors="coerce")
        valid = (budget > 0) & budget.notna()
        out = np.full(len(budget), self._fallback_log_pred, dtype=float)
        if valid.any():
            x_log = np.log1p(budget[valid].to_numpy()).reshape(-1, 1)
            out[valid.to_numpy()] = self._model.predict(x_log)
        return out


def compute_baseline_per_year(
    df: pd.DataFrame,
    *,
    target_col: str,
    budget_col: str,
    year_col: str,
    eval_years: Sequence[int],
) -> list[BaselineFoldResult]:
    """For each year Y in `eval_years`, fit baseline on `< Y`, score on `== Y`.

    Target is on the dollar scale; the baseline fits in log space and we
    score R² on dollar scale (matching how model R² is reported).
    """
    results: list[BaselineFoldResult] = []
    target_log = np.log1p(df[target_col].astype(float))

    for year in sorted(eval_years):
        train_mask = df[year_col] < year
        val_mask = df[year_col] == year

        n_train = int(train_mask.sum())
        n_val = int(val_mask.sum())
        if n_train < 2 or n_val < 1:
            continue

        baseline = LogBudgetBaseline().fit(
            df.loc[train_mask, budget_col],
            target_log.loc[train_mask],
        )
        y_pred_log = baseline.predict_log(df.loc[val_mask, budget_col])
        y_true_log = target_log.loc[val_mask].to_numpy()
        y_true_dollars = df.loc[val_mask, target_col].to_numpy()
        y_pred_dollars = np.expm1(y_pred_log)

        # Score on the same three lenses as the model: log-space R² (matches
        # the model's objective), rank correlation (ordering quality), and
        # dollar-space R² (absolute calibration).
        results.append(
            BaselineFoldResult(
                year=int(year),
                n_train=n_train,
                n_val=n_val,
                baseline_r2_dollars=float(r2_score(y_true_dollars, y_pred_dollars)),
                baseline_r2_log=float(r2_score(y_true_log, y_pred_log)),
                baseline_spearman=spearman_rank_corr(y_true_log, y_pred_log),
            )
        )
    return results


def assemble_per_year_metrics_table(
    *,
    model_fold_results: Iterable[dict],
    baseline_results: Iterable[BaselineFoldResult],
) -> pd.DataFrame:
    """Combine model + baseline per-year results into one report DataFrame.

    Required columns on each model fold dict (added by the CV loop):
    ``eval_year``, ``val_samples``, ``model_r2_log``, ``model_spearman``,
    ``model_r2_dollars``, ``rmsle_score``, ``model_median_ape``. Folds with
    errors are skipped. Returns columns: ``year``, ``n_movies``,
    ``baseline_r2_log``, ``model_r2_log``, ``gain_r2_log``, ``baseline_spearman``,
    ``model_spearman``, ``baseline_r2``, ``model_r2``, ``gain_r2``,
    ``model_rmsle``, ``model_median_ape`` (``_r2`` columns are dollar-space).
    """
    base_dollars = {b.year: b.baseline_r2_dollars for b in baseline_results}
    base_log = {b.year: b.baseline_r2_log for b in baseline_results}
    base_spear = {b.year: b.baseline_spearman for b in baseline_results}

    def _gain(model: float, baseline: float) -> float:
        return model - baseline if not np.isnan(baseline) else float("nan")

    rows = []
    for fold in model_fold_results:
        if fold.get("error") is not None:
            continue
        year = int(fold["eval_year"])
        baseline_r2 = base_dollars.get(year, float("nan"))
        baseline_r2_log = base_log.get(year, float("nan"))
        model_r2 = float(fold.get("model_r2_dollars", float("nan")))
        model_r2_log = float(fold.get("model_r2_log", float("nan")))
        rows.append(
            {
                "year": year,
                "n_movies": int(fold["val_samples"]),
                "baseline_r2_log": baseline_r2_log,
                "model_r2_log": model_r2_log,
                "gain_r2_log": _gain(model_r2_log, baseline_r2_log),
                "baseline_spearman": base_spear.get(year, float("nan")),
                "model_spearman": float(fold.get("model_spearman", float("nan"))),
                "baseline_r2": baseline_r2,
                "model_r2": model_r2,
                "gain_r2": _gain(model_r2, baseline_r2),
                "model_rmsle": float(fold.get("rmsle_score", float("nan"))),
                "model_median_ape": float(fold.get("model_median_ape", float("nan"))),
            }
        )
    return pd.DataFrame(rows).sort_values("year").reset_index(drop=True)


def render_metrics_table_markdown(table: pd.DataFrame) -> str:
    """Render the per-year table as a Markdown block for the README / model card.

    Log-space R² leads: it matches the model's training objective
    (``reg:squarederror`` on ``log1p`` revenue) and is robust to the heavy
    revenue tail. Rank ρ (Spearman) shows ordering quality. Dollar-space R² and
    median APE expose absolute calibration — where a market-wide shock such as
    the COVID-2020 theatrical shutdown surfaces — and are kept in view rather
    than dropped.
    """
    if table.empty:
        return "_No per-year metrics available._"

    header = (
        "| Year | n | Model R² (log) | Baseline R² (log) | Gain (log) "
        "| Model ρ | Baseline ρ | Model R² ($) | Median APE |\n"
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|"
    )
    rows = []
    for _, r in table.iterrows():
        rows.append(
            f"| {int(r['year'])} | {int(r['n_movies'])} "
            f"| {r['model_r2_log']:.3f} | {r['baseline_r2_log']:.3f} "
            f"| {r['gain_r2_log']:+.3f} "
            f"| {r['model_spearman']:.3f} | {r['baseline_spearman']:.3f} "
            f"| {r['model_r2']:.3f} | {r['model_median_ape']:.1%} |"
        )
    return "\n".join([header, *rows])
