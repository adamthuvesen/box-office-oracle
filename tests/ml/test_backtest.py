"""Tests for the per-year backtest reporting module."""

import numpy as np
import pandas as pd
import pytest

from box_office.ml.backtest import (
    LogBudgetBaseline,
    assemble_per_year_metrics_table,
    compute_baseline_per_year,
    render_metrics_table_markdown,
)
from box_office.ml.backtest_report import build_report


@pytest.fixture
def synthetic_yearly_movies() -> pd.DataFrame:
    rng = np.random.default_rng(7)
    rows = []
    for year in (2020, 2021, 2022, 2023, 2024):
        for _ in range(30):
            budget = float(rng.uniform(5_000_000, 250_000_000))
            noise = float(rng.normal(0, 0.4))
            log_gross = np.log1p(budget) * 0.85 + noise
            rows.append(
                {
                    "RELEASE_YEAR": year,
                    "production_budget": budget,
                    "worldwide_gross": float(np.expm1(log_gross)),
                }
            )
    return pd.DataFrame(rows)


class TestLogBudgetBaseline:
    def test_fits_and_predicts(self, synthetic_yearly_movies):
        df = synthetic_yearly_movies
        baseline = LogBudgetBaseline().fit(
            df["production_budget"], np.log1p(df["worldwide_gross"])
        )
        preds = baseline.predict_log(df["production_budget"])
        assert preds.shape == (len(df),)
        assert np.all(np.isfinite(preds))

    def test_zero_budget_falls_back_to_mean(self):
        budget = pd.Series([10_000_000.0, 50_000_000.0, 100_000_000.0])
        target_log = pd.Series([16.0, 18.0, 19.0])
        baseline = LogBudgetBaseline().fit(budget, target_log)

        preds = baseline.predict_log(pd.Series([0.0, np.nan, 25_000_000.0]))
        assert preds[0] == pytest.approx(np.mean(target_log), rel=1e-6)
        assert preds[1] == pytest.approx(np.mean(target_log), rel=1e-6)
        assert preds[2] != pytest.approx(np.mean(target_log), rel=1e-6)

    def test_fit_rejects_too_few_rows(self):
        with pytest.raises(ValueError, match="at least 2 rows"):
            LogBudgetBaseline().fit(
                pd.Series([1_000_000.0]),
                pd.Series([14.0]),
            )


class TestComputeBaselinePerYear:
    def test_returns_one_row_per_eval_year(self, synthetic_yearly_movies):
        results = compute_baseline_per_year(
            synthetic_yearly_movies,
            target_col="worldwide_gross",
            budget_col="production_budget",
            year_col="RELEASE_YEAR",
            eval_years=[2022, 2023, 2024],
        )
        assert [r.year for r in results] == [2022, 2023, 2024]
        for r in results:
            assert r.n_train > 0
            assert r.n_val > 0
            # Synthetic data is highly budget-driven; baseline should explain plenty.
            assert r.baseline_r2_dollars > 0.3
            assert r.baseline_r2_log > 0.3
            assert -1.0 <= r.baseline_spearman <= 1.0

    def test_skips_year_with_no_train_data(self, synthetic_yearly_movies):
        # 2020 is the earliest year; with eval_years=[2020], train_mask is empty.
        results = compute_baseline_per_year(
            synthetic_yearly_movies,
            target_col="worldwide_gross",
            budget_col="production_budget",
            year_col="RELEASE_YEAR",
            eval_years=[2020],
        )
        assert results == []


class TestAssemblePerYearTable:
    def test_combines_model_and_baseline(self):
        fold_results = [
            {
                "eval_year": 2023,
                "val_samples": 100,
                "rmsle_score": 1.2,
                "model_r2_log": 0.80,
                "model_spearman": 0.88,
                "model_r2_dollars": 0.65,
                "model_median_ape": 0.4,
                "error": None,
            },
            {
                "eval_year": 2024,
                "val_samples": 95,
                "rmsle_score": 1.4,
                "model_r2_log": 0.78,
                "model_spearman": 0.86,
                "model_r2_dollars": 0.55,
                "model_median_ape": 0.5,
                "error": None,
            },
        ]
        baseline_results = compute_baseline_per_year(
            pd.DataFrame(
                {
                    "RELEASE_YEAR": [2022, 2023, 2024] * 5,
                    "production_budget": [50e6] * 15,
                    "worldwide_gross": [120e6] * 15,
                }
            ),
            target_col="worldwide_gross",
            budget_col="production_budget",
            year_col="RELEASE_YEAR",
            eval_years=[2023, 2024],
        )
        table = assemble_per_year_metrics_table(
            model_fold_results=fold_results,
            baseline_results=baseline_results,
        )
        assert list(table["year"]) == [2023, 2024]
        assert set(table.columns) == {
            "year",
            "n_movies",
            "baseline_r2_log",
            "model_r2_log",
            "gain_r2_log",
            "baseline_spearman",
            "model_spearman",
            "baseline_r2",
            "model_r2",
            "gain_r2",
            "model_rmsle",
            "model_median_ape",
        }
        assert (table["gain_r2"] == table["model_r2"] - table["baseline_r2"]).all()
        assert (
            table["gain_r2_log"] == table["model_r2_log"] - table["baseline_r2_log"]
        ).all()

    def test_skips_failed_folds(self):
        fold_results = [
            {"eval_year": 2023, "error": "boom"},
            {
                "eval_year": 2024,
                "val_samples": 50,
                "rmsle_score": 1.1,
                "model_r2_dollars": 0.7,
                "model_median_ape": 0.3,
                "error": None,
            },
        ]
        table = assemble_per_year_metrics_table(
            model_fold_results=fold_results,
            baseline_results=[],
        )
        assert list(table["year"]) == [2024]


class TestMarkdownRendering:
    def test_renders_header_and_rows(self):
        table = pd.DataFrame(
            [
                {
                    "year": 2024,
                    "n_movies": 100,
                    "baseline_r2_log": 0.5,
                    "model_r2_log": 0.7,
                    "gain_r2_log": 0.2,
                    "baseline_spearman": 0.6,
                    "model_spearman": 0.85,
                    "baseline_r2": 0.4,
                    "model_r2": 0.65,
                    "gain_r2": 0.25,
                    "model_rmsle": 1.1,
                    "model_median_ape": 0.35,
                }
            ]
        )
        md = render_metrics_table_markdown(table)
        assert "Year" in md and "Baseline R²" in md
        assert "2024" in md
        assert "+0.200" in md

    def test_empty_table(self):
        assert "No per-year" in render_metrics_table_markdown(pd.DataFrame())


class TestBuildReportEndToEnd:
    def test_report_combines_inputs(self, synthetic_yearly_movies):
        cv_results = {
            "fold_results": [
                {
                    "eval_year": 2023,
                    "val_samples": 30,
                    "rmsle_score": 1.0,
                    "model_r2_log": 0.83,
                    "model_spearman": 0.9,
                    "model_r2_dollars": 0.78,
                    "model_median_ape": 0.32,
                    "error": None,
                },
                {
                    "eval_year": 2024,
                    "val_samples": 30,
                    "rmsle_score": 1.05,
                    "model_r2_log": 0.81,
                    "model_spearman": 0.88,
                    "model_r2_dollars": 0.74,
                    "model_median_ape": 0.36,
                    "error": None,
                },
            ]
        }
        table = build_report(
            raw_df=synthetic_yearly_movies,
            cv_results=cv_results,
            target_col="worldwide_gross",
            budget_col="production_budget",
            year_col="RELEASE_YEAR",
        )
        assert len(table) == 2
        # Model should beat baseline on synthetic data with strong budget signal + noise
        assert (table["model_r2"] >= table["baseline_r2"]).all()
        # Log-space + rank columns flow end-to-end alongside the dollar columns.
        assert {"model_r2_log", "gain_r2_log", "model_spearman"} <= set(table.columns)
        assert table[["model_r2_log", "model_spearman"]].notna().all().all()
