"""Data phase computes the v9 contract and wires leakage-free CV downstream.

``run_data_phase`` must apply the shared quality gate + v9 IP/franchise
features so ``X_TRAIN`` (X_train_processed) carries all 13 SELECTED_FEATURES,
and the SageMaker container must run CV with a per-fold preprocessor factory.
"""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pandas as pd

from box_office.ml.feature_pipeline.constants import SELECTED_FEATURES


class _Logger:
    def info(self, *a, **k):
        pass

    def warning(self, *a, **k):
        pass


def _staging_frame() -> pd.DataFrame:
    rows = []
    # Years span the split boundary (config end_year=2024) so both train and
    # val are non-empty; <2026 so the future-year gate keeps them.
    for i in range(40):
        year = 2016 + (i % 9)  # 2016..2024
        rows.append(
            {
                "TMDB_ID": 1000 + i,
                "IMDB_ID": f"tt{i:07d}",
                "TITLE": f"Movie {i}",
                "ORIGINAL_TITLE": f"Movie {i}",
                "RELEASE_YEAR": year,
                "RELEASE_DATE": f"{year}-06-15",
                "WORLDWIDE_GROSS": 5e7 + i * 1e6,
                "PRODUCTION_BUDGET": 1e7 + i * 1e6,
                "PRODUCTION_BUDGET_SOURCE": "tmdb",
                "PRODUCTION_BUDGET_WAS_MISSING": False,
                "RUNTIME": 100 + (i % 40),
                "MPAA": ["G", "PG", "PG-13", "R"][i % 4],
                "GENRES": ["Action", "Drama", "Comedy, Horror"][i % 3],
                "DIRECTOR": f"Director {i % 5}",
                "PRODUCTION_COMPANY": f"Studio {i % 3}",
                "ACTORS": f"Actor {i}, Actor {(i + 1) % 40}",
                "KEYWORDS": "",
                "OVERVIEW": "",
                "TAGLINE": "",
                "COLLECTION_ID": np.nan,
                "COLLECTION_NAME": None,
            }
        )
    return pd.DataFrame(rows)


def test_run_data_phase_x_train_has_13_v9_features():
    from box_office.orchestration.phases import data_phase

    captured: dict[str, pd.DataFrame] = {}

    def fake_save_tables(specs, _impl):
        for spec in specs:
            captured[spec.table_name] = spec.df
        from box_office.orchestration.persistence import TableSaveReport

        return TableSaveReport(
            results={s.table_name: True for s in specs},
            operations=[],
        )

    from unittest.mock import MagicMock

    import box_office.orchestration.tasks.data_tasks as data_tasks

    # Run the pure Prefect tasks as plain functions (no Prefect API round-trip).
    with (
        patch.object(data_tasks, "get_run_logger", lambda: MagicMock()),
        patch.object(data_phase, "run_raw_to_staging_dbt_transformations"),
        patch.object(
            data_phase,
            "load_staging_box_office_from_snowflake",
            return_value=_staging_frame(),
        ),
        patch.object(data_phase, "split_data", data_phase.split_data.fn),
        patch.object(
            data_phase,
            "apply_feature_engineering",
            data_phase.apply_feature_engineering.fn,
        ),
        patch.object(data_phase, "scale_features", data_phase.scale_features.fn),
        patch.object(data_phase, "transform_targets", data_phase.transform_targets.fn),
        patch.object(
            data_phase,
            "create_feature_metadata",
            data_phase.create_feature_metadata.fn,
        ),
        patch.object(data_phase, "save_tables", side_effect=fake_save_tables),
        patch.object(data_phase, "log_table_operations_summary"),
        patch.object(
            data_phase,
            "validate_snowflake_tables",
            return_value={"X_TRAIN": 30},
        ),
        patch.object(data_phase, "save_artifacts", return_value=("/tmp/p", "/tmp/s")),
    ):
        result = data_phase.run_data_phase(_Logger())

    assert list(result.X_train_processed.columns) == list(SELECTED_FEATURES)
    assert result.X_train_processed.shape[1] == 13
    assert "X_TRAIN" in captured
    assert list(captured["X_TRAIN"].columns) == list(SELECTED_FEATURES)
    # RAW frame kept for the leakage-free SageMaker upload.
    assert "ACTORS" in result.X_train_raw.columns
    assert "RELEASE_YEAR" in result.X_train_raw.columns


def test_container_train_passes_preprocessor_factory():
    """model_training.train must run CV with FeaturePreprocessorHigh as the
    per-fold factory, so each fold refits on train-year rows only (leakage-free)."""
    from box_office.ml import model_training
    from box_office.ml.feature_preprocessor import FeaturePreprocessorHigh

    raw = _staging_frame()
    # Container load_data shape: raw preprocessor inputs + RELEASE_YEAR, GROSS_LOG.
    from box_office.training_frame import (
        PREPROCESSOR_INPUT_COLUMNS,
        build_production_training_frame,
    )

    kept, _ = build_production_training_frame(raw, overrides_path=None)
    X = kept[list(PREPROCESSOR_INPUT_COLUMNS)]
    y_log = pd.Series(np.log1p(kept["WORLDWIDE_GROSS"].astype(float)), name="GROSS_LOG")
    dates = kept["RELEASE_YEAR"].astype(int)

    seen = {}

    def fake_cross_validate(model_class, X_train, y_train_log, dates, **kwargs):
        seen["preprocessor_factory"] = kwargs.get("preprocessor_factory")
        seen["X_columns"] = list(X_train.columns)
        return {
            "mean_best_iteration": 10,
            "mean_cv_mae": 0.5,
            "std_cv_mae": 0.1,
            "mean_cv_rmsle": 0.5,
            "std_cv_rmsle": 0.1,
            "cv_scores": [0.5],
            "cv_rmsle_scores": [0.5],
            "oof_predictions": {},
            "oof_records": [],
            "fold_results": [],
            "feature_importances": None,
            "feature_names": list(X_train.columns),
        }

    args = type("Args", (), {})()
    args.train = "/unused"
    args.cv_folds = 3
    args.start_eval_year = 2022
    args.end_year = 2024
    args.backtest_years = 3
    for k, v in {
        "n_estimators": 10,
        "learning_rate": 0.04,
        "max_depth": 2,
        "min_child_weight": 2,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.01,
        "reg_lambda": 0.2,
        "early_stopping_rounds": 5,
        "model_dir": "/tmp/m",
        "output_data_dir": "/tmp/o",
        "processor_s3_uri": "s3://b/p",
        "scaler_s3_uri": "s3://b/s",
    }.items():
        setattr(args, k, v)

    preprocessor = FeaturePreprocessorHigh().fit(X)
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler().fit(preprocessor.transform(X))

    with (
        patch.object(model_training, "load_data", return_value=(X, y_log, dates)),
        patch.object(
            model_training.TimeSeriesCrossValidator,
            "cross_validate",
            side_effect=fake_cross_validate,
            autospec=False,
        ),
        patch.object(
            model_training,
            "load_preprocessing_artifacts",
            return_value=(preprocessor, scaler, "/tmp/m/p", "/tmp/m/s"),
        ),
        patch.object(
            model_training.ModelEvaluator, "evaluate_oof_performance", return_value={}
        ),
        patch.object(model_training, "save_results"),
    ):
        model_training.train(args)

    assert seen["preprocessor_factory"] is FeaturePreprocessorHigh
    assert "ACTORS" in seen["X_columns"]  # raw frame reached CV
