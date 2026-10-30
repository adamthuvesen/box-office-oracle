"""Data preparation phase: dbt, the v9 training frame, features, and artifacts."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from box_office.config import config
from box_office.orchestration.tasks.data_tasks import (
    apply_feature_engineering,
    load_staging_box_office_from_snowflake,
    run_raw_to_staging_dbt_transformations,
    save_artifacts,
    scale_features,
    split_data,
    transform_targets,
)
from box_office.training_frame import (
    PREPROCESSOR_INPUT_COLUMNS,
    build_production_training_frame,
)

TARGET_COLUMN = "WORLDWIDE_GROSS"


@dataclass
class DataPhaseResult:
    target_column: str
    X_train_raw: pd.DataFrame
    X_train_processed: pd.DataFrame
    X_train_scaled: pd.DataFrame
    y_train_log: pd.Series
    X_train_shape: tuple[int, int]
    X_val_shape: tuple[int, int]
    processor_path: str
    scaler_path: str
    feature_names: list[str]


def run_data_phase(logger, approved: bool = False) -> DataPhaseResult:
    """Run dbt, build the v9 training frame, and return in-memory frames for training."""
    run_raw_to_staging_dbt_transformations(approved=approved)

    staging_data = load_staging_box_office_from_snowflake()

    # Apply the shared quality gate + v9 IP/franchise computation so X_TRAIN
    # carries the full v9 contract (SELECTED_FEATURES). IP is classified
    # in-pipeline here; the rules match scripts/prepare_training_frame.py
    # exactly (box_office.training_frame). NaN budgets pass through un-imputed.
    training_frame, dropped = build_production_training_frame(staging_data)
    logger.info(
        "v9 training frame: %d rows kept, %d dropped by the quality gate",
        len(training_frame),
        len(dropped),
    )
    model_frame = training_frame[[*PREPROCESSOR_INPUT_COLUMNS, TARGET_COLUMN]]
    X_train, X_val, y_train, y_val = split_data(model_frame, TARGET_COLUMN)

    X_train_processed, X_val_processed, processor = apply_feature_engineering(X_train, X_val)
    X_train_scaled, _, scaler = scale_features(X_train_processed, X_val_processed)
    y_train_log, _ = transform_targets(y_train, y_val)

    processor_path, scaler_path = save_artifacts(
        processor, scaler, artifact_dir=config.model.artifacts_dir
    )
    logger.info("Data phase complete: %d features", X_train_processed.shape[1])

    return DataPhaseResult(
        target_column=TARGET_COLUMN,
        X_train_raw=X_train,
        X_train_processed=X_train_processed,
        X_train_scaled=X_train_scaled,
        y_train_log=y_train_log,
        X_train_shape=X_train.shape,
        X_val_shape=X_val.shape,
        processor_path=processor_path,
        scaler_path=scaler_path,
        feature_names=X_train_processed.columns.tolist(),
    )


def sagemaker_training_frames(
    data: DataPhaseResult,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build in-memory training frames for SageMaker upload.

    Uploads the RAW v9 preprocessor-input frame (not the engineered/scaled
    matrix) so the training container fits ``FeaturePreprocessorHigh`` per CV
    fold on train-years rows only — leakage-free frequency encodings, matching
    scripts/train_local.py. ``RELEASE_YEAR`` rides along as both a feature and
    the chronological CV key; ``GROSS_LOG`` is the log-transformed target.
    """
    X = data.X_train_raw.copy()
    if "RELEASE_YEAR" not in X.columns:
        raise ValueError("RELEASE_YEAR is required for SageMaker time-series CV")
    y = data.y_train_log.to_frame("GROSS_LOG")
    return X, y
