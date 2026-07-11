"""Data preparation phase: dbt, features, Snowflake persistence."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from box_office.config import config
from box_office.orchestration.persistence import (
    TableSaveSpec,
    log_table_operations_summary,
    save_tables,
)
from box_office.orchestration.tasks.data_tasks import (
    apply_feature_engineering,
    create_feature_metadata,
    load_staging_box_office_from_snowflake,
    run_raw_to_staging_dbt_transformations,
    save_artifacts,
    save_dataset_to_snowflake_impl,
    scale_features,
    split_data,
    transform_targets,
    validate_snowflake_tables,
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
    save_results: dict[str, bool]
    validation_results: dict[str, Any]
    feature_names: list[str]


def run_data_phase(logger) -> DataPhaseResult:
    """Run dbt through Snowflake validation; return in-memory frames for training."""
    run_raw_to_staging_dbt_transformations()

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

    X_train_processed, X_val_processed, processor = apply_feature_engineering(
        X_train, X_val
    )
    X_train_scaled, X_val_scaled, scaler = scale_features(
        X_train_processed, X_val_processed
    )
    y_train_log, y_val_log = transform_targets(y_train, y_val)

    processor_path, scaler_path = save_artifacts(
        processor, scaler, artifact_dir=config.model.artifacts_dir
    )
    feature_metadata = create_feature_metadata(
        X_train_processed.columns.tolist(), processor_path, scaler_path
    )

    logger.info("Saving ML tables to Snowflake")
    ml_schema = config.snowflake.schemas.ml_training
    fs_schema = config.snowflake.schemas.feature_store

    specs = [
        TableSaveSpec(X_train_processed, "X_TRAIN", ml_schema),
        TableSaveSpec(X_train_scaled, "X_TRAIN_SCALED", ml_schema),
        TableSaveSpec(y_train.to_frame("WORLDWIDE_GROSS"), "Y_TRAIN", ml_schema),
        TableSaveSpec(y_train_log.to_frame("GROSS_LOG"), "Y_TRAIN_LOG", ml_schema),
        TableSaveSpec(X_val_processed, "X_VAL", ml_schema),
        TableSaveSpec(X_val_scaled, "X_VAL_SCALED", ml_schema),
        TableSaveSpec(y_val.to_frame("WORLDWIDE_GROSS"), "Y_VAL", ml_schema),
        TableSaveSpec(y_val_log.to_frame("GROSS_LOG"), "Y_VAL_LOG", ml_schema),
        TableSaveSpec(feature_metadata, "FEATURE_METADATA", fs_schema),
    ]
    report = save_tables(specs, save_dataset_to_snowflake_impl)
    log_table_operations_summary(report.operations, logger)

    expected_tables = [s.table_name for s in specs if s.schema == ml_schema]
    validation_results = validate_snowflake_tables(expected_tables, schema=ml_schema)

    feature_store_validation = validate_snowflake_tables(
        ["FEATURE_METADATA"], schema=fs_schema
    )
    if "error" in feature_store_validation:
        logger.warning(
            "FEATURE_STORE validation failed (non-fatal): %s",
            feature_store_validation["error"],
        )

    if "error" in validation_results:
        raise RuntimeError(
            f"Snowflake validation failed: {validation_results['error']}"
        )

    validation_errors = [
        k for k, v in validation_results.items() if isinstance(v, str) and "Error" in v
    ]
    if validation_errors:
        raise RuntimeError(
            f"Snowflake validation failed for tables: {validation_errors}"
        )

    successful_saves = sum(1 for success in report.results.values() if success)
    logger.info(
        "Data phase complete: %d features, Snowflake saves %d/%d",
        X_train_processed.shape[1],
        successful_saves,
        len(report.results),
    )

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
        save_results=report.results,
        validation_results=validation_results,
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
