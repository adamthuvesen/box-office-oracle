"""
Orchestration tasks module.

Contains Prefect task definitions for data processing, training, and metrics.
Tasks are organized by domain: data_tasks, training_tasks, metrics_tasks.
"""

__all__ = [
    # Data tasks
    "run_raw_to_staging_dbt_transformations",
    "load_staging_box_office_from_snowflake",
    "split_data",
    "apply_feature_engineering",
    "scale_features",
    "transform_targets",
    "save_artifacts",
    "create_feature_metadata",
    "validate_snowflake_tables",
    # Training tasks
    "upload_processed_data_to_s3",
    "train_model",
    "download_and_analyze_results",
    "register_model_in_registry",
    "validate_model_for_promotion",
    "promote_model_in_aws_registry",
    "upload_preprocessing_artifacts_to_s3",
    # Metrics tasks
    "log_pipeline_start_metrics",
    "log_data_processing_metrics",
    "log_feature_engineering_metrics",
    "log_model_training_summary",
    "log_pipeline_completion_metrics",
    "save_metrics_to_json",
]
