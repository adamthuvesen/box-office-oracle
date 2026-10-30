#!/usr/bin/env python3

import argparse
import json
import logging
import os
import pickle
import time
from pathlib import Path

import boto3
import joblib
import numpy as np
import pandas as pd

from box_office.config import MODEL_HYPERPARAMETERS, ModelHyperparameters
from box_office.eval_windows import ITERATION_CV_FOLDS, ITERATION_END_YEAR, START_EVAL_YEAR
from box_office.ml.artifacts import (
    FEATURE_PREPROCESSOR_PKL,
    FEATURE_SCALER_PKL,
    MODEL_PKL,
)
from box_office.ml.cv import ModelEvaluator, TimeSeriesCrossValidator
from box_office.ml.feature_preprocessor import FeaturePreprocessorHigh
from box_office.ml.model import BoxOfficeXGBoostModel, fit_final_model
from box_office.utils.aws_helpers import parse_s3_uri, resolve_aws_region

try:
    from smexperiments.experiment_context import load_run_context  # pyright: ignore[reportMissingImports]
except ImportError:
    load_run_context = None


def setup_logging_once():
    """Configure logging only if not already configured."""
    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


setup_logging_once()
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command-line arguments passed to the training script."""
    parser = argparse.ArgumentParser()

    # SageMaker channel paths; fall back to local defaults outside SageMaker.
    parser.add_argument("--model-dir", type=str, default=os.environ.get("SM_MODEL_DIR", "./model"))
    parser.add_argument(
        "--train", type=str, default=os.environ.get("SM_CHANNEL_TRAIN", "./data/train")
    )
    parser.add_argument(
        "--output-data-dir",
        type=str,
        default=os.environ.get("SM_OUTPUT_DATA_DIR", "./output"),
    )

    # XGBoost hyperparameters injected via Estimator.hyperparameters.
    for name, default in MODEL_HYPERPARAMETERS.model_dump().items():
        parser.add_argument(f"--{name}", type=type(default), default=default)

    args = parser.parse_args()

    # Custom parameters injected via Estimator.environment. The lowercase names are the
    # contract with the training job's environment dict (see sagemaker_train_job.py).
    args.cv_folds = int(os.environ.get("cv_folds", str(ITERATION_CV_FOLDS)))  # noqa: SIM112
    args.end_year = int(os.environ.get("end_year", str(ITERATION_END_YEAR)))  # noqa: SIM112
    args.start_eval_year = int(os.environ.get("start_eval_year", str(START_EVAL_YEAR)))  # noqa: SIM112

    processor_uri = os.environ.get("processor_s3_uri", "None")  # noqa: SIM112
    args.processor_s3_uri = None if processor_uri == "None" else processor_uri

    scaler_uri = os.environ.get("scaler_s3_uri", "None")  # noqa: SIM112
    args.scaler_s3_uri = None if scaler_uri == "None" else scaler_uri

    return args


def load_data(train_path):
    """Load training data from the directory specified by SageMaker."""
    logger.info(f"Loading training data from: {train_path}")

    # SageMaker stages a channel as a directory of files.
    try:
        parquet_files = [f.name for f in Path(train_path).iterdir() if f.name.endswith(".parquet")]
        if not parquet_files:
            raise FileNotFoundError(
                f"No .parquet file found in the training directory: {train_path}"
            )

        data_path = Path(train_path) / parquet_files[0]
        data = pd.read_parquet(data_path)
        logger.info(f"Loaded data with shape: {data.shape} from {data_path}")

    except Exception as e:
        logger.error(f"Error loading data from {train_path}: {e}")
        raise

    date_col = "RELEASE_YEAR"
    target_col = "GROSS_LOG"

    if date_col not in data.columns or target_col not in data.columns:
        raise ValueError(f"Required columns '{date_col}' or '{target_col}' not in dataset.")

    dates = data[date_col].copy()
    y_train_log = data[target_col]
    # RELEASE_YEAR stays in X: it is a v9 feature (SELECTED_FEATURES) as well as
    # the chronological CV key. The uploaded frame is the RAW preprocessor
    # input, so the per-fold FeaturePreprocessorHigh consumes RELEASE_YEAR here.
    X_train = data.drop([target_col], axis=1)

    logger.info(f"Features shape: {X_train.shape}")
    logger.info(f"Target shape: {y_train_log.shape}")
    logger.info(f"Date range for CV: {dates.min()} - {dates.max()}")

    # The cross-validator does integer year comparisons, so coerce upfront.
    if not pd.api.types.is_integer_dtype(dates):
        logger.warning(f"Date column '{date_col}' is not integer. Attempting conversion.")
        dates = dates.astype(int)

    return X_train, y_train_log, dates


def hyperparameters_from_args(args) -> ModelHyperparameters:
    """The XGBoost hyperparameters the estimator passed, validated by their owner."""
    return ModelHyperparameters(
        **{name: getattr(args, name) for name in ModelHyperparameters.model_fields}
    )


def load_preprocessing_artifacts(args):
    """Download the pipeline-fitted preprocessor + scaler from S3 into model_dir.

    These were fit on all training rows in the data phase; the container reuses
    them (not a fresh fit) so the deployed model's features match inference
    exactly. Returns the loaded objects and their on-disk paths (the paths are
    already inside ``model_dir``, so they ship in the model tarball).
    """
    if not (args.processor_s3_uri and args.scaler_s3_uri):
        raise FileNotFoundError(
            "No S3 URIs provided for preprocessing artifacts. "
            "Set processor_s3_uri and scaler_s3_uri environment variables."
        )

    Path(args.model_dir).mkdir(parents=True, exist_ok=True)
    logger.info("Downloading fitted preprocessing artifacts from S3...")
    aws_region = resolve_aws_region()
    s3_client = boto3.client("s3", region_name=aws_region)

    try:
        processor_bucket, processor_key = parse_s3_uri(args.processor_s3_uri)
        processor_path = Path(args.model_dir) / FEATURE_PREPROCESSOR_PKL
        s3_client.download_file(processor_bucket, processor_key, str(processor_path))

        scaler_bucket, scaler_key = parse_s3_uri(args.scaler_s3_uri)
        scaler_path = Path(args.model_dir) / FEATURE_SCALER_PKL
        s3_client.download_file(scaler_bucket, scaler_key, str(scaler_path))
    except Exception as e:
        raise RuntimeError(
            f"Failed to download preprocessing artifacts: {e}. "
            f"Ensure artifacts exist at processor_s3_uri={args.processor_s3_uri} "
            f"and scaler_s3_uri={args.scaler_s3_uri}"
        ) from e

    logger.info("Loaded fitted preprocessing artifacts from training pipeline")
    return (
        joblib.load(processor_path),
        joblib.load(scaler_path),
        processor_path,
        scaler_path,
    )


def transform_for_final_fit(X_raw, preprocessor, scaler):
    """Engineer + scale the raw frame with the pipeline-fitted artifacts.

    NaN-aware: StandardScaler uses nanmean/nanvar, so missing budgets stay NaN
    (never imputed to 0), matching scripts/train_local.py.
    """
    X_processed = preprocessor.transform(X_raw)
    return pd.DataFrame(
        scaler.transform(X_processed),
        columns=X_processed.columns,
        index=X_processed.index,
    )


def save_results(model, cv_results, oof_results, args, processor_path, scaler_path):
    """Save model artifacts and evaluation metrics.

    ``processor_path`` / ``scaler_path`` were downloaded into ``model_dir`` by
    ``load_preprocessing_artifacts`` and are already staged for the tarball.
    """

    model_dir = Path(args.model_dir)
    output_data_dir = Path(args.output_data_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    output_data_dir.mkdir(parents=True, exist_ok=True)

    # Pickle preserves the BoxOfficeXGBoostModel wrapper used at inference.
    model_path = model_dir / MODEL_PKL
    with model_path.open("wb") as f:
        pickle.dump(model, f)
    logger.info(f"Final model saved to: {model_path}")
    logger.info(f"Bundled preprocessing artifacts: {processor_path}, {scaler_path}")

    def convert_for_json(obj):
        if isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        if isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    cv_results_path = output_data_dir / "cv_results.json"
    with cv_results_path.open("w") as f:
        json.dump(cv_results, f, indent=4, default=convert_for_json)
    logger.info(f"CV results saved to: {cv_results_path}")

    oof_results_path = output_data_dir / "oof_evaluation.json"
    with oof_results_path.open("w") as f:
        json.dump(oof_results, f, indent=4, default=convert_for_json)
    logger.info(f"OOF evaluation results saved to: {oof_results_path}")

    logger.info("Training metrics summary")

    cv_mae = cv_results.get("mean_cv_mae", 0)
    cv_mae_std = cv_results.get("std_cv_mae", 0)
    cv_rmsle = cv_results.get("mean_cv_rmsle", 0)
    cv_rmsle_std = cv_results.get("std_cv_rmsle", 0)
    mean_best_iteration = cv_results.get("mean_best_iteration", 0)
    oof_r2 = oof_results.get("oof_r2", 0)
    oof_mae = oof_results.get("oof_mae", 0)
    oof_rmsle = oof_results.get("oof_rmsle", 0)
    oof_num_samples = oof_results.get("num_oof_samples", 0)
    cv_folds = len(cv_results.get("cv_scores", []))

    # 1. Detailed console output
    logger.info("CROSS-VALIDATION PERFORMANCE METRICS:")
    logger.info(f"Mean CV MAE (log scale): {cv_mae:.4f}")
    logger.info(f"CV MAE Std Dev: ±{cv_mae_std:.4f}")
    logger.info(f"Mean CV RMSLE: {cv_rmsle:.4f}")
    logger.info(f"CV RMSLE Std Dev: ±{cv_rmsle_std:.4f}")
    logger.info(f"Average Best Iteration: {mean_best_iteration:.1f}")
    logger.info(f"Cross-Validation Folds: {cv_folds}")

    logger.info("OUT-OF-FOLD EVALUATION METRICS:")
    logger.info(f"OOF R² Score: {oof_r2:.4f}")
    logger.info(f"OOF MAE (USD): ${oof_mae:,.0f}")
    logger.info(f"OOF RMSLE: {oof_rmsle:.4f}")
    logger.info(f"OOF Samples: {oof_num_samples}")

    # 2. Structured metrics for parsing
    logger.info("STRUCTURED METRICS (for automated parsing):")
    metrics_summary = {
        "cv_mean_mae": cv_mae,
        "cv_std_mae": cv_mae_std,
        "cv_mean_rmsle": cv_rmsle,
        "cv_std_rmsle": cv_rmsle_std,
        "cv_mean_best_iteration": mean_best_iteration,
        "oof_r2": oof_r2,
        "oof_mae_dollars": oof_mae,
        "oof_rmsle": oof_rmsle,
        "oof_num_samples": oof_num_samples,
        "cv_folds_completed": cv_folds,
    }

    for metric_name, metric_value in metrics_summary.items():
        # SageMaker metric extraction pattern: METRIC:name=value
        logger.info(f"METRIC:{metric_name}={metric_value:.6f}")

    # 3. Detailed metrics export to file
    detailed_metrics = {
        "training_summary": {
            "model_type": "XGBoost",
            "objective": "reg:squarederror",
            "cv_folds": cv_folds,
            "oof_eval_samples": oof_num_samples,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
        "cross_validation_metrics": {
            "mean_mae": cv_mae,
            "std_mae": cv_mae_std,
            "mean_rmsle": cv_rmsle,
            "std_rmsle": cv_rmsle_std,
            "mean_best_iteration": mean_best_iteration,
            "individual_fold_scores": cv_results.get("cv_scores", []),
            "individual_rmsle_scores": cv_results.get("cv_rmsle_scores", []),
            "individual_best_iterations": [
                fr.get("best_iteration")
                for fr in cv_results.get("fold_results", [])
                if fr.get("best_iteration") is not None
            ],
        },
        "out_of_fold_evaluation": {
            "r2_score": oof_r2,
            "mae_dollars": oof_mae,
            "rmsle": oof_rmsle,
            "num_samples": oof_num_samples,
        },
        "training_metadata": {
            "n_estimators": args.n_estimators,
            "learning_rate": args.learning_rate,
            "max_depth": args.max_depth,
            "early_stopping_rounds": args.early_stopping_rounds,
        },
    }

    comprehensive_metrics_file = output_data_dir / "comprehensive_metrics.json"
    with comprehensive_metrics_file.open("w") as f:
        json.dump(detailed_metrics, f, indent=2, default=convert_for_json)
    logger.info(f"Detailed metrics saved to: {comprehensive_metrics_file}")

    # Per-year model-side metrics table. The log-budget baseline column is
    # populated post-training by ``box_office.ml.backtest_report`` because
    # the raw production_budget lives outside this container.
    from box_office.ml.backtest import assemble_per_year_metrics_table

    per_year_model_table = assemble_per_year_metrics_table(
        model_fold_results=cv_results.get("fold_results", []),
        baseline_results=[],
    )
    per_year_path = output_data_dir / "per_year_model_metrics.json"
    per_year_model_table.to_json(per_year_path, orient="records", indent=2)
    logger.info(f"Per-year model metrics saved to: {per_year_path}")

    # 4. CloudWatch custom metrics
    try:
        aws_region = resolve_aws_region()
        cloudwatch = boto3.client("cloudwatch", region_name=aws_region)
        logger.info(f"CloudWatch client created for region: {aws_region}")

        dimensions = [
            {"Name": "Environment", "Value": "dev"},
            {"Name": "ModelType", "Value": "XGBoost"},
        ]
        cloudwatch_metrics = [
            {
                "MetricName": "OOF_R2_Score",
                "Value": oof_r2,
                "Unit": "None",
                "Dimensions": dimensions,
            },
            {
                "MetricName": "OOF_MAE_Millions",
                "Value": oof_mae / 1e6,
                "Unit": "None",
                "Dimensions": dimensions,
            },
            {
                "MetricName": "CV_Mean_MAE",
                "Value": cv_mae,
                "Unit": "None",
                "Dimensions": dimensions,
            },
            {
                "MetricName": "CV_Mean_RMSLE",
                "Value": cv_rmsle,
                "Unit": "None",
                "Dimensions": dimensions,
            },
            {
                "MetricName": "CV_Best_Iteration",
                "Value": mean_best_iteration,
                "Unit": "Count",
                "Dimensions": dimensions,
            },
        ]

        # CloudWatch accepts up to 20 data points per call; one batched put
        # replaces five sequential round-trips.
        cloudwatch.put_metric_data(
            Namespace="BoxOffice/ModelTraining",
            MetricData=cloudwatch_metrics,
        )

        logger.info(f"Sent {len(cloudwatch_metrics)} metrics to CloudWatch")

    except Exception as e:
        logger.warning(f"Could not send CloudWatch metrics: {e}")

    # 5. Performance summary
    logger.info("TRAINING PERFORMANCE SUMMARY:")
    logger.info(f"Model Quality: R² = {oof_r2:.3f} (Target: > 0.55)")
    logger.info(f"Prediction Error: ${oof_mae / 1e6:.1f}M average error")
    logger.info(f"Cross-Validation Stability: MAE std = ±{cv_mae_std:.3f}")
    logger.info(f"Training Efficiency: {mean_best_iteration:.0f} avg iterations")

    logger.info("Training metrics summary complete")

    # Log custom metrics to SageMaker Experiment from inside the container.
    try:
        run_ctx = load_run_context() if load_run_context is not None else None
        if run_ctx:
            custom_metrics = {
                "cv_mean_mae": cv_results.get("mean_cv_mae"),
                "cv_mean_rmsle": cv_results.get("mean_cv_rmsle"),
                "cv_std_mae": cv_results.get("std_cv_mae"),
                "cv_std_rmsle": cv_results.get("std_cv_rmsle"),
                "cv_mean_best_iteration": cv_results.get("mean_best_iteration"),
                "oof_r2": oof_results.get("oof_r2"),
                "oof_mae": oof_results.get("oof_mae"),
                "oof_rmsle": oof_results.get("oof_rmsle"),
                "oof_num_samples": oof_results.get("num_oof_samples"),
            }
            for k, v in custom_metrics.items():
                if v is not None:
                    run_ctx.log_metric(k, float(v))
            logger.info("Custom metrics logged to SageMaker Experiment")
    except Exception as e:
        logger.warning(f"Could not log custom metrics via run context: {e}")


def train(args):
    """Main training orchestrator function."""
    logger.info("Starting Box Office Prediction Model Training")
    logger.info("Running in XGBoost framework mode with custom metric capture")
    logger.info(f"Training parameters: {vars(args)}")

    X_train, y_train_log, dates = load_data(args.train)

    cv = TimeSeriesCrossValidator(
        cv_folds=args.cv_folds,
        start_eval_year=args.start_eval_year,
        end_year=args.end_year,
        early_stopping_rounds=args.early_stopping_rounds,
    )

    hyperparameters = hyperparameters_from_args(args)
    model_kwargs = {
        **hyperparameters.model_dump(exclude={"early_stopping_rounds"}),
        "random_state": 42,
    }

    # Leakage-free CV: each fold fits a fresh FeaturePreprocessorHigh on
    # train-years rows only, so frequency encodings can't see the eval year
    # (matches scripts/train_local.py). X_train is the RAW preprocessor input.
    cv_results = cv.cross_validate(
        model_class=BoxOfficeXGBoostModel,
        X_train=X_train,
        y_train_log=y_train_log,
        dates=dates,
        preprocessor_factory=FeaturePreprocessorHigh,
        **model_kwargs,
    )

    oof_results = ModelEvaluator.evaluate_oof_performance(cv_results, y_train_log)

    # Final model trains on the full frame engineered + scaled with the
    # pipeline-fitted artifacts (correct for serving: at prediction time every
    # training row is legitimately past).
    preprocessor, scaler, processor_path, scaler_path = load_preprocessing_artifacts(args)
    X_train_scaled = transform_for_final_fit(X_train, preprocessor, scaler)
    logger.info(
        "Training final model with %d estimators based on CV.",
        int(cv_results["mean_best_iteration"]),
    )
    final_model = fit_final_model(
        X_train_scaled, y_train_log, cv_results["mean_best_iteration"], hyperparameters
    )

    save_results(final_model, cv_results, oof_results, args, processor_path, scaler_path)

    logger.info("Training script completed successfully!")


if __name__ == "__main__":
    args = parse_args()
    train(args)
