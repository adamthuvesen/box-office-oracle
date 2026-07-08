"""Local training driver for the 1980-2026 TMDB training frame.

Pattern-matched to ``scripts/run_backtest.py`` (offline CV with the
production ``TimeSeriesCrossValidator`` + ``BoxOfficeXGBoostModel``) and to
``box_office.ml.model_training.train_final_model`` (final fit on all data at
the CV-mean best iteration). Runs entirely locally: no Snowflake, no
SageMaker, no AWS.

Steps:

1. Load the Phase-1 frame (``scripts/prepare_training_frame.py`` output),
   ``y = log1p(worldwide_gross)``.
2. Expanding-window CV, eval years 2015-2023 (iteration mode). 2024-2025 are
   a SPENT confirmation set: v8's frozen confirmation stands, v9 is adopted
   on <=2023 evidence, and 2026 actuals will confirm it. Evaluating them
   again requires the explicit ``--i-know-this-burns-the-holdout`` flag.
   2026 rows were excluded by the prep script because their gross is not
   final. Each fold fits a fresh
   ``FeaturePreprocessorHigh`` on train-years rows only (leakage-free
   frequency features); the final deployment preprocessor is fit on all data.
3. Fit a ``StandardScaler`` and train the final model on scaled features —
   the same convention as the production pipeline
   (``box_office/orchestration/tasks/data_tasks.py``), which the inference
   predictor's transform->scale->predict path assumes.
4. Save ``artifacts/local/``: ``model.pkl``, ``feature_preprocessor.pkl``,
   ``feature_scaler.pkl``, ``cv_results.json``, and ``metadata.json`` stamped
   ``feature_schema_version = "9"`` with a sha256 manifest, mirroring what the
   registry stores in ``CustomerMetadataProperties``.
5. Smoke-check the artifact through the inference path: enforce the same
   schema-version check the Lambda model loader applies, load all three
   artifacts with ``PredictionEngine.load_model_artifacts`` (the loader's
   joblib path), and run one prediction, asserting a finite dollar value.

Run:  uv run python scripts/train_local.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from box_office.inference.app.integrity import compute_sha256
from box_office.inference.app.predictor import PredictionEngine, PredictionRequest
from box_office.ml.artifacts import (
    FEATURE_PREPROCESSOR_PKL,
    FEATURE_SCALER_PKL,
    MODEL_PKL,
)
from box_office.ml.cv import TimeSeriesCrossValidator
from box_office.ml.feature_pipeline.constants import SELECTED_FEATURES
from box_office.ml.feature_preprocessor import FeaturePreprocessorHigh
from box_office.ml.model import BoxOfficeXGBoostModel
from box_office.ml.registry_constants import (
    CURRENT_FEATURE_SCHEMA_VERSION,
    SCHEMA_VERSION_METADATA_KEY,
    FeatureSchemaVersionMismatch,
)

FRAME_PATH = Path("data/generated/training/train_frame_1980_2026.parquet")
ARTIFACT_DIR = Path("artifacts/local")

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
HOLDOUT_END_YEAR = 2025  # 2024-2025: spent confirmation set; 2026 not final

EVAL_DISCIPLINE_NOTE = (
    "v9 adopted on 2015-2023 iteration evidence; 2024-2025 are a spent "
    "confirmation set (v8's frozen confirmation stands) and were not "
    "evaluated. 2026 actuals will confirm v9."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the local model (iteration mode, eval 2015-2023)."
    )
    parser.add_argument(
        "--i-know-this-burns-the-holdout",
        action="store_true",
        dest="burn_holdout",
        help=(
            "Also evaluate the spent 2024-2025 confirmation years. Doing this "
            "turns the confirmation set into a validation set; only for a "
            "deliberate, final confirmation run."
        ),
    )
    return parser.parse_args()


def json_default(obj: object) -> object:
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"not JSON serializable: {type(obj).__name__}")


def run_cross_validation(
    X_input: pd.DataFrame, y_log: pd.Series, dates: pd.Series, end_eval_year: int
) -> dict:
    """Leakage-free CV: each fold fits a fresh preprocessor on train years
    only, so frequency features cannot see the eval year."""
    cv = TimeSeriesCrossValidator(
        cv_folds=end_eval_year - START_EVAL_YEAR + 1,
        start_eval_year=START_EVAL_YEAR,
        end_year=end_eval_year,
    )
    return cv.cross_validate(
        BoxOfficeXGBoostModel,
        X_input,
        y_log,
        dates,
        preprocessor_factory=FeaturePreprocessorHigh,
    )


def train_final_model(
    X_scaled: pd.DataFrame, y_log: pd.Series, cv_results: dict
) -> BoxOfficeXGBoostModel:
    """Final fit on all data at the CV-mean best iteration (production convention)."""
    best_iteration = int(cv_results["mean_best_iteration"])
    model = BoxOfficeXGBoostModel(n_estimators=best_iteration, random_state=42)
    model.fit(X_scaled, y_log)
    return model


def save_artifact(
    model: BoxOfficeXGBoostModel,
    preprocessor: FeaturePreprocessorHigh,
    scaler: StandardScaler,
    cv_results: dict,
    n_rows: int,
    end_eval_year: int,
) -> dict[str, Path]:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    paths = {
        "model": ARTIFACT_DIR / MODEL_PKL,
        "preprocessor": ARTIFACT_DIR / FEATURE_PREPROCESSOR_PKL,
        "scaler": ARTIFACT_DIR / FEATURE_SCALER_PKL,
    }
    joblib.dump(model, paths["model"])
    joblib.dump(preprocessor, paths["preprocessor"])
    joblib.dump(scaler, paths["scaler"])

    (ARTIFACT_DIR / "cv_results.json").write_text(
        json.dumps(cv_results, indent=2, default=json_default) + "\n"
    )

    metadata = {
        SCHEMA_VERSION_METADATA_KEY: CURRENT_FEATURE_SCHEMA_VERSION,
        "selected_features": list(SELECTED_FEATURES),
        "training_rows": n_rows,
        "cv_mean_mae_log": float(cv_results["mean_cv_mae"]),
        "cv_mean_rmsle": float(cv_results["mean_cv_rmsle"]),
        "mean_best_iteration": float(cv_results["mean_best_iteration"]),
        "eval_years": f"{START_EVAL_YEAR}-{end_eval_year}",
        "eval_discipline": EVAL_DISCIPLINE_NOTE,
        "sha256": {name: compute_sha256(path) for name, path in paths.items()},
    }
    (ARTIFACT_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return paths


def smoke_check_inference_path(paths: dict[str, Path]) -> float:
    """Load the artifact back the way inference does and run one prediction.

    Applies the same feature-schema-version gate as
    ``ModelLoader._validated_manifest_sha256`` (the AWS loader itself needs a
    SageMaker registry, so the metadata check + ``PredictionEngine`` joblib
    load is the full locally-exercisable inference path).
    """
    metadata = json.loads((ARTIFACT_DIR / "metadata.json").read_text())
    artifact_version = metadata.get(SCHEMA_VERSION_METADATA_KEY)
    if artifact_version != CURRENT_FEATURE_SCHEMA_VERSION:
        raise FeatureSchemaVersionMismatch(
            f"artifact has {SCHEMA_VERSION_METADATA_KEY}={artifact_version!r}; "
            f"runtime requires {CURRENT_FEATURE_SCHEMA_VERSION!r}"
        )

    engine = PredictionEngine()
    engine.load_model_artifacts(
        model_path=str(paths["model"]),
        preprocessor_path=str(paths["preprocessor"]),
        scaler_path=str(paths["scaler"]),
        model_metadata={"model_id": "local-tmdb-1980-2026", "version": 1},
    )
    response = engine.predict(
        PredictionRequest(
            budget=150_000_000,
            runtime=130,
            genre=["Action", "Adventure"],
            release_month=6,
            release_year=2025,
            mpaa="PG-13",
            director="Christopher Nolan",
            actors=["Tom Cruise", "Emily Blunt"],
            production_company="Warner Bros. Pictures",
        )
    )
    prediction = float(response.prediction)
    if not (np.isfinite(prediction) and prediction > 0):
        raise AssertionError(
            f"smoke prediction is not a finite dollar value: {prediction}"
        )
    return prediction


def main() -> None:
    args = parse_args()
    end_eval_year = HOLDOUT_END_YEAR if args.burn_holdout else ITERATION_END_YEAR
    if args.burn_holdout:
        print(
            "WARNING: evaluating the spent 2024-2025 confirmation set "
            "(--i-know-this-burns-the-holdout).",
            file=sys.stderr,
        )
    if not FRAME_PATH.exists():
        raise SystemExit(
            f"training frame not found: {FRAME_PATH}. "
            "Run `uv run python scripts/prepare_training_frame.py` first."
        )

    frame = pd.read_parquet(FRAME_PATH)
    X_input = frame[list(PREPROCESSOR_INPUT_COLUMNS)]
    y_log = pd.Series(
        np.log1p(frame["WORLDWIDE_GROSS"].astype(float)), name="GROSS_LOG"
    )
    dates = frame["RELEASE_YEAR"].astype(int)

    cv_results = run_cross_validation(X_input, y_log, dates, end_eval_year)

    # Deployment artifact: one preprocessor fit on ALL data (correct for
    # serving — at prediction time every training row is legitimately past).
    preprocessor = FeaturePreprocessorHigh()
    X = preprocessor.fit_transform(X_input)
    assert list(X.columns) == list(SELECTED_FEATURES)

    # StandardScaler is NaN-aware (nanmean/nanvar); missing budgets stay NaN.
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)
    model = train_final_model(X_scaled, y_log, cv_results)

    paths = save_artifact(
        model,
        preprocessor,
        scaler,
        cv_results,
        n_rows=len(frame),
        end_eval_year=end_eval_year,
    )
    prediction = smoke_check_inference_path(paths)

    print(f"trained on {len(frame)} rows, {X.shape[1]} features ({list(X.columns)})")
    print(EVAL_DISCIPLINE_NOTE)
    print(
        f"CV ({START_EVAL_YEAR}-{end_eval_year}): mean MAE (log) = "
        f"{cv_results['mean_cv_mae']:.4f}, mean RMSLE = "
        f"{cv_results['mean_cv_rmsle']:.4f}, mean best iteration = "
        f"{cv_results['mean_best_iteration']:.0f}"
    )
    print(f"artifact saved under {ARTIFACT_DIR}/")
    print(
        "inference smoke check passed: loaded via PredictionEngine, "
        f"predicted ${prediction:,.0f}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
