"""
Prediction engine for the serverless inference API.
Handles feature preprocessing, input validation, and model inference.
"""

import ast
import json
import logging
from datetime import UTC, datetime
from typing import Any, Self

import joblib
import numpy as np
import pandas as pd
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_serializer,
    field_validator,
)

from box_office.ml.feature_schema import FeatureContractMismatch
from box_office.ml.text_utils import MAX_LITERAL_EVAL_BYTES, LiteralEvalTooLarge

logger = logging.getLogger(__name__)


def _parse_json_or_python_repr_str_list(raw: str) -> list[str]:
    """Parse a JSON array or Python repr list string (e.g. \"['a','b']\") into strings."""
    s = raw.strip()
    if s == "" or s == "[]":
        return []
    # Bound the input before it reaches ast.literal_eval, which is recursive and
    # can be driven to exhaust CPU/memory by a deeply-nested attacker payload.
    # LiteralEvalTooLarge is a ValueError, so it surfaces as a 400 via pydantic.
    if len(s.encode("utf-8")) > MAX_LITERAL_EVAL_BYTES:
        raise LiteralEvalTooLarge(
            f"Input exceeds MAX_LITERAL_EVAL_BYTES={MAX_LITERAL_EVAL_BYTES}"
        )
    for parser in (json.loads, ast.literal_eval):
        try:
            parsed = parser(s)
        except (
            json.JSONDecodeError,
            ValueError,
            SyntaxError,
            TypeError,
        ):
            continue
        if isinstance(parsed, list):
            return [str(x) for x in parsed]
        if isinstance(parsed, str):
            return [parsed]
    return [s]


def _normalize_str_list(value: Any) -> list[str]:
    if isinstance(value, str):
        return _parse_json_or_python_repr_str_list(value)
    if isinstance(value, list):
        return [str(x) for x in value]
    return [] if value is None else [str(value)]


class PredictionRequest(BaseModel):
    """Input schema for prediction requests."""

    model_config = ConfigDict(protected_namespaces=(), extra="forbid")

    # Core movie features
    budget: float = Field(..., ge=0, description="Production budget in USD")
    runtime: float = Field(..., ge=1, le=500, description="Movie runtime in minutes")
    genre: str | list[str] = Field(..., description="Movie genre(s)")
    release_month: int = Field(..., ge=1, le=12, description="Release month (1-12)")
    release_year: int = Field(..., ge=1900, le=2030, description="Release year")

    # Categorical features
    mpaa: str | None = Field(default="Not Rated", description="MPAA rating")
    director: str | None = Field(default="Unknown", description="Director name")
    actors: str | list[str] | None = Field(
        default=[], description="List of main actors"
    )
    production_company: str | None = Field(
        default="Unknown", description="Production company"
    )

    # Pre-release IP/franchise strength (defaults describe an original movie
    # with no pre-sold IP and no earlier films in its collection).
    ip_tier: int = Field(
        default=5,
        ge=1,
        le=5,
        description="Pre-sold IP tier at release (1 = strongest, 5 = no IP)",
    )
    prior_franchise_gross: float = Field(
        default=0.0,
        ge=0,
        description=(
            "Sum of worldwide gross (USD) of strictly-earlier films in the "
            "same collection; 0 for a first film or no collection"
        ),
    )
    is_franchise_followup: bool = Field(
        default=False,
        description="Whether any strictly-earlier film exists in the collection",
    )

    # Additional options
    return_confidence: bool | None = Field(
        default=True, description="Return confidence interval"
    )
    model_version: str | None = Field(
        default=None, description="Specific model version to use"
    )

    @field_validator("genre")
    @classmethod
    def validate_genre(cls, v):
        """Convert genre to list format."""
        return _normalize_str_list(v)

    @field_validator("actors")
    @classmethod
    def validate_actors(cls, v):
        """Convert actors to list format."""
        return _normalize_str_list(v)


class PredictionResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    prediction: float = Field(..., description="Predicted box office gross in USD")
    model_id: str = Field(..., description="Model identifier used for prediction")
    model_version: int = Field(..., description="Model version used")
    prediction_interval_heuristic: list[float] | None = Field(
        default=None,
        description=(
            "Heuristic interval (lower, upper) around the point prediction. "
            "NOT a calibrated confidence interval — derived from a fixed "
            "percentage of the predicted value, not from model uncertainty."
        ),
    )
    timestamp: str = Field(..., description="Prediction timestamp")
    processing_time_ms: float = Field(
        ..., description="Processing time in milliseconds"
    )

    @field_serializer("prediction", "processing_time_ms")
    def _serialize_floats(self, value: float) -> float:
        """Coerce numpy float scalars to native Python float for JSON output."""
        return float(value)

    @field_serializer("model_version")
    def _serialize_ints(self, value: int) -> int:
        """Coerce numpy int scalars to native Python int for JSON output."""
        return int(value)


class ModelInfo(BaseModel):
    model_config = {"protected_namespaces": ()}
    model_id: str
    version: int
    status: str
    created_at: str
    metrics: dict[str, float]
    framework: str = "scikit-learn"

    @classmethod
    def from_registry_package(cls, package: dict[str, Any]) -> Self:
        """Normalize a SageMaker model package summary for API and runtime use."""
        created_at = package.get("CreationTime")
        return cls(
            model_id=package.get("ModelPackageArn", "unknown"),
            version=package.get("ModelPackageVersion", 1),
            status=package.get("ModelApprovalStatus", "unknown"),
            created_at=(
                created_at.isoformat()
                if hasattr(created_at, "isoformat")
                else str(created_at or "unknown")
            ),
            metrics=package.get("metrics", {}),
            framework=package.get("framework", "unknown"),
        )


class PredictionEngine:
    def __init__(self):
        self.model = None
        self.preprocessor = None
        self.scaler = None
        self.model_info = None
        self._is_loaded = False

    def load_model_artifacts(
        self,
        model_path: str,
        preprocessor_path: str,
        scaler_path: str,
        model_metadata: dict[str, Any],
    ):
        try:
            logger.info(f"Loading model artifacts from {model_path}")

            self.model = joblib.load(model_path)
            logger.info("Model loaded successfully")

            self.preprocessor = joblib.load(preprocessor_path)
            logger.info("Feature preprocessor loaded successfully")

            self.scaler = joblib.load(scaler_path)
            logger.info("Feature scaler loaded successfully")

            self.model_info = ModelInfo(
                model_id=model_metadata.get("model_id", "unknown"),
                version=model_metadata.get("version", 1),
                status=model_metadata.get("status", "unknown"),
                created_at=model_metadata.get(
                    "created_at", datetime.now(UTC).isoformat()
                ),
                metrics=model_metadata.get("metrics", {}),
                framework=model_metadata.get("framework", "scikit-learn"),
            )

            self._is_loaded = True
            logger.info(
                f"Model {self.model_info.model_id} v{self.model_info.version} loaded successfully"
            )

        except Exception as e:
            logger.error(f"Failed to load model artifacts: {str(e)}", exc_info=True)
            raise RuntimeError(f"Model loading failed: {str(e)}") from e

    def is_loaded(self) -> bool:
        """Check if model is loaded and ready for predictions."""
        return (
            self._is_loaded and self.model is not None and self.preprocessor is not None
        )

    def get_model_info(self) -> ModelInfo | None:
        """Get current model information."""
        return self.model_info

    def validate_input(self, request_data: dict[str, Any]) -> PredictionRequest:
        try:
            return PredictionRequest(**request_data)
        except ValidationError as e:
            logger.warning(f"Input validation failed: {str(e)}")
            raise

    def _prepare_dataframe(self, request: PredictionRequest) -> pd.DataFrame:
        release_date = f"{request.release_year}-{request.release_month:02d}-01"

        data = {
            "RELEASE_YEAR": request.release_year,
            "PRODUCTION_BUDGET": request.budget,
            "RUNTIME": request.runtime,
            "MPAA": request.mpaa,
            # Pass genre/actors lists through directly; the preprocessor accepts
            # list inputs and keeps apostrophes inside names intact.
            "GENRES": list(request.genre) if request.genre else [],
            "RELEASE_DATE": release_date,
            "DIRECTOR": request.director,
            "PRODUCTION_COMPANY": request.production_company,
            "ACTORS": list(request.actors) if request.actors else [],
            "IP_TIER": float(request.ip_tier),
            # Training stores this column already log1p-transformed
            # (scripts/prepare_training_frame.py); the request carries raw
            # dollars, so apply the same log1p here.
            "PRIOR_FRANCHISE_GROSS_LOG": float(np.log1p(request.prior_franchise_gross)),
            "IS_FRANCHISE_FOLLOWUP": 1.0 if request.is_franchise_followup else 0.0,
        }

        df = pd.DataFrame([data])
        logger.debug(f"Prepared DataFrame with shape {df.shape}")
        return df

    def _assert_feature_contract(self, features: pd.DataFrame) -> None:
        """Fail loudly when preprocessor output and the model disagree.

        Guards the training↔serving contract at predict time: the schema-version
        check at load handles wrong-vintage artifacts, this catches an artifact
        whose components were assembled inconsistently — before XGBoost raises
        a cryptic shape error deep in ``predict``.
        """
        expected = self.preprocessor.get_feature_names()
        actual = list(features.columns)
        if actual != expected:
            # A length mismatch must not reach zip(strict=True) — that raises a
            # bare ValueError before the FeatureContractMismatch below. Report
            # the length divergence explicitly; only scan for a token mismatch
            # when the two sequences are the same length.
            if len(actual) != len(expected):
                divergence = "<length>"
            else:
                divergence = next(
                    (
                        f"{a!r}!={e!r}"
                        for a, e in zip(actual, expected, strict=True)
                        if a != e
                    ),
                    "<length>",
                )
            raise FeatureContractMismatch(
                f"Preprocessor produced {len(actual)} features but the artifact "
                f"contract declares {len(expected)} (first divergence: {divergence}). "
                "Retrain and re-register so training and serving agree."
            )
        n_model = getattr(self.model, "n_features_in_", None)
        if n_model is not None and n_model != len(actual):
            raise FeatureContractMismatch(
                f"Model expects {n_model} features but the preprocessor produced "
                f"{len(actual)}. Artifact components disagree; retrain and re-register."
            )

    def preprocess_features(self, request: PredictionRequest) -> np.ndarray:
        """Preprocess features for prediction."""
        if not self.is_loaded():
            raise RuntimeError("Model not loaded. Call load_model_artifacts first.")

        try:
            df = self._prepare_dataframe(request)

            logger.debug("Applying feature preprocessing")
            features_processed = self.preprocessor.transform(df)
            self._assert_feature_contract(features_processed)

            logger.debug("Applying feature scaling")
            features_scaled = self.scaler.transform(features_processed)

            logger.debug(f"Preprocessed features shape: {features_scaled.shape}")
            return features_scaled

        except FeatureContractMismatch:
            raise
        except Exception as e:
            logger.error(f"Feature preprocessing failed: {str(e)}", exc_info=True)
            raise RuntimeError(f"Feature preprocessing failed: {str(e)}") from e

    def predict(self, request: PredictionRequest) -> PredictionResponse:
        if not self.is_loaded():
            raise RuntimeError("Model not loaded. Call load_model_artifacts first.")

        start_time = datetime.now(UTC)

        try:
            features = self.preprocess_features(request)

            logger.debug("Making model prediction")
            prediction = self.model.predict(features)[0]
            prediction_value = np.expm1(prediction)  # Inverse of log1p

            prediction_interval_heuristic = None
            if request.return_confidence:
                prediction_interval_heuristic = self._prediction_interval_heuristic(
                    prediction_value
                )

            processing_time = (datetime.now(UTC) - start_time).total_seconds() * 1000

            response = PredictionResponse(
                prediction=float(prediction_value),
                model_id=self.model_info.model_id,
                model_version=self.model_info.version,
                prediction_interval_heuristic=prediction_interval_heuristic,
                timestamp=datetime.now(UTC).isoformat() + "Z",
                processing_time_ms=processing_time,
            )

            logger.info(
                f"Prediction completed: ${prediction_value:,.0f} (processing time: {processing_time:.2f}ms)"
            )
            return response

        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}", exc_info=True)
            raise RuntimeError(f"Prediction failed: {str(e)}") from e

    @staticmethod
    def _prediction_interval_heuristic(prediction_value: float) -> list[float]:
        std_error = prediction_value * (0.20 if prediction_value < 1000000 else 0.15)
        return [
            max(0, prediction_value - 1.96 * std_error),
            prediction_value + 1.96 * std_error,
        ]
