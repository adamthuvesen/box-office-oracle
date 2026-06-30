"""
Prediction engine for the serverless inference API.
Handles feature preprocessing, input validation, and model inference.
"""

import ast
import logging
import numpy as np
import pandas as pd
import joblib
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Union
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_serializer,
    field_validator,
    ValidationError,
)
import json

from box_office.ml.feature_schema import FeatureContractMismatch

logger = logging.getLogger(__name__)


def _parse_json_or_python_repr_str_list(raw: str) -> List[str]:
    """Parse a JSON array or Python repr list string (e.g. \"['a','b']\") into strings."""
    s = raw.strip()
    if s == "" or s == "[]":
        return []
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


def _normalize_str_list(value: Any) -> List[str]:
    if isinstance(value, str):
        return _parse_json_or_python_repr_str_list(value)
    if isinstance(value, list):
        return [str(x) for x in value]
    return [] if value is None else [str(value)]


class PredictionRequest(BaseModel):
    """Input schema for prediction requests."""

    model_config = {"protected_namespaces": ()}

    # Core movie features
    budget: float = Field(..., ge=0, description="Production budget in USD")
    runtime: float = Field(..., ge=1, le=500, description="Movie runtime in minutes")
    genre: Union[str, List[str]] = Field(..., description="Movie genre(s)")
    release_month: int = Field(..., ge=1, le=12, description="Release month (1-12)")
    release_year: int = Field(..., ge=1900, le=2030, description="Release year")

    # Optional features with defaults
    ad_budget: Optional[float] = Field(
        default=0, ge=0, description="Advertising budget in USD"
    )
    rating: Optional[float] = Field(default=5.0, ge=0, le=10, description="IMDB rating")
    votes: Optional[float] = Field(
        default=100, ge=0, description="Number of IMDB votes"
    )
    franchise_rating: Optional[float] = Field(
        default=0, ge=0, description="Franchise rating"
    )

    # Categorical features
    mpaa: Optional[str] = Field(default="Not Rated", description="MPAA rating")
    director: Optional[str] = Field(default="Unknown", description="Director name")
    actors: Optional[Union[str, List[str]]] = Field(
        default=[], description="List of main actors"
    )
    production_company: Optional[str] = Field(
        default="Unknown", description="Production company"
    )

    # Additional options
    return_confidence: Optional[bool] = Field(
        default=True, description="Return confidence interval"
    )
    model_version: Optional[str] = Field(
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
    prediction_interval_heuristic: Optional[List[float]] = Field(
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
    metrics: Dict[str, float]
    framework: str = "scikit-learn"


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
        model_metadata: Dict[str, Any],
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
                    "created_at", datetime.now(timezone.utc).isoformat()
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
            raise RuntimeError(f"Model loading failed: {str(e)}")

    def is_loaded(self) -> bool:
        """Check if model is loaded and ready for predictions."""
        return (
            self._is_loaded and self.model is not None and self.preprocessor is not None
        )

    def get_model_info(self) -> Optional[ModelInfo]:
        """Get current model information."""
        return self.model_info

    def validate_input(self, request_data: Dict[str, Any]) -> PredictionRequest:
        try:
            return PredictionRequest(**request_data)
        except ValidationError as e:
            logger.warning(f"Input validation failed: {str(e)}")
            raise

    def _prepare_dataframe(self, request: PredictionRequest) -> pd.DataFrame:
        release_date = f"{request.release_year}-{request.release_month:02d}-01"

        data = {
            "RELEASE_YEAR": request.release_year,
            "RATING": request.rating,
            "VOTES": request.votes,
            "AD_BUDGET": request.ad_budget,
            "PRODUCTION_BUDGET": request.budget,
            "FRANCHISE_RATING": request.franchise_rating,
            "RUNTIME": request.runtime,
            "MPAA": request.mpaa,
            # Pass genre/actors lists through directly. The preprocessor's
            # _normalize_to_list helper accepts list inputs and skips
            # ast.literal_eval entirely. The previous str(list) round-trip
            # broke on apostrophes (e.g. "Children's") because str([...]) emits
            # mixed quotes that literal_eval rejects.
            "GENRES": list(request.genre) if request.genre else [],
            "RELEASE_DATE": release_date,
            "DIRECTOR": request.director,
            "PRODUCTION_COMPANY": request.production_company,
            "ACTORS": list(request.actors) if request.actors else [],
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
            divergence = next(
                (f"{a!r}!={e!r}" for a, e in zip(actual, expected) if a != e),
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
            raise RuntimeError(f"Feature preprocessing failed: {str(e)}")

    def predict(self, request: PredictionRequest) -> PredictionResponse:
        if not self.is_loaded():
            raise RuntimeError("Model not loaded. Call load_model_artifacts first.")

        start_time = datetime.now(timezone.utc)

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

            processing_time = (
                datetime.now(timezone.utc) - start_time
            ).total_seconds() * 1000

            response = PredictionResponse(
                prediction=float(prediction_value),
                model_id=self.model_info.model_id,
                model_version=self.model_info.version,
                prediction_interval_heuristic=prediction_interval_heuristic,
                timestamp=datetime.now(timezone.utc).isoformat() + "Z",
                processing_time_ms=processing_time,
            )

            logger.info(
                f"Prediction completed: ${prediction_value:,.0f} (processing time: {processing_time:.2f}ms)"
            )
            return response

        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}", exc_info=True)
            raise RuntimeError(f"Prediction failed: {str(e)}")

    @staticmethod
    def _prediction_interval_heuristic(prediction_value: float) -> List[float]:
        std_error = prediction_value * (0.20 if prediction_value < 1000000 else 0.15)
        return [
            max(0, prediction_value - 1.96 * std_error),
            prediction_value + 1.96 * std_error,
        ]
