"""
Unit tests for PredictionEngine class.

Tests feature preprocessing, input validation, model inference,
and error handling with comprehensive mocking.
"""

import os
import pickle
import tempfile
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

# Import the module under test
from box_office.inference.app.predictor import (
    ModelInfo,
    PredictionEngine,
    PredictionRequest,
    PredictionResponse,
)
from box_office.ml.feature_pipeline.constants import SELECTED_FEATURES

_BUDGET_IDX = list(SELECTED_FEATURES).index("PRODUCTION_BUDGET")


class MockModel:
    """Mock ML model for testing."""

    def __init__(self, prediction_value=12.5):  # Log-transformed value
        self.prediction_value = prediction_value
        self.n_features_in_ = len(SELECTED_FEATURES)

    def predict(self, X):
        """Mock predict method that returns log-transformed values."""
        if hasattr(X, "shape"):
            return np.array([self.prediction_value] * X.shape[0])
        return np.array([self.prediction_value])


class MockPreprocessor:
    """Mock feature preprocessor for testing."""

    def get_feature_names(self):
        return list(SELECTED_FEATURES)

    def transform(self, df):
        """Mock transform: a deterministic frame keyed by the feature contract."""
        cols = list(SELECTED_FEATURES)
        return pd.DataFrame(np.ones((len(df), len(cols))), columns=cols)


class MockScaler:
    """Mock feature scaler for testing."""

    def transform(self, features):
        """Mock transform that returns scaled features as an ndarray."""
        return np.asarray(features, dtype=float) * 0.5


class TestPredictionRequest:
    """Test cases for PredictionRequest validation."""

    def test_valid_request_minimal(self):
        """Test valid request with minimal required fields."""
        request_data = {
            "budget": 50000000,
            "runtime": 120,
            "genre": "Action",
            "release_month": 6,
            "release_year": 2024,
        }

        request = PredictionRequest(**request_data)

        assert request.budget == 50000000
        assert request.runtime == 120
        assert request.genre == ["Action"]  # Should be converted to list
        assert request.release_month == 6
        assert request.release_year == 2024
        assert request.return_confidence is True
        # IP/franchise defaults describe an original movie with no pre-sold IP.
        assert request.ip_tier == 5
        assert request.prior_franchise_gross == 0.0
        assert request.is_franchise_followup is False

    def test_ip_franchise_fields(self):
        """Optional IP/franchise fields validate range and pass through."""
        base = {
            "budget": 50000000,
            "runtime": 120,
            "genre": "Action",
            "release_month": 6,
            "release_year": 2024,
        }

        request = PredictionRequest(
            **base,
            ip_tier=1,
            prior_franchise_gross=2_000_000_000,
            is_franchise_followup=True,
        )
        assert request.ip_tier == 1
        assert request.prior_franchise_gross == 2_000_000_000
        assert request.is_franchise_followup is True

        for bad in ({"ip_tier": 0}, {"ip_tier": 6}, {"prior_franchise_gross": -1}):
            with pytest.raises(ValidationError):
                PredictionRequest(**base, **bad)

    def test_valid_request_full(self):
        """Test valid request with all fields."""
        request_data = {
            "budget": 100000000,
            "runtime": 150,
            "genre": ["Action", "Adventure"],
            "release_month": 12,
            "release_year": 2024,
            "mpaa": "PG-13",
            "director": "Christopher Nolan",
            "actors": ["Tom Hardy", "Anne Hathaway"],
            "production_company": "Warner Bros",
            "return_confidence": False,
            "model_version": "v2.1",
        }

        request = PredictionRequest(**request_data)

        assert request.budget == 100000000
        assert request.genre == ["Action", "Adventure"]
        assert request.actors == ["Tom Hardy", "Anne Hathaway"]
        assert request.director == "Christopher Nolan"
        assert request.return_confidence is False

    def test_genre_string_conversion(self):
        """Test genre conversion from string to list."""
        request_data = {
            "budget": 50000000,
            "runtime": 120,
            "genre": "Comedy",
            "release_month": 6,
            "release_year": 2024,
        }

        request = PredictionRequest(**request_data)
        assert request.genre == ["Comedy"]

        # Genre as list string
        request_data["genre"] = "['Action', 'Comedy']"
        request = PredictionRequest(**request_data)
        assert request.genre == ["Action", "Comedy"]

    def test_actors_string_conversion(self):
        """Test actors conversion from string to list."""
        request_data = {
            "budget": 50000000,
            "runtime": 120,
            "genre": "Action",
            "release_month": 6,
            "release_year": 2024,
            "actors": "['Tom Cruise', 'Emily Blunt']",
        }

        request = PredictionRequest(**request_data)
        assert request.actors == ["Tom Cruise", "Emily Blunt"]

        request_data["actors"] = "[]"
        request = PredictionRequest(**request_data)
        assert request.actors == []

        request_data["actors"] = "Tom Cruise"
        request = PredictionRequest(**request_data)
        assert request.actors == ["Tom Cruise"]

    def test_invalid_budget_negative(self):
        """Test validation error for negative budget."""
        request_data = {
            "budget": -1000000,
            "runtime": 120,
            "genre": "Action",
            "release_month": 6,
            "release_year": 2024,
        }

        with pytest.raises(ValidationError) as exc_info:
            PredictionRequest(**request_data)

        errors = exc_info.value.errors()
        assert any(error["loc"] == ("budget",) for error in errors)

    def test_invalid_runtime_out_of_range(self):
        """Test validation error for invalid runtime."""
        request_data = {
            "budget": 50000000,
            "runtime": 600,
            "genre": "Action",
            "release_month": 6,
            "release_year": 2024,
        }

        with pytest.raises(ValidationError) as exc_info:
            PredictionRequest(**request_data)

        errors = exc_info.value.errors()
        assert any(error["loc"] == ("runtime",) for error in errors)

    def test_invalid_release_month(self):
        """Test validation error for invalid release month."""
        request_data = {
            "budget": 50000000,
            "runtime": 120,
            "genre": "Action",
            "release_month": 13,
            "release_year": 2024,
        }

        with pytest.raises(ValidationError) as exc_info:
            PredictionRequest(**request_data)

        errors = exc_info.value.errors()
        assert any(error["loc"] == ("release_month",) for error in errors)

    def test_post_release_fields_rejected(self):
        """Post-release/leakage request fields are rejected."""
        request_data = {
            "budget": 50000000,
            "runtime": 120,
            "genre": "Action",
            "release_month": 6,
            "release_year": 2024,
            "rating": 8.0,
            "votes": 1000,
            "franchise_rating": 2,
        }

        with pytest.raises(ValidationError) as exc_info:
            PredictionRequest(**request_data)

        errors = exc_info.value.errors()
        rejected = {error["loc"][0] for error in errors}
        assert {"rating", "votes", "franchise_rating"}.issubset(rejected)

    def test_missing_required_fields(self):
        """Test validation error for missing required fields."""
        request_data = {"budget": 50000000}

        with pytest.raises(ValidationError) as exc_info:
            PredictionRequest(**request_data)

        errors = exc_info.value.errors()
        required_fields = {"runtime", "genre", "release_month", "release_year"}
        error_fields = {error["loc"][0] for error in errors}
        assert required_fields.issubset(error_fields)


class TestPredictionResponse:
    """Test cases for PredictionResponse model."""

    def test_valid_response(self):
        """Test valid prediction response creation."""
        response_data = {
            "prediction": 125000000.0,
            "model_id": "model_003",
            "model_version": 3,
            "prediction_interval_heuristic": [100000000.0, 150000000.0],
            "timestamp": "2025-01-24T10:30:00Z",
            "processing_time_ms": 45.2,
        }

        response = PredictionResponse(**response_data)

        assert response.prediction == 125000000.0
        assert response.model_id == "model_003"
        assert response.model_version == 3
        assert response.prediction_interval_heuristic == [100000000.0, 150000000.0]
        assert response.timestamp == "2025-01-24T10:30:00Z"
        assert response.processing_time_ms == 45.2

    def test_response_without_prediction_interval_heuristic(self):
        """Test response without confidence interval."""
        response_data = {
            "prediction": 125000000.0,
            "model_id": "model_003",
            "model_version": 3,
            "timestamp": "2025-01-24T10:30:00Z",
            "processing_time_ms": 45.2,
        }

        response = PredictionResponse(**response_data)
        assert response.prediction_interval_heuristic is None

    def test_numpy_type_conversion(self):
        """Test numpy type conversion in response."""
        response_data = {
            "prediction": np.float64(125000000.0),
            "model_id": "model_003",
            "model_version": np.int32(3),
            "timestamp": "2025-01-24T10:30:00Z",
            "processing_time_ms": np.float32(45.2),
        }

        response = PredictionResponse(**response_data)

        assert isinstance(response.prediction, float)
        assert isinstance(response.model_version, int)
        assert isinstance(response.processing_time_ms, float)


class TestModelInfo:
    """Test cases for ModelInfo model."""

    def test_valid_model_info(self):
        """Test valid model info creation."""
        info_data = {
            "model_id": "box-office-model-v3",
            "version": 3,
            "status": "Approved",
            "created_at": "2025-01-24T10:00:00Z",
            "metrics": {"rmse": 0.15, "mae": 0.12, "r2": 0.85},
            "framework": "xgboost",
        }

        info = ModelInfo(**info_data)

        assert info.model_id == "box-office-model-v3"
        assert info.version == 3
        assert info.status == "Approved"
        assert info.metrics == {"rmse": 0.15, "mae": 0.12, "r2": 0.85}
        assert info.framework == "xgboost"


class TestPredictionEngine:
    """Test cases for PredictionEngine class."""

    @pytest.fixture
    def mock_model(self):
        """Mock ML model for testing."""
        return MockModel(prediction_value=12.5)  # Log-transformed value

    @pytest.fixture
    def mock_preprocessor(self):
        """Mock feature preprocessor for testing."""
        return MockPreprocessor()

    @pytest.fixture
    def mock_scaler(self):
        """Mock feature scaler for testing."""
        return MockScaler()

    @pytest.fixture
    def sample_model_metadata(self):
        """Sample model metadata for testing."""
        return {
            "model_id": "test-model-v1",
            "version": 1,
            "status": "Approved",
            "created_at": "2025-01-24T10:00:00Z",
            "metrics": {"rmse": 0.15, "mae": 0.12},
            "framework": "xgboost",
        }

    @pytest.fixture
    def prediction_engine(self):
        """Create a fresh PredictionEngine instance for testing."""
        return PredictionEngine()

    @pytest.fixture
    def sample_request(self):
        """Sample prediction request for testing."""
        return PredictionRequest(
            budget=50000000,
            runtime=120,
            genre=["Action"],
            release_month=6,
            release_year=2024,
            mpaa="PG-13",
            director="Test Director",
            actors=["Actor 1", "Actor 2"],
            production_company="Test Studios",
        )

    def test_init(self, prediction_engine):
        """Test PredictionEngine initialization."""
        assert prediction_engine.model is None
        assert prediction_engine.preprocessor is None
        assert prediction_engine.scaler is None
        assert prediction_engine.model_info is None
        assert prediction_engine._is_loaded is False

    def test_load_model_artifacts_success(
        self,
        prediction_engine,
        mock_model,
        mock_preprocessor,
        mock_scaler,
        sample_model_metadata,
    ):
        """Test successful model artifacts loading."""
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = os.path.join(temp_dir, "model.pkl")
            preprocessor_path = os.path.join(temp_dir, "preprocessor.pkl")
            scaler_path = os.path.join(temp_dir, "scaler.pkl")

            with open(model_path, "wb") as f:
                pickle.dump(mock_model, f)
            with open(preprocessor_path, "wb") as f:
                pickle.dump(mock_preprocessor, f)
            with open(scaler_path, "wb") as f:
                pickle.dump(mock_scaler, f)

            prediction_engine.load_model_artifacts(
                model_path=model_path,
                preprocessor_path=preprocessor_path,
                scaler_path=scaler_path,
                model_metadata=sample_model_metadata,
            )

            assert prediction_engine.model is not None
            assert prediction_engine.preprocessor is not None
            assert prediction_engine.scaler is not None
            assert prediction_engine.model_info is not None
            assert prediction_engine._is_loaded is True
            assert prediction_engine.model_info.model_id == "test-model-v1"
            assert prediction_engine.model_info.version == 1

    def test_load_model_artifacts_file_not_found(
        self, prediction_engine, sample_model_metadata
    ):
        """Test model artifacts loading failure with missing files."""
        with pytest.raises(RuntimeError, match="Model loading failed"):
            prediction_engine.load_model_artifacts(
                model_path="/nonexistent/model.pkl",
                preprocessor_path="/nonexistent/preprocessor.pkl",
                scaler_path="/nonexistent/scaler.pkl",
                model_metadata=sample_model_metadata,
            )

    def test_is_loaded(
        self, prediction_engine, mock_model, mock_preprocessor, mock_scaler
    ):
        """Test is_loaded method."""
        assert prediction_engine.is_loaded() is False

        prediction_engine.model = mock_model
        prediction_engine.preprocessor = mock_preprocessor
        prediction_engine.scaler = mock_scaler
        prediction_engine._is_loaded = True

        assert prediction_engine.is_loaded() is True

    def test_get_model_info(self, prediction_engine, sample_model_metadata):
        """Test get_model_info method."""
        assert prediction_engine.get_model_info() is None

        prediction_engine.model_info = ModelInfo(**sample_model_metadata)

        info = prediction_engine.get_model_info()
        assert info is not None
        assert info.model_id == "test-model-v1"
        assert info.version == 1

    def test_validate_input_success(self, prediction_engine):
        """Test successful input validation."""
        request_data = {
            "budget": 50000000,
            "runtime": 120,
            "genre": "Action",
            "release_month": 6,
            "release_year": 2024,
        }

        validated = prediction_engine.validate_input(request_data)

        assert isinstance(validated, PredictionRequest)
        assert validated.budget == 50000000
        assert validated.genre == ["Action"]

    def test_validate_input_failure(self, prediction_engine):
        """Test input validation failure."""
        request_data = {
            "budget": -1000000,
            "runtime": 120,
            "genre": "Action",
            "release_month": 6,
            "release_year": 2024,
        }

        with pytest.raises(ValidationError):
            prediction_engine.validate_input(request_data)

    def test_prepare_dataframe(self, prediction_engine, sample_request):
        """Test DataFrame preparation from request."""
        df = prediction_engine._prepare_dataframe(sample_request)

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1
        assert "RELEASE_YEAR" in df.columns
        assert "PRODUCTION_BUDGET" in df.columns
        assert "RATING" not in df.columns
        assert "VOTES" not in df.columns
        assert "FRANCHISE_RATING" not in df.columns
        assert "RUNTIME" in df.columns
        assert "SOCIAL_MEDIA_BUZZ" not in df.columns
        assert "MPAA" in df.columns
        assert "GENRES" in df.columns
        assert "RELEASE_DATE" in df.columns
        assert "DIRECTOR" in df.columns
        assert "PRODUCTION_COMPANY" in df.columns
        assert "ACTORS" in df.columns

        # Check values
        assert df.iloc[0]["RELEASE_YEAR"] == 2024
        assert df.iloc[0]["PRODUCTION_BUDGET"] == 50000000
        assert df.iloc[0]["RELEASE_DATE"] == "2024-06-01"
        # Lists are passed through directly instead of round-tripping through
        # str(list) and ast.literal_eval.
        assert df.iloc[0]["GENRES"] == ["Action"]
        assert df.iloc[0]["ACTORS"] == ["Actor 1", "Actor 2"]
        # IP/franchise defaults: original movie, no prior franchise history.
        assert df.iloc[0]["IP_TIER"] == 5.0
        assert df.iloc[0]["PRIOR_FRANCHISE_GROSS_LOG"] == 0.0
        assert df.iloc[0]["IS_FRANCHISE_FOLLOWUP"] == 0.0

    def test_prepare_dataframe_logs_prior_franchise_gross(self, prediction_engine):
        """The request carries raw dollars; the frame carries log1p(dollars)."""
        request = PredictionRequest(
            budget=50_000_000,
            runtime=120,
            genre="Action",
            release_month=6,
            release_year=2024,
            ip_tier=2,
            prior_franchise_gross=1_000_000_000,
            is_franchise_followup=True,
        )
        df = prediction_engine._prepare_dataframe(request)
        assert df.iloc[0]["IP_TIER"] == 2.0
        assert df.iloc[0]["PRIOR_FRANCHISE_GROSS_LOG"] == pytest.approx(
            np.log1p(1_000_000_000)
        )
        assert df.iloc[0]["IS_FRANCHISE_FOLLOWUP"] == 1.0

    def test_preprocess_features_success(
        self,
        prediction_engine,
        mock_model,
        mock_preprocessor,
        mock_scaler,
        sample_request,
    ):
        """Test successful feature preprocessing."""
        prediction_engine.model = mock_model
        prediction_engine.preprocessor = mock_preprocessor
        prediction_engine.scaler = mock_scaler
        prediction_engine._is_loaded = True

        features = prediction_engine.preprocess_features(sample_request)

        assert isinstance(features, np.ndarray)
        assert features.shape[0] == 1
        assert features.shape[1] == len(SELECTED_FEATURES)

    def test_preprocess_features_contract_mismatch_raises(
        self, prediction_engine, mock_model, mock_scaler, sample_request
    ):
        """A preprocessor output off-contract fails loudly, not as a shape error."""
        from box_office.ml.feature_schema import FeatureContractMismatch

        off_contract = MockPreprocessor()
        off_contract.transform = lambda df: pd.DataFrame(
            np.ones((len(df), 3)), columns=["A", "B", "C"]
        )
        prediction_engine.model = mock_model
        prediction_engine.preprocessor = off_contract
        prediction_engine.scaler = mock_scaler
        prediction_engine._is_loaded = True

        with pytest.raises(FeatureContractMismatch):
            prediction_engine.preprocess_features(sample_request)

    def test_preprocess_features_not_loaded(self, prediction_engine, sample_request):
        """Test feature preprocessing failure when model not loaded."""
        with pytest.raises(RuntimeError, match="Model not loaded"):
            prediction_engine.preprocess_features(sample_request)

    def test_preprocess_features_error(self, prediction_engine, sample_request):
        """Feature preprocessing error handling.

        The predictor calls ``self.preprocessor.transform(df)``, not the
        preprocessor mock directly. Attaching the side_effect to ``transform``
        ensures this test exercises the preprocessing-failure branch.
        """
        prediction_engine.model = MockModel()
        prediction_engine.preprocessor = Mock(
            transform=Mock(side_effect=Exception("Preprocessing failed"))
        )
        prediction_engine.scaler = MockScaler()
        prediction_engine._is_loaded = True

        with pytest.raises(RuntimeError, match="Feature preprocessing failed"):
            prediction_engine.preprocess_features(sample_request)

    def test_predict_success(
        self,
        prediction_engine,
        mock_model,
        mock_preprocessor,
        mock_scaler,
        sample_request,
        sample_model_metadata,
    ):
        """Test successful prediction."""
        prediction_engine.model = mock_model
        prediction_engine.preprocessor = mock_preprocessor
        prediction_engine.scaler = mock_scaler
        prediction_engine.model_info = ModelInfo(**sample_model_metadata)
        prediction_engine._is_loaded = True

        response = prediction_engine.predict(sample_request)

        assert isinstance(response, PredictionResponse)
        assert response.prediction > 0
        assert response.model_id == "test-model-v1"
        assert response.model_version == 1
        assert response.prediction_interval_heuristic is not None
        assert len(response.prediction_interval_heuristic) == 2
        assert (
            response.prediction_interval_heuristic[0]
            < response.prediction_interval_heuristic[1]
        )
        assert response.processing_time_ms > 0
        assert response.timestamp is not None

    def test_predict_without_confidence(
        self,
        prediction_engine,
        mock_model,
        mock_preprocessor,
        mock_scaler,
        sample_model_metadata,
    ):
        """Test prediction without confidence interval."""
        prediction_engine.model = mock_model
        prediction_engine.preprocessor = mock_preprocessor
        prediction_engine.scaler = mock_scaler
        prediction_engine.model_info = ModelInfo(**sample_model_metadata)
        prediction_engine._is_loaded = True

        request = PredictionRequest(
            budget=50000000,
            runtime=120,
            genre=["Action"],
            release_month=6,
            release_year=2024,
            return_confidence=False,
        )

        response = prediction_engine.predict(request)

        assert response.prediction_interval_heuristic is None

    def test_predict_not_loaded(self, prediction_engine, sample_request):
        """Test prediction failure when model not loaded."""
        with pytest.raises(RuntimeError, match="Model not loaded"):
            prediction_engine.predict(sample_request)

    def test_predict_model_error(
        self,
        prediction_engine,
        mock_preprocessor,
        mock_scaler,
        sample_request,
        sample_model_metadata,
    ):
        """Test prediction error handling."""
        failing_model = Mock(side_effect=Exception("Model prediction failed"))
        prediction_engine.model = failing_model
        prediction_engine.preprocessor = mock_preprocessor
        prediction_engine.scaler = mock_scaler
        prediction_engine.model_info = ModelInfo(**sample_model_metadata)
        prediction_engine._is_loaded = True

        with pytest.raises(RuntimeError, match="Prediction failed"):
            prediction_engine.predict(sample_request)

    def test_prediction_interval_heuristic_calculation_error(
        self,
        prediction_engine,
        mock_model,
        mock_preprocessor,
        mock_scaler,
        sample_request,
        sample_model_metadata,
    ):
        """NaN model output is propagated as NaN, not silently coerced to 0.

        ``pd.isna`` makes the test fail if the predictor ever stops propagating
        NaN from the underlying model.
        """
        import pandas as pd

        prediction_engine.model = mock_model
        prediction_engine.preprocessor = mock_preprocessor
        prediction_engine.scaler = mock_scaler
        prediction_engine.model_info = ModelInfo(**sample_model_metadata)
        prediction_engine._is_loaded = True

        # Force the model to emit NaN; the predictor should propagate NaN
        # rather than coerce to 0 (or any sentinel).
        mock_model.prediction_value = float("nan")

        response = prediction_engine.predict(sample_request)
        assert isinstance(response, PredictionResponse)
        assert pd.isna(response.prediction), (
            f"Expected NaN propagation, got {response.prediction!r}"
        )


class TestRuntimeEngine:
    """The runtime exposes a single shared PredictionEngine across warm starts."""

    def test_runtime_is_singleton(self):
        from box_office.inference.app import runtime as app_runtime

        app_runtime._runtime = None

        runtime1 = app_runtime.get_runtime()
        runtime2 = app_runtime.get_runtime()

        assert runtime1 is runtime2
        assert isinstance(runtime1._engine, PredictionEngine)

    def test_engine_starts_unloaded(self):
        from box_office.inference.app import runtime as app_runtime

        app_runtime._runtime = None

        engine = app_runtime.get_runtime()._engine

        assert isinstance(engine, PredictionEngine)
        assert not engine.is_loaded()


# Realistic components for integration testing (defined at module level for pickling)
class RealisticModel:
    """Realistic model for integration testing."""

    n_features_in_ = len(SELECTED_FEATURES)

    def predict(self, X):
        budget_feature = X[0, _BUDGET_IDX] if X.shape[1] > _BUDGET_IDX else 50000000
        log_prediction = np.log1p(budget_feature * 2.5)
        return np.array([log_prediction])


class RealisticPreprocessor:
    """Realistic preprocessor for integration testing."""

    def get_feature_names(self):
        return list(SELECTED_FEATURES)

    def transform(self, df):
        # Emit the full SELECTED_FEATURES contract; populate the columns this
        # mock knows, zero-fill the rest. Order matches the contract.
        row = {c: 0.0 for c in SELECTED_FEATURES}
        for col in ("PRODUCTION_BUDGET",):
            if col in df.columns:
                row[col] = df[col].iloc[0]
        return pd.DataFrame([row], columns=list(SELECTED_FEATURES))


class RealisticScaler:
    """Realistic scaler for integration testing (identity-preserving)."""

    def transform(self, features):
        return np.asarray(features, dtype=float)


class TestPredictionEngineIntegration:
    """Integration tests for PredictionEngine with realistic scenarios."""

    def test_full_prediction_workflow(self):
        """Test complete prediction workflow with realistic data."""
        engine = PredictionEngine()

        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = os.path.join(temp_dir, "model.pkl")
            preprocessor_path = os.path.join(temp_dir, "preprocessor.pkl")
            scaler_path = os.path.join(temp_dir, "scaler.pkl")

            with open(model_path, "wb") as f:
                pickle.dump(RealisticModel(), f)
            with open(preprocessor_path, "wb") as f:
                pickle.dump(RealisticPreprocessor(), f)
            with open(scaler_path, "wb") as f:
                pickle.dump(RealisticScaler(), f)

            model_metadata = {
                "model_id": "realistic-model-v1",
                "version": 1,
                "status": "Approved",
                "created_at": "2025-01-24T10:00:00Z",
                "metrics": {"rmse": 0.15, "mae": 0.12, "r2": 0.85},
                "framework": "xgboost",
            }

            engine.load_model_artifacts(
                model_path=model_path,
                preprocessor_path=preprocessor_path,
                scaler_path=scaler_path,
                model_metadata=model_metadata,
            )

            request_data = {
                "budget": 100000000,
                "runtime": 150,
                "genre": ["Action", "Adventure"],
                "release_month": 6,
                "release_year": 2024,
                "mpaa": "PG-13",
                "director": "Christopher Nolan",
                "actors": ["Christian Bale", "Tom Hardy"],
                "production_company": "Warner Bros",
            }

            validated_request = engine.validate_input(request_data)
            response = engine.predict(validated_request)

            assert isinstance(response, PredictionResponse)
            assert response.prediction > 0
            assert response.prediction > 100000
            assert response.model_id == "realistic-model-v1"
            assert response.model_version == 1
            assert response.prediction_interval_heuristic is not None
            assert len(response.prediction_interval_heuristic) == 2
            assert response.processing_time_ms > 0

            model_info = engine.get_model_info()
            assert model_info.model_id == "realistic-model-v1"
            assert model_info.metrics["rmse"] == pytest.approx(0.15)

    def test_edge_case_inputs(self):
        """Test prediction with edge case inputs."""
        engine = PredictionEngine()

        edge_cases = [
            # Minimum budget
            {
                "budget": 1,
                "runtime": 1,
                "genre": "Documentary",
                "release_month": 1,
                "release_year": 1900,
            },
            # Maximum values
            {
                "budget": 500000000,
                "runtime": 500,
                "genre": ["Action", "Adventure", "Comedy", "Drama"],
                "release_month": 12,
                "release_year": 2030,
            },
            # Empty optional fields
            {
                "budget": 50000000,
                "runtime": 120,
                "genre": "Horror",
                "release_month": 10,
                "release_year": 2024,
                "actors": [],
                "director": "",
                "production_company": "",
            },
        ]

        for case in edge_cases:
            try:
                validated = engine.validate_input(case)
                assert isinstance(validated, PredictionRequest)

                df = engine._prepare_dataframe(validated)
                assert isinstance(df, pd.DataFrame)
                assert len(df) == 1

            except ValidationError as e:
                # Some edge cases might fail validation, which is expected
                assert len(e.errors()) > 0
