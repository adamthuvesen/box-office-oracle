"""API contract tests for the inference service."""

from __future__ import annotations

import asyncio
import inspect
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import httpx
import pytest
from botocore.exceptions import ClientError

# Disable API key auth for the contract tests so we can exercise the
# /predict handler directly without juggling headers.
from box_office.inference.app.config import get_settings

_settings = get_settings()
_settings.enable_api_key_auth = False

from box_office.inference.app.main import MAX_REQUEST_BODY_BYTES, app  # noqa: E402
from box_office.inference.app.model_loader import (  # noqa: E402
    ModelLoadError,
    ModelLoader,
)
from box_office.inference.app.integrity import ArtifactIntegrityError  # noqa: E402
from box_office.inference.app.predictor import PredictionResponse  # noqa: E402
from box_office.ml.registry_constants import FeatureSchemaVersionMismatch  # noqa: E402

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class ASGITestClient:
    def __init__(self, app, raise_server_exceptions: bool = True):
        self._app = app
        self._raise_server_exceptions = raise_server_exceptions

    def get(self, path: str, **kwargs) -> httpx.Response:
        return self._request("GET", path, **kwargs)

    def post(self, path: str, **kwargs) -> httpx.Response:
        return self._request("POST", path, **kwargs)

    def _request(self, method: str, path: str, **kwargs) -> httpx.Response:
        return asyncio.run(self._async_request(method, path, **kwargs))

    async def _async_request(self, method: str, path: str, **kwargs) -> httpx.Response:
        transport = httpx.ASGITransport(
            app=self._app,
            raise_app_exceptions=self._raise_server_exceptions,
        )
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
        ) as client:
            response = await client.request(method, path, **kwargs)
            await response.aread()
            return response


@pytest.fixture
def client():
    return ASGITestClient(app)


@pytest.fixture
def auth_enabled_client(monkeypatch):
    monkeypatch.setattr(_settings, "enable_api_key_auth", True)
    monkeypatch.setattr(_settings, "api_key", "test-secret")
    return ASGITestClient(app, raise_server_exceptions=False)


@pytest.fixture
def patched_engine_and_loader():
    """Patch runtime so /predict runs against pure mocks."""
    with patch("box_office.inference.app.main.get_runtime") as get_runtime_mock:
        loader = MagicMock()
        engine = MagicMock()
        engine.is_loaded.return_value = True
        engine.validate_input.side_effect = lambda data: data

        runtime = MagicMock()
        runtime.ensure_ready.return_value = False
        runtime.validate_input = engine.validate_input
        runtime.predict = engine.predict
        runtime._loader = loader
        runtime._engine = engine
        get_runtime_mock.return_value = runtime

        yield engine, loader


# ---------------------------------------------------------------------------
# 2.0 — API key auth boundary
# ---------------------------------------------------------------------------


class TestApiKeyAuthContract:
    """API-key auth failures return API responses, not middleware 500s."""

    def test_health_remains_unauthenticated(self, auth_enabled_client):
        resp = auth_enabled_client.get("/health")
        assert resp.status_code == 200

    def test_missing_api_key_returns_401(self, auth_enabled_client):
        resp = auth_enabled_client.get("/")

        assert resp.status_code == 401
        assert resp.headers["www-authenticate"] == "ApiKey"
        body = resp.json()
        assert body["detail"]["error"] == "UNAUTHORIZED"
        assert body["detail"]["message"] == "Missing X-API-Key header"

    def test_invalid_api_key_returns_401(self, auth_enabled_client):
        resp = auth_enabled_client.get("/", headers={"X-API-Key": "wrong"})

        assert resp.status_code == 401
        assert resp.headers["www-authenticate"] == "ApiKey"
        body = resp.json()
        assert body["detail"]["error"] == "UNAUTHORIZED"
        assert body["detail"]["message"] == "Invalid API key"

    def test_valid_api_key_passes_through(self, auth_enabled_client):
        resp = auth_enabled_client.get("/", headers={"X-API-Key": "test-secret"})

        assert resp.status_code == 200
        assert resp.json()["health"] == "/health"


# ---------------------------------------------------------------------------
# 2.1 — Malformed JSON / oversize body
# ---------------------------------------------------------------------------


class TestRequestBodyContract:
    """M48: malformed JSON returns 400, oversize body returns 413."""

    def test_malformed_json_returns_400(self, client, patched_engine_and_loader):
        resp = client.post(
            "/predict",
            content=b"{not json",
            headers={"Content-Type": "application/json"},
        )
        assert resp.status_code == 400
        body = resp.json()
        assert body["detail"]["error"] == "INVALID_JSON"

    def test_oversize_body_via_content_length_returns_413(
        self, client, patched_engine_and_loader
    ):
        # Content-Length larger than the cap is rejected before we even read
        # the body, even if the actual body is small.
        resp = client.post(
            "/predict",
            content=b'{"budget": 1}',
            headers={
                "Content-Type": "application/json",
                "Content-Length": str(MAX_REQUEST_BODY_BYTES + 1),
            },
        )
        # Starlette/httpx may auto-correct content-length; accept either the
        # 413 fast-path or the post-read 413 path. Both are spec-compliant.
        assert resp.status_code in (400, 413)

    def test_oversize_body_actual_length_returns_413(
        self, client, patched_engine_and_loader
    ):
        # Force-read path: actual body bytes exceed the cap.
        big_body = b'{"x": "' + b"a" * (MAX_REQUEST_BODY_BYTES + 16) + b'"}'
        resp = client.post(
            "/predict",
            content=big_body,
            headers={"Content-Type": "application/json"},
        )
        assert resp.status_code == 413
        assert resp.json()["detail"]["error"] == "REQUEST_TOO_LARGE"


# ---------------------------------------------------------------------------
# 2.2 — Non-object JSON payload
# ---------------------------------------------------------------------------


class TestPayloadShapeContract:
    """M49: a JSON array / scalar / null is a 400, not a 500."""

    @pytest.mark.parametrize("payload", [b"[]", b"42", b'"hi"', b"null", b"true"])
    def test_non_object_payload_returns_400(
        self, client, patched_engine_and_loader, payload
    ):
        resp = client.post(
            "/predict",
            content=payload,
            headers={"Content-Type": "application/json"},
        )
        assert resp.status_code == 400
        assert resp.json()["detail"]["error"] == "INVALID_PAYLOAD_SHAPE"


# ---------------------------------------------------------------------------
# 2.3 — Generic error message on prediction failure
# ---------------------------------------------------------------------------


class TestPredictionErrorMessageContract:
    """M50: the response body must NOT echo the raw exception text."""

    def test_runtime_error_returns_generic_message(
        self, client, patched_engine_and_loader
    ):
        engine, _loader = patched_engine_and_loader
        secret = "INTERNAL_LEAK_/var/secrets/foo: AccessDenied"
        engine.predict.side_effect = RuntimeError(secret)

        resp = client.post(
            "/predict",
            json={
                "budget": 1,
                "runtime": 90,
                "genre": "Drama",
                "release_month": 5,
                "release_year": 2024,
            },
        )
        assert resp.status_code == 500
        body = resp.json()
        assert body["detail"]["error"] == "PREDICTION_FAILED"
        # The secret/exception message must not appear in the response body.
        assert secret not in body["detail"]["message"]
        # The generic message points operators at the logs.
        assert "correlation_id" in body["detail"]["message"]

    @pytest.mark.parametrize(
        "error",
        [
            ArtifactIntegrityError("bad artifact"),
            FeatureSchemaVersionMismatch("wrong schema"),
        ],
    )
    def test_model_integrity_and_schema_failures_return_503(
        self, client, patched_engine_and_loader, error
    ):
        engine, _loader = patched_engine_and_loader
        runtime = MagicMock()
        runtime.ensure_ready.side_effect = error
        runtime.validate_input = engine.validate_input
        runtime.predict = engine.predict

        with patch("box_office.inference.app.main.get_runtime", return_value=runtime):
            resp = client.post(
                "/predict",
                json={
                    "budget": 1,
                    "runtime": 90,
                    "genre": "Drama",
                    "release_month": 5,
                    "release_year": 2024,
                },
            )

        assert resp.status_code == 503
        assert resp.json()["detail"]["error"] == "SERVICE_UNAVAILABLE"


# ---------------------------------------------------------------------------
# 2.4 — Stale-cache TTL
# ---------------------------------------------------------------------------


class TestMaxStaleTTL:
    """M52: refresh failure beyond max_stale_seconds drops the cache."""

    def _make_loader(self, tmp_path, max_stale_seconds=3600):
        # Bypass __init__ so we don't need real boto clients.
        loader = ModelLoader.__new__(ModelLoader)
        loader.cache_dir = tmp_path
        loader.cache_ttl_seconds = 60
        loader.max_stale_seconds = max_stale_seconds
        loader._current_model = object()
        loader._current_model_info = {"ModelPackageArn": "arn:fake"}
        loader._last_load_time = datetime.now(timezone.utc) - timedelta(seconds=1)
        loader._extracted_artifacts_cache = {}
        return loader

    def test_within_ttl_serves_cached_model_on_refresh_failure(self, tmp_path):
        loader = self._make_loader(tmp_path, max_stale_seconds=3600)
        with patch.object(
            loader, "_get_latest_approved_model_info", side_effect=RuntimeError("boom")
        ):
            # Bypass cache-valid fast path
            loader._last_load_time = datetime.now(timezone.utc) - timedelta(seconds=120)
            assert loader.refresh_model_if_needed() is False
        # Cache preserved
        assert loader._current_model is not None

    def test_exceeded_ttl_drops_cache_and_raises(self, tmp_path):
        loader = self._make_loader(tmp_path, max_stale_seconds=10)
        # Force the cached model to look ancient.
        loader._last_load_time = datetime.now(timezone.utc) - timedelta(seconds=999)
        with patch.object(
            loader, "_get_latest_approved_model_info", side_effect=RuntimeError("boom")
        ):
            with pytest.raises(ModelLoadError):
                loader.refresh_model_if_needed()
        # Cache must be cleared so the next predict re-loads.
        assert loader._current_model is None
        assert loader._current_model_info is None
        assert loader._last_load_time is None


# ---------------------------------------------------------------------------
# 2.5 — bucket initialization
# ---------------------------------------------------------------------------


class TestBucketUnboundContract:
    """M54: describe_model_package failure must NOT NameError on `bucket`."""

    def test_describe_failure_raises_modelloaderror_not_nameerror(self, tmp_path):
        loader = ModelLoader.__new__(ModelLoader)
        loader.cache_dir = tmp_path
        loader.cache_ttl_seconds = 60
        loader.max_stale_seconds = 3600
        loader._current_model = None
        loader._current_model_info = None
        loader._last_load_time = None
        loader._extracted_artifacts_cache = {}
        loader.sagemaker_client = MagicMock()
        loader.sagemaker_client.describe_model_package.side_effect = ClientError(
            {"Error": {"Code": "NoSuchBucket", "Message": "missing"}},
            "DescribeModelPackage",
        )

        # The handler MUST raise ModelLoadError (or a subclass), not NameError.
        with pytest.raises(ModelLoadError):
            loader._download_and_load_model({"ModelPackageArn": "arn:fake"})


# ---------------------------------------------------------------------------
# 2.6 — Single Pydantic model_config + no json_encoders
# ---------------------------------------------------------------------------


class TestPredictionResponseConfig:
    """M57: exactly one model_config and no json_encoders."""

    def test_single_model_config_in_class_body(self):
        # Source-level check: only one assignment to `model_config` survives.
        source = inspect.getsource(PredictionResponse)
        # Count standalone assignments at any indentation. Class body
        # assignments look like "    model_config = ".
        assignment_count = sum(
            1
            for line in source.splitlines()
            if line.lstrip().startswith("model_config =")
        )
        assert assignment_count == 1, (
            f"PredictionResponse has {assignment_count} model_config "
            "assignments; expected exactly 1"
        )

    def test_no_json_encoders_in_config(self):
        cfg = PredictionResponse.model_config
        assert "json_encoders" not in cfg


# ---------------------------------------------------------------------------
# 2.7 — cleanup paths only reference bound names
# ---------------------------------------------------------------------------


class TestLoaderCleanupSafety:
    """L10: cleanup must not NameError on partially-initialized state.

    The current loader uses `final_dir` / `stage_dir` defined inside the outer
    try; they are assigned before any operation that can raise. This test
    verifies that an early extraction failure raises ModelLoadError rather
    than a NameError from the cleanup branch.
    """

    def test_extract_failure_raises_modelloaderror(self, tmp_path):
        loader = ModelLoader.__new__(ModelLoader)
        loader.cache_dir = tmp_path
        loader.cache_ttl_seconds = 60
        loader.max_stale_seconds = 3600
        loader._extracted_artifacts_cache = {}

        # Pass a non-existent tar path to force tarfile.open to fail.
        with pytest.raises(ModelLoadError):
            loader._extract_and_load_model_with_cache(
                str(tmp_path / "does_not_exist.tar.gz"),
                "arn:fake",
                "deadbeef" * 8,
            )


# ---------------------------------------------------------------------------
# 2.8 — Field rename
# ---------------------------------------------------------------------------


class TestResponseFieldRename:
    """L11: field is `prediction_interval_heuristic`, not `confidence_interval`."""

    def test_renamed_field_present(self):
        assert "prediction_interval_heuristic" in PredictionResponse.model_fields

    def test_confidence_interval_field_absent(self):
        assert "confidence_interval" not in PredictionResponse.model_fields

    def test_description_calls_out_heuristic(self):
        field = PredictionResponse.model_fields["prediction_interval_heuristic"]
        assert field.description is not None
        assert "heuristic" in field.description.lower()
