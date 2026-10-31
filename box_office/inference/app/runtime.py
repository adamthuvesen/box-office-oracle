"""Unified inference runtime: registry load, artifact cache, and prediction."""

from __future__ import annotations

import logging
from typing import Any

from box_office.inference.app.model_loader import (
    ModelLoader,
    ModelLoadError,
)
from box_office.inference.app.predictor import (
    ModelInfo,
    PredictionEngine,
    PredictionRequest,
    PredictionResponse,
)

logger = logging.getLogger(__name__)

_runtime: ModelRuntime | None = None


class ModelRuntime:
    """Single entry point for Lambda inference: load approved model and predict."""

    def __init__(self, loader: ModelLoader, engine: PredictionEngine | None = None):
        self._loader = loader
        self._engine = engine or PredictionEngine()

    def ensure_ready(self) -> bool:
        """Refresh model if needed and load artifacts into the prediction engine."""
        refreshed = self._loader.refresh_model_if_needed()
        current = self._loader.get_current_model()
        if not current:
            raise ModelLoadError("No approved model available")

        if refreshed or not self._engine.is_loaded():
            paths = self._loader.get_model_artifacts_paths()
            self._engine.load_model_artifacts(
                model_path=paths["model"],
                preprocessor_path=paths["preprocessor"],
                scaler_path=paths["scaler"],
                model_info=ModelInfo.from_registry_package(
                    current[1], data_fingerprint=self._loader.get_data_fingerprint()
                ),
            )
        return refreshed

    def validate_input(self, request_data: dict[str, Any]) -> PredictionRequest:
        return self._engine.validate_input(request_data)

    def predict(self, request: PredictionRequest) -> PredictionResponse:
        return self._engine.predict(request)

    def get_model_info(self):
        return self._engine.get_model_info()

    def describe_model(self) -> tuple[ModelInfo, bool] | None:
        """The loaded model's info, else the latest approved package's; and whether it is loaded."""
        if self._engine.is_loaded():
            return self._engine.get_model_info(), True
        package = self._loader.get_latest_approved_model_info()
        if not package:
            return None
        return ModelInfo.from_registry_package(package), False


def get_runtime() -> ModelRuntime:
    """Global runtime for Lambda warm starts."""
    global _runtime
    if _runtime is None:
        from box_office.inference.app.config import get_settings

        settings = get_settings()
        loader = ModelLoader(
            model_package_group_name=settings.model_registry_group_name,
            aws_region=settings.aws_region,
            cache_ttl_seconds=settings.model_cache_ttl,
            max_stale_seconds=settings.max_stale_seconds,
        )
        _runtime = ModelRuntime(loader)
    return _runtime
