"""Typed test double for the inference loader. Imports nothing train-only (see the
"Inference tests import nothing train-only" contract in pyproject.toml)."""

from pathlib import Path
from unittest.mock import MagicMock

from box_office.inference.app.model_loader import ModelLoader


class FakeModelLoader(ModelLoader):
    def __init__(self) -> None:
        self.sagemaker_mock = MagicMock()
        self.s3_mock = MagicMock()
        self.sagemaker_client = self.sagemaker_mock
        self.s3_client = self.s3_mock
        self.cache_dir = Path("unused-cache-dir")
        self._extracted_artifacts_cache = {}
