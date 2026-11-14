"""Serve one set of trained artifacts through the inference app over ASGI and print what it answered.

Run inside a checkout copy: ``python serve_probe.py <artifact_dir> <api_key> <answers.json>``.
The loader is the boundary to S3, so a stand-in that points at the local artifact files replaces it.
Everything above it (auth, validation, the prediction engine, the response model) is the real code.
"""

import asyncio
import json
import sys
from pathlib import Path
from typing import Any, cast
from unittest.mock import patch

import httpx

from box_office.inference.app import main as app_main
from box_office.inference.app.model_loader import ModelLoader
from box_office.inference.app.runtime import ModelRuntime
from box_office.ml.artifacts import FEATURE_PREPROCESSOR_PKL, FEATURE_SCALER_PKL, MODEL_PKL
from box_office.ml.model_metadata import LocalModelMetadata

VALID_REQUEST: dict[str, Any] = {
    "budget": 50_000_000,
    "runtime": 120,
    "genre": "Action",
    "release_month": 6,
    "release_year": 2024,
}


class LocalArtifactLoader:
    """The five loader methods the runtime calls, answered from a local artifact directory."""

    def __init__(self, artifact_dir: Path) -> None:
        self._dir = artifact_dir
        metadata = LocalModelMetadata.model_validate_json(
            (artifact_dir / "metadata.json").read_text()
        )
        self._fingerprint = metadata.data_fingerprint

    def refresh_model_if_needed(self) -> bool:
        return False

    def get_current_model(self) -> tuple[object, dict[str, str]]:
        return object(), {"ModelPackageArn": "local-verify-model"}

    def get_model_artifacts_paths(self) -> dict[str, str]:
        return {
            "model": str(self._dir / MODEL_PKL),
            "preprocessor": str(self._dir / FEATURE_PREPROCESSOR_PKL),
            "scaler": str(self._dir / FEATURE_SCALER_PKL),
        }

    def get_data_fingerprint(self) -> Any:
        return self._fingerprint

    def get_latest_approved_model_info(self) -> None:
        return None


async def ask(client: httpx.AsyncClient, method: str, path: str, **kwargs: Any) -> dict[str, Any]:
    response = await client.request(method, path, **kwargs)
    return {"status": response.status_code, "body": response.json()}


async def probe(artifact_dir: Path, api_key: str) -> dict[str, Any]:
    runtime = ModelRuntime(cast("ModelLoader", LocalArtifactLoader(artifact_dir)))
    transport = httpx.ASGITransport(app=app_main.app, raise_app_exceptions=False)
    good = {"X-API-Key": api_key}
    with patch.object(app_main, "get_runtime", return_value=runtime):
        async with httpx.AsyncClient(transport=transport, base_url="http://verify") as client:
            return {
                "predict": await ask(client, "POST", "/predict", json=VALID_REQUEST, headers=good),
                "missing_key": await ask(client, "POST", "/predict", json=VALID_REQUEST),
                "wrong_key": await ask(
                    client, "POST", "/predict", json=VALID_REQUEST, headers={"X-API-Key": "wrong"}
                ),
                "runtime_out_of_bounds": await ask(
                    client,
                    "POST",
                    "/predict",
                    json={**VALID_REQUEST, "runtime": 9999},
                    headers=good,
                ),
                "month_out_of_bounds": await ask(
                    client,
                    "POST",
                    "/predict",
                    json={**VALID_REQUEST, "release_month": 13},
                    headers=good,
                ),
                "model_info": await ask(client, "GET", "/model/info", headers=good),
            }


if __name__ == "__main__":
    # The app logs to stdout, so the answers go to a file.
    Path(sys.argv[3]).write_text(json.dumps(asyncio.run(probe(Path(sys.argv[1]), sys.argv[2]))))
