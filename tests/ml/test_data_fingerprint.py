"""The data fingerprint identifies the exact training rows a model saw."""

from unittest.mock import MagicMock

import pandas as pd
import pytest

from box_office.inference.app.runtime import ModelRuntime
from box_office.ml.data_fingerprint import DataFingerprint, fingerprint_training_frame


@pytest.fixture
def frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "RELEASE_DATE": ["2019-03-01", "2023-11-24", "2021-07-09"],
            "GENRES": [["Action"], ["Drama", "Comedy"], []],
            "PRODUCTION_BUDGET": [5e7, float("nan"), 1.2e8],
            "WORLDWIDE_GROSS": [1.5e8, 3e7, 4.4e8],
        }
    )


def test_fingerprint_records_rows_and_latest_release(frame):
    fingerprint = fingerprint_training_frame(frame)
    assert (fingerprint.rows, fingerprint.max_release_date) == (3, "2023-11-24")
    assert fingerprint == fingerprint_training_frame(frame.copy())


@pytest.mark.parametrize(
    "change",
    [
        lambda f: f.assign(WORLDWIDE_GROSS=f["WORLDWIDE_GROSS"] + 1),
        lambda f: f.iloc[::-1],
        lambda f: f.iloc[:2],
        lambda f: f[list(reversed(f.columns))],
    ],
)
def test_any_change_to_the_rows_changes_the_hash(frame, change):
    assert (
        fingerprint_training_frame(change(frame)).content_sha256
        != fingerprint_training_frame(frame).content_sha256
    )


def test_registry_metadata_round_trips(frame):
    fingerprint = fingerprint_training_frame(frame)
    metadata = fingerprint.as_registry_metadata()
    assert all(isinstance(value, str) for value in metadata.values())
    assert DataFingerprint.from_registry_metadata(metadata) == fingerprint


def test_packages_without_a_fingerprint_read_as_none():
    assert DataFingerprint.from_registry_metadata({"sha256": "x"}) is None


def test_runtime_loads_the_fingerprint_into_the_model_info(frame):
    fingerprint = fingerprint_training_frame(frame)
    loader = MagicMock()
    loader.refresh_model_if_needed.return_value = True
    loader.get_current_model.return_value = (object(), {"ModelPackageArn": "arn:pkg"})
    loader.get_model_artifacts_paths.return_value = {
        "model": "m",
        "preprocessor": "p",
        "scaler": "s",
    }
    loader.get_data_fingerprint.return_value = fingerprint
    engine = MagicMock()

    ModelRuntime(loader, engine).ensure_ready()

    assert (
        engine.load_model_artifacts.call_args.kwargs["model_info"].data_fingerprint == fingerprint
    )
