"""Shared fixtures for inference tests."""

import pickle
import tempfile
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

from box_office.ml.feature_pipeline.constants import SELECTED_FEATURES


@pytest.fixture
def valid_prediction_payload() -> dict:
    return {
        "budget": 50_000_000,
        "runtime": 120,
        "genre": "Action",
        "release_month": 6,
        "release_year": 2024,
    }


@pytest.fixture
def mock_model():
    model = Mock()
    model.n_features_in_ = len(SELECTED_FEATURES)

    def predict(X):
        n = X.shape[0] if hasattr(X, "shape") else 1
        return np.array([12.5] * n)

    model.predict = predict
    return model


@pytest.fixture
def mock_preprocessor():
    pre = Mock()
    # Mirror the real contract: a DataFrame keyed by SELECTED_FEATURES so the
    # predictor's feature-contract guard sees the expected names and width.
    cols = list(SELECTED_FEATURES)
    pre.get_feature_names = Mock(return_value=cols)
    pre.transform = Mock(
        return_value=pd.DataFrame(np.ones((1, len(cols))), columns=cols)
    )
    return pre


@pytest.fixture
def mock_scaler():
    scaler = Mock()
    scaler.transform = Mock(side_effect=lambda x: x * 0.5)
    return scaler


@pytest.fixture
def artifact_paths(mock_model, mock_preprocessor, mock_scaler):
    with tempfile.TemporaryDirectory() as temp_dir:
        paths = {
            "model": f"{temp_dir}/model.pkl",
            "preprocessor": f"{temp_dir}/feature_preprocessor.pkl",
            "scaler": f"{temp_dir}/feature_scaler.pkl",
        }
        with open(paths["model"], "wb") as f:
            pickle.dump(mock_model, f)
        with open(paths["preprocessor"], "wb") as f:
            pickle.dump(mock_preprocessor, f)
        with open(paths["scaler"], "wb") as f:
            pickle.dump(mock_scaler, f)
        yield paths
