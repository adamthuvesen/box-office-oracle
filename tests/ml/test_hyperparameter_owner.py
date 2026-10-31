"""Every training path takes its XGBoost hyperparameters from config.ModelHyperparameters."""

import json
import logging
import sys
from unittest.mock import patch

import pytest

from box_office.config import MODEL_HYPERPARAMETERS
from box_office.ml import model_training
from box_office.ml.model import BoxOfficeXGBoostModel
from box_office.sagemaker.sagemaker_train_job import _validate_hyperparameters

_MODEL_PARAMS = MODEL_HYPERPARAMETERS.model_dump(exclude={"early_stopping_rounds"})


def test_model_defaults_come_from_the_owner():
    params = BoxOfficeXGBoostModel().params
    assert {k: params[k] for k in _MODEL_PARAMS} == _MODEL_PARAMS


def test_container_argparse_defaults_come_from_the_owner():
    with patch.object(sys, "argv", ["model_training.py"]):
        args = model_training.parse_args()
    assert {k: getattr(args, k) for k in MODEL_HYPERPARAMETERS.model_dump()} == (
        MODEL_HYPERPARAMETERS.model_dump()
    )


@pytest.mark.parametrize(
    "override",
    [{"max_depth": 0}, {"learning_rate": 2.0}, {"max_depth": 4.5}],
)
def test_training_job_validation_uses_the_owner_ranges(override):
    hyperparameters = {**MODEL_HYPERPARAMETERS.model_dump(), **override}
    with pytest.raises(ValueError, match="Invalid hyperparameters"):
        _validate_hyperparameters(hyperparameters, logging.getLogger(__name__))


def test_container_reads_sagemaker_hyperparameters_file(tmp_path, monkeypatch):
    path = tmp_path / "hyperparameters.json"
    path.write_text(json.dumps({"n_estimators": "300", "learning_rate": "0.05"}))
    monkeypatch.setattr(model_training, "SAGEMAKER_HYPERPARAMETERS_PATH", path)
    with patch.object(sys, "argv", ["train"]):
        args = model_training.parse_args()
    assert (args.n_estimators, args.learning_rate) == (300, 0.05)
    assert args.max_depth == MODEL_HYPERPARAMETERS.max_depth
