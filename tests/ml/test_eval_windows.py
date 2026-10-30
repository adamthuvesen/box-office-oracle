"""No tuning or CV path selects a confirmation-set year unless the caller burns the holdout."""

import os
import sys
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from pydantic import ValidationError

from box_office.config import CROSS_VALIDATION_WINDOW, CrossValidationWindow
from box_office.eval_windows import (
    CONFIRMATION_START_YEAR,
    CONFIRMATION_YEARS,
    ConfirmationSetTouched,
)
from box_office.ml import model_training
from box_office.ml.cv import TimeSeriesCrossValidator
from box_office.orchestration.tasks import data_tasks
from box_office.orchestration.tasks.data_tasks import split_data


@pytest.mark.parametrize("end_year", CONFIRMATION_YEARS)
def test_cv_refuses_a_window_that_reaches_the_confirmation_set(end_year):
    with pytest.raises(ConfirmationSetTouched):
        TimeSeriesCrossValidator(start_eval_year=2015, end_year=end_year)


def test_cv_scores_the_confirmation_set_only_when_the_holdout_is_burned():
    cv = TimeSeriesCrossValidator(end_year=2025, allow_confirmation_years=True)
    assert cv.end_year == 2025


def test_default_cv_window_ends_before_the_confirmation_set():
    assert TimeSeriesCrossValidator().end_year < CONFIRMATION_START_YEAR
    assert CROSS_VALIDATION_WINDOW.end_year < CONFIRMATION_START_YEAR


def test_cloud_config_refuses_a_confirmation_year():
    with pytest.raises(ValidationError, match="confirmation set"):
        CrossValidationWindow(end_year=CONFIRMATION_START_YEAR)


def test_training_container_defaults_to_the_iteration_window():
    env = {k: v for k, v in os.environ.items() if k not in {"end_year", "start_eval_year"}}
    with patch.dict(os.environ, env, clear=True), patch.object(sys, "argv", ["x"]):
        args = model_training.parse_args()
    assert (args.start_eval_year, args.end_year, args.cv_folds) == (
        CROSS_VALIDATION_WINDOW.start_eval_year,
        CROSS_VALIDATION_WINDOW.end_year,
        CROSS_VALIDATION_WINDOW.cv_folds,
    )


def test_cloud_split_keeps_the_confirmation_set_out_of_training(monkeypatch):
    monkeypatch.setattr(data_tasks, "get_run_logger", MagicMock)
    frame = pd.DataFrame({"RELEASE_YEAR": range(2018, 2026), "WORLDWIDE_GROSS": 1.0})
    X_train, X_val, _, _ = split_data.fn(frame, "WORLDWIDE_GROSS")
    assert X_train["RELEASE_YEAR"].max() < CONFIRMATION_START_YEAR
    assert set(CONFIRMATION_YEARS) <= set(X_val["RELEASE_YEAR"])
