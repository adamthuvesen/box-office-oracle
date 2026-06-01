"""Pipeline failure semantics: exit codes, retry boundaries, orchestrator/task contract."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest

from box_office.ml.exceptions import CrossValidationFailed
from box_office.ml.model import TimeSeriesCrossValidator
from box_office.utils.snowflake_loader import (
    SnowflakeLoader,
    validate_sql_identifier,
)


# ---------------------------------------------------------------------------
# CV loop failure handling (tasks 3.1 / 3.2 / 3.3)
# ---------------------------------------------------------------------------


class _FlakyModel:
    """Test double whose ``fit`` can raise based on the eval-set year."""

    def __init__(self, fail_when_eval_year=None, **kwargs):
        self._fail_when_eval_year = fail_when_eval_year
        self._kwargs = kwargs
        self.is_fitted = False

    def fit(self, X, y, eval_set=None, verbose=False):
        # The eval set carries the validation fold (one fold = one year in
        # our synthetic dataset). Fail if the validation y matches the
        # injection trigger.
        if self._fail_when_eval_year is not None and eval_set:
            _, y_val = eval_set[0]
            if len(y_val) > 0 and float(y_val.iloc[0]) == float(
                self._fail_when_eval_year
            ):
                raise RuntimeError(
                    f"injected failure for eval year {self._fail_when_eval_year}"
                )
        self.is_fitted = True
        return self

    def predict(self, X):
        return np.zeros(len(X))

    @property
    def feature_importances_(self):
        return np.zeros(2)

    @property
    def best_iteration(self):
        return 50


def _make_synthetic_cv_dataset(years=(2018, 2019, 2020, 2021)):
    """Build a tiny dataset where ``y_train`` per-row equals the eval year, so
    ``_FlakyModel.fit`` can decide whether to fail based on which fold it's in."""
    rows = []
    for year in years:
        for _ in range(20):
            rows.append({"year": year, "f1": 1.0, "f2": 2.0})
    df = pd.DataFrame(rows)
    X = df[["f1", "f2"]].copy()
    # y per row encodes the year as a float so _FlakyModel can compare.
    y = pd.Series(df["year"].astype(float).to_numpy())
    dates = pd.Series(df["year"].to_numpy())
    return X, y, dates


def test_cv_one_fold_fails_others_succeed(caplog):
    """A single fold raising must not abort the run; the failure is logged."""
    X, y, dates = _make_synthetic_cv_dataset()
    cv = TimeSeriesCrossValidator(
        cv_folds=10,
        start_eval_year=2018,
        end_year=2021,
        early_stopping_rounds=5,
    )

    with caplog.at_level(logging.ERROR, logger="box_office.ml.model"):
        results = cv.cross_validate(
            model_class=lambda **kw: _FlakyModel(fail_when_eval_year=2020, **kw),
            X_train=X,
            y_train_log=y,
            dates=dates,
        )

    fold_results = results["fold_results"]
    successful = [r for r in fold_results if r["error"] is None]
    failed = [r for r in fold_results if r["error"] is not None]
    # Year 2018 is the first eval year and has no training data before it,
    # so the validator skips it without recording a fold_result. That leaves
    # 2019 (success), 2020 (injected failure), 2021 (success).
    assert len(failed) == 1, f"expected 1 failure, got {failed}"
    assert failed[0]["eval_year"] == 2020
    assert len(successful) >= 2, f"expected >=2 successes, got {successful}"
    # The error log carried exc_info=True (full traceback in CloudWatch).
    assert any("CV fold" in rec.getMessage() for rec in caplog.records)


def test_cv_all_folds_fail_raises_with_cause():
    X, y, dates = _make_synthetic_cv_dataset()
    cv = TimeSeriesCrossValidator(
        cv_folds=10,
        start_eval_year=2018,
        end_year=2021,
        early_stopping_rounds=5,
    )

    class _AlwaysFails(_FlakyModel):
        def fit(self, X, y, eval_set=None, verbose=False):
            raise RuntimeError("always fails")

    with pytest.raises(CrossValidationFailed) as excinfo:
        cv.cross_validate(
            model_class=lambda **kw: _AlwaysFails(**kw),
            X_train=X,
            y_train_log=y,
            dates=dates,
        )

    assert excinfo.value.__cause__ is not None
    assert isinstance(excinfo.value.__cause__, RuntimeError)
    assert "always fails" in str(excinfo.value.__cause__)


# ---------------------------------------------------------------------------
# SQL identifier validation (task 3.6 / H12-H16)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "value",
    [
        "BOX_OFFICE",
        "STAGING",
        "stg_box_office",
        "_underscore_first",
        "A1B2C3",
    ],
)
def test_validate_sql_identifier_accepts_safe_names(value):
    assert validate_sql_identifier(value, "table") == value


@pytest.mark.parametrize(
    "value",
    [
        "BOX_OFFICE; DROP TABLE FOO",
        "1starts_with_digit",
        "has space",
        "a-dash",
        "',--",
        "",
        None,
    ],
)
def test_validate_sql_identifier_rejects_unsafe_names(value):
    with pytest.raises(ValueError):
        validate_sql_identifier(value, "table")


def test_snowflake_loader_init_rejects_compromised_database_env(monkeypatch):
    """A SnowflakeLoader constructed with an injection-y database string must
    raise at __init__, not at SQL-execute time."""
    with pytest.raises(ValueError):
        SnowflakeLoader(schema="RAW", database="BOX_OFFICE; DROP TABLE FOO")


def test_snowflake_loader_init_rejects_unsafe_schema():
    with pytest.raises(ValueError):
        SnowflakeLoader(schema="RAW; DROP TABLE FOO", database="BOX_OFFICE")


# ---------------------------------------------------------------------------
# Private-key fallback narrowing (task 3.8)
# ---------------------------------------------------------------------------


def test_private_key_fallback_only_on_missing_file(tmp_path, monkeypatch):
    """A missing key path falls back; any other key-load error re-raises.

    We pass ``use_browser_auth=True`` so the fallback path takes the
    well-defined external-browser branch instead of the password branch
    — the password branch depends on ``config.snowflake.password`` being
    set, which is environment-dependent (CI doesn't set SNOWFLAKE_PASSWORD).
    What we're actually testing is: does ``except FileNotFoundError`` set
    ``use_private_key = False``, and does ``except Exception`` re-raise?
    """
    from box_office.utils import snowflake_connection as sc

    real_key = tmp_path / "key.p8"
    real_key.write_text("ignored")  # path exists; loader is mocked

    # The connection helper resolves the key path from SNOWFLAKE_PRIVATE_KEY_PATH
    # (env), not the kwarg. Set it explicitly so the key-load branch runs
    # deterministically — otherwise the test only passes on hosts that happen to
    # have the env/.env configured (it skips key loading entirely without it).
    monkeypatch.setenv("SNOWFLAKE_PRIVATE_KEY_PATH", str(real_key))

    captured = {}

    def fake_connect(**kwargs):
        captured["params"] = kwargs

        class _Conn:
            def close(self):
                pass

        return _Conn()

    monkeypatch.setattr(sc.snowflake.connector, "connect", fake_connect)

    # Case A: load raises FileNotFoundError -> falls back. With
    # use_browser_auth=True, the fallback engages the external-browser
    # branch (no password lookup needed).
    monkeypatch.setattr(
        sc,
        "load_private_key_from_file",
        lambda *a, **kw: (_ for _ in ()).throw(
            FileNotFoundError("simulated missing key")
        ),
    )
    captured.clear()
    result = sc.create_snowflake_connection(
        schema="ML_TRAINING",
        use_private_key=True,
        use_browser_auth=True,
        private_key_path=str(real_key),
    )
    assert result is not None
    assert "private_key" not in captured["params"], (
        "FileNotFoundError on key load must fall back, not silently send a "
        "(None) private_key to Snowflake."
    )
    assert captured["params"].get("authenticator") == "externalbrowser"

    # Case B: load raises ValueError (malformed PEM) -> re-raise, no fallback
    monkeypatch.setattr(
        sc,
        "load_private_key_from_file",
        lambda *a, **kw: (_ for _ in ()).throw(ValueError("malformed PEM")),
    )
    captured.clear()
    with pytest.raises(ValueError, match="malformed PEM"):
        sc.create_snowflake_connection(
            schema="ML_TRAINING",
            use_private_key=True,
            use_browser_auth=True,
            private_key_path=str(real_key),
        )
    # Critically, connect was never reached on this path.
    assert "params" not in captured
