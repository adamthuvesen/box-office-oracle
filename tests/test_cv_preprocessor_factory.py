"""Per-fold preprocessing in TimeSeriesCrossValidator (leakage fix).

``cross_validate(preprocessor_factory=...)`` must fit a fresh preprocessor on
train-years rows only for every fold, and omitting the kwarg must leave the
existing (pre-engineered X) behavior untouched.
"""

import numpy as np
import pandas as pd

from box_office.ml.cv import TimeSeriesCrossValidator
from box_office.ml.model import BoxOfficeXGBoostModel

FAST_MODEL_KWARGS = {"n_estimators": 10, "max_depth": 2}

UNSEEN_DEFAULT_FREQ = -123.0


class _IdentityPreprocessor:
    """fit_transform/transform that return the input unchanged."""

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X


class _RecordingFrequencyPreprocessor:
    """Frequency-encodes DIRECTOR the way IndustryTransformer does (map fitted
    counts, default for unseen) and records every fit/transform input plus the
    encoded output so the test can inspect per-fold behavior."""

    def __init__(self, calls: list[tuple[str, pd.DataFrame, pd.DataFrame]]):
        self._calls = calls
        self._freq: dict[str, int] = {}

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        self._freq = X["DIRECTOR"].value_counts().to_dict()
        return self._encode("fit_transform", X)

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return self._encode("transform", X)

    def _encode(self, call: str, X: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=X.index)
        out["DIRECTOR_FREQ"] = (
            X["DIRECTOR"].map(self._freq).fillna(UNSEEN_DEFAULT_FREQ).astype(float)
        )
        out["RELEASE_YEAR"] = X["RELEASE_YEAR"].astype(float)
        out["NOISE"] = X["NOISE"].astype(float)
        self._calls.append((call, X.copy(), out.copy()))
        return out


def _engineered_frame(seed: int = 42) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    rng = np.random.default_rng(seed)
    years = np.repeat([2018, 2019, 2020, 2021], 15)
    X = pd.DataFrame(
        {
            "f1": rng.normal(size=len(years)),
            "f2": rng.normal(size=len(years)),
        }
    )
    y = pd.Series(rng.normal(size=len(years)) + 10)
    return X, y, pd.Series(years)


def test_omitted_factory_matches_identity_factory():
    """The kwarg is a pure opt-in: no factory and an identity factory must
    produce identical results on an already-engineered matrix."""
    X, y, dates = _engineered_frame()
    cv_kwargs = dict(
        cv_folds=2, start_eval_year=2020, end_year=2021, early_stopping_rounds=5
    )

    without = TimeSeriesCrossValidator(**cv_kwargs).cross_validate(
        BoxOfficeXGBoostModel, X, y, dates, **FAST_MODEL_KWARGS
    )
    with_identity = TimeSeriesCrossValidator(**cv_kwargs).cross_validate(
        BoxOfficeXGBoostModel,
        X,
        y,
        dates,
        preprocessor_factory=_IdentityPreprocessor,
        **FAST_MODEL_KWARGS,
    )

    assert without["cv_scores"] == with_identity["cv_scores"]
    assert without["oof_records"] == with_identity["oof_records"]
    assert without["feature_names"] == with_identity["feature_names"]
    assert without["mean_cv_mae"] == with_identity["mean_cv_mae"]
    assert [f["mae_score"] for f in without["fold_results"]] == [
        f["mae_score"] for f in with_identity["fold_results"]
    ]


def test_factory_fits_on_train_years_only_and_eval_only_name_gets_default():
    """Each fold's preprocessor sees only pre-eval-year rows, so a director
    who appears only in the eval year encodes to the unseen default in that
    fold's eval rows — not to their global count."""
    rng = np.random.default_rng(7)
    years = np.repeat([2018, 2019, 2020, 2021], 12)
    directors = np.array([f"Director {i % 4}" for i in range(len(years))], dtype=object)
    # This director exists ONLY in 2021 rows, 3 times (global count = 3).
    eval_only_positions = np.flatnonzero(years == 2021)[:3]
    directors[eval_only_positions] = "Eval Only Director"

    X_raw = pd.DataFrame(
        {
            "DIRECTOR": directors,
            "RELEASE_YEAR": years,
            "NOISE": rng.normal(size=len(years)),
        }
    )
    y = pd.Series(rng.normal(size=len(years)) + 10)
    dates = pd.Series(years)

    calls: list[tuple[str, pd.DataFrame, pd.DataFrame]] = []
    cv = TimeSeriesCrossValidator(
        cv_folds=2, start_eval_year=2020, end_year=2021, early_stopping_rounds=5
    )
    cv.cross_validate(
        BoxOfficeXGBoostModel,
        X_raw,
        y,
        dates,
        preprocessor_factory=lambda: _RecordingFrequencyPreprocessor(calls),
        **FAST_MODEL_KWARGS,
    )

    # Folds run in eval-year order, one fit_transform + one transform each.
    assert [c[0] for c in calls] == [
        "fit_transform",
        "transform",
        "fit_transform",
        "transform",
    ]
    fold_eval_years = [2020, 2021]
    for fold, eval_year in enumerate(fold_eval_years):
        fit_input = calls[2 * fold][1]
        val_input = calls[2 * fold + 1][1]
        assert (fit_input["RELEASE_YEAR"] < eval_year).all()
        assert (val_input["RELEASE_YEAR"] == eval_year).all()

    # 2021 fold: the eval-only director is unseen at fit time.
    val_2021_output = calls[3][2]
    val_2021_input = calls[3][1]
    eval_only_rows = val_2021_input["DIRECTOR"] == "Eval Only Director"
    assert eval_only_rows.sum() == 3
    encoded = val_2021_output.loc[eval_only_rows.to_numpy(), "DIRECTOR_FREQ"]
    assert (encoded == UNSEEN_DEFAULT_FREQ).all()
    assert not (encoded == 3).any()


def test_factory_composes_with_real_feature_preprocessor():
    """cross_validate runs end to end with FeaturePreprocessorHigh as the
    factory on a raw staging-shaped frame."""
    from box_office.ml.feature_pipeline.constants import SELECTED_FEATURES
    from box_office.ml.feature_preprocessor import FeaturePreprocessorHigh

    rng = np.random.default_rng(11)
    years = np.repeat([2018, 2019, 2020, 2021], 10)
    n = len(years)
    X_raw = pd.DataFrame(
        {
            "RELEASE_YEAR": years,
            "RELEASE_DATE": [f"{y}-06-15" for y in years],
            "PRODUCTION_BUDGET": rng.uniform(1e6, 2e8, size=n),
            "RUNTIME": rng.integers(80, 180, size=n),
            "MPAA": rng.choice(["PG", "PG-13", "R"], size=n),
            "GENRES": rng.choice(["Action", "Drama", "Comedy, Horror"], size=n),
            "DIRECTOR": rng.choice([f"Director {i}" for i in range(6)], size=n),
            "PRODUCTION_COMPANY": rng.choice([f"Studio {i}" for i in range(4)], size=n),
            "ACTORS": ["['Actor A', 'Actor B']"] * n,
        }
    )
    y = pd.Series(rng.normal(size=n) + 17)
    dates = pd.Series(years)

    cv = TimeSeriesCrossValidator(
        cv_folds=2, start_eval_year=2020, end_year=2021, early_stopping_rounds=5
    )
    results = cv.cross_validate(
        BoxOfficeXGBoostModel,
        X_raw,
        y,
        dates,
        preprocessor_factory=FeaturePreprocessorHigh,
        **FAST_MODEL_KWARGS,
    )

    assert [f["error"] for f in results["fold_results"]] == [None, None]
    assert results["feature_names"] == list(SELECTED_FEATURES)
    assert len(results["oof_records"]) == int((years >= 2020).sum())
