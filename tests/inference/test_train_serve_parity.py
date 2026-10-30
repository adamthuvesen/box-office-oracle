"""The serving engine scores a frame exactly as the training path does."""

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler

from box_office.inference.app.predictor import PredictionEngine, PredictionRequest
from box_office.ml.feature_preprocessor import FeaturePreprocessorHigh
from box_office.ml.model import fit_final_model
from box_office.ml.target import from_log_target, to_log_target
from box_office.training_frame import PREPROCESSOR_INPUT_COLUMNS


def _synthetic_frame(seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = 120
    years = rng.integers(2005, 2024, size=n)
    budgets = rng.uniform(5e6, 2.5e8, size=n)
    budgets[::17] = np.nan  # missing budgets stay NaN through training and serving
    return pd.DataFrame(
        {
            "RELEASE_YEAR": years,
            "RELEASE_DATE": [
                f"{y}-{m:02d}-15" for y, m in zip(years, rng.integers(1, 13, n), strict=True)
            ],
            "PRODUCTION_BUDGET": budgets,
            "RUNTIME": rng.uniform(80, 180, size=n),
            "MPAA": rng.choice(["PG", "PG-13", "R"], size=n),
            "GENRES": [list(rng.choice(["Action", "Comedy", "Drama"], size=2)) for _ in range(n)],
            "DIRECTOR": [f"Director_{i % 9}" for i in range(n)],
            "PRODUCTION_COMPANY": [f"Studio_{i % 4}" for i in range(n)],
            "ACTORS": [[f"Actor_{i % 11}", f"Actor_{(i + 3) % 11}"] for i in range(n)],
            "IP_TIER": rng.integers(1, 6, size=n).astype(float),
            "PRIOR_FRANCHISE_GROSS_LOG": to_log_target(rng.choice([0.0, 3e8], size=n)),
            "IS_FRANCHISE_FOLLOWUP": rng.integers(0, 2, size=n).astype(float),
            "WORLDWIDE_GROSS": budgets * rng.uniform(0.5, 4.0, size=n),
        }
    ).fillna({"WORLDWIDE_GROSS": 4e7})


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    frame = _synthetic_frame()
    X_input = frame[list(PREPROCESSOR_INPUT_COLUMNS)]
    preprocessor = FeaturePreprocessorHigh()
    X = preprocessor.fit_transform(X_input)
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)
    model = fit_final_model(X_scaled, to_log_target(frame["WORLDWIDE_GROSS"]), 40)

    artifact_dir = tmp_path_factory.mktemp("artifact")
    for name, obj in {"model": model, "preprocessor": preprocessor, "scaler": scaler}.items():
        joblib.dump(obj, artifact_dir / f"{name}.pkl")
    engine = PredictionEngine()
    engine.load_model_artifacts(
        model_path=str(artifact_dir / "model.pkl"),
        preprocessor_path=str(artifact_dir / "preprocessor.pkl"),
        scaler_path=str(artifact_dir / "scaler.pkl"),
        model_metadata={"model_id": "parity", "version": 1},
    )
    training_path = from_log_target(model.predict(X_scaled))
    return frame, engine, training_path


def test_engine_frame_scores_match_the_training_path(trained):
    frame, engine, training_path = trained
    served = engine.predict_frame(frame[list(PREPROCESSOR_INPUT_COLUMNS)])
    np.testing.assert_allclose(served, training_path, rtol=1e-6)


def test_single_request_matches_the_frame_path(trained):
    frame, engine, _ = trained
    row = frame.iloc[1]
    request = PredictionRequest(
        budget=float(row.PRODUCTION_BUDGET),
        runtime=float(row.RUNTIME),
        genre=list(row.GENRES),
        release_month=int(row.RELEASE_DATE[5:7]),
        release_year=int(row.RELEASE_YEAR),
        mpaa=row.MPAA,
        director=row.DIRECTOR,
        actors=list(row.ACTORS),
        production_company=row.PRODUCTION_COMPANY,
        ip_tier=int(row.IP_TIER),
        prior_franchise_gross=float(from_log_target(row.PRIOR_FRANCHISE_GROSS_LOG)),
        is_franchise_followup=bool(row.IS_FRANCHISE_FOLLOWUP),
    )
    request_frame = frame.iloc[[1]][list(PREPROCESSOR_INPUT_COLUMNS)].assign(
        RELEASE_DATE=f"{int(row.RELEASE_YEAR)}-{int(row.RELEASE_DATE[5:7]):02d}-01"
    )
    expected = engine.predict_frame(request_frame)[0]
    assert engine.predict(request).prediction == pytest.approx(float(expected), rel=1e-6)
