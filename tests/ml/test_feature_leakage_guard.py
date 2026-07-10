"""Guard against feature columns whose values are derived from the target.

A feature with |corr| > 0.99 with the (log-transformed) target on any single
year is almost certainly a deterministic function of the target — i.e. target
leakage. This test fails loudly if such a column reappears.
"""

import numpy as np
import pandas as pd
import pytest

from box_office.ml.feature_preprocessor import FeaturePreprocessorHigh

LEAKAGE_THRESHOLD = 0.99
MIN_ROWS_PER_YEAR = 8


@pytest.fixture
def synthetic_movies() -> pd.DataFrame:
    """Three years of synthetic movies with realistic, target-independent inputs."""
    rng = np.random.default_rng(42)
    rows = []
    for year in (2021, 2022, 2023):
        for i in range(20):
            budget = float(rng.uniform(5_000_000, 250_000_000))
            runtime = float(rng.uniform(80, 180))
            # Target is a noisy-nonlinear function of budget+runtime, NOT a feature.
            log_gross = (
                np.log1p(budget) * 0.7 + runtime * 0.015 + float(rng.normal(0, 0.5))
            )
            rows.append(
                {
                    "RELEASE_DATE": f"{year}-{int(rng.integers(1, 13)):02d}-15",
                    "RELEASE_YEAR": year,
                    "PRODUCTION_BUDGET": budget,
                    "RUNTIME": runtime,
                    "MPAA": rng.choice(["PG", "PG-13", "R"]),
                    "GENRES": rng.choice(["Action", "Comedy", "Drama"]),
                    "DIRECTOR": f"Director_{i % 5}",
                    "ACTORS": f"Actor_{i % 7}",
                    "PRODUCTION_COMPANY": f"Studio_{i % 3}",
                    "worldwide_gross": float(np.expm1(log_gross)),
                }
            )
    return pd.DataFrame(rows)


def test_no_feature_correlates_perfectly_with_target(synthetic_movies):
    """No feature column has |corr| > 0.99 with log1p(target) within any year."""
    df = synthetic_movies
    target_log = np.log1p(df["worldwide_gross"])

    pre = FeaturePreprocessorHigh()
    pre.fit(df)
    features = pre.transform(df)
    features = features.select_dtypes(include=[np.number])

    leaks: list[tuple[str, int, float]] = []
    for year, group_idx in df.groupby("RELEASE_YEAR").groups.items():
        if len(group_idx) < MIN_ROWS_PER_YEAR:
            continue
        y = target_log.loc[group_idx]
        for col in features.columns:
            x = features.loc[group_idx, col]
            if x.nunique(dropna=True) < 2 or y.nunique(dropna=True) < 2:
                continue
            corr = float(x.corr(y))
            if not np.isfinite(corr):
                continue
            if abs(corr) > LEAKAGE_THRESHOLD:
                leaks.append((col, int(year), corr))

    assert not leaks, (
        f"Suspected target leakage — {len(leaks)} (column, year, corr) "
        f"pairs exceed |corr| > {LEAKAGE_THRESHOLD}: {leaks[:5]}"
    )


def test_known_leaked_feature_names_absent_from_pipeline(synthetic_movies):
    """Forbidden leakage names must not appear in the feature contract."""
    pre = FeaturePreprocessorHigh().fit(synthetic_movies)
    feature_names = pre.get_feature_names()
    forbidden = {
        "SOCIAL_MEDIA_BUZZ",
        "RATING",
        "VOTES",
        "FRANCHISE_RATING",
        "RANK",
        "MOVIE_RANK",
        "DOMESTIC_GROSS",
        "BUDGET_TO_VOTES_RATIO",
        "VOTES_PER_BUDGET",
        "RATING_PER_BUDGET",
        "RATING_VOTES_INTERACTION",
        "YEAR_TO_VOTES_RATIO",
        "VOTES_ERA_ADJUSTED",
        "SOCIAL_BUZZ_TO_BUDGET",
        "MARKETING_EFFICIENCY",
        "VIRAL_POTENTIAL",
        "BUZZ_TO_VOTES_RATIO",
    }
    found = forbidden & set(feature_names)
    assert not found, f"Leaked features reintroduced: {found}"
