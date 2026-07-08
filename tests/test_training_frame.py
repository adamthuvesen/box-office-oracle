"""Shared v9 frame-building rules (box_office.training_frame).

Covers the production path: an uppercase staging frame -> quality gate + v9
IP/franchise features (IP classified in-pipeline), and that the kept frame
satisfies the SELECTED_FEATURES contract through FeaturePreprocessorHigh.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from box_office.ml.feature_pipeline.constants import SELECTED_FEATURES
from box_office.ml.feature_preprocessor import FeaturePreprocessorHigh
from box_office.training_frame import (
    PREPROCESSOR_INPUT_COLUMNS,
    add_franchise_features,
    build_production_training_frame,
    check_frame_against_preprocessor,
)


def _staging_row(**overrides) -> dict:
    row = {
        "TMDB_ID": 1,
        "IMDB_ID": "tt0000001",
        "TITLE": "A Movie",
        "ORIGINAL_TITLE": "A Movie",
        "RELEASE_YEAR": 2015,
        "RELEASE_DATE": "2015-06-15",
        "WORLDWIDE_GROSS": 150_000_000.0,
        "PRODUCTION_BUDGET": 50_000_000.0,
        "PRODUCTION_BUDGET_SOURCE": "tmdb",
        "PRODUCTION_BUDGET_WAS_MISSING": False,
        "RUNTIME": 120,
        "MPAA": "PG-13",
        "GENRES": "Action, Comedy",
        "DIRECTOR": "A Director",
        "PRODUCTION_COMPANY": "A Studio",
        "ACTORS": "First Actor, Second Actor",
        "KEYWORDS": "",
        "OVERVIEW": "",
        "TAGLINE": "",
        "COLLECTION_ID": np.nan,
        "COLLECTION_NAME": None,
    }
    row.update(overrides)
    return row


def _staging_frame(n: int = 12) -> pd.DataFrame:
    rows = []
    for i in range(n):
        rows.append(
            _staging_row(
                TMDB_ID=100 + i,
                TITLE=f"Movie {i}",
                RELEASE_YEAR=2010 + (i % 8),
                RELEASE_DATE=f"{2010 + (i % 8)}-06-15",
                DIRECTOR=f"Director {i % 3}",
                PRODUCTION_COMPANY=f"Studio {i % 2}",
                ACTORS=f"Actor {i}, Actor {(i + 1) % n}",
                PRODUCTION_BUDGET=1e7 + i * 1e6,
                WORLDWIDE_GROSS=5e7 + i * 1e6,
            )
        )
    return pd.DataFrame(rows)


class TestBuildProductionTrainingFrame:
    def test_produces_the_13_v9_features_through_the_preprocessor(self):
        staging = _staging_frame()
        kept, _dropped = build_production_training_frame(
            staging, overrides_path=None
        )

        for col in ("IP_TIER", "PRIOR_FRANCHISE_GROSS_LOG", "IS_FRANCHISE_FOLLOWUP"):
            assert col in kept.columns
            assert not kept[col].isna().any()

        features = FeaturePreprocessorHigh().fit_transform(
            kept[list(PREPROCESSOR_INPUT_COLUMNS)]
        )
        assert list(features.columns) == list(SELECTED_FEATURES)
        assert len(list(SELECTED_FEATURES)) == 13
        # The full contract check the local script also runs.
        check_frame_against_preprocessor(kept)

    def test_quality_gate_drops_short_runtime_and_future_year(self):
        staging = pd.DataFrame(
            [
                _staging_row(TMDB_ID=1, RUNTIME=45),
                _staging_row(TMDB_ID=2, RELEASE_YEAR=2026, RELEASE_DATE="2026-01-01"),
                _staging_row(TMDB_ID=3),
            ]
        )
        kept, dropped = build_production_training_frame(staging, overrides_path=None)
        assert kept["TMDB_ID"].tolist() == [3]
        assert set(dropped["TMDB_ID"]) == {1, 2}

    def test_null_budget_passes_through_as_nan(self):
        staging = pd.DataFrame(
            [_staging_row(PRODUCTION_BUDGET=np.nan, WORLDWIDE_GROSS=8_000_000.0)]
        )
        kept, _dropped = build_production_training_frame(staging, overrides_path=None)
        assert len(kept) == 1
        assert np.isnan(kept.loc[0, "PRODUCTION_BUDGET"])

    def test_actors_become_list_literal(self):
        staging = pd.DataFrame([_staging_row(ACTORS="Mark Hamill, Harrison Ford")])
        kept, _dropped = build_production_training_frame(staging, overrides_path=None)
        assert kept.loc[0, "ACTORS"] == "['Mark Hamill', 'Harrison Ford']"


class TestFranchiseFeatures:
    def test_second_film_in_collection_is_a_followup_with_prior_gross(self):
        frame = pd.DataFrame(
            {
                "TMDB_ID": [10, 11],
                "RELEASE_DATE": ["2000-06-01", "2003-06-01"],
                "WORLDWIDE_GROSS": [200_000_000.0, 300_000_000.0],
            }
        )
        collection_map = {10: "collection:7", 11: "collection:7"}
        out = add_franchise_features(frame, collection_map)

        # First film: no prior; second film: one prior worth $200M.
        assert out.loc[0, "IS_FRANCHISE_FOLLOWUP"] == 0.0
        assert out.loc[0, "PRIOR_FRANCHISE_GROSS_LOG"] == 0.0
        assert out.loc[1, "IS_FRANCHISE_FOLLOWUP"] == 1.0
        assert out.loc[1, "PRIOR_FRANCHISE_GROSS_LOG"] == np.log1p(200_000_000.0)

    def test_ineligible_rows_never_contribute_but_still_receive_values(self):
        frame = pd.DataFrame(
            {
                "TMDB_ID": [10, 11, 12],
                "RELEASE_DATE": ["2000-06-01", "2003-06-01", "2006-06-01"],
                "WORLDWIDE_GROSS": [200_000_000.0, 426_900_000.0, 50_000_000.0],
            }
        )
        collection_map = {t: "collection:7" for t in (10, 11, 12)}
        # Middle film is a known-bad gross artifact the quality gate drops.
        eligible = pd.Series([True, False, True])
        out = add_franchise_features(
            frame, collection_map, eligible_as_prior=eligible
        )

        # The third film's prior counts only the first film's gross.
        assert out.loc[2, "PRIOR_FRANCHISE_GROSS_LOG"] == np.log1p(200_000_000.0)
        # The ineligible row still receives its own feature values.
        assert out.loc[1, "IS_FRANCHISE_FOLLOWUP"] == 1.0
        assert out.loc[1, "PRIOR_FRANCHISE_GROSS_LOG"] == np.log1p(200_000_000.0)

    def test_quality_gated_rows_do_not_feed_priors_in_production_build(self):
        # Same collection, three films; the middle one fails the gate
        # (gross > $100M with no documented budget).
        staging = pd.DataFrame(
            [
                _staging_row(
                    TMDB_ID=20,
                    RELEASE_YEAR=2000,
                    RELEASE_DATE="2000-06-01",
                    WORLDWIDE_GROSS=200_000_000.0,
                    COLLECTION_ID=9,
                    COLLECTION_NAME="A Collection",
                ),
                _staging_row(
                    TMDB_ID=21,
                    RELEASE_YEAR=2003,
                    RELEASE_DATE="2003-06-01",
                    WORLDWIDE_GROSS=426_900_000.0,
                    PRODUCTION_BUDGET=np.nan,
                    COLLECTION_ID=9,
                    COLLECTION_NAME="A Collection",
                ),
                _staging_row(
                    TMDB_ID=22,
                    RELEASE_YEAR=2006,
                    RELEASE_DATE="2006-06-01",
                    WORLDWIDE_GROSS=150_000_000.0,
                    COLLECTION_ID=9,
                    COLLECTION_NAME="A Collection",
                ),
            ]
        )
        kept, dropped = build_production_training_frame(staging, overrides_path=None)
        assert 21 in set(dropped["TMDB_ID"])

        third = kept[kept["TMDB_ID"] == 22].iloc[0]
        # Prior counts only film 20's $200M — not the dropped artifact's gross.
        assert third["PRIOR_FRANCHISE_GROSS_LOG"] == np.log1p(200_000_000.0)
