"""Smoke tests for the consolidated feature pipeline."""

import pandas as pd
import unittest

from box_office.ml.feature_pipeline.constants import SELECTED_FEATURES


class TestCoreNumericalTransformer(unittest.TestCase):
    def test_transform_works(self):
        from box_office.ml.feature_pipeline import (
            CORE_NUMERICAL_FEATURES,
            CoreNumericalTransformer,
        )

        data = pd.DataFrame(
            {
                "RELEASE_YEAR": [2020, 2021],
                "AD_BUDGET": [500000, 600000],
                "PRODUCTION_BUDGET": [1000000, 2000000],
                "RUNTIME": [120, 135],
            }
        )

        out = CoreNumericalTransformer().transform(data)
        self.assertEqual(len(out), 2)
        for c in CORE_NUMERICAL_FEATURES:
            self.assertIn(c, out.columns)


class TestTemporalTransformer(unittest.TestCase):
    def test_transform_works(self):
        from box_office.ml.feature_pipeline import TemporalTransformer

        data = pd.DataFrame({"RELEASE_DATE": ["2023-07-15", "2020-04-15"]})
        out = TemporalTransformer().transform(data)

        self.assertEqual(len(out), 2)
        self.assertIn("IS_SUMMER_RELEASE", out.columns)
        self.assertIn("IS_COVID_ERA", out.columns)


class TestGenreTransformer(unittest.TestCase):
    def test_transform_works(self):
        from box_office.ml.feature_pipeline import GenreTransformer

        data = pd.DataFrame({"GENRES": ["Action, Adventure", "Comedy", "Drama"]})
        t = GenreTransformer().fit(data)
        out = t.transform(data)

        self.assertEqual(len(out), 3)
        self.assertIn("SUPER_GENRE_ENCODED", out.columns)

    def test_science_fiction_is_one_token(self):
        """'Science Fiction' must survive as a single vocab entry."""
        from box_office.ml.feature_pipeline import GenreTransformer

        data = pd.DataFrame({"GENRES": ["Action, Science Fiction, Comedy", "Drama"]})
        t = GenreTransformer().fit(data)
        out = t.transform(data)

        self.assertIn("GENRE_science_fiction", out.columns)
        self.assertEqual(int(out["GENRE_science_fiction"].iloc[0]), 1)
        self.assertEqual(int(out["GENRE_science_fiction"].iloc[1]), 0)

    def test_super_genre_scifi_blockbuster_fires(self):
        """super_genre lookup matches Action/Adventure/Sci-Fi."""
        from box_office.ml.feature_pipeline import _map_super_genre, _split_genre_field

        canon = _split_genre_field("Action|Adventure|Science Fiction")
        self.assertEqual(_map_super_genre(canon), "SciFi_Blockbuster")

    def test_super_genre_rules_keep_priority_order(self):
        from box_office.ml.feature_pipeline import _map_super_genre, _split_genre_field

        canon = _split_genre_field("Action|Adventure|Science Fiction|Thriller")
        self.assertEqual(_map_super_genre(canon), "SciFi_Blockbuster")

    def test_super_genre_romantic_comedy_alias(self):
        from box_office.ml.feature_pipeline import _map_super_genre, _split_genre_field

        canon = _split_genre_field("Romantic Comedy")
        self.assertEqual(_map_super_genre(canon), "RomCom")


class TestIndustryTransformer(unittest.TestCase):
    def test_transform_works(self):
        from box_office.ml.feature_pipeline import IndustryTransformer

        data = pd.DataFrame(
            {
                "DIRECTOR": ["Director A", "Director B", "Director A"],
                "PRODUCTION_COMPANY": ["Warner Bros", "Disney", "Warner Bros"],
                "ACTORS": ["Actor A, Actor B", "Actor C", "Actor A"],
                "MPAA": ["PG-13", "R", "PG-13"],
            }
        )
        out = IndustryTransformer().fit(data).transform(data)

        self.assertEqual(len(out), 3)
        for c in ("DIRECTOR_FREQ", "COMPANY_FREQ", "MPAA_ENCODED"):
            self.assertIn(c, out.columns)


class TestFinancialTransformer(unittest.TestCase):
    def test_safe_budget_interactions_are_emitted(self):
        from box_office.ml.feature_pipeline import FinancialTransformer

        data = pd.DataFrame(
            {
                "RELEASE_YEAR": [2023],
                "AD_BUDGET": [10_000_000],
                "PRODUCTION_BUDGET": [50_000_000],
                "IS_BLOCKBUSTER_SEASON": [1],
                "IS_MEMORIAL_DAY_WEEKEND": [0],
                "IS_JULY_4TH_WEEKEND": [1],
                "IS_THANKSGIVING_WEEK": [0],
                "IS_CHRISTMAS_WEEK": [0],
                "IS_SUMMER_RELEASE": [1],
                "GENRE_horror": [0],
                "GENRE_adventure": [1],
                "GENRE_comedy": [0],
                "DIRECTOR_FREQ": [3],
                "COMPANY_FREQ": [5],
                "LEAD_ACTOR_FREQ": [7],
                "MAX_ACTOR_FREQ": [11],
                "AVG_ACTOR_FREQ": [4],
            }
        )

        out = FinancialTransformer().transform(data)

        expected = {
            "LOG_PRODUCTION_BUDGET",
            "LOG_TOTAL_BUDGET_X_ADVENTURE",
            "DIRECTOR_BUDGET_CONFIDENCE",
            "CREATIVE_FREQ_SCORE",
            "BLOCKBUSTER_BUDGET_MULTIPLIER",
            "LOG1P_LEAD_ACTOR_FREQ",
            "LOG_TOTAL_BUDGET_X_COMPANY_FREQ",
            "LOG_TOTAL_BUDGET_X_SUMMER",
        }
        self.assertTrue(expected <= set(out.columns))
        self.assertGreater(out["BLOCKBUSTER_BUDGET_MULTIPLIER"].iloc[0], 0)


class TestFeaturePipeline(unittest.TestCase):
    def test_pipeline_steps(self):
        from box_office.ml.feature_pipeline import build_feature_pipeline

        p = build_feature_pipeline()
        self.assertEqual(
            list(p.named_steps),
            [
                "drop_pre_engineered",
                "core",
                "temporal",
                "genre",
                "industry",
                "financial",
                "select",
                "feature_selector",
            ],
        )

    def test_preprocessor_get_feature_names_after_fit(self):
        from box_office.ml.feature_preprocessor import FeaturePreprocessorHigh

        data = pd.DataFrame(
            {
                "RELEASE_YEAR": [2020, 2021, 2022, 2023],
                "RELEASE_DATE": [
                    "2020-06-15",
                    "2021-07-04",
                    "2022-12-25",
                    "2023-05-26",
                ],
                "AD_BUDGET": [1e6, 2e6, 5e5, 1.5e6],
                "PRODUCTION_BUDGET": [1e7, 2e7, 5e6, 1.5e7],
                "RUNTIME": [120, 130, 95, 105],
                "MPAA": ["PG-13", "R", "G", "PG"],
                "GENRES": ["Action", "Comedy, Romance", "Animation, Family", "Drama"],
                "DIRECTOR": ["A", "B", "C", "A"],
                "PRODUCTION_COMPANY": ["X", "Y", "X", "Z"],
                "ACTORS": ["a, b", "c", "d, e", "a"],
            }
        )
        pre = FeaturePreprocessorHigh().fit(data)
        names = pre.get_feature_names()
        self.assertIsInstance(names, list)
        # Contract lock: the fitted pipeline must emit exactly SELECTED_FEATURES,
        # in order. This is the single guard that training and serving agree.
        self.assertEqual(names, list(SELECTED_FEATURES))

    def test_pipeline_strips_pre_engineered_columns_from_input(self):
        """Input snapshots may contain encoded columns; the pipeline must drop
        them before its own transformers emit the same names."""
        from box_office.ml.feature_preprocessor import FeaturePreprocessorHigh

        data = pd.DataFrame(
            {
                "RELEASE_YEAR": [2020, 2021],
                "RELEASE_DATE": ["2020-06-15", "2021-07-04"],
                "AD_BUDGET": [1e6, 2e6],
                "PRODUCTION_BUDGET": [1e7, 2e7],
                "RUNTIME": [120, 130],
                "MPAA": ["PG-13", "R"],
                "MPAA_ENCODED": [3, 4],
                "RELEASE_TYPE_ENCODED": [1, 1],
                "PRODUCTION_COMPANY_ENCODED": [12, 7],
                "GENRES": ["Action", "Drama"],
                "DIRECTOR": ["A", "B"],
                "PRODUCTION_COMPANY": ["X", "Y"],
                "ACTORS": ["a, b", "c"],
            }
        )
        pre = FeaturePreprocessorHigh().fit(data)
        out = pre.transform(data)
        self.assertTrue(out.columns.is_unique)
        self.assertEqual(list(out.columns), list(SELECTED_FEATURES))
        self.assertNotIn("RELEASE_TYPE_ENCODED", out.columns)
        self.assertNotIn("PRODUCTION_COMPANY_ENCODED", out.columns)


if __name__ == "__main__":
    unittest.main()
