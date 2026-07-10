"""Unit tests for the pure transformations in scripts/prepare_training_frame.py."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from prepare_training_frame import (  # noqa: E402
    COLUMN_MAPPING,
    actors_to_list_literal,
    apply_quality_gate,
    flag_extreme_gross_multiplier,
    map_to_staging_columns,
)

from box_office.ml.text_utils import process_text_list  # noqa: E402


def _source_row(**overrides) -> dict:
    row = {
        "tmdb_id": 1,
        "imdb_id": "tt0000001",
        "title": "A Movie",
        "release_year": 2015,
        "release_date": "2015-06-15",
        "production_budget": 50_000_000.0,
        "production_budget_source": "tmdb",
        "production_budget_was_missing": False,
        "runtime": 120,
        "mpaa": "PG-13",
        "genres": "Action, Comedy",
        "director": "A Director",
        "production_company": "A Studio",
        "actors": "First Actor, Second Actor",
        "worldwide_gross": 150_000_000,
    }
    row.update(overrides)
    return row


class TestActorsToListLiteral:
    def test_comma_separated_names_become_list_literal(self):
        literal = actors_to_list_literal("Mark Hamill, Harrison Ford")
        assert literal == "['Mark Hamill', 'Harrison Ford']"
        assert process_text_list(literal) == ["mark hamill", "harrison ford"]

    def test_apostrophes_survive_the_round_trip(self):
        literal = actors_to_list_literal("Conan O'Brien, N!xau")
        assert process_text_list(literal) == ["conan o'brien", "n!xau"]

    def test_nan_and_empty_become_empty_list(self):
        assert actors_to_list_literal(float("nan")) == "[]"
        assert actors_to_list_literal(None) == "[]"
        assert actors_to_list_literal("") == "[]"
        assert process_text_list("[]") == []

    def test_non_string_input_raises(self):
        with pytest.raises(TypeError):
            actors_to_list_literal(42)


class TestMapToStagingColumns:
    def test_maps_and_uppercases_all_columns(self):
        frame = map_to_staging_columns(pd.DataFrame([_source_row()]))
        assert list(frame.columns) == list(COLUMN_MAPPING.values())
        assert frame.loc[0, "PRODUCTION_BUDGET"] == 50_000_000.0
        assert frame.loc[0, "ACTORS"] == "['First Actor', 'Second Actor']"

    def test_missing_source_column_raises(self):
        source = pd.DataFrame([_source_row()]).drop(columns=["actors"])
        with pytest.raises(ValueError, match="actors"):
            map_to_staging_columns(source)

    def test_null_budget_stays_nan(self):
        source = pd.DataFrame([_source_row(production_budget=np.nan)])
        frame = map_to_staging_columns(source)
        assert np.isnan(frame.loc[0, "PRODUCTION_BUDGET"])


class TestQualityGate:
    def test_keeps_ordinary_rows(self):
        frame = map_to_staging_columns(pd.DataFrame([_source_row()]))
        kept, dropped = apply_quality_gate(frame)
        assert len(kept) == 1
        assert dropped.empty

    def test_drops_short_runtime(self):
        frame = map_to_staging_columns(pd.DataFrame([_source_row(runtime=45)]))
        kept, dropped = apply_quality_gate(frame)
        assert kept.empty
        assert dropped["DROP_REASON"].tolist() == ["runtime_under_60_non_feature"]

    def test_drops_big_gross_with_missing_budget(self):
        frame = map_to_staging_columns(
            pd.DataFrame(
                [_source_row(production_budget=np.nan, worldwide_gross=426_900_000)]
            )
        )
        kept, dropped = apply_quality_gate(frame)
        assert kept.empty
        assert dropped["DROP_REASON"].tolist() == [
            "gross_over_100m_with_no_documented_budget"
        ]

    def test_keeps_missing_budget_with_plausible_gross(self):
        frame = map_to_staging_columns(
            pd.DataFrame(
                [_source_row(production_budget=np.nan, worldwide_gross=8_000_000)]
            )
        )
        kept, dropped = apply_quality_gate(frame)
        assert len(kept) == 1
        assert np.isnan(kept.loc[0, "PRODUCTION_BUDGET"])
        assert dropped.empty

    def test_drops_rows_without_reliable_gross(self):
        frame = map_to_staging_columns(
            pd.DataFrame([_source_row(worldwide_gross=np.nan)])
        )
        kept, dropped = apply_quality_gate(frame)
        assert kept.empty
        assert dropped["DROP_REASON"].tolist() == ["no_reliable_worldwide_gross"]

    def test_drops_future_year_placeholder_gross(self):
        frame = map_to_staging_columns(
            pd.DataFrame([_source_row(release_year=2026, release_date="2026-05-13")])
        )
        kept, dropped = apply_quality_gate(frame)
        assert kept.empty
        assert dropped["DROP_REASON"].tolist() == ["gross_not_final_future_year"]

    def test_multiple_reasons_are_joined(self):
        frame = map_to_staging_columns(
            pd.DataFrame(
                [
                    _source_row(
                        runtime=5,
                        production_budget=np.nan,
                        worldwide_gross=200_000_000,
                    )
                ]
            )
        )
        _, dropped = apply_quality_gate(frame)
        reason = dropped["DROP_REASON"].iloc[0]
        assert "runtime_under_60_non_feature" in reason
        assert "gross_over_100m_with_no_documented_budget" in reason
        assert ";" in reason


class TestExtremeMultiplierFlag:
    def test_flags_sleeper_hit_but_gate_keeps_it(self):
        # The Blair Witch Project shape: tiny known budget, huge gross.
        frame = map_to_staging_columns(
            pd.DataFrame(
                [_source_row(production_budget=60_000.0, worldwide_gross=248_639_099)]
            )
        )
        assert flag_extreme_gross_multiplier(frame).tolist() == [True]
        kept, _ = apply_quality_gate(frame)
        assert len(kept) == 1

    def test_does_not_flag_missing_budget(self):
        frame = map_to_staging_columns(
            pd.DataFrame(
                [_source_row(production_budget=np.nan, worldwide_gross=200_000_000)]
            )
        )
        assert flag_extreme_gross_multiplier(frame).tolist() == [False]
