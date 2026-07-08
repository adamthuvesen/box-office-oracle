"""Tests for the Wikipedia infobox budget recovery."""

from __future__ import annotations

import pandas as pd
import pytest

from box_office.ingestion import wikipedia_budget_fill as wiki

INFOBOX = """{{Infobox film
| name = Test Movie
| budget = %s
| gross = $50 million
}}
Plot text here.
"""


def test_extract_infobox_budget() -> None:
    assert wiki.extract_infobox_budget(INFOBOX % "$25 million") == "$25 million"


def test_extract_infobox_budget_missing() -> None:
    wikitext = "{{Infobox film\n| name = No Budget\n}}"
    assert wiki.extract_infobox_budget(wikitext) is None


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("$25 million", 25_000_000),
        ("US$7.7 million", 7_700_000),
        ("$20,500,000", 20_500_000),
        ("$6–7 million", 6_500_000),
        ("$60-70 million<ref>Mojo</ref>", 65_000_000),
        ("{{US$|14 million|link=yes}}", 14_000_000),
        ("$1.2 billion", None),  # above sanity bound
        ("$102.5 million (equivalent to $250 million in 2023)", 102_500_000),
        ("[[United States dollar|$]]3 million", 3_000_000),
    ],
)
def test_parse_budget_values(raw: str, expected: float | None) -> None:
    value, _ = wiki.classify_budget_text(raw)
    assert value == expected


def test_empty_budget_never_bleeds_into_gross_line() -> None:
    wikitext = (
        "{{Infobox film\n"
        "| budget = \n"
        "| gross          = $34.7 million<ref name=mojo/>\n"
        "}}"
    )
    assert wiki.extract_infobox_budget(wikitext) is None


def test_gross_mention_in_budget_value_is_rejected() -> None:
    value, status = wiki.classify_budget_text("| gross = $30.1 million<ref name=mojo/>")
    assert value is None
    assert status == "gross_line_guard"


def test_minus_sign_range_and_thousand_scale() -> None:
    value, _ = wiki.classify_budget_text("$6.5−$8 million")
    assert value == 7_250_000
    value, _ = wiki.classify_budget_text(
        "[[United States dollar|US$]]440{{ndash}}600{{nbsp}}thousand"
    )
    assert value == 520_000


def test_non_usd_is_skipped() -> None:
    value, status = wiki.classify_budget_text("£3 million")
    assert value is None
    assert status == "non_usd"


def test_ambiguous_bare_number_is_rejected() -> None:
    value, status = wiki.classify_budget_text("$12")
    assert value is None
    assert status == "parse_failed"


def test_parenthetical_usd_conversion_is_used() -> None:
    value, _ = wiki.classify_budget_text("£3 million ($4.9 million)")
    assert value == 4_900_000


def _dataset() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "tmdb_id": [1, 2, 3, 4],
            "title": ["Known", "Fillable", "No Article", "No Budget Field"],
            "release_year": [2001, 2002, 2003, 2004],
            "imdb_id": ["tt0000001", "tt0000002", "tt0000003", "tt0000004"],
            "production_budget": [10_000_000.0, None, None, None],
            "production_budget_source": [
                "tmdb",
                "missing",
                "missing",
                "missing",
            ],
            "worldwide_gross": [30_000_000, 24_000_000, 9_000_000, 6_110_461],
        }
    )


def test_fill_rejects_budget_equal_to_own_gross() -> None:
    titles = {"tt0000004": "Gross In Budget"}
    wikitexts = {"Gross In Budget": INFOBOX % "$6,110,461"}

    filled, audit = wiki.fill_budgets(_dataset(), titles, wikitexts)

    assert audit[audit["tmdb_id"] == 4]["status"].iloc[0] == ("budget_equals_gross")
    assert filled[filled["tmdb_id"] == 4]["production_budget"].isna().all()


def test_fill_budgets_fills_and_audits() -> None:
    titles = {"tt0000002": "Fillable (film)", "tt0000003": "No Article"}
    wikitexts = {"Fillable (film)": INFOBOX % "$8 million"}

    filled, audit = wiki.fill_budgets(_dataset(), titles, wikitexts)

    fillable = filled[filled["tmdb_id"] == 2].iloc[0]
    assert fillable["production_budget"] == 8_000_000
    assert fillable["production_budget_source"] == "wikipedia"
    legacy = "a" + "d_" + "budget"
    assert legacy not in filled.columns

    statuses = dict(zip(audit["tmdb_id"], audit["status"], strict=True))
    assert statuses == {
        2: "filled",
        3: "no_wikipedia_article",
        4: "no_wikipedia_article",
    }
    # Unfilled rows stay null, never zero.
    assert filled[filled["tmdb_id"] == 3]["production_budget"].isna().all()


def test_fill_budgets_leaves_known_rows_untouched() -> None:
    filled, audit = wiki.fill_budgets(_dataset(), {}, {})
    known = filled[filled["tmdb_id"] == 1]["production_budget"].iloc[0]
    assert known == 10_000_000
    assert (audit["status"] == "no_wikipedia_article").all()
