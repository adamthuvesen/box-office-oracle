"""Tests for movie IP tier classification.

ip_tier = pre-sold magnitude at the movie's release date, from as-of-date
brand rules, prior-franchise gross (strictly earlier films), or documented
source-work success — never from total collection gross (leakage).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from box_office.ip_classification import classify_movies, load_rules


def _row(
    tmdb_id: int,
    title: str,
    *,
    year: int = 2024,
    keywords: str = "",
    overview: str = "",
    company: str = "",
    gross: int = 50_000_000,
    release_date: str | None = "default",
) -> dict:
    return {
        "tmdb_id": tmdb_id,
        "imdb_id": f"tt{tmdb_id:07d}",
        "title": title,
        "release_year": year,
        "release_date": f"{year}-06-15" if release_date == "default" else release_date,
        "worldwide_gross": gross,
        "keywords": keywords,
        "overview": overview,
        "production_company": company,
    }


def _raw(
    tmdb_id: int,
    *,
    collection_id: int | None = None,
    collection_name: str | None = None,
) -> dict:
    return {
        "tmdb_id": tmdb_id,
        "collection_id": collection_id,
        "collection_name": collection_name,
        "wikidata_id": f"Q{tmdb_id}",
    }


def _classify(rows: list[dict], raw_rows: list[dict] | None = None) -> pd.DataFrame:
    rules = load_rules(Path("data/ip_rules.yml"))
    raw_metadata = pd.DataFrame(raw_rows or [_raw(row["tmdb_id"]) for row in rows])
    output, _ = classify_movies(pd.DataFrame(rows), raw_metadata, rules)
    return output.set_index("title")


def test_global_brands_include_non_film_origin_ip() -> None:
    output = _classify(
        [
            _row(1, "Barbie", company="Mattel Films"),
            _row(2, "The Super Mario Bros. Movie", keywords="nintendo, plumber"),
            _row(3, "Black Panther", company="Marvel Studios"),
            _row(4, "Joker", keywords="gotham city"),
            _row(5, "Minions", keywords="minion"),
            _row(6, "Sonic the Hedgehog", keywords="based on video game"),
            _row(7, "Transformers", keywords="based on toy"),
            _row(8, "The Flash"),
            _row(9, "Blade"),
        ]
    )

    assert output.loc["Barbie", "ip_tier"] == 2
    assert output.loc["Barbie", "brand_origin"] == "toy"
    assert output.loc["The Super Mario Bros. Movie", "ip_name"] == "Super Mario"
    assert output.loc["The Super Mario Bros. Movie", "ip_tier"] == 1
    assert output.loc["The Super Mario Bros. Movie", "brand_origin"] == "video_game"
    assert output.loc["Black Panther", "ip_name"] == "Marvel"
    assert output.loc["Black Panther", "ip_tier"] == 1
    assert output.loc["Joker", "ip_name"] == "DC"
    assert output.loc["Minions", "ip_name"] == "Despicable Me"
    assert output.loc["Sonic the Hedgehog", "ip_tier"] == 2
    assert output.loc["Transformers", "ip_tier"] == 2
    assert output.loc["Transformers", "brand_origin"] == "toy"
    assert output.loc["The Flash", "ip_name"] == "DC"
    assert output.loc["Blade", "ip_name"] == "Marvel"


def test_public_domain_stories_are_established_ip() -> None:
    output = _classify(
        [
            _row(10, "Cinderella"),
            _row(11, "Sherlock Holmes"),
            _row(12, "Dracula Untold"),
        ]
    )

    assert output.loc["Cinderella", "ip_tier"] == 3
    assert output.loc["Cinderella", "ip_scope"] == "public_domain"
    assert output.loc["Cinderella", "ip_rights_status"] == "public_domain"
    assert output.loc["Sherlock Holmes", "ip_tier"] == 3
    assert output.loc["Dracula Untold", "ip_name"] == "Dracula"


def test_true_story_is_awareness_source_not_ip() -> None:
    output = _classify(
        [
            _row(
                20,
                "Oppenheimer",
                keywords="biography, based on true story, based on novel or book",
            )
        ]
    )

    row = output.loc["Oppenheimer"]
    assert row["ip_tier"] == 5
    assert row["ip_scope"] == "none"
    assert row["awareness_source"] == "true_story,biography"


def test_cinderella_man_is_not_public_domain_cinderella_ip() -> None:
    output = _classify(
        [
            _row(21, "Cinderella Man", keywords="biography, based on true story"),
            _row(22, "Ricki and the Flash"),
        ]
    )

    row = output.loc["Cinderella Man"]
    assert row["ip_tier"] == 5
    assert row["ip_scope"] == "none"
    assert row["awareness_source"] == "true_story,biography"

    flash_row = output.loc["Ricki and the Flash"]
    assert flash_row["ip_tier"] == 5
    assert flash_row["ip_scope"] == "none"


def test_public_domain_keyword_mentions_are_not_ip() -> None:
    output = _classify(
        [
            _row(23, "Rob Roy", keywords="robin hood, biography"),
            _row(24, "Gods and Monsters", keywords="frankenstein, biography"),
        ]
    )

    assert output.loc["Rob Roy", "ip_tier"] == 5
    assert output.loc["Rob Roy", "ip_scope"] == "none"
    assert output.loc["Gods and Monsters", "ip_tier"] == 5
    assert output.loc["Gods and Monsters", "ip_scope"] == "none"


def test_awareness_only_suppresses_weak_collection_and_source_fallbacks() -> None:
    output = _classify(
        [
            _row(25, "The Social Network", keywords="biography, based on true story"),
            _row(26, "Hamilton", keywords="biography, based on play or musical"),
        ],
        [
            _raw(
                25, collection_id=901, collection_name="The Social Network Collection"
            ),
            _raw(26),
        ],
    )

    assert output.loc["The Social Network", "ip_tier"] == 5
    assert output.loc["The Social Network", "ip_scope"] == "none"
    assert output.loc["Hamilton", "ip_tier"] == 5
    assert output.loc["Hamilton", "ip_scope"] == "none"


def test_collection_fallback_and_keyword_adaptation() -> None:
    output = _classify(
        [
            _row(30, "Tiny Franchise 2", keywords="sequel", gross=600_000_000),
            _row(31, "The Notebook", keywords="based on novel or book"),
            _row(32, "Original Comedy"),
        ],
        [
            _raw(30, collection_id=900, collection_name="Tiny Franchise Collection"),
            _raw(31),
            _raw(32),
        ],
    )

    assert output.loc["Tiny Franchise 2", "ip_tier"] == 4
    assert output.loc["Tiny Franchise 2", "ip_scope"] == "direct_collection"
    assert output.loc["Tiny Franchise 2", "is_sequel_or_spinoff"]
    assert output.loc["The Notebook", "ip_tier"] == 4
    assert output.loc["The Notebook", "ip_source_type"] == "book"
    assert output.loc["Original Comedy", "ip_tier"] == 5


def test_first_harry_potter_is_tier_1_via_period_not_gross() -> None:
    output = _classify(
        [_row(40, "Harry Potter and the Philosopher's Stone", year=2001)]
    )

    row = output.loc["Harry Potter and the Philosopher's Stone"]
    assert row["ip_tier"] == 1
    assert row["prior_franchise_gross"] == 0.0
    assert row["tier_basis"].startswith("brand:Wizarding World")


def test_first_despicable_me_does_not_inherit_later_success() -> None:
    output = _classify(
        [
            _row(50, "Despicable Me", year=2010, gross=543_000_000),
            _row(51, "Despicable Me 2", year=2013, gross=970_000_000),
            _row(52, "Minions", year=2015, gross=1_159_000_000),
        ],
        [
            _raw(50, collection_id=650, collection_name="Despicable Me Collection"),
            _raw(51, collection_id=650, collection_name="Despicable Me Collection"),
            _raw(52, collection_id=650, collection_name="Despicable Me Collection"),
        ],
    )

    assert output.loc["Despicable Me", "ip_tier"] == 5
    assert output.loc["Despicable Me", "ip_name"] == "Despicable Me"
    # Follow-ups rise only via prior gross: 543M -> tier 2; 1.513B -> tier 1
    assert output.loc["Despicable Me 2", "ip_tier"] == 2
    assert output.loc["Minions", "ip_tier"] == 1


def test_followup_tier_reflects_only_earlier_films_gross() -> None:
    output = _classify(
        [
            _row(60, "Saga", year=2010, gross=600_000_000),
            _row(61, "Saga 2", year=2015, gross=1_000_000_000),
            _row(62, "Saga 3", year=2020, gross=100_000_000),
        ],
        [
            _raw(60, collection_id=800, collection_name="Saga Collection"),
            _raw(61, collection_id=800, collection_name="Saga Collection"),
            _raw(62, collection_id=800, collection_name="Saga Collection"),
        ],
    )

    assert output.loc["Saga", "ip_tier"] == 5
    assert output.loc["Saga", "prior_franchise_gross"] == 0.0
    assert output.loc["Saga 2", "ip_tier"] == 2
    assert output.loc["Saga 2", "prior_franchise_gross"] == 600_000_000
    assert output.loc["Saga 3", "ip_tier"] == 1
    assert output.loc["Saga 3", "prior_franchise_gross"] == 1_600_000_000


def test_tier_by_period_respects_from_year() -> None:
    output = _classify(
        [
            _row(70, "Fantastic Four", year=2005, company="Marvel Enterprises"),
            _row(71, "Ant-Man", year=2015, company="Marvel Studios"),
        ]
    )

    assert output.loc["Fantastic Four", "ip_tier"] == 2
    assert output.loc["Ant-Man", "ip_tier"] == 1
    assert "from 2013" in output.loc["Ant-Man", "tier_basis"]


def test_source_work_bestseller_is_tier_3_from_its_year() -> None:
    output = _classify(
        [
            _row(80, "The Da Vinci Code", year=2006, keywords="based on novel or book"),
            _row(81, "Gone Girl", year=2014, keywords="based on novel or book"),
        ]
    )

    assert output.loc["The Da Vinci Code", "ip_tier"] == 3
    assert output.loc["The Da Vinci Code", "tier_basis"].startswith("source_work:")
    assert output.loc["Gone Girl", "ip_tier"] == 3


def test_total_collection_gross_path_is_abolished() -> None:
    # A lone movie with a huge gross must not raise its own tier: prior
    # franchise gross is strictly earlier films only.
    output = _classify(
        [_row(90, "One Hit Wonder", gross=6_000_000_000)],
        [_raw(90, collection_id=700, collection_name="One Hit Wonder Collection")],
    )

    assert output.loc["One Hit Wonder", "ip_tier"] == 5
    assert output.loc["One Hit Wonder", "prior_franchise_gross"] == 0.0


def test_brand_matched_predecessor_still_counts_via_shared_collection() -> None:
    # The predecessor matches a public-domain brand ("snow white"), but the
    # follow-up must still see its gross through their shared TMDB collection.
    output = _classify(
        [
            _row(100, "Snow White and the Huntsman", year=2012, gross=396_000_000),
            _row(101, "The Huntsman: Winter's War", year=2016, keywords="prequel"),
        ],
        [
            _raw(100, collection_id=500, collection_name="The Huntsman Collection"),
            _raw(101, collection_id=500, collection_name="The Huntsman Collection"),
        ],
    )

    predecessor = output.loc["Snow White and the Huntsman"]
    assert predecessor["ip_tier"] == 3  # public-domain brand rule, not pooled gross
    assert predecessor["prior_franchise_gross"] == 0.0

    followup = output.loc["The Huntsman: Winter's War"]
    assert followup["ip_tier"] == 3  # $396M prior >= $100M threshold
    assert followup["prior_franchise_gross"] == 396_000_000
    assert followup["is_sequel_or_spinoff"]


def test_unrelated_same_brand_films_do_not_pool_gross() -> None:
    # Both match the Marvel brand, but with no shared collection the earlier
    # film's gross must not raise the later one: tier comes from the brand
    # rule only (tier 2 before 2013).
    output = _classify(
        [
            _row(110, "Spider-Man", year=2002, gross=2_000_000_000),
            _row(111, "Elektra", year=2005, company="Marvel Enterprises"),
        ]
    )

    elektra = output.loc["Elektra"]
    assert elektra["prior_franchise_gross"] == 0.0
    assert elektra["ip_tier"] == 2
    assert elektra["tier_basis"].startswith("brand:Marvel tier 2")


def test_null_release_date_films_never_count_as_prior() -> None:
    # A same-collection film without a real release_date may actually be
    # later than its year suggests, so it must not contribute prior gross.
    output = _classify(
        [
            _row(
                120, "Mystery Origin", year=2010, gross=900_000_000, release_date=None
            ),
            _row(121, "Mystery Origin 2", year=2015, keywords="sequel"),
        ],
        [
            _raw(120, collection_id=400, collection_name="Mystery Collection"),
            _raw(121, collection_id=400, collection_name="Mystery Collection"),
        ],
    )

    followup = output.loc["Mystery Origin 2"]
    assert followup["prior_franchise_gross"] == 0.0
    assert followup["ip_tier"] == 4  # sequel keyword floor, no prior magnitude


def test_nan_collection_name_does_not_inherit_umbrella_scope() -> None:
    # A brand match with a NaN collection link must not be treated as an
    # umbrella-inherited collection nor leak NaN into the evidence.
    output = _classify(
        [_row(130, "Black Panther", company="Marvel Studios")],
        [_raw(130, collection_id=None, collection_name=float("nan"))],
    )

    row = output.loc["Black Panther"]
    assert row["ip_scope"] == "brand_origin"
    assert "collection_name" not in row["evidence_json"]


def test_source_work_only_match_has_book_provenance() -> None:
    # "Jaws" matches a source_works rule by title with no source_mappings
    # keyword, so provenance must still be filled in (book).
    output = _classify([_row(140, "Jaws", year=2000)])

    row = output.loc["Jaws"]
    assert row["ip_scope"] == "adaptation"
    assert row["ip_source_type"] == "book"
    assert row["brand_origin"] == "book"


def test_rules_with_collection_gross_thresholds_are_rejected(
    tmp_path: Path,
) -> None:
    rules_path = tmp_path / "rules.yml"
    rules_path.write_text(
        "tier_thresholds:\n  collection_gross:\n    tier_1: 5000000000\n"
    )
    with pytest.raises(ValueError, match="collection_gross is abolished"):
        load_rules(rules_path)
