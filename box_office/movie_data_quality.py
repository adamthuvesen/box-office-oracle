"""Shared source-data cleanup for the local movie dataset."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

UNSET = object()


@dataclass(frozen=True)
class FinancialCorrection:
    reason: str
    production_budget: float | None | object = UNSET
    worldwide_gross: float | None | object = UNSET


@dataclass(frozen=True)
class SourceExclusion:
    reason: str


MIN_ROI_BUDGET = 1_000

# Hand-checked financial corrections. Null means "unknown/untrusted", not a
# real zero-dollar value.
FINANCIAL_CORRECTIONS: dict[int, FinancialCorrection] = {
    # 6-minute digital short; $5/$100M appears only on TMDB-derived mirrors.
    1175807: FinancialCorrection(
        reason="digital_short_tmdb_mirror_financials",
        production_budget=None,
        worldwide_gross=None,
    ),
    # No credible theatrical footprint; $1/$10M appears only on mirrors.
    1248416: FinancialCorrection(
        reason="tmdb_mirror_financials_no_theatrical_source",
        production_budget=None,
        worldwide_gross=None,
    ),
    # Box Office Mojo: domestic/worldwide gross $89,507; budget unsupported.
    270650: FinancialCorrection(
        reason="box_office_mojo_gross_89507_budget_unknown",
        production_budget=None,
        worldwide_gross=89_507,
    ),
    # The Numbers: worldwide $5,727; IMDb budget is A$350K (~current USD value).
    1289601: FinancialCorrection(
        reason="the_numbers_gross_5727",
        worldwide_gross=5_727,
    ),
    # $30M appears only on mirrors; no credible gross source found.
    1679323: FinancialCorrection(
        reason="unsupported_30m_gross",
        worldwide_gross=None,
    ),
    1494943: FinancialCorrection(
        reason="unsupported_super_vixens_gross",
        worldwide_gross=None,
    ),
    1494978: FinancialCorrection(
        reason="unsupported_super_vixens_gross",
        worldwide_gross=None,
    ),
    1494947: FinancialCorrection(
        reason="unsupported_super_vixens_gross",
        worldwide_gross=None,
    ),
    538831: FinancialCorrection(
        reason="unbound_gross_only_found_on_tmdb_mirrors",
        worldwide_gross=None,
    ),
    1723282: FinancialCorrection(
        reason="no_credible_source_for_400m_gross",
        worldwide_gross=None,
    ),
    1724676: FinancialCorrection(
        reason="no_credible_source_for_46m_gross",
        worldwide_gross=None,
    ),
}

# Rows that are not useful theatrical feature-film observations for training.
SOURCE_EXCLUSIONS: dict[int, SourceExclusion] = {
    565916: SourceExclusion("dvd_extra_inherits_scary_movie_4_financials"),
    715904: SourceExclusion("concert_recording_uses_tour_box_office"),
    120454: SourceExclusion("concert_recording_uses_tour_box_office"),
    123052: SourceExclusion("concert_recording_uses_tour_box_office"),
    92060: SourceExclusion("music_video_short_not_feature_film"),
    1462164: SourceExclusion("short_prank_video_not_feature_film"),
    1693620: SourceExclusion("one_minute_short_not_feature_film"),
    1467514: SourceExclusion("two_minute_short_not_feature_film"),
    1533252: SourceExclusion("three_minute_short_not_feature_film"),
    1674278: SourceExclusion("short_film_with_placeholder_financials"),
    1679376: SourceExclusion("dirty_scraped_title_and_unsupported_financials"),
    1697520: SourceExclusion("tv_episode_not_movie"),
    1699787: SourceExclusion("future_zero_footprint_placeholder_financials"),
    1700118: SourceExclusion("future_zero_footprint_placeholder_financials"),
}


def _is_missing(value: Any) -> bool:
    return value is None or pd.isna(value)


def _optional_str(value: Any) -> str | None:
    if _is_missing(value):
        return None
    text = str(value).strip()
    return text or None


def _set_missing_source(row: pd.Series) -> None:
    if "production_budget_source" in row.index:
        row["production_budget_source"] = "missing"
    if "production_budget_was_missing" in row.index:
        row["production_budget_was_missing"] = True


def clean_movie_source_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply curated source fixes and return (cleaned_rows, audit_rows)."""
    cleaned_rows: list[pd.Series] = []
    audit_rows: list[dict[str, Any]] = []

    for _, original in df.iterrows():
        row = original.copy()
        tmdb_id = int(row["tmdb_id"])
        actions: list[str] = []
        reasons: list[str] = []

        exclusion = SOURCE_EXCLUSIONS.get(tmdb_id)
        if exclusion is not None:
            audit_rows.append(_audit_row(row, "exclude_row", exclusion.reason))
            continue

        correction = FINANCIAL_CORRECTIONS.get(tmdb_id)
        if correction is not None:
            if correction.production_budget is not UNSET:
                row["production_budget"] = correction.production_budget
                if correction.production_budget is None:
                    _set_missing_source(row)
                actions.append("set_production_budget")
            if correction.worldwide_gross is not UNSET:
                row["worldwide_gross"] = correction.worldwide_gross
                actions.append("set_worldwide_gross")
            reasons.append(correction.reason)

        budget = row.get("production_budget")
        if not _is_missing(budget) and float(budget) < MIN_ROI_BUDGET:
            row["production_budget"] = None
            _set_missing_source(row)
            actions.append("set_production_budget")
            reasons.append("budget_under_1000_placeholder")

        gross = row.get("worldwide_gross")
        imdb_id = _optional_str(row.get("imdb_id"))
        vote_count = row.get("vote_count")
        has_zero_votes = "vote_count" in row.index and (
            _is_missing(vote_count) or int(vote_count) == 0
        )
        if (
            not _is_missing(gross)
            and float(gross) >= 1_000_000
            and imdb_id is None
            and has_zero_votes
        ):
            row["worldwide_gross"] = None
            actions.append("set_worldwide_gross")
            reasons.append("high_gross_without_imdb_or_votes")

        if actions:
            audit_rows.append(
                _audit_row(row, ",".join(sorted(set(actions))), ";".join(reasons))
            )
        cleaned_rows.append(row)

    cleaned = pd.DataFrame(cleaned_rows, columns=df.columns).reset_index(drop=True)
    audit = pd.DataFrame(audit_rows)
    return cleaned, audit


def _audit_row(row: pd.Series, action: str, reason: str) -> dict[str, Any]:
    return {
        "tmdb_id": int(row["tmdb_id"]),
        "imdb_id": _optional_str(row.get("imdb_id")),
        "title": _optional_str(row.get("title")),
        "release_year": row.get("release_year"),
        "action": action,
        "reason": reason,
        "production_budget": row.get("production_budget"),
        "worldwide_gross": row.get("worldwide_gross"),
        "runtime": row.get("runtime"),
        "vote_count": row.get("vote_count"),
        "popularity": row.get("popularity"),
    }
