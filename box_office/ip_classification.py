"""Classify movies by pre-release IP strength.

Two separate axes:

- Provenance (``ip_source_type``): where the property came from — book,
  comic, video game, toy, TV, remake, sequel, original film. Factual.
- Magnitude (``ip_tier`` 1-5): how pre-sold the property was AT THE MOVIE'S
  RELEASE DATE, computed only from information available before that date —
  an as-of-date umbrella-brand rule, the franchise's prior-films gross
  (strictly earlier releases in our own dataset), or a documented
  pre-release source-work success. No total-collection gross: that includes
  the movie's own outcome and future sequels (target leakage).
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from box_office.franchise_history import prior_franchise_stats
from box_office.movie_data_quality import clean_movie_source_data

# The same cleaned source parquet the training frame is built from, so the
# classifier's row set and gross values match training (curated exclusions
# and financial fixes applied in main() via clean_movie_source_data).
DEFAULT_MOVIES_PATH = Path(
    "data/generated/tmdb/rich_backfill_1980_2026/"
    "tmdb_budget_wikipedia_5m_1980_2026.parquet"
)
DEFAULT_RAW_JSONL_PATH = Path(
    "data/generated/tmdb/rich_backfill_1980_2026/tmdb_rich_raw_5m_1980_2026.jsonl"
)
DEFAULT_RULES_PATH = Path("data/ip_rules.yml")
DEFAULT_OUTPUT_DIR = Path("data/generated/ip")

NOMINAL_IP_TIER = 4
NO_IP_TIER = 5
SOURCE_WORK_TIER = 3

OUTPUT_COLUMNS = [
    "tmdb_id",
    "imdb_id",
    "title",
    "release_year",
    "ip_tier",
    "ip_name",
    "ip_sub_ip_name",
    "ip_scope",
    "ip_source_type",
    "brand_origin",
    "brand_power_tier",
    "ip_rights_status",
    "awareness_source",
    "is_sequel_or_spinoff",
    "prior_franchise_gross",
    "tier_basis",
    "confidence",
    "evidence_json",
]


@dataclass(frozen=True)
class PatternGroup:
    title: tuple[re.Pattern[str], ...] = ()
    collection: tuple[re.Pattern[str], ...] = ()
    keyword: tuple[re.Pattern[str], ...] = ()
    company: tuple[re.Pattern[str], ...] = ()


@dataclass(frozen=True)
class TierPeriod:
    from_year: int
    tier: int
    basis: str


@dataclass(frozen=True)
class BrandRule:
    name: str
    origin: str
    rights_status: str
    periods: tuple[TierPeriod, ...]
    patterns: PatternGroup


@dataclass(frozen=True)
class SourceWorkRule:
    name: str
    from_year: int
    basis: str
    patterns: PatternGroup


@dataclass(frozen=True)
class SourceRule:
    source_type: str
    sequel_or_spinoff: bool
    patterns: PatternGroup


@dataclass(frozen=True)
class AwarenessRule:
    name: str
    patterns: PatternGroup


@dataclass(frozen=True)
class TierThresholds:
    prior_tier_1: int
    prior_tier_2: int
    prior_tier_3: int


@dataclass(frozen=True)
class IpRules:
    tier_thresholds: TierThresholds
    brand_rules: tuple[BrandRule, ...]
    source_work_rules: tuple[SourceWorkRule, ...]
    source_rules: tuple[SourceRule, ...]
    awareness_rules: tuple[AwarenessRule, ...]


@dataclass(frozen=True)
class MatchContext:
    title_text: str
    collection_text: str
    keyword_text: str
    company_text: str


@dataclass(frozen=True)
class PatternMatch:
    field: str
    pattern: str


@dataclass(frozen=True)
class Classification:
    ip_tier: int
    ip_name: str | None
    ip_sub_ip_name: str | None
    ip_scope: str
    ip_source_type: str | None
    brand_origin: str | None
    brand_power_tier: int | None
    ip_rights_status: str | None
    awareness_source: str | None
    is_sequel_or_spinoff: bool
    prior_franchise_gross: float
    tier_basis: str
    confidence: str
    evidence: dict[str, Any]


def load_rules(path: Path = DEFAULT_RULES_PATH) -> IpRules:
    raw = yaml.safe_load(path.read_text()) or {}
    if "collection_gross" in raw.get("tier_thresholds", {}):
        raise ValueError(
            "tier_thresholds.collection_gross is abolished (target leakage); "
            "use tier_thresholds.prior_franchise_gross"
        )
    thresholds = raw["tier_thresholds"]["prior_franchise_gross"]
    tier_thresholds = TierThresholds(
        prior_tier_1=int(thresholds["tier_1"]),
        prior_tier_2=int(thresholds["tier_2"]),
        prior_tier_3=int(thresholds["tier_3"]),
    )

    brand_rules = [
        _brand_rule(item)
        for item in [
            *raw.get("umbrella_brands", []),
            *[
                {
                    **item,
                    "origin": "public_domain",
                    "rights_status": "public_domain",
                }
                for item in raw.get("public_domain_brands", [])
            ],
        ]
    ]
    source_work_rules = [
        SourceWorkRule(
            name=str(item["name"]),
            from_year=int(item["from_year"]),
            basis=_required_basis(item, f"source_works[{item.get('name')}]"),
            patterns=_pattern_group(item.get("patterns", {})),
        )
        for item in raw.get("source_works", [])
    ]
    source_rules = [
        SourceRule(
            source_type=str(item["source_type"]),
            sequel_or_spinoff=bool(item.get("sequel_or_spinoff", False)),
            patterns=_pattern_group(item.get("patterns", {})),
        )
        for item in raw.get("source_mappings", [])
    ]
    awareness_rules = [
        AwarenessRule(
            name=str(item["name"]),
            patterns=_pattern_group(item.get("patterns", {})),
        )
        for item in raw.get("awareness_sources", [])
    ]

    return IpRules(
        tier_thresholds=tier_thresholds,
        brand_rules=tuple(brand_rules),
        source_work_rules=tuple(source_work_rules),
        source_rules=tuple(source_rules),
        awareness_rules=tuple(awareness_rules),
    )


def load_raw_movie_metadata(path: Path = DEFAULT_RAW_JSONL_PATH) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    with path.open() as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}") from exc

            payload = record.get("payload")
            if not isinstance(payload, dict):
                raise ValueError(f"Missing payload object at {path}:{line_number}")

            collection = payload.get("belongs_to_collection") or {}
            external_ids = payload.get("external_ids") or {}
            rows.append(
                {
                    "tmdb_id": int(payload.get("id") or record["tmdb_id"]),
                    "collection_id": collection.get("id"),
                    "collection_name": collection.get("name"),
                    "wikidata_id": external_ids.get("wikidata_id"),
                }
            )
    return pd.DataFrame(rows)


def classify_movies(
    movies: pd.DataFrame,
    raw_metadata: pd.DataFrame,
    rules: IpRules,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    normalized = _normalize_columns(movies)
    metadata = raw_metadata.drop_duplicates("tmdb_id")
    df = normalized.merge(metadata, on="tmdb_id", how="left")
    collection_counts = _collection_movie_counts(df)

    records = df.to_dict(orient="records")
    contexts = [_match_context(row) for row in records]
    brand_matches = [_brand_match(context, rules) for context in contexts]
    prior = prior_franchise_stats(
        pd.DataFrame(
            {
                "franchise_key": [_franchise_key(row) for row in records],
                "release_date": _release_dates(df),
                "worldwide_gross": df["worldwide_gross"].astype(float),
                # Rows without a real release_date may actually be later than
                # their Jan-1 fallback suggests, so they never count as prior.
                "counts_as_prior": pd.to_datetime(
                    df["release_date"], errors="coerce"
                ).notna(),
            }
        )
    )

    output_rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    for i, row in enumerate(records):
        classification = _classify_row(
            row,
            contexts[i],
            brand_matches[i],
            rules,
            collection_counts,
            prior_gross=float(prior["prior_gross"].iloc[i]),
            prior_count=int(prior["prior_count"].iloc[i]),
        )
        output_rows.append(_output_row(row, classification))
        audit_rows.extend(_audit_rows(row, classification))

    output = pd.DataFrame(output_rows, columns=OUTPUT_COLUMNS)
    audit = pd.DataFrame(
        audit_rows,
        columns=["tmdb_id", "title", "issue", "details_json"],
    )
    return output, audit


def write_classification_outputs(
    classification: pd.DataFrame,
    audit: pd.DataFrame,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "ip_classification_1980_2026.csv"
    parquet_path = output_dir / "ip_classification_1980_2026.parquet"
    audit_path = output_dir / "ip_classification_audit_1980_2026.csv"

    classification = classification.copy()
    classification["brand_power_tier"] = classification["brand_power_tier"].astype(
        "Int64"
    )
    classification.to_csv(csv_path, index=False)
    classification.to_parquet(parquet_path, index=False)
    audit.to_csv(audit_path, index=False)
    return {
        "csv": str(csv_path),
        "parquet": str(parquet_path),
        "audit_csv": str(audit_path),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Classify movies by pre-release IP strength."
    )
    parser.add_argument("--movies", type=Path, default=DEFAULT_MOVIES_PATH)
    parser.add_argument("--raw-jsonl", type=Path, default=DEFAULT_RAW_JSONL_PATH)
    parser.add_argument("--rules", type=Path, default=DEFAULT_RULES_PATH)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    movies_raw = pd.read_parquet(args.movies)
    movies, source_audit = clean_movie_source_data(movies_raw)
    print(
        f"source cleanup: {len(movies_raw)} -> {len(movies)} rows "
        f"({len(source_audit)} audited fixes/exclusions)"
    )
    raw_metadata = load_raw_movie_metadata(args.raw_jsonl)
    rules = load_rules(args.rules)
    classification, audit = classify_movies(movies, raw_metadata, rules)
    paths = write_classification_outputs(classification, audit, args.out_dir)

    tier_counts = classification["ip_tier"].value_counts().sort_index().to_dict()
    print(f"classified {len(classification)} movies")
    print(f"tier counts: {tier_counts}")
    print(f"audit rows: {len(audit)}")
    print(f"wrote {paths['csv']}")
    print(f"wrote {paths['parquet']}")
    print(f"wrote {paths['audit_csv']}")


def _required_basis(item: dict[str, Any], where: str) -> str:
    basis = str(item.get("basis", "")).strip()
    if not basis:
        raise ValueError(f"{where}: 'basis' is required and must not be empty")
    return basis


def _brand_rule(item: dict[str, Any]) -> BrandRule:
    name = str(item["name"])
    periods = tuple(
        sorted(
            (
                TierPeriod(
                    from_year=int(period["from_year"]),
                    tier=int(period["tier"]),
                    basis=_required_basis(period, f"umbrella_brands[{name}]"),
                )
                for period in item.get("tier_by_period", [])
            ),
            key=lambda period: period.from_year,
        )
    )
    return BrandRule(
        name=name,
        origin=str(item["origin"]),
        rights_status=str(item["rights_status"]),
        periods=periods,
        patterns=_pattern_group(item.get("patterns", {})),
    )


def _pattern_group(raw: dict[str, list[str]]) -> PatternGroup:
    return PatternGroup(
        title=_compile_patterns(raw.get("title", [])),
        collection=_compile_patterns(raw.get("collection", [])),
        keyword=_compile_patterns(raw.get("keyword", [])),
        company=_compile_patterns(raw.get("company", [])),
    )


def _compile_patterns(patterns: list[str]) -> tuple[re.Pattern[str], ...]:
    return tuple(re.compile(pattern, re.IGNORECASE) for pattern in patterns)


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    renamed = df.rename(columns={column: column.lower() for column in df.columns})
    required = ["tmdb_id", "title", "release_year", "worldwide_gross"]
    missing = [column for column in required if column not in renamed.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")

    normalized = renamed.copy()
    for column in [
        "imdb_id",
        "original_title",
        "keywords",
        "overview",
        "tagline",
        "production_company",
        "release_date",
    ]:
        if column not in normalized.columns:
            normalized[column] = None
    normalized["tmdb_id"] = normalized["tmdb_id"].astype(int)
    return normalized


def _release_dates(df: pd.DataFrame) -> pd.Series:
    """release_date as datetime; missing dates fall back to Jan 1 of the year."""
    dates = pd.to_datetime(df["release_date"], errors="coerce")
    fallback = pd.to_datetime(
        df["release_year"].astype("Int64").astype(str) + "-01-01", errors="coerce"
    )
    return dates.fillna(fallback)


def _franchise_key(row: dict[str, Any]) -> str | None:
    """TMDB collection only. Umbrella brands contribute solely via
    tier_by_period; pooling gross across a whole brand (all Marvel films,
    all fairy tales) would inflate tiers with unrelated films' money."""
    collection_id = _optional_int(row.get("collection_id"))
    if collection_id is not None:
        return f"collection:{collection_id}"
    return None


def _collection_movie_counts(df: pd.DataFrame) -> dict[int, int]:
    rows = df[df["collection_id"].notna()]
    if rows.empty:
        return {}
    counts = rows.groupby("collection_id")["tmdb_id"].size()
    return {int(cid): int(count) for cid, count in counts.items()}


def _brand_tier_at(rule: BrandRule, year: int) -> TierPeriod | None:
    applicable = [period for period in rule.periods if period.from_year <= year]
    if not applicable:
        return None
    return max(applicable, key=lambda period: period.from_year)


def _tier_from_prior_gross(
    gross: float, thresholds: TierThresholds
) -> tuple[int, str] | None:
    for tier, threshold in (
        (1, thresholds.prior_tier_1),
        (2, thresholds.prior_tier_2),
        (3, thresholds.prior_tier_3),
    ):
        if gross >= threshold:
            return tier, f"prior_franchise_gross ${gross:,.0f} >= ${threshold:,.0f}"
    return None


def _source_work_match(
    context: MatchContext,
    rules: IpRules,
    year: int,
) -> SourceWorkRule | None:
    for rule in rules.source_work_rules:
        if year >= rule.from_year and _first_match(rule.patterns, context):
            return rule
    return None


def _resolve_tier(
    *,
    brand_period: TierPeriod | None,
    brand_name: str | None,
    gross_tier: tuple[int, str] | None,
    source_work: SourceWorkRule | None,
    floor: tuple[int, str],
) -> tuple[int, str]:
    candidates: list[tuple[int, str]] = [floor]
    if brand_period is not None:
        candidates.append(
            (
                brand_period.tier,
                f"brand:{brand_name} tier {brand_period.tier} "
                f"from {brand_period.from_year}: {brand_period.basis}",
            )
        )
    if gross_tier is not None:
        candidates.append(gross_tier)
    if source_work is not None:
        candidates.append(
            (SOURCE_WORK_TIER, f"source_work:{source_work.name}: {source_work.basis}")
        )
    return min(candidates, key=lambda candidate: candidate[0])


def _floor_tier(
    brand_rule: BrandRule | None,
    source_match: tuple[SourceRule, PatternMatch] | None,
    prior_count: int,
) -> tuple[int, str]:
    if source_match:
        return NOMINAL_IP_TIER, f"nominal_ip:{source_match[0].source_type}"
    if prior_count > 0:
        return NOMINAL_IP_TIER, "nominal_ip:franchise_followup"
    if brand_rule is not None and brand_rule.origin != "original_film":
        return NOMINAL_IP_TIER, f"nominal_ip:{brand_rule.origin}"
    return NO_IP_TIER, "original"


def _classify_row(
    row: dict[str, Any],
    context: MatchContext,
    brand_match: tuple[BrandRule, PatternMatch] | None,
    rules: IpRules,
    collection_counts: dict[int, int],
    *,
    prior_gross: float,
    prior_count: int,
) -> Classification:
    year = int(row["release_year"])
    awareness = _awareness_source(context, rules)
    source_match = _source_match(context, rules)
    source_work = _source_work_match(context, rules, year)
    gross_tier = _tier_from_prior_gross(prior_gross, rules.tier_thresholds)

    def resolve(brand_rule: BrandRule | None) -> tuple[int, str]:
        return _resolve_tier(
            brand_period=_brand_tier_at(brand_rule, year) if brand_rule else None,
            brand_name=brand_rule.name if brand_rule else None,
            gross_tier=gross_tier,
            source_work=source_work,
            floor=_floor_tier(brand_rule, source_match, prior_count),
        )

    if brand_match:
        rule, match = brand_match
        tier, tier_basis = resolve(rule)
        brand_period = _brand_tier_at(rule, year)
        evidence: dict[str, Any] = {
            "matched_rule": rule.name,
            "matched_field": match.field,
            "matched_pattern": match.pattern,
            "prior_franchise_gross": prior_gross,
            "prior_franchise_film_count": prior_count,
            "tier_basis": tier_basis,
        }
        if row.get("collection_name"):
            evidence["collection_name"] = row.get("collection_name")
            evidence["collection_id"] = _optional_int(row.get("collection_id"))
        return Classification(
            ip_tier=tier,
            ip_name=rule.name,
            ip_sub_ip_name=_sub_ip_name(rule.name, row),
            ip_scope=_brand_scope(rule, row),
            ip_source_type=rule.origin,
            brand_origin=rule.origin,
            brand_power_tier=brand_period.tier if brand_period else None,
            ip_rights_status=rule.rights_status,
            awareness_source=awareness,
            is_sequel_or_spinoff=_is_sequel_or_spinoff(source_match) or prior_count > 0,
            prior_franchise_gross=prior_gross,
            tier_basis=tier_basis,
            confidence="high",
            evidence=evidence,
        )

    collection_id = _optional_int(row.get("collection_id"))
    if collection_id is not None:
        movie_count = collection_counts.get(collection_id, 0)
        if _is_awareness_only(awareness) and movie_count <= 1 and source_work is None:
            return _awareness_only_classification(
                awareness,
                {
                    "collection_id": collection_id,
                    "collection_name": row.get("collection_name"),
                    "reason": "single_movie_collection_awareness_only",
                },
            )
        tier, tier_basis = resolve(None)
        source_type = source_match[0].source_type if source_match else "original_film"
        return Classification(
            ip_tier=tier,
            ip_name=_clean_collection_name(row.get("collection_name")),
            ip_sub_ip_name=None,
            ip_scope="direct_collection",
            ip_source_type=source_type,
            brand_origin=source_type,
            brand_power_tier=None,
            ip_rights_status=None,
            awareness_source=awareness,
            is_sequel_or_spinoff=_is_sequel_or_spinoff(source_match) or prior_count > 0,
            prior_franchise_gross=prior_gross,
            tier_basis=tier_basis,
            confidence="high",
            evidence={
                "collection_id": collection_id,
                "collection_name": row.get("collection_name"),
                "collection_movie_count": movie_count,
                "prior_franchise_gross": prior_gross,
                "prior_franchise_film_count": prior_count,
                "tier_basis": tier_basis,
            },
        )

    if source_match or source_work:
        if _is_awareness_only(awareness) and source_work is None:
            rule, match = source_match  # type: ignore[misc]
            return _awareness_only_classification(
                awareness,
                {
                    "suppressed_source": rule.source_type,
                    "matched_field": match.field,
                    "matched_pattern": match.pattern,
                    "reason": "awareness_source_not_ip",
                },
            )
        tier, tier_basis = resolve(None)
        source_type = source_match[0].source_type if source_match else None
        evidence = {
            "prior_franchise_gross": prior_gross,
            "tier_basis": tier_basis,
        }
        if source_match:
            evidence["matched_source"] = source_match[0].source_type
            evidence["matched_field"] = source_match[1].field
            evidence["matched_pattern"] = source_match[1].pattern
        if source_work:
            evidence["matched_source_work"] = source_work.name
        return Classification(
            ip_tier=tier,
            ip_name=source_work.name if source_work else None,
            ip_sub_ip_name=None,
            ip_scope="adaptation",
            ip_source_type=source_type,
            brand_origin=source_type,
            brand_power_tier=None,
            ip_rights_status=None,
            awareness_source=awareness,
            is_sequel_or_spinoff=_is_sequel_or_spinoff(source_match),
            prior_franchise_gross=prior_gross,
            tier_basis=tier_basis,
            confidence="medium",
            evidence=evidence,
        )

    return Classification(
        ip_tier=NO_IP_TIER,
        ip_name=None,
        ip_sub_ip_name=None,
        ip_scope="none",
        ip_source_type=None,
        brand_origin=None,
        brand_power_tier=None,
        ip_rights_status=None,
        awareness_source=awareness,
        is_sequel_or_spinoff=False,
        prior_franchise_gross=prior_gross,
        tier_basis="original",
        confidence="high",
        evidence={},
    )


def _is_awareness_only(awareness: str | None) -> bool:
    if not awareness:
        return False
    sources = set(awareness.split(","))
    return bool(sources & {"true_story", "biography", "historical_event"})


def _awareness_only_classification(
    awareness: str,
    evidence: dict[str, Any],
) -> Classification:
    return Classification(
        ip_tier=NO_IP_TIER,
        ip_name=None,
        ip_sub_ip_name=None,
        ip_scope="none",
        ip_source_type=None,
        brand_origin=None,
        brand_power_tier=None,
        ip_rights_status=None,
        awareness_source=awareness,
        is_sequel_or_spinoff=False,
        prior_franchise_gross=0.0,
        tier_basis="awareness_only",
        confidence="high",
        evidence=evidence,
    )


def _match_context(row: dict[str, Any]) -> MatchContext:
    title_parts = [row.get("title"), row.get("original_title")]
    keyword_parts = [row.get("keywords"), row.get("overview"), row.get("tagline")]
    return MatchContext(
        title_text=_join_text(title_parts),
        collection_text=_join_text([row.get("collection_name")]),
        keyword_text=_join_text(keyword_parts),
        company_text=_join_text([row.get("production_company")]),
    )


def _join_text(values: list[Any]) -> str:
    parts: list[str] = []
    for value in values:
        if value is None:
            continue
        if isinstance(value, list):
            parts.extend(str(item) for item in value if item is not None)
            continue
        if pd.isna(value):
            continue
        parts.append(str(value))
    return " | ".join(parts).lower()


def _brand_match(
    context: MatchContext,
    rules: IpRules,
) -> tuple[BrandRule, PatternMatch] | None:
    for rule in rules.brand_rules:
        match = _first_match(rule.patterns, context)
        if match:
            return rule, match
    return None


def _source_match(
    context: MatchContext,
    rules: IpRules,
) -> tuple[SourceRule, PatternMatch] | None:
    for rule in rules.source_rules:
        match = _first_match(rule.patterns, context)
        if match:
            return rule, match
    return None


def _awareness_source(context: MatchContext, rules: IpRules) -> str | None:
    matches = [
        rule.name
        for rule in rules.awareness_rules
        if _first_match(rule.patterns, context)
    ]
    if not matches:
        return None
    return ",".join(matches)


def _first_match(
    patterns: PatternGroup,
    context: MatchContext,
) -> PatternMatch | None:
    fields = {
        "title": (patterns.title, context.title_text),
        "collection": (patterns.collection, context.collection_text),
        "keyword": (patterns.keyword, context.keyword_text),
        "company": (patterns.company, context.company_text),
    }
    for field, (compiled_patterns, text) in fields.items():
        for pattern in compiled_patterns:
            if pattern.search(text):
                return PatternMatch(field=field, pattern=pattern.pattern)
    return None


def _clean_collection_name(value: Any) -> str | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    return re.sub(r"\s+collection$", "", text, flags=re.IGNORECASE)


def _sub_ip_name(brand_name: str, row: dict[str, Any]) -> str | None:
    collection_name = _clean_collection_name(row.get("collection_name"))
    if collection_name and collection_name.lower() != brand_name.lower():
        return collection_name
    title = row.get("title")
    return str(title).strip() if title else None


def _brand_scope(rule: BrandRule, row: dict[str, Any]) -> str:
    if rule.rights_status == "public_domain":
        return "public_domain"
    if row.get("collection_name"):
        return "umbrella_inherited"
    return "brand_origin"


def _is_sequel_or_spinoff(
    source_match: tuple[SourceRule, PatternMatch] | None,
) -> bool:
    return bool(source_match and source_match[0].sequel_or_spinoff)


def _optional_int(value: Any) -> int | None:
    if value is None or pd.isna(value):
        return None
    return int(value)


def _output_row(row: dict[str, Any], classification: Classification) -> dict[str, Any]:
    return {
        "tmdb_id": int(row["tmdb_id"]),
        "imdb_id": row.get("imdb_id"),
        "title": row.get("title"),
        "release_year": row.get("release_year"),
        "ip_tier": classification.ip_tier,
        "ip_name": classification.ip_name,
        "ip_sub_ip_name": classification.ip_sub_ip_name,
        "ip_scope": classification.ip_scope,
        "ip_source_type": classification.ip_source_type,
        "brand_origin": classification.brand_origin,
        "brand_power_tier": classification.brand_power_tier,
        "ip_rights_status": classification.ip_rights_status,
        "awareness_source": classification.awareness_source,
        "is_sequel_or_spinoff": classification.is_sequel_or_spinoff,
        "prior_franchise_gross": classification.prior_franchise_gross,
        "tier_basis": classification.tier_basis,
        "confidence": classification.confidence,
        "evidence_json": json.dumps(classification.evidence, sort_keys=True),
    }


def _audit_rows(
    row: dict[str, Any],
    classification: Classification,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if classification.confidence == "low":
        rows.append(_audit_row(row, "low_confidence", classification.evidence))
    if classification.ip_scope == "adaptation":
        rows.append(_audit_row(row, "source_mapping_only", classification.evidence))
    if (
        classification.awareness_source
        and classification.ip_tier < NO_IP_TIER
        and classification.ip_scope not in {"public_domain", "umbrella_inherited"}
    ):
        rows.append(
            _audit_row(
                row,
                "awareness_plus_ip",
                {
                    "awareness_source": classification.awareness_source,
                    "ip_name": classification.ip_name,
                    "ip_scope": classification.ip_scope,
                },
            )
        )
    if (
        classification.ip_scope == "direct_collection"
        and classification.evidence.get("collection_movie_count") == 1
    ):
        rows.append(_audit_row(row, "single_movie_collection", classification.evidence))
    if (
        classification.ip_scope in {"brand_origin", "umbrella_inherited"}
        and classification.evidence.get("matched_field") == "keyword"
    ):
        rows.append(_audit_row(row, "keyword_brand_match", classification.evidence))
    return rows


def _audit_row(
    row: dict[str, Any],
    issue: str,
    details: dict[str, Any],
) -> dict[str, Any]:
    return {
        "tmdb_id": int(row["tmdb_id"]),
        "title": row.get("title"),
        "issue": issue,
        "details_json": json.dumps(details, sort_keys=True),
    }


if __name__ == "__main__":
    main()
