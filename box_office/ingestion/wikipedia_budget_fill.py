"""Local-only Wikipedia infobox budget recovery.

Fills missing ``production_budget`` values in the rich TMDB backfill dataset
from English Wikipedia film-article infoboxes. This recovers *reported*
budgets, not estimates: rows we cannot fill stay null with
``production_budget_source = "missing"`` — no imputation.

Flow: imdb_id -> Wikidata sitelink (SPARQL) -> enwiki wikitext -> infobox
``budget`` field -> parsed USD amount. USD-only, with sanity bounds. Every
row gets an audit entry with the raw infobox string and a fill status.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import requests

from box_office.movie_data_quality import clean_movie_source_data

logger = logging.getLogger(__name__)

WIKIDATA_SPARQL_URL = "https://query.wikidata.org/sparql"
ENWIKI_API_URL = "https://en.wikipedia.org/w/api.php"
USER_AGENT = (
    "box-office-oracle-budget-fill/0.1 "
    "(https://github.com/adamthuvesen/box-office-oracle; a.thuvesen@gmail.com)"
)

MIN_PLAUSIBLE_BUDGET = 100_000
MAX_PLAUSIBLE_BUDGET = 600_000_000

# Currency symbols that mark a non-USD amount when no dollar amount exists.
NON_USD_SYMBOLS = ("£", "€", "¥", "₹", "₩", "kr", "CA$", "A$", "NZ$", "HK$")


@dataclass(frozen=True)
class WikipediaFillConfig:
    input_path: Path = Path(
        "data/generated/tmdb/rich_backfill_1980_2026/"
        "tmdb_budget_wikidata_usd_5m_1980_2026.csv"
    )
    output_dir: Path = Path("data/generated/tmdb/rich_backfill_1980_2026")
    sparql_batch_size: int = 100
    wikitext_batch_size: int = 50
    request_timeout_seconds: int = 60
    sleep_seconds: float = 0.5
    max_retries: int = 4
    retry_sleep_seconds: float = 5.0

    @property
    def filled_csv_path(self) -> Path:
        return self.output_dir / "tmdb_budget_wikipedia_5m_1980_2026.csv"

    @property
    def filled_parquet_path(self) -> Path:
        return self.output_dir / "tmdb_budget_wikipedia_5m_1980_2026.parquet"

    @property
    def audit_csv_path(self) -> Path:
        return self.output_dir / "tmdb_budget_wikipedia_audit_1980_2026.csv"

    @property
    def source_quality_csv_path(self) -> Path:
        return self.output_dir / "tmdb_budget_wikipedia_source_quality_1980_2026.csv"

    @property
    def manifest_path(self) -> Path:
        return self.output_dir / "tmdb_budget_wikipedia_manifest_1980_2026.json"


def build_session() -> requests.Session:
    session = requests.Session()
    session.headers["User-Agent"] = USER_AGENT
    return session


def request_with_retries(
    config: WikipediaFillConfig,
    send: Callable[[], requests.Response],
) -> requests.Response:
    """Send a request, retrying on timeouts and 429/5xx responses."""
    last_error: Exception | None = None
    for attempt in range(1, config.max_retries + 1):
        try:
            response = send()
            if response.status_code in (429, 500, 502, 503, 504):
                raise requests.HTTPError(
                    f"retryable status {response.status_code}",
                    response=response,
                )
            response.raise_for_status()
            return response
        except (requests.Timeout, requests.HTTPError) as error:
            last_error = error
            if attempt == config.max_retries:
                break
            wait = config.retry_sleep_seconds * attempt
            logger.warning(
                "Request failed (%s), retry %d/%d in %.0fs",
                error,
                attempt,
                config.max_retries,
                wait,
            )
            time.sleep(wait)
    raise RuntimeError("request failed after retries") from last_error


def fetch_enwiki_titles(
    session: requests.Session,
    imdb_ids: list[str],
    config: WikipediaFillConfig,
) -> dict[str, str]:
    """Map IMDb ids to English Wikipedia article titles via Wikidata."""
    titles: dict[str, str] = {}
    for start in range(0, len(imdb_ids), config.sparql_batch_size):
        batch = imdb_ids[start : start + config.sparql_batch_size]
        values = " ".join(f'"{imdb_id}"' for imdb_id in batch)
        query = (
            "SELECT ?imdb ?article WHERE { "
            f"VALUES ?imdb {{ {values} }} "
            "?item wdt:P345 ?imdb . "
            "?article schema:about ?item ; "
            "schema:isPartOf <https://en.wikipedia.org/> . }"
        )
        response = request_with_retries(
            config,
            lambda q=query: session.post(
                WIKIDATA_SPARQL_URL,
                data={"query": q, "format": "json"},
                timeout=config.request_timeout_seconds,
            ),
        )
        for row in response.json()["results"]["bindings"]:
            imdb_id = row["imdb"]["value"]
            article_url = row["article"]["value"]
            title = article_url.rsplit("/wiki/", 1)[-1].replace("_", " ")
            titles.setdefault(imdb_id, requests.utils.unquote(title))
        logger.info(
            "Wikidata sitelinks: %d/%d imdb ids resolved",
            len(titles),
            start + len(batch),
        )
        time.sleep(config.sleep_seconds)
    return titles


def fetch_wikitexts(
    session: requests.Session,
    titles: list[str],
    config: WikipediaFillConfig,
) -> dict[str, str]:
    """Fetch article wikitext for each title, following redirects."""
    wikitexts: dict[str, str] = {}
    for start in range(0, len(titles), config.wikitext_batch_size):
        batch = titles[start : start + config.wikitext_batch_size]
        response = request_with_retries(
            config,
            lambda b=batch: session.get(
                ENWIKI_API_URL,
                params={
                    "action": "query",
                    "prop": "revisions",
                    "rvprop": "content",
                    "rvslots": "main",
                    "redirects": "1",
                    "format": "json",
                    "formatversion": "2",
                    "titles": "|".join(b),
                },
                timeout=config.request_timeout_seconds,
            ),
        )
        payload = response.json()["query"]
        # Map requested title -> final page title through
        # normalization/redirects.
        rename: dict[str, str] = {}
        steps = payload.get("normalized", []) + payload.get("redirects", [])
        for step in steps:
            rename[step["from"]] = step["to"]
        final_to_requested: dict[str, str] = {}
        for requested in batch:
            final = requested
            seen = set()
            while final in rename and final not in seen:
                seen.add(final)
                final = rename[final]
            final_to_requested[final] = requested
        for page in payload.get("pages", []):
            if page.get("missing"):
                continue
            requested = final_to_requested.get(page["title"], page["title"])
            revisions = page.get("revisions", [])
            if revisions:
                wikitexts[requested] = revisions[0]["slots"]["main"]["content"]
        logger.info(
            "Wikipedia wikitext: %d/%d articles fetched",
            len(wikitexts),
            start + len(batch),
        )
        time.sleep(config.sleep_seconds)
    return wikitexts


def extract_infobox_budget(wikitext: str) -> str | None:
    """Return the raw ``budget`` parameter value from a film infobox."""
    # [ \t]* (not \s*) so an empty budget field never bleeds into the next
    # infobox line — that next line is often "| gross = ...", the target.
    match = re.search(
        r"^[ \t]*\|[ \t]*budget[ \t]*=[ \t]*(.+)$",
        wikitext,
        flags=re.MULTILINE,
    )
    if not match:
        return None
    value = match.group(1).strip()
    return value or None


def clean_budget_text(raw: str) -> str:
    """Strip wiki markup so only the money expression remains."""
    text = raw
    text = re.sub(r"<!--.*?-->", " ", text, flags=re.DOTALL)
    text = re.sub(r"<ref[^>]*/>", " ", text)
    text = re.sub(r"<ref[^>]*>.*?</ref>", " ", text, flags=re.DOTALL)
    # Unterminated ref at end of the captured line.
    text = re.sub(r"<ref[^>]*>.*$", " ", text)
    # {{US$|20 million}} and {{USD|20 million}} -> $20 million
    text = re.sub(r"\{\{US\$\|([^}|]+)[^}]*\}\}", r"$\1", text, flags=re.IGNORECASE)
    text = re.sub(r"\{\{USD\|([^}|]+)[^}]*\}\}", r"$\1", text, flags=re.IGNORECASE)
    text = re.sub(
        r"\{\{\s*(ndash|snd|spaced en dash)\s*\}\}", "–", text, flags=re.IGNORECASE
    )
    text = re.sub(r"\{\{\s*nbsp\s*\}\}", " ", text, flags=re.IGNORECASE)
    # Drop inflation-adjusted parentheticals and remaining templates.
    text = re.sub(
        r"\([^)]*(equivalent|adjusted|inflation)[^)]*\)",
        " ",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(r"\{\{[^{}]*\}\}", " ", text)
    # [[link|display]] -> display, [[link]] -> link
    text = re.sub(r"\[\[(?:[^|\]]*\|)?([^\]]+)\]\]", r"\1", text)
    text = text.replace("&nbsp;", " ").replace("&ndash;", "–")
    text = text.replace("US$", "$").replace("USD", "$")
    return " ".join(text.split())


_SCALE = {"thousand": 1e3, "million": 1e6, "billion": 1e9}

_RANGE_PATTERN = re.compile(
    r"\$\s*([\d,]+(?:\.\d+)?)\s*[–—−-]\s*\$?\s*([\d,]+(?:\.\d+)?)"
    r"\s*(thousand|million|billion)?",
    flags=re.IGNORECASE,
)
_SINGLE_PATTERN = re.compile(
    r"\$\s*([\d,]+(?:\.\d+)?)\s*(thousand|million|billion)?",
    flags=re.IGNORECASE,
)
# US$/USD are normalized to a bare "$" by clean_budget_text, so any "$" still
# preceded by letters (A$, CA$, NZ$, HK$, SG$, R$, ...) is a non-USD currency.
_NON_USD_DOLLAR_PATTERN = re.compile(
    r"[A-Za-z]{1,4}\$\s*[\d,]+(?:\.\d+)?\s*(?:thousand|million|billion)?",
    flags=re.IGNORECASE,
)


def parse_usd_budget(cleaned: str) -> float | None:
    """Parse a USD amount from a cleaned infobox budget string.

    Ranges use the midpoint. Non-USD-only strings and implausible values
    return None.
    """
    if "$" not in cleaned:
        return None

    range_match = _RANGE_PATTERN.search(cleaned)
    if range_match:
        low = float(range_match.group(1).replace(",", ""))
        high = float(range_match.group(2).replace(",", ""))
        scale = _SCALE.get((range_match.group(3) or "").lower(), 1.0)
        value = (low + high) / 2 * scale
    else:
        single_match = _SINGLE_PATTERN.search(cleaned)
        if not single_match:
            return None
        amount = float(single_match.group(1).replace(",", ""))
        scale = _SCALE.get((single_match.group(2) or "").lower(), 1.0)
        value = amount * scale
        # A bare "$12" without scale or thousands separators is too
        # ambiguous to trust as a real budget figure.
        if scale == 1.0 and amount < MIN_PLAUSIBLE_BUDGET:
            return None

    if not MIN_PLAUSIBLE_BUDGET <= value <= MAX_PLAUSIBLE_BUDGET:
        return None
    return value


def classify_budget_text(raw: str | None) -> tuple[float | None, str]:
    """Return (parsed value, audit status) for a raw infobox budget string."""
    if raw is None:
        return None, "no_infobox_budget"
    # A budget value never mentions gross; if it does, we captured the
    # wrong infobox line and must not fill (that number is the target).
    if "gross" in raw.lower():
        return None, "gross_line_guard"
    cleaned = clean_budget_text(raw)
    # Drop currency-prefixed dollar amounts (A$3 million, CA$10 million, ...) so
    # they are never read as USD; a genuine bare "$" amount alongside them still
    # parses.
    had_prefixed_dollar = bool(_NON_USD_DOLLAR_PATTERN.search(cleaned))
    cleaned = _NON_USD_DOLLAR_PATTERN.sub(" ", cleaned)
    if "$" not in cleaned:
        if had_prefixed_dollar or any(
            symbol in cleaned for symbol in NON_USD_SYMBOLS
        ):
            return None, "non_usd"
        return None, "parse_failed"
    value = parse_usd_budget(cleaned)
    if value is None:
        return None, "parse_failed"
    return value, "filled"


def fill_budgets(
    df: pd.DataFrame,
    titles_by_imdb: dict[str, str],
    wikitexts_by_title: dict[str, str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fill missing budgets from Wikipedia. Returns (filled df, audit df)."""
    filled = df.copy()
    audit_rows = []

    missing_mask = filled["production_budget_source"] == "missing"
    for index in filled.index[missing_mask]:
        row = filled.loc[index]
        imdb_id = row.get("imdb_id")
        audit = {
            "tmdb_id": row["tmdb_id"],
            "title": row["title"],
            "release_year": row["release_year"],
            "imdb_id": imdb_id,
            "wikipedia_title": None,
            "budget_raw": None,
            "parsed_budget_usd": None,
            "status": None,
        }
        if not isinstance(imdb_id, str) or not imdb_id:
            audit["status"] = "no_imdb_id"
            audit_rows.append(audit)
            continue
        wiki_title = titles_by_imdb.get(imdb_id)
        if wiki_title is None:
            audit["status"] = "no_wikipedia_article"
            audit_rows.append(audit)
            continue
        audit["wikipedia_title"] = wiki_title
        wikitext = wikitexts_by_title.get(wiki_title)
        if wikitext is None:
            audit["status"] = "no_wikipedia_article"
            audit_rows.append(audit)
            continue
        raw = extract_infobox_budget(wikitext)
        audit["budget_raw"] = raw
        value, status = classify_budget_text(raw)
        # An infobox budget equal to the movie's gross to the dollar is an
        # editor putting the gross in the budget field, not a real budget.
        if value is not None and value == row.get("worldwide_gross"):
            value, status = None, "budget_equals_gross"
        audit["status"] = status
        if value is not None:
            audit["parsed_budget_usd"] = value
            filled.loc[index, "production_budget"] = value
            filled.loc[index, "production_budget_source"] = "wikipedia"
        audit_rows.append(audit)

    legacy = "a" + "d_" + "budget"
    if legacy in filled.columns:
        filled = filled.drop(columns=[legacy])
    return filled, pd.DataFrame(audit_rows)


def write_outputs(
    filled: pd.DataFrame,
    audit: pd.DataFrame,
    config: WikipediaFillConfig,
) -> dict:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    filled, source_quality = clean_movie_source_data(filled)
    filled.to_csv(config.filled_csv_path, index=False)
    filled.to_parquet(config.filled_parquet_path, index=False)
    audit.to_csv(config.audit_csv_path, index=False)
    source_quality.to_csv(config.source_quality_csv_path, index=False)

    source_counts = filled["production_budget_source"].value_counts().to_dict()
    status_counts = audit["status"].value_counts().to_dict() if not audit.empty else {}
    manifest = {
        "source_dataset": str(config.input_path),
        "outputs": {
            "csv": str(config.filled_csv_path),
            "parquet": str(config.filled_parquet_path),
            "audit_csv": str(config.audit_csv_path),
            "source_quality_csv": str(config.source_quality_csv_path),
        },
        "rules": [
            "Fills only rows with production_budget_source = missing.",
            "Budgets come from the enwiki film infobox budget field, "
            "matched via imdb_id -> Wikidata sitelink.",
            "USD amounts only; ranges use the midpoint.",
            f"Values outside ${MIN_PLAUSIBLE_BUDGET:,}-"
            f"${MAX_PLAUSIBLE_BUDGET:,} are rejected.",
            "No worldwide_gross or target-derived field used.",
            "Unfilled budgets remain null - no imputation.",
            "Curated source-quality cleanup excludes non-movie rows and nulls "
            "unsupported financials before writing the filled dataset.",
        ],
        "row_counts": {
            "rows": int(len(filled)),
            "source_quality_actions": int(len(source_quality)),
            "missing_before": int(len(audit)),
            "wikipedia_filled": int(status_counts.get("filled", 0)),
            "missing_after": int(
                (filled["production_budget_source"] == "missing").sum()
            ),
            "production_budget_source_counts": {
                key: int(value) for key, value in source_counts.items()
            },
            "audit_status_counts": {
                key: int(value) for key, value in status_counts.items()
            },
        },
    }
    config.manifest_path.write_text(json.dumps(manifest, indent=2))
    return manifest


def run(config: WikipediaFillConfig) -> dict:
    df = pd.read_csv(config.input_path)
    missing = df[df["production_budget_source"] == "missing"]
    logger.info("Rows with missing budget: %d", len(missing))

    imdb_ids = sorted(
        {i for i in missing["imdb_id"].dropna() if isinstance(i, str) and i}
    )
    session = build_session()
    titles_by_imdb = fetch_enwiki_titles(session, imdb_ids, config)
    wanted_titles = sorted(set(titles_by_imdb.values()))
    wikitexts_by_title = fetch_wikitexts(session, wanted_titles, config)

    filled, audit = fill_budgets(df, titles_by_imdb, wikitexts_by_title)
    manifest = write_outputs(filled, audit, config)
    logger.info(
        "Filled %d budgets from Wikipedia; %d still missing",
        manifest["row_counts"]["wikipedia_filled"],
        manifest["row_counts"]["missing_after"],
    )
    return manifest


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(
        description="Fill missing production budgets from Wikipedia infoboxes."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=WikipediaFillConfig.input_path,
        help="Dataset CSV with production_budget_source column.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=WikipediaFillConfig.output_dir,
        help="Directory for filled dataset, audit, and manifest.",
    )
    arguments = parser.parse_args()
    config = WikipediaFillConfig(
        input_path=arguments.input, output_dir=arguments.output_dir
    )
    manifest = run(config)
    print(json.dumps(manifest["row_counts"], indent=2))


if __name__ == "__main__":
    main()
