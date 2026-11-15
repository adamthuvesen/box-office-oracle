"""Seeded synthetic movie fixtures for the verify drives. Nothing here is real movie data.

Run inside a checkout copy: ``python fixtures.py``. It writes the three inputs that
``scripts/prepare_training_frame.py`` reads, at the paths that script names, and prints a JSON
manifest of the facts the drives assert against.
"""

import json
import sys

import numpy as np
import pandas as pd

from box_office.ml.feature_pipeline.constants import GENRE_VOCABULARY
from box_office.training_frame import COLUMN_MAPPING
from scripts.prepare_training_frame import DATASET, IP_CLASSIFICATION_PATH, RAW_JSONL_PATH

SEED = 20261112
FIRST_YEAR = 1990
LAST_FINAL_YEAR = 2025
FILMS_PER_YEAR = 12
FIRST_TMDB_ID = 9_000_001  # above any real TMDB id, so curated source fixes never match
FRANCHISE_COUNT = 40
FRANCHISE_SIZE = 3
MPAA_RATINGS = ("G", "PG", "PG-13", "R", "Not Rated")
MISSING_BUDGET_SHARE = 0.08
MISSING_BUDGET_GROSS_CAP = 90_000_000.0


def _title_case(token: str) -> str:
    return token.replace("_", " ").title()


def _pick_names(rng: np.random.Generator, prefix: str, pool: int, count: int) -> str:
    chosen = rng.choice(pool, size=count, replace=False)
    return ", ".join(f"{prefix} {int(n) + 1:02d}" for n in chosen)


def _normal_rows(rng: np.random.Generator) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    genre_labels = [_title_case(token) for token in GENRE_VOCABULARY]
    for year in range(FIRST_YEAR, LAST_FINAL_YEAR + 1):
        for _ in range(FILMS_PER_YEAR):
            budget = float(np.exp(rng.normal(17.5, 1.0)))
            genres = rng.choice(genre_labels, size=int(rng.integers(1, 4)), replace=False)
            log_gross = (
                np.log(budget)
                + 0.9
                + 0.25 * ("Action" in genres)
                - 0.2 * ("Drama" in genres)
                + rng.normal(0, 0.6)
            )
            gross = float(np.exp(log_gross))
            budget_missing = bool(rng.random() < MISSING_BUDGET_SHARE)
            if budget_missing:
                gross = min(gross, MISSING_BUDGET_GROSS_CAP)
            rows.append(
                {
                    "release_year": year,
                    "release_date": f"{year}-{int(rng.integers(1, 13)):02d}-{int(rng.integers(1, 29)):02d}",
                    "production_budget": np.nan if budget_missing else round(budget),
                    "production_budget_was_missing": budget_missing,
                    "runtime": int(rng.integers(80, 160)),
                    "mpaa": str(rng.choice(MPAA_RATINGS)),
                    "genres": ", ".join(genres),
                    "director": f"Synthetic Director {int(rng.integers(1, 26)):02d}",
                    "production_company": f"Synthetic Studio {int(rng.integers(1, 11)):02d}",
                    "actors": _pick_names(rng, "Synthetic Actor", 40, int(rng.integers(1, 4))),
                    "worldwide_gross": round(gross),
                }
            )
    return rows


def _designed_drops(template: dict[str, object]) -> list[dict[str, object]]:
    """Rows the quality gate must drop: short runtime, no budget over $100M, future year, no gross."""
    drops: list[dict[str, object]] = []
    for _ in range(4):
        drops.append(
            {**template, "runtime": 40, "release_year": 2005, "release_date": "2005-03-03"}
        )
    for _ in range(3):
        drops.append(
            {
                **template,
                "production_budget": np.nan,
                "production_budget_was_missing": True,
                "worldwide_gross": 300_000_000,
                "release_year": 2010,
                "release_date": "2010-07-07",
            }
        )
    for _ in range(6):
        drops.append({**template, "release_year": 2026, "release_date": "2026-06-06"})
    for _ in range(3):
        drops.append(
            {
                **template,
                "worldwide_gross": np.nan,
                "release_year": 2012,
                "release_date": "2012-02-02",
            }
        )
    return drops


def build_source() -> tuple[pd.DataFrame, dict[str, object]]:
    rng = np.random.default_rng(SEED)
    normal = _normal_rows(rng)
    drops = _designed_drops(normal[0])
    frame = pd.DataFrame([*normal, *drops])
    frame.insert(0, "tmdb_id", np.arange(FIRST_TMDB_ID, FIRST_TMDB_ID + len(frame)))
    frame["imdb_id"] = [f"tt{i:07d}" for i in frame["tmdb_id"] - FIRST_TMDB_ID + 1]
    frame["title"] = [f"Synthetic Film {i:04d}" for i in frame["tmdb_id"] - FIRST_TMDB_ID + 1]
    frame["overview"] = "A synthetic fixture film."
    frame["tagline"] = ""
    frame["keywords"] = "synthetic, fixture"
    frame["poster_path"] = [f"/synthetic-{i}.jpg" for i in frame["tmdb_id"]]
    frame["backdrop_path"] = [f"/synthetic-bd-{i}.jpg" for i in frame["tmdb_id"]]
    frame["production_budget_source"] = [
        None if was_missing else "synthetic"
        for was_missing in frame["production_budget_was_missing"]
    ]
    missing = [c for c in COLUMN_MAPPING if c not in frame.columns]
    if missing:
        raise SystemExit(f"fixture generator lacks source columns the pipeline needs: {missing}")
    manifest: dict[str, object] = {
        "seed": SEED,
        "source_rows": len(frame),
        "kept_rows": len(normal),
        "dropped_rows": len(drops),
        "first_tmdb_id": FIRST_TMDB_ID,
    }
    return frame, manifest


def _franchise_lines(frame: pd.DataFrame) -> list[str]:
    rng = np.random.default_rng(SEED + 1)
    ids = frame["tmdb_id"].to_numpy()
    members = rng.choice(len(ids), size=FRANCHISE_COUNT * FRANCHISE_SIZE, replace=False)
    lines: list[str] = []
    for k in range(FRANCHISE_COUNT):
        for pos in members[k * FRANCHISE_SIZE : (k + 1) * FRANCHISE_SIZE]:
            collection = {"id": 9_100_000 + k, "name": f"Synthetic Collection {k + 1:02d}"}
            lines.append(
                json.dumps(
                    {"tmdb_id": int(ids[pos]), "payload": {"belongs_to_collection": collection}}
                )
            )
    return lines


def write_fixture() -> dict[str, object]:
    frame, manifest = build_source()
    rng = np.random.default_rng(SEED + 2)
    ip = pd.DataFrame(
        {
            "tmdb_id": frame["tmdb_id"],
            "ip_tier": rng.choice([5, 4, 3, 2, 1], size=len(frame), p=[0.8, 0.1, 0.06, 0.03, 0.01]),
        }
    )
    for path in (DATASET, RAW_JSONL_PATH, IP_CLASSIFICATION_PATH):
        path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(DATASET, index=False)
    ip.to_parquet(IP_CLASSIFICATION_PATH, index=False)
    RAW_JSONL_PATH.write_text("\n".join(_franchise_lines(frame)) + "\n")
    return manifest


if __name__ == "__main__":
    json.dump(write_fixture(), sys.stdout)
