"""Historical OpenAI-assisted enrichment helper.

This script is kept as reference material for the original data-gathering
workflow. The maintained ingestion path is the package CLI documented in
``data/README.md``.
"""

import os
import json
import re
import logging
from typing import Dict, List, Tuple, Union

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client: OpenAI | None = None


NUMERIC_FIELDS = {
    "domestic_gross",
    "worldwide_gross",
    "production_budget",
    "ad_budget",
    "social_media_buzz",
    "franchise_rating",
}

def _collapse_us_int(s: str) -> str:
    """ '52_500_000' -> '52500000'  (Box‑office sites sometimes use underscores) """
    return re.sub(r"(\d+)_(\d+)", r"\1\2", s)

def _normalise_movie_dict(movie: Dict[str, Union[str, int, list, None]]) -> Dict:
    """Post‑process model JSON to consistent pythonic types."""
    for k, v in list(movie.items()):
        # convert "unknown" → None
        if isinstance(v, str) and v.lower() == "unknown":
            movie[k] = None
            continue

        # numbers may arrive as strings, possibly with underscores or commas
        if k in NUMERIC_FIELDS and isinstance(v, str):
            v_clean = _collapse_us_int(v.replace(",", ""))
            if v_clean.isdigit():
                movie[k] = int(v_clean)
            else:
                movie[k] = None

        # trim actor list to 3
        if k == "actors" and isinstance(v, list):
            movie[k] = v[:3]

    return movie


def _get_openai_client() -> OpenAI:
    global client
    if client is None:
        if not OPENAI_API_KEY:
            raise ValueError("Please set OPENAI_API_KEY in your environment")
        client = OpenAI(api_key=OPENAI_API_KEY)
    return client


def _extract_json_block(text: str) -> str:
    """
    Return the substring between <<JSON_OUTPUT_ONLY>> … <<END>>.
    Raises ValueError if the markers are missing.
    """
    m = re.search(r"<<JSON_OUTPUT_ONLY>>(.*?)<<END>>", text, flags=re.S)
    if not m:
        raise ValueError("JSON markers not found in model output")
    block = m.group(1).strip()

    # Strip code‑block fences if the model wrapped the JSON in ```json … ```
    if block.startswith("```"):
        block = re.sub(r"^```[a-zA-Z]*\n?", "", block)
    if block.endswith("```"):
        block = block[:-3]
    return block.strip()


def batch_enrich_movies(
    movies: List[Dict[str, Union[str, int]]],
    *,
    max_retries: int = 1
) -> Tuple[Dict[str, Dict], int, int, int]:
    """
    Ask GPT‑4.1 for missing financial / credit data on a batch of movies.
    Returns: (results_dict, input_tokens, output_tokens, total_tokens)
    """
    logging.info("Processing batch of %s movies", len(movies))

    search_prompt = f"""
You are a **senior movie‑data researcher**.  For each film below, run Google searches
and extract the fields listed. EXTREMELY IMPORTANT: RETURN IN JSON FORMAT.

############  SEARCH STRATEGY  ############
For every film do AT LEAST these four focused probes:
1. "\"{{title}}\" {{year}} movie worldwide box office " (for revenue data)
2. "\"{{title}}\" {{year}} movie production budget cost" (for production costs)
3. "\"{{title}}\" {{year}} movie marketing budget advertising P&A" (for promotional spend)
4. "\"{{title}}\" {{year}} movie director cast MPAA rating release type" (for metadata & credits)

If a probe returns irrelevant results (e.g. “civil war history”) ADD the director surname
or the word “film” to the query and retry once.

############  EXTRACTION CUES  ############
- Prefer **Box Office Mojo** or **TheNumbers** tables for grosses.
- Prefer IMDb → “Box office” → “Budget” / “Opening weekend US & Canada” for theatre count.
- For `production_budget` also scan Variety / Deadline sentences starting “Budget:”.
- List the top 3 billed actors from IMDb.
- Release is “wide” if opening‑week theatre count ≥ 2 000, else “limited”.
- Franchise rating: 2 if part of Marvel/DC/Star Wars/major IP; 1 for sequels or shared universes; 0 otherwise.
- Estimate social media buzz from the number of likes, comments, shares, etc. on social media platforms and youtube views, 1-3 for low, 4-6 for medium, 7-10 for high (rare).

############  FALL‑BACK RULES  ############
If production_budget missing:
    estimate using worldwide_gross (60% for blockbusters; 40% mid‑range; 25% small).
If ad_budget missing:
    ad_budget = round(0.5 * production_budget)

############  MOVIES TO RESEARCH  ############
{json.dumps([{"tmdb_id": m["tmdb_id"], "title": m["title"], "year": m["year"]} for m in movies], indent=2)}

############  OUTPUT FORMAT  ############
Return **one** JSON object.
**EXTREMELY IMPORTANT:** Every TMDB id listed above **MUST** be present as a key, followed by a dictionary of values.

Example (schema only):
{{
"<tmdb_id>": {{
    "domestic_gross": <int|"unknown">,
    "worldwide_gross": <int|"unknown">,
    "production_budget": <int|"unknown">,
    "director": "<string|unknown>",
    "actors": ["<actor_1>", "<actor_2>", "<actor_3>"],
    "mpaa": "<G|PG|PG-13|R|NC-17|unknown>",
    "social_media_buzz": <1‑10>,
    "release_type": "<wide|limited>",
    "franchise_rating": <0|1|2>,
    "ad_budget": <int>,
    "production_company": "<string|unknown>"
}},
...
}}
<<END>>
""".strip()

    # -------------------------------------------------------------------------
    def _call_openai(temp: float):
        return _get_openai_client().responses.create(
            model="gpt-4.1",
            input=search_prompt,
            tools=[{"type": "web_search"}],
            temperature=temp
        )

    for attempt in range(max_retries + 1):
        response = _call_openai(0.9 if attempt == 0 else 0.0)

        usage = getattr(response, "usage", None) or {}
        in_tok = getattr(usage, "input_tokens", 0)
        out_tok = getattr(usage, "output_tokens", 0)
        tot_tok = getattr(usage, "total_tokens", 0)

        text_chunks = []
        for msg in response.output:
            if getattr(msg, "content", None):
                for part in msg.content:
                    if getattr(part, "text", None):
                        text_chunks.append(part.text)
        full_text = "".join(text_chunks)

        try:
            json_match = re.search(r'\{.*\}', full_text, re.S)
            if not json_match:
                raise ValueError("Unable to locate JSON braces in model output")
            json_block = json_match.group(0)
            raw = json.loads(_collapse_us_int(json_block))
            break  # JSON parsed OK
        except Exception as e:
            logging.warning("Parsing failed on attempt %s: %s", attempt + 1, e)
            raw = {}

    if not raw:
        logging.error("Could not parse model output after %s attempt(s)", attempt + 1)
        return (
            {str(m["tmdb_id"]): {f: None for f in NUMERIC_FIELDS} for m in movies},
            in_tok,
            out_tok,
            tot_tok,
        )

    parsed: Dict[str, Dict] = {}
    for tmdb, data in raw.items():
        parsed[tmdb] = _normalise_movie_dict(data)

    for m in movies:
        t_id = str(m["tmdb_id"])
        if t_id not in parsed:
            logging.warning("Movie %s absent from model output", t_id)
            parsed[t_id] = {f: None for f in NUMERIC_FIELDS}

    return parsed, in_tok, out_tok, tot_tok

def chunked(iterable: List, size: int):
    for i in range(0, len(iterable), size):
        yield iterable[i : i + size]


def process_csv(
    input_csv: str,
    output_csv: str,
    *,
    batch_size: int = 4,
    num_films: int = 20,
) -> None:

    print(f"\nLoading data from: {input_csv}")
    df = pd.read_csv(input_csv)

    required_cols = ["id", "title", "release_date"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"CSV must contain columns: {missing}")
    print("Required columns present:", ", ".join(required_cols))

    if "budget" in df.columns:
        df["production_budget"] = df["budget"]
    elif "production_budget" not in df.columns:
        df["production_budget"] = None

    if "revenue" in df.columns:
        df["worldwide_gross"] = df["revenue"]
    elif "worldwide_gross" not in df.columns:
        df["worldwide_gross"] = None

    df["release_year"] = pd.to_datetime(
        df["release_date"], errors="coerce"
    ).dt.year
    df = df.dropna(subset=["release_year", "title"])

    print(f"Dataset contains {len(df)} rows after date cleanup")

    df = df.sort_values("vote_count", ascending=False)

    processed_ids = set()
    if os.path.exists(output_csv):
        try:
            prev = pd.read_csv(output_csv)
            processed_ids = set(prev["tmdb_id"].dropna().astype(int))
            print(f"Found {len(processed_ids)} movies already enriched")
        except Exception as e:
            print(f"Could not read {output_csv}: {e}")

    mask_new = ~df["id"].astype(int).isin(processed_ids)
    df_to_process = df[mask_new].head(num_films).copy()
    if df_to_process.empty:
        print("Nothing new to enrich.")
        return

    movies = [
        {
            "tmdb_id": int(row.id),
            "title": row.title,
            "year": int(row.release_year),
        }
        for row in df_to_process.itertuples(index=False)
    ]

    all_results: Dict[str, Dict] = {}
    total_in = total_out = total_tot = 0

    for i, batch in enumerate(chunked(movies, batch_size), 1):
        print(f"\nBatch {i}/{-(-len(movies)//batch_size)}")
        res, in_tok, out_tok, tot_tok = batch_enrich_movies(batch)
        all_results.update(res)
        total_in += in_tok
        total_out += out_tok
        total_tot += tot_tok

    print(
        f"\nTokens - input: {total_in:,}, output: {total_out:,}, "
        f"total: {total_tot:,}"
    )

    def _pick(row, field):
        """Prefer existing data; otherwise take model answer."""
        existing = getattr(row, field, None)
        if pd.notna(existing):
            return existing
        tmdb_id = str(int(row.id))
        return all_results.get(tmdb_id, {}).get(field)

    enrich_fields = [
        "domestic_gross",
        "worldwide_gross",
        "production_budget",
        "director",
        "actors",
        "mpaa",
        "social_media_buzz",
        "release_type",
        "franchise_rating",
        "ad_budget",
        "production_company",
    ]

    for f in enrich_fields:
        df_to_process[f] = df_to_process.apply(_pick, axis=1, field=f)

    df_enriched = df_to_process.reset_index(drop=True)
    df_enriched["rank"] = df_enriched.index + 1

    df_enriched = df_enriched.rename(
        columns={
            "id": "tmdb_id",
            "vote_average": "rating",
            "vote_count": "votes",
        }
    )

    target_cols = [
        "tmdb_id",
        "imdb_id",
        "rank",
        "title",
        "worldwide_gross",
        "domestic_gross",
        "release_date",
        "rating",
        "votes",
        "original_language",
        "production_countries",
        "genres",
        "production_budget",
        "director",
        "actors",
        "mpaa",
        "social_media_buzz",
        "release_type",
        "franchise_rating",
        "runtime",
        "overview",
        "tagline",
        "keywords",
        "ad_budget",
        "production_company",
        "release_year",
    ]
    for c in target_cols:
        if c not in df_enriched.columns:
            df_enriched[c] = None
    df_enriched = df_enriched[target_cols]

    if processed_ids:
        prev = pd.read_csv(output_csv)
        # Ensure schema parity
        for c in target_cols:
            if c not in prev.columns:
                prev[c] = None
        final_df = pd.concat([prev, df_enriched], ignore_index=True)
        final_df = final_df.sort_values("votes", ascending=False).reset_index(
            drop=True
        )
        final_df["rank"] = final_df.index + 1
    else:
        final_df = df_enriched

    print(f"\nWriting {len(final_df)} rows to {output_csv}")
    final_df.to_csv(output_csv, index=False)
    print("Done.")


if __name__ == "__main__":
    INPUT_CSV_PATH = "data/external/tmdb/suggested_movies_2000_2019_extra.csv"
    OUTPUT_CSV_PATH = "data/external/tmdb/enriched_movies_2000_2019_extra.csv"
    NUM_FILMS = 300  
    BATCH_SIZE = 1  
    
    print(f"Input file: {INPUT_CSV_PATH}")
    print(f"Output file: {OUTPUT_CSV_PATH}")
    print(f"Processing {NUM_FILMS} unprocessed films")
    
    process_csv(INPUT_CSV_PATH, OUTPUT_CSV_PATH, batch_size=BATCH_SIZE, num_films=NUM_FILMS)
