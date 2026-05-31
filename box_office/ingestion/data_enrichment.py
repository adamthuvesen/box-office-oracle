"""
Heuristic data enrichment for movie datasets.

Fills missing fields using rule-based estimation when TMDB data is incomplete.
Refactored from data/external/tmdb/3_process_enriched_dataset_v2.py.
"""

import logging
import random
import re

import pandas as pd

logger = logging.getLogger(__name__)

# Major franchise keywords for franchise_rating detection
MAJOR_FRANCHISE_KEYWORDS = [
    # Marvel Cinematic Universe & Marvel properties
    "Star Wars",
    "Harry Potter",
    "Fantastic Beasts",
    "Avengers",
    "Spider-Man",
    "Iron Man",
    "Captain America",
    "Thor",
    "Guardians of the Galaxy",
    "Black Panther",
    "Doctor Strange",
    "Ant-Man",
    "Captain Marvel",
    "Black Widow",
    "Shang-Chi",
    "Eternals",
    "The Marvels",
    "The Incredible Hulk",
    "Deadpool",
    "Wolverine",
    "Logan",
    "X-Men",
    "Venom",
    "Morbius",
    "Silver Surfer",
    "Fantastic Four",
    # DC Extended Universe & DC properties
    "The Dark Knight",
    "Batman",
    "Joker",
    "Superman",
    "Man of Steel",
    "Wonder Woman",
    "Aquaman",
    "The Flash",
    "Shazam!",
    "Suicide Squad",
    "Justice League",
    "Birds of Prey",
    "Black Adam",
    "Blue Beetle",
    "Green Lantern",
    "Catwoman",
    "The Batman",
    "Batgirl",
    # Fantasy & Adventure franchises
    "The Lord of the Rings",
    "The Hobbit",
    "The Chronicles of Narnia",
    "Pirates of the Caribbean",
    "Indiana Jones",
    "The Matrix",
    "Avatar",
    "Dune",
    "John Carter",
    "Warcraft",
    "Mortal Kombat",
    "Tomb Raider",
    "Resident Evil",
    "Silent Hill",
    "Assassin's Creed",
    # Action franchises
    "Jurassic Park",
    "Jurassic World",
    "Mission: Impossible",
    "Fast & Furious",
    "Hobbs & Shaw",
    "F9",
    "Fast X",
    "2 Fast 2 Furious",
    "Tokyo Drift",
    "Fast Five",
    "Fast & Furious 6",
    "Furious 7",
    "The Fate of the Furious",
    "Transformers",
    "Bumblebee",
    "Top Gun",
    "Die Hard",
    "Rambo",
    "The Expendables",
    "Taken",
    "John Wick",
    "Kingsman",
    "Bad Boys",
    "Terminator",
    "Alien",
    "Predator",
    "RoboCop",
    "Total Recall",
    # Horror franchises
    "It",
    "The Conjuring",
    "Annabelle",
    "The Nun",
    "Insidious",
    "Paranormal Activity",
    "Saw",
    "Scream",
    "Halloween",
    "Friday the 13th",
    "A Nightmare on Elm Street",
    "Child's Play",
    "The Ring",
    "The Grudge",
    "Final Destination",
    "Scary Movie",
    "The Purge",
    "Sinister",
    # Comedy franchises
    "The Hangover",
    "Ted",
    "22 Jump Street",
    "Anchorman",
    "Zoolander",
    "Meet the Parents",
    "American Pie",
    "Scary Movie",
    "The Grown Ups",
    "Night at the Museum",
    "Alvin and the Chipmunks",
    "The Smurfs",
    # Animated franchises
    "Frozen",
    "The Lion King",
    "The Super Mario Bros.",
    "Sonic the Hedgehog",
    "Minions",
    "Despicable Me",
    "Toy Story",
    "Incredibles",
    "Cars",
    "Finding Nemo",
    "Finding Dory",
    "Monsters",
    "Ice Age",
    "Shrek",
    "How to Train Your Dragon",
    "Madagascar",
    "Kung Fu Panda",
    "Rio",
    "The Secret Life of Pets",
    "Sing",
    "Hotel Transylvania",
    "Wreck-It Ralph",
    "Inside Out",
    "Zootopia",
    "Moana",
    "Coco",
    "Onward",
    "Soul",
    "Luca",
    "Turning Red",
    "Lightyear",
    "The Boss Baby",
    "Trolls",
    "The Croods",
    "Turbo",
    "Epic",
    "SpongeBob",
    "The Simpsons",
    "Family Guy",
    "South Park",
    # Monster & Kaiju franchises
    "Godzilla",
    "Kong",
    "Pacific Rim",
    "Cloverfield",
    "The Mummy",
    "Van Helsing",
    "Underworld",
    "Blade",
    "Ghost Rider",
    # Spy & Espionage franchises
    "James Bond",
    "Skyfall",
    "Spectre",
    "Casino Royale",
    "No Time to Die",
    "Quantum of Solace",
    "Goldeneye",
    "Tomorrow Never Dies",
    "The World Is Not Enough",
    "Die Another Day",
    "The Bourne Identity",
    "The Bourne Supremacy",
    "The Bourne Ultimatum",
    "The Bourne Legacy",
    "Jason Bourne",
    "xXx",
    "Men in Black",
    "Spy Kids",
    # Teen & Young Adult franchises
    "Hunger Games",
    "Divergent",
    "The Maze Runner",
    "Percy Jackson",
    "Mortal Instruments",
    "Vampire Academy",
    "The Host",
    "Beautiful Creatures",
    "Eragon",
    "The Golden Compass",
    # Planet of the Apes franchise
    "Planet of the Apes",
    "Rise of the Planet of the Apes",
    "Dawn of the Planet of the Apes",
    "War for the Planet of the Apes",
    # Star Trek franchise
    "Star Trek",
    "Into Darkness",
    "Beyond",
    "The Motion Picture",
    # Miscellaneous franchises
    "Step Up",
    "The Fast and the Furious",
    "Need for Speed",
    "G.I. Joe",
    "Battleship",
    "The A-Team",
    "S.W.A.T.",
    "Charlie's Angels",
    "Lara Croft",
    "The Mummy Returns",
    "Blade Runner",
    "Ghost in the Shell",
    "Alita",
    "Ready Player One",
]

# Pre-compile the keyword list into a single alternation. Replaces ~200
# `re.search` calls per row with one match against a combined pattern.
_FRANCHISE_PATTERN = re.compile(
    r"\b(?:"
    + "|".join(re.escape(kw.lower()) for kw in MAJOR_FRANCHISE_KEYWORDS)
    + r")\b"
)


class HeuristicEnricher:
    """
    Fill missing fields using rule-based estimation.

    Provides fallback values for fields that TMDB doesn't provide:
    - ad_budget: Calculated as 20-70% of production_budget
    - franchise_rating: Based on title keyword matching
    """

    def __init__(self, seed: int = 42):
        """Initialize enricher with a *local* RNG.

        Uses ``random.Random(seed)`` rather than ``random.seed(seed)`` so two
        enrichers (or any other code that uses the module-level ``random``)
        in the same process don't perturb each other. Reproducibility is
        guaranteed per-instance, not per-process.
        """
        self.seed = seed
        self._rng = random.Random(seed)

    def enrich(self, df: pd.DataFrame) -> pd.DataFrame:
        """Run all heuristic enrichment steps and return the enriched copy."""
        df = df.copy()
        logger.info(f"Enriching {len(df)} movies with heuristics...")

        df = self._fill_ad_budget(df)
        df = self._fill_franchise_rating(df)

        logger.info("Heuristic enrichment complete")
        return df

    def _fill_ad_budget(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate ad_budget as percentage of production_budget."""
        if "ad_budget" not in df.columns:
            df["ad_budget"] = None

        budget_col = self._get_budget_column(df)
        if budget_col is None:
            logger.warning("Budget column not found, skipping ad_budget")
            return df

        missing_mask = df["ad_budget"].isna()
        missing_count = missing_mask.sum()

        if missing_count > 0:
            logger.info(f"Generating ad_budget for {missing_count} movies")
            df.loc[missing_mask, "ad_budget"] = df.loc[missing_mask, budget_col].apply(
                self._generate_ad_budget
            )

        return df

    def _fill_franchise_rating(self, df: pd.DataFrame) -> pd.DataFrame:
        """Assign franchise_rating based on title keywords and revenue."""
        if "franchise_rating" not in df.columns:
            df["franchise_rating"] = None

        revenue_col = self._get_revenue_column(df)
        missing_mask = df["franchise_rating"].isna()
        missing_count = int(missing_mask.sum())
        if missing_count == 0:
            return df

        logger.info(f"Generating franchise_rating for {missing_count} movies")

        titles = (
            df.loc[missing_mask, "title"].astype(str)
            if "title" in df.columns
            else pd.Series("", index=df.index[missing_mask])
        )
        revenues = (
            df.loc[missing_mask, revenue_col]
            if revenue_col
            else pd.Series(0, index=df.index[missing_mask])
        )
        df.loc[missing_mask, "franchise_rating"] = [
            self._assign_franchise_rating(t, r) for t, r in zip(titles, revenues)
        ]

        return df

    def _get_budget_column(self, df: pd.DataFrame) -> str:
        """Find the budget column (production_budget or budget)."""
        for col in ["production_budget", "budget"]:
            if col in df.columns:
                return col
        return None

    def _get_revenue_column(self, df: pd.DataFrame) -> str:
        """Find the revenue column (worldwide_gross or revenue)."""
        for col in ["worldwide_gross", "revenue"]:
            if col in df.columns:
                return col
        return None

    def _generate_normal_int(self, min_val: int, max_val: int) -> int:
        """Generate normally distributed integer within range using ``self._rng``."""
        mu = (min_val + max_val) / 2
        sigma = (max_val - min_val) / 3.5

        if sigma == 0:
            return round(mu)

        while True:
            value = self._rng.gauss(mu, sigma)
            int_value = round(value)
            if min_val <= int_value <= max_val:
                return int_value

    def _generate_ad_budget(self, production_budget: float) -> float:
        """Generate ad_budget as percentage of production_budget (uses ``self._rng``)."""
        if pd.isna(production_budget) or production_budget <= 0:
            return 0

        if production_budget < 10_000_000:
            multiplier = self._rng.uniform(0.20, 0.35)
        elif production_budget < 50_000_000:
            multiplier = self._rng.uniform(0.30, 0.40)
        elif production_budget < 100_000_000:
            multiplier = self._rng.uniform(0.40, 0.50)
        elif production_budget < 200_000_000:
            multiplier = self._rng.uniform(0.45, 0.55)
        else:
            multiplier = self._rng.uniform(0.5, 0.70)

        return round(production_budget * multiplier)

    @staticmethod
    def _assign_franchise_rating(title: str, worldwide_gross: float) -> int:
        """
        Assign franchise rating based on title keywords and revenue.

        Returns:
            0: Not a franchise
            1: Minor franchise or sequel
            2: Major franchise (Marvel, DC, Star Wars, etc. with high revenue)
        """
        if not title:
            return 0

        title_lower = title.lower()
        is_major_by_gross = (
            worldwide_gross >= 300_000_000 if pd.notna(worldwide_gross) else False
        )
        is_major_by_keyword = bool(_FRANCHISE_PATTERN.search(title_lower))

        if is_major_by_gross and is_major_by_keyword:
            return 2
        elif is_major_by_keyword:
            return 1
        else:
            return 0


def enrich_dataframe(df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """Convenience wrapper around :class:`HeuristicEnricher`."""
    enricher = HeuristicEnricher(seed=seed)
    return enricher.enrich(df)
