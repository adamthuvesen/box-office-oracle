"""Genre feature transformers."""

from __future__ import annotations

from typing import Dict, List

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer

from box_office.config import config
from box_office.ml.feature_pipeline.constants import GENRE_VOCABULARY
from box_office.ml.text_utils import process_text_list


def _pipe_split(s: str) -> List[str]:
    """Module-level tokenizer for ``CountVectorizer`` so artifacts can pickle.

    ``lambda s: s.split("|")`` worked for in-process fitting but pickle can't
    locate a lambda by qualified name, which broke joblib serialization of
    the trained model.
    """
    return s.split("|")


class GenreTransformer(BaseEstimator, TransformerMixin):
    """Binary genre indicators + super-genre encoding from `GENRES` column."""

    def __init__(self) -> None:
        self.max_features = config.feature_engineering.max_genre_features
        self.vocabulary = list(GENRE_VOCABULARY[: self.max_features])
        self.vectorizer = CountVectorizer(
            vocabulary=self.vocabulary,
            binary=True,
            tokenizer=_pipe_split,
            token_pattern=None,
            lowercase=True,
        )
        self.super_genre_map: Dict[str, int] = {}
        self.super_genre_other_val = -1

    def fit(self, X: pd.DataFrame, y=None) -> "GenreTransformer":
        genre_lists = X["GENRES"].apply(_split_genre_field)
        genre_texts = genre_lists.apply(
            lambda g: "|".join(g).lower() if g else "unknown"
        )
        self.vectorizer.fit(genre_texts)
        super_genres = genre_lists.apply(_map_super_genre)
        top = super_genres.value_counts().index
        self.super_genre_map = {g: i for i, g in enumerate(top)}
        self.super_genre_other_val = -1
        self.is_fitted_ = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        genre_lists = X["GENRES"].apply(_split_genre_field)
        genre_texts = genre_lists.apply(
            lambda g: "|".join(g).lower() if g else "unknown"
        )
        mat = self.vectorizer.transform(genre_texts).toarray()
        cols = [f"GENRE_{n}" for n in self.vectorizer.get_feature_names_out()]
        new = pd.DataFrame(mat, columns=cols, index=X.index)
        new["SUPER_GENRE_ENCODED"] = (
            genre_lists.apply(_map_super_genre)
            .map(self.super_genre_map)
            .fillna(self.super_genre_other_val)
        )
        return pd.concat([X, new], axis=1)


def _split_genre_field(value) -> List[str]:
    """Lowercase + canonicalize multi-word genres to underscored tokens."""
    out: List[str] = []
    for tok in process_text_list(value):
        for piece in tok.replace("|", ",").split(","):
            piece = piece.strip().lower()
            if piece:
                out.append(piece.replace(" ", "_"))
    return out


def _map_super_genre(genres) -> str:
    g = set(genres)
    if {"action", "adventure", "science_fiction"}.issubset(g):
        return "SciFi_Blockbuster"
    if "superhero" in g and "action" in g:
        return "Superhero_Blockbuster"
    if {"adventure", "fantasy"}.issubset(g):
        return "Epic_Fantasy"
    if "animation" in g and "family" in g:
        return "Family_Animation"
    if {"animation", "comedy"}.issubset(g):
        return "Animated_Comedy"
    if {"action", "adventure", "thriller"}.issubset(g):
        return "Action_Thriller_Vehicle"
    if {"horror", "thriller"}.issubset(g):
        return "Horror_Suspense"
    if {"comedy", "romance"}.issubset(g) or "romantic_comedy" in g:
        return "RomCom"
    if {"action", "comedy"}.issubset(g):
        return "Action_Comedy"
    for k, label in (
        ("action", "Action"),
        ("adventure", "Adventure"),
        ("science_fiction", "SciFi"),
        ("fantasy", "Fantasy"),
        ("animation", "Animation"),
        ("horror", "Horror"),
        ("thriller", "Thriller"),
        ("comedy", "Comedy"),
    ):
        if k in g:
            return label
    return "Drama" if "drama" in g else "Other"
