"""Industry frequency-encoding transformers (director, company)."""

from __future__ import annotations

from collections import Counter
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder

from box_office.ml.text_utils import process_text_list


class IndustryTransformer(BaseEstimator, TransformerMixin):
    """Frequency features for directors, companies, actors + MPAA encoding."""

    UNKNOWN_MPAA = "unknown"

    def __init__(self) -> None:
        self.director_freq_map: Dict[str, int] = {}
        self.company_freq_map: Dict[str, int] = {}
        self.actor_freq_map: Counter[str] = Counter()
        self.global_director_freq = 0
        self.global_company_freq = 0
        self.global_actor_freq = 0
        self.mpaa_encoder = LabelEncoder()

    def fit(self, X: pd.DataFrame, y=None) -> "IndustryTransformer":
        # Seed the unknown bucket so transform() can route unseen ratings.
        mpaa_train = X["MPAA"].fillna("Not Rated").tolist()
        mpaa_train.append(self.UNKNOWN_MPAA)
        self.mpaa_encoder.fit(mpaa_train)

        self.director_freq_map = (
            X["DIRECTOR"].fillna("Unknown").value_counts().to_dict()
        )
        self.company_freq_map = (
            X["PRODUCTION_COMPANY"].fillna("Unknown").value_counts().to_dict()
        )

        actor_lists = X["ACTORS"].apply(process_text_list)
        self.actor_freq_map = Counter(a for sub in actor_lists for a in sub)

        director_counts = X["DIRECTOR"].value_counts()
        self.global_director_freq = (
            director_counts.median() if len(director_counts) else 0
        )
        company_counts = X["PRODUCTION_COMPANY"].value_counts()
        self.global_company_freq = company_counts.median() if len(company_counts) else 0
        self.global_actor_freq = (
            np.median(list(self.actor_freq_map.values())) if self.actor_freq_map else 0
        )
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        new = pd.DataFrame(index=X.index)
        mpaa_values = X["MPAA"].fillna("Not Rated")
        known = set(self.mpaa_encoder.classes_)
        new["MPAA_ENCODED"] = self.mpaa_encoder.transform(
            mpaa_values.where(mpaa_values.isin(known), self.UNKNOWN_MPAA)
        )
        new["DIRECTOR_FREQ"] = (
            X["DIRECTOR"]
            .fillna("Unknown")
            .map(self.director_freq_map)
            .fillna(self.global_director_freq)
        )
        new["COMPANY_FREQ"] = (
            X["PRODUCTION_COMPANY"]
            .fillna("Unknown")
            .map(self.company_freq_map)
            .fillna(self.global_company_freq)
        )
        actor_lists = X["ACTORS"].apply(process_text_list)
        actor_freq_map = self.actor_freq_map
        global_actor_freq = self.global_actor_freq

        def _actor_freqs(xs):
            if not xs:
                return (global_actor_freq, 0.0, 0.0)
            freqs = [actor_freq_map.get(a, 0) for a in xs]
            return (
                actor_freq_map.get(xs[0], global_actor_freq),
                float(np.mean(freqs)),
                float(np.max(freqs)),
            )

        actor_freq_triples = actor_lists.apply(_actor_freqs)
        new["LEAD_ACTOR_FREQ"] = actor_freq_triples.apply(lambda t: t[0])
        new["AVG_ACTOR_FREQ"] = actor_freq_triples.apply(lambda t: t[1])
        new["MAX_ACTOR_FREQ"] = actor_freq_triples.apply(lambda t: t[2])
        return pd.concat([X, new], axis=1)
