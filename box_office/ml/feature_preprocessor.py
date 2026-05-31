"""Feature preprocessing coordinator.

``FeaturePreprocessorHigh`` is the stable, picklable interface over
``box_office.ml.feature_pipeline.build_feature_pipeline()``: it adds the
fit / transform / fit_transform / get_feature_names surface plus the column
collision and NaN guards the orchestration and inference paths depend on.
Trained inference artifacts pickle an instance of this class, so its public
shape must stay stable.
"""

import logging
from typing import List, Optional

import pandas as pd

from box_office.ml.data_prep import DataSplitter, TargetTransformer
from box_office.ml.feature_pipeline import build_feature_pipeline
from box_office.ml.feature_pipeline.constants import SELECTED_FEATURES
from box_office.utils.feature_flags import (
    strict_features_enabled as _strict_features_enabled,
)

logger = logging.getLogger(__name__)


class FeatureNameCollisionError(ValueError):
    """Raised when the feature pipeline emits duplicate column names.

    Two transformers claiming the same name is a config bug; silent
    deduplication would hide the offender from CI.
    """


class FeaturePreprocessorHigh:
    """Coordinator for the engineered feature pipeline.

    Wraps ``build_feature_pipeline()`` with the historical fit / transform /
    fit_transform / get_feature_names surface plus the collision and NaN
    guards the orchestration relies on.
    """

    def __init__(self):
        self.pipeline = build_feature_pipeline()
        self._feature_names: Optional[List[str]] = None

    def fit(self, X: pd.DataFrame) -> "FeaturePreprocessorHigh":
        # Capture column names from a single fit_transform pass; the transformed
        # output is discarded because fit() only needs the schema.
        self._feature_names = list(self.pipeline.fit_transform(X).columns)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return self._guard_output(self.pipeline.transform(X))

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        # Single pipeline pass for the training path: fit + transform together,
        # rather than fit() (which fit_transforms) followed by a second transform.
        out = self.pipeline.fit_transform(X)
        self._feature_names = list(out.columns)
        return self._guard_output(out)

    def _guard_output(self, out: pd.DataFrame) -> pd.DataFrame:
        if not out.columns.is_unique:
            duplicates = sorted(set(out.columns[out.columns.duplicated()].tolist()))
            raise FeatureNameCollisionError(
                f"Feature pipeline emitted duplicate column names: {duplicates}. "
                "Each transformer must produce a unique set of feature names."
            )

        # NaN propagates by design: XGBoost handles it natively, and a blanket
        # fillna(0) would conflate missing with legitimate zero. Strict mode
        # raises so CI catches sudden NaN spikes.
        nan_counts = out.isna().sum()
        nan_columns = nan_counts[nan_counts > 0]
        if not nan_columns.empty:
            for col, count in nan_columns.items():
                logger.warning(
                    "Feature column %r has %d NaN values; propagating (XGBoost handles NaN).",
                    col,
                    int(count),
                )
            if _strict_features_enabled():
                offenders = {str(c): int(n) for c, n in nan_columns.items()}
                raise ValueError(
                    f"ML_STRICT_FEATURES=true and NaN values present in: {offenders}"
                )

        return out

    def get_feature_names(self) -> List[str]:
        """Return engineered column names; lazily fits on a synthetic schema
        fixture if the caller hasn't fit on real data yet (model registry,
        dbt feature metadata, etc.).
        """
        if self._feature_names is None:
            self.fit(_SCHEMA_FIXTURE)
        if self._feature_names != list(SELECTED_FEATURES):
            raise FeatureNameCollisionError(
                "Pipeline feature names drifted from the SELECTED_FEATURES contract. "
                f"Got {self._feature_names}; expected {list(SELECTED_FEATURES)}."
            )
        return list(self._feature_names)


_SCHEMA_FIXTURE = pd.DataFrame(
    [
        {
            "RELEASE_YEAR": 2020,
            "RELEASE_DATE": "2020-06-15",
            "RATING": 7.0,
            "VOTES": 1000,
            "AD_BUDGET": 1_000_000,
            "PRODUCTION_BUDGET": 10_000_000,
            "FRANCHISE_RATING": 0,
            "RUNTIME": 120,
            "MPAA": "PG-13",
            "GENRES": "Action",
            "DIRECTOR": "Director",
            "PRODUCTION_COMPANY": "Studio",
            "ACTORS": "Actor",
        }
    ]
)

__all__ = [
    "FeaturePreprocessorHigh",
    "FeatureNameCollisionError",
    "DataSplitter",
    "TargetTransformer",
]
