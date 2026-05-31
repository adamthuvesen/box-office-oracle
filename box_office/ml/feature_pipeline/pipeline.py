"""Compose the sklearn feature-engineering pipeline."""

from sklearn.pipeline import Pipeline

from box_office.ml.feature_pipeline.transformers.core import CoreNumericalTransformer
from box_office.ml.feature_pipeline.transformers.financial import FinancialTransformer
from box_office.ml.feature_pipeline.transformers.genre import GenreTransformer
from box_office.ml.feature_pipeline.transformers.industry import IndustryTransformer
from box_office.ml.feature_pipeline.transformers.select import (
    FeatureSelector,
    _DropPreEngineered,
    _SelectEngineered,
)
from box_office.ml.feature_pipeline.transformers.temporal import TemporalTransformer


def build_feature_pipeline() -> Pipeline:
    """Compose the feature-engineering pipeline.

    Step order matters: financial reads ``YEARS_SINCE_2000`` from temporal +
    ``AVG_ACTOR_FREQ`` from industry. The trailing ``FeatureSelector`` projects
    the full engineered frame onto the canonical ``SELECTED_FEATURES`` contract.
    """
    return Pipeline(
        steps=[
            ("drop_pre_engineered", _DropPreEngineered()),
            ("core", CoreNumericalTransformer()),
            ("temporal", TemporalTransformer()),
            ("genre", GenreTransformer()),
            ("industry", IndustryTransformer()),
            ("financial", FinancialTransformer()),
            ("select", _SelectEngineered()),
            ("feature_selector", FeatureSelector()),
        ]
    )
