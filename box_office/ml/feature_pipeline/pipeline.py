"""Compose the sklearn feature-engineering pipeline."""

from sklearn.pipeline import Pipeline

from box_office.ml.feature_pipeline.transformers.core import CoreNumericalTransformer
from box_office.ml.feature_pipeline.transformers.financial import FinancialTransformer
from box_office.ml.feature_pipeline.transformers.genre import GenreTransformer
from box_office.ml.feature_pipeline.transformers.industry import IndustryTransformer
from box_office.ml.feature_pipeline.transformers.interactions import (
    InteractionTransformer,
)
from box_office.ml.feature_pipeline.transformers.select import (
    _DropPreEngineered,
    _SelectEngineered,
)
from box_office.ml.feature_pipeline.transformers.temporal import TemporalTransformer


def build_feature_pipeline() -> Pipeline:
    """Compose the full feature-engineering pipeline.

    Step order matters: financial reads ``YEARS_SINCE_2000`` from temporal +
    ``AVG_ACTOR_FREQ`` from industry; interactions reads everything upstream.
    """
    return Pipeline(
        steps=[
            ("drop_pre_engineered", _DropPreEngineered()),
            ("core", CoreNumericalTransformer()),
            ("temporal", TemporalTransformer()),
            ("genre", GenreTransformer()),
            ("industry", IndustryTransformer()),
            ("financial", FinancialTransformer()),
            ("interactions", InteractionTransformer()),
            ("select", _SelectEngineered()),
        ]
    )
