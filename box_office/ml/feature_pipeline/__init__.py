"""Box-office feature engineering as a composed sklearn Pipeline."""

from box_office.ml.feature_pipeline.constants import (
    CORE_NUMERICAL_FEATURES,
    GENRE_VOCABULARY,
    RAW_INPUT_COLUMNS_TO_DROP,
)
from box_office.ml.feature_pipeline.cpi import CPI_ANCHOR_YEAR, CPI_BY_YEAR
from box_office.ml.feature_pipeline._helpers import _column
from box_office.ml.feature_pipeline.pipeline import build_feature_pipeline
from box_office.ml.feature_pipeline.transformers.core import CoreNumericalTransformer
from box_office.ml.feature_pipeline.transformers.financial import FinancialTransformer
from box_office.ml.feature_pipeline.transformers.genre import (
    GenreTransformer,
    _map_super_genre,
    _split_genre_field,
)
from box_office.ml.feature_pipeline.transformers.industry import IndustryTransformer
from box_office.ml.feature_pipeline.transformers.interactions import (
    InteractionTransformer,
)
from box_office.ml.feature_pipeline.transformers.select import (
    _DropPreEngineered,
    _SelectEngineered,
)
from box_office.ml.feature_pipeline.transformers.temporal import TemporalTransformer

__all__ = [
    "build_feature_pipeline",
    "CPI_BY_YEAR",
    "CPI_ANCHOR_YEAR",
    "RAW_INPUT_COLUMNS_TO_DROP",
    "CORE_NUMERICAL_FEATURES",
    "GENRE_VOCABULARY",
    "CoreNumericalTransformer",
    "TemporalTransformer",
    "GenreTransformer",
    "IndustryTransformer",
    "FinancialTransformer",
    "InteractionTransformer",
    "_DropPreEngineered",
    "_SelectEngineered",
    "_map_super_genre",
    "_split_genre_field",
    "_column",
]
