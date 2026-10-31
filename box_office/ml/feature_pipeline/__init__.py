"""Box-office feature engineering as a composed sklearn Pipeline."""

from box_office.ml.feature_pipeline._helpers import last_column
from box_office.ml.feature_pipeline.constants import (
    CORE_NUMERICAL_FEATURES,
    GENRE_VOCABULARY,
    RAW_INPUT_COLUMNS_TO_DROP,
    SELECTED_FEATURES,
)
from box_office.ml.feature_pipeline.cpi import CPI_ANCHOR_YEAR, CPI_BY_YEAR
from box_office.ml.feature_pipeline.pipeline import build_feature_pipeline
from box_office.ml.feature_pipeline.transformers.core import CoreNumericalTransformer
from box_office.ml.feature_pipeline.transformers.financial import FinancialTransformer
from box_office.ml.feature_pipeline.transformers.genre import (
    GenreTransformer,
    map_super_genre,
    split_genre_field,
)
from box_office.ml.feature_pipeline.transformers.industry import IndustryTransformer
from box_office.ml.feature_pipeline.transformers.select import (
    DropPreEngineered,
    FeatureSelector,
    SelectEngineered,
)
from box_office.ml.feature_pipeline.transformers.temporal import TemporalTransformer

__all__ = [
    "CORE_NUMERICAL_FEATURES",
    "CPI_ANCHOR_YEAR",
    "CPI_BY_YEAR",
    "GENRE_VOCABULARY",
    "RAW_INPUT_COLUMNS_TO_DROP",
    "SELECTED_FEATURES",
    "CoreNumericalTransformer",
    "DropPreEngineered",
    "FeatureSelector",
    "FinancialTransformer",
    "GenreTransformer",
    "IndustryTransformer",
    "SelectEngineered",
    "TemporalTransformer",
    "build_feature_pipeline",
    "last_column",
    "map_super_genre",
    "split_genre_field",
]
