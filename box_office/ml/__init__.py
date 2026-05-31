"""
Machine Learning module for Box Office Prediction

Contains feature engineering, model training, and model registry functionality.
"""

from .feature_preprocessor import FeaturePreprocessorHigh
from .model import BoxOfficeXGBoostModel

__all__ = ["FeaturePreprocessorHigh", "BoxOfficeXGBoostModel"]
