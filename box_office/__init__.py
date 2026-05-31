"""
Box Office Prediction ML Pipeline

A machine learning system for predicting movie box office revenue using AWS
SageMaker, Snowflake, dbt, and Prefect orchestration.
"""

__version__ = "0.1.0"

from .config import config

__all__ = [
    "__version__",
    "config",
]
