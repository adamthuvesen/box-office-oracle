"""
Simple Model Registry for Box Office Prediction ML Pipeline

This package provides core data structures and functionality for tracking
trained ML models, their metadata, performance metrics, and versioning.
It also includes SageMaker integration functions for automatic model registration.
"""

from .aws_model_registry import AWSModelRegistry
from .metadata import ModelMetadata

__all__ = ["ModelMetadata", "AWSModelRegistry"]
