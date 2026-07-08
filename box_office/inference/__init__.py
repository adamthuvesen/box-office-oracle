"""Inference module for Box Office Prediction."""

from .app.main import app
from .app.model_loader import ModelLoader
from .app.predictor import PredictionEngine

__all__ = ["app", "PredictionEngine", "ModelLoader"]
