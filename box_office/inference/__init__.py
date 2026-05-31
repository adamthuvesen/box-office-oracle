"""Inference module for Box Office Prediction."""

from .app.main import app
from .app.predictor import PredictionEngine
from .app.model_loader import ModelLoader

__all__ = ["app", "PredictionEngine", "ModelLoader"]
