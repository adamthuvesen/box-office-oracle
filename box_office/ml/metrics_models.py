"""Typed SageMaker training-output metrics."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class _Strict(BaseModel):
    # NaN is a legitimate metric value (an overflowing dollar-scale fold), so
    # it round-trips as a JSON NaN constant instead of null.
    model_config = ConfigDict(extra="forbid", ser_json_inf_nan="constants")


class FoldResult(_Strict):
    """One CV fold: the model scored on one eval year."""

    fold_number: int
    eval_year: int
    mae_score: float
    rmsle_score: float
    model_r2_log: float
    model_spearman: float
    model_r2_dollars: float
    model_median_ape: float
    best_iteration: int
    train_samples: int
    val_samples: int
    # Artifacts written before a failed fold failed the run carry ``error: null``.
    error: None = None


class OofRecord(_Strict):
    """One out-of-fold prediction on the log target, keyed by frame position."""

    fold: int
    idx: int
    pred: float


class CrossValidationResults(_Strict):
    """What ``TimeSeriesCrossValidator.cross_validate`` returns and ``cv_results.json`` holds."""

    mean_cv_mae: float
    std_cv_mae: float
    mean_cv_rmsle: float
    std_cv_rmsle: float
    cv_scores: list[float]
    cv_rmsle_scores: list[float]
    mean_best_iteration: float
    oof_predictions: dict[str, float]
    oof_records: list[OofRecord]
    feature_importances: list[float] | None
    feature_names: list[str]
    fold_results: list[FoldResult]


class OofEvaluation(_Strict):
    """Pooled out-of-fold metrics; ``oof_r2`` is the promotion-gate number."""

    oof_r2: float
    oof_mae: float
    oof_rmsle: float
    num_oof_samples: int


class TrainingMetrics(BaseModel):
    """Metrics extracted from a SageMaker training job and output tarball."""

    model_config = ConfigDict(extra="allow")

    job_name: str = ""
    duration: float = 0.0
    model_data_url: str | None = None
    training_duration_seconds: float | None = None
    training_duration_minutes: float | None = None
    cv_mean_mae: float | None = None
    cv_std_mae: float | None = None
    cv_mean_rmsle: float | None = None
    cv_std_rmsle: float | None = None
    cv_mean_best_iteration: float | None = None
    oof_r2: float | None = None
    oof_mae: float | None = None
    oof_rmsle: float | None = None
    oof_num_samples: int | None = None
    hyperparameters: dict[str, Any] = Field(default_factory=dict)
    cv_results: CrossValidationResults | None = None
    oof_results: OofEvaluation | None = None
    training_time: float | None = None
    estimated_cost: float | None = None

    @classmethod
    def from_performance_dict(cls, data: dict[str, Any]) -> TrainingMetrics:
        """Build from the flat dict produced by training output parsing."""
        return cls.model_validate(data)

    def to_performance_dict(self) -> dict[str, Any]:
        return self.model_dump(exclude_none=True)
