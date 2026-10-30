"""``metadata.json``: what the local training run records next to its artifacts.

``scripts/train_local.py`` writes it; ``scripts/register_local_model.py`` reads
it into the registry's ``CustomerMetadataProperties``.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class LocalModelMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")

    feature_schema_version: str
    selected_features: list[str]
    training_rows: int
    cv_mean_mae_log: float
    cv_mean_rmsle: float
    mean_best_iteration: float
    eval_years: str
    eval_discipline: str
    sha256: dict[str, str]
