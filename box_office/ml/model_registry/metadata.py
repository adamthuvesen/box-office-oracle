"""
Model metadata data structure for the model registry.
"""

import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from typing import Any


def _isoformat_utc_z(dt: datetime) -> str:
    """Format ``dt`` as ISO 8601 UTC with a trailing ``Z`` (registry wire format)."""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _validate_metrics_dict(metrics: Any) -> None:
    if not isinstance(metrics, dict):
        raise TypeError("Metrics must be a dictionary")
    for metric_name, metric_value in metrics.items():
        if not isinstance(metric_value, (int, float)):
            raise ValueError(
                f"Metric '{metric_name}' must be numeric, got {type(metric_value).__name__}"
            )


@dataclass
class ModelMetadata:
    """Core data structure for storing model metadata in the registry."""

    model_id: str
    version: int
    training_job_name: str
    model_artifacts_path: str
    hyperparameters: dict[str, Any]
    status: str
    metrics: dict[str, float]
    created_at: datetime
    updated_at: datetime

    # Valid status values as defined in requirements
    VALID_STATUSES = {"development", "staging", "production", "archived"}
    DEFAULT_STATUS = "development"

    def __post_init__(self):
        self.validate_status()
        self.validate_metrics()

    def validate_status(self) -> None:
        if self.status not in self.VALID_STATUSES:
            raise ValueError(
                f"Invalid status '{self.status}'. Must be one of: {', '.join(self.VALID_STATUSES)}"
            )

    def validate_metrics(self) -> None:
        _validate_metrics_dict(self.metrics)

    def update_status(self, new_status: str) -> None:
        self.status = new_status
        self.validate_status()
        self.updated_at = datetime.now(UTC)

    def update_metrics(self, new_metrics: dict[str, float]) -> None:
        _validate_metrics_dict(new_metrics)
        self.metrics = new_metrics.copy()
        self.updated_at = datetime.now(UTC)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["created_at"] = _isoformat_utc_z(self.created_at)
        data["updated_at"] = _isoformat_utc_z(self.updated_at)
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ModelMetadata":
        data = data.copy()

        if isinstance(data.get("created_at"), str):
            data["created_at"] = datetime.fromisoformat(data["created_at"])

        if isinstance(data.get("updated_at"), str):
            data["updated_at"] = datetime.fromisoformat(data["updated_at"])

        if "metrics" not in data:
            data["metrics"] = {}

        if "status" not in data:
            data["status"] = cls.DEFAULT_STATUS

        return cls(**data)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_json(cls, json_str: str) -> "ModelMetadata":
        return cls.from_dict(json.loads(json_str))

    def __str__(self) -> str:
        return f"ModelMetadata(id={self.model_id}, version={self.version}, status={self.status})"

    def __repr__(self) -> str:
        return (
            f"ModelMetadata(model_id='{self.model_id}', version={self.version}, "
            f"status='{self.status}', training_job='{self.training_job_name}', "
            f"created_at='{self.created_at.isoformat()}')"
        )
