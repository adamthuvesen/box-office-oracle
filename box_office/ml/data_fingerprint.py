"""Which training data a model saw: row count, latest release date, and a content hash.

The training frame is not pinned anywhere else (the local parquet is gitignored,
and the Snowflake staging table is mutable), so every training path records this
fingerprint in the model metadata and the serving loader reports it.
"""

from __future__ import annotations

import hashlib

import pandas as pd
from pydantic import BaseModel, ConfigDict

_REGISTRY_KEYS = {
    "rows": "data_rows",
    "max_release_date": "data_max_release_date",
    "content_sha256": "data_sha256",
}


class DataFingerprint(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    rows: int
    max_release_date: str
    content_sha256: str

    def as_registry_metadata(self) -> dict[str, str]:
        """``CustomerMetadataProperties`` entries (string values only)."""
        return {key: str(getattr(self, field)) for field, key in _REGISTRY_KEYS.items()}

    @classmethod
    def from_registry_metadata(cls, metadata: dict[str, str]) -> DataFingerprint | None:
        """Read the fingerprint back; ``None`` for packages registered before it existed."""
        if not all(key in metadata for key in _REGISTRY_KEYS.values()):
            return None
        return cls(
            rows=int(metadata[_REGISTRY_KEYS["rows"]]),
            max_release_date=metadata[_REGISTRY_KEYS["max_release_date"]],
            content_sha256=metadata[_REGISTRY_KEYS["content_sha256"]],
        )


def fingerprint_training_frame(frame: pd.DataFrame) -> DataFingerprint:
    """Fingerprint the frame a model trains on. Needs a ``RELEASE_DATE`` column.

    The hash covers column names, column order, row order, and every value, so
    any change to the training rows changes it.
    """
    latest = pd.to_datetime(frame["RELEASE_DATE"], errors="coerce").max()
    if pd.isna(latest):
        raise ValueError("training frame has no parseable RELEASE_DATE")
    canonical = frame.reset_index(drop=True).to_json(
        orient="split", index=False, date_format="iso", double_precision=15
    )
    return DataFingerprint(
        rows=len(frame),
        max_release_date=latest.date().isoformat(),
        content_sha256=hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
    )
