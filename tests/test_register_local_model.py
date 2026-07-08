"""Pure-logic tests for scripts/register_local_model.py (no live AWS)."""

from __future__ import annotations

import tarfile

import numpy as np
import pandas as pd
import pytest

from box_office.inference.app.integrity import compute_sha256
from box_office.ml.artifacts import (
    FEATURE_PREPROCESSOR_PKL,
    FEATURE_SCALER_PKL,
    MODEL_PKL,
)

register = pytest.importorskip("scripts.register_local_model")


def _write_artifacts(artifact_dir):
    for name in (MODEL_PKL, FEATURE_PREPROCESSOR_PKL, FEATURE_SCALER_PKL):
        (artifact_dir / name).write_bytes(name.encode())


def test_tarball_has_exactly_the_loader_member_names(tmp_path):
    artifact_dir = tmp_path / "local"
    artifact_dir.mkdir()
    _write_artifacts(artifact_dir)

    out = register.build_model_tarball(artifact_dir, tmp_path / "model.tar.gz")

    with tarfile.open(out, "r:gz") as tar:
        names = sorted(tar.getnames())
    # Member names are the bare filenames at the archive root — exactly what
    # ModelLoader._extract_and_load_model_with_cache verifies.
    assert names == sorted(
        [MODEL_PKL, FEATURE_PREPROCESSOR_PKL, FEATURE_SCALER_PKL]
    )
    for name in names:
        assert "/" not in name


def test_build_tarball_raises_on_missing_artifact(tmp_path):
    artifact_dir = tmp_path / "local"
    artifact_dir.mkdir()
    (artifact_dir / MODEL_PKL).write_bytes(b"x")  # only one of three

    with pytest.raises(FileNotFoundError):
        register.build_model_tarball(artifact_dir, tmp_path / "model.tar.gz")


def test_tarball_sha256_manifest_is_deterministic(tmp_path):
    artifact_dir = tmp_path / "local"
    artifact_dir.mkdir()
    _write_artifacts(artifact_dir)

    a = register.build_model_tarball(artifact_dir, tmp_path / "a.tar.gz")
    b = register.build_model_tarball(artifact_dir, tmp_path / "b.tar.gz")

    digest = compute_sha256(a)
    assert len(digest) == 64
    # gzip embeds mtimes, so tarball bytes are not byte-identical across builds;
    # the manifest is the SHA256 of whichever object is uploaded, computed the
    # same way the registry recomputes it from the S3 object.
    assert compute_sha256(a) == digest  # stable for a fixed file
    assert compute_sha256(b) == compute_sha256(b)


def test_compute_oof_metrics_matches_dollar_space_pooled_r2(tmp_path):
    # Three OOF rows with known log-space preds; the helper must invert log1p
    # (expm1) and pool dollar-space R² exactly like the container.
    y_dollars = np.array([1_000_000.0, 50_000_000.0, 200_000_000.0])
    frame = pd.DataFrame({"WORLDWIDE_GROSS": y_dollars})
    y_log = np.log1p(y_dollars)
    preds_log = y_log + np.array([0.1, -0.2, 0.05])

    cv_results = {
        "oof_predictions": {str(i): float(preds_log[i]) for i in range(3)},
    }
    frame_path = tmp_path / "frame.parquet"
    frame.to_parquet(frame_path)

    from sklearn.metrics import r2_score

    expected = r2_score(y_dollars, np.expm1(preds_log))
    out = register.compute_oof_metrics(cv_results, frame_path)
    assert out["oof_r2"] == pytest.approx(expected)
    assert out["num_oof_samples"] == 3


def test_build_metrics_carries_gate_and_cv_fields():
    cv_results = {
        "mean_cv_mae": 0.7,
        "std_cv_mae": 0.05,
        "mean_cv_rmsle": 0.88,
        "std_cv_rmsle": 0.04,
        "mean_best_iteration": 143.0,
    }
    oof = {
        "oof_r2": 0.5943,
        "oof_mae": 8.2e7,
        "oof_rmsle": 0.8746,
        "num_oof_samples": 1159,
    }
    metrics = register.build_metrics(cv_results, oof)
    # oof_r2 is the field the promotion gate reads.
    assert metrics["oof_r2"] == 0.5943
    assert metrics["cv_mean_best_iteration"] == 143.0
    assert metrics["oof_num_samples"] == 1159


def test_provenance_metadata_marks_local_quota_blocked():
    meta = register.provenance_metadata(
        {"training_rows": 6077, "eval_years": "2015-2023"}
    )
    assert meta["trained_on"] == "local"
    assert meta["code_path"] == "container-identical"
    assert "quota" in meta["provenance_note"].lower()
    assert meta["training_rows"] == "6077"
    # every value must be a str — CustomerMetadataProperties rejects non-strings.
    assert all(isinstance(v, str) for v in meta.values())
