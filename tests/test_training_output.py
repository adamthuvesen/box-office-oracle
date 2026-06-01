"""Tests for SageMaker training output parsing."""

import json
import tarfile
from pathlib import Path

import pytest
from box_office.sagemaker.training_output import parse_training_output_tarball


def _write_tar(path: Path, files: dict[str, str]) -> None:
    with tarfile.open(path, "w:gz") as tar:
        for name, content in files.items():
            data = content.encode("utf-8")

            import io

            info = tarfile.TarInfo(name=name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))


def test_parse_training_output_tarball_extracts_cv_and_oof(tmp_path: Path) -> None:
    tar_path = tmp_path / "output.tar.gz"
    _write_tar(
        tar_path,
        {
            "cv_results.json": json.dumps(
                {"mean_cv_mae": 1.2, "std_cv_mae": 0.1, "cv_scores": [1.0, 1.4]}
            ),
            "oof_evaluation.json": json.dumps(
                {"oof_r2": 0.8, "oof_mae": 1e6, "num_oof_samples": 100}
            ),
        },
    )

    metrics = parse_training_output_tarball(tar_path)
    assert metrics["cv_mean_mae"] == pytest.approx(1.2)
    assert metrics["oof_r2"] == pytest.approx(0.8)
    assert metrics["oof_num_samples"] == 100
    assert "cv_results" in metrics
    assert "oof_results" in metrics
