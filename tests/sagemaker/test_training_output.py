"""Tests for SageMaker training output parsing."""

import json
import tarfile
from pathlib import Path

import pytest
from pydantic import ValidationError

from box_office.ml.metrics_models import CrossValidationResults, OofEvaluation
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
            "cv_results.json": CrossValidationResults(
                mean_cv_mae=1.2,
                std_cv_mae=0.1,
                mean_cv_rmsle=0.9,
                std_cv_rmsle=0.1,
                cv_scores=[1.0, 1.4],
                cv_rmsle_scores=[0.8, 1.0],
                mean_best_iteration=120,
                oof_predictions={},
                oof_records=[],
                feature_importances=None,
                feature_names=[],
                fold_results=[],
            ).model_dump_json(),
            "oof_evaluation.json": OofEvaluation(
                oof_r2=0.8, oof_mae=1e6, oof_rmsle=0.9, num_oof_samples=100
            ).model_dump_json(),
        },
    )

    metrics = parse_training_output_tarball(tar_path)
    assert metrics["cv_mean_mae"] == pytest.approx(1.2)
    assert metrics["oof_r2"] == pytest.approx(0.8)
    assert metrics["oof_num_samples"] == 100
    assert metrics["cv_results"]["cv_scores"] == [1.0, 1.4]
    assert metrics["oof_results"]["oof_rmsle"] == pytest.approx(0.9)


def test_parse_training_output_tarball_rejects_an_unknown_cv_key(tmp_path: Path) -> None:
    tar_path = tmp_path / "output.tar.gz"
    _write_tar(tar_path, {"cv_results.json": json.dumps({"mean_cv_mae": 1.2, "surprise": 1})})
    with pytest.raises(ValidationError):
        parse_training_output_tarball(tar_path)
