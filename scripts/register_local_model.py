"""Register the locally-trained v9 model in the SageMaker Model Registry.

SageMaker training is quota-blocked (account training quotas are 0), so the
production pipeline cannot produce a model package. This script registers the
locally-trained artifact in ``artifacts/local/`` *exactly as a pipeline run
would have*, so the inference Lambda can serve it:

1. Build ``model.tar.gz`` in the layout the training container produces —
   ``model.pkl``, ``feature_preprocessor.pkl``, ``feature_scaler.pkl`` at the
   archive root (the member names the inference ``ModelLoader`` verifies).
2. Compute the pooled OOF metrics the container computes: dollar-space pooled
   OOF R² via ``ModelEvaluator.evaluate_oof_performance`` (the same code path
   the container runs), reconstructing ``y_train_log`` from the training frame.
3. Upload the tarball to S3 and register a model package in the existing group.
   ``AWSModelRegistry.register_model_package`` stamps the SHA256 manifest,
   ``size_bytes`` and ``feature_schema_version=9`` into
   ``CustomerMetadataProperties`` — the same fields the loader checks.
4. Run the pipeline's promotion gate (``validate_model_for_promotion``,
   min pooled OOF R² = ``config.model.promotion_threshold``). Promote to
   Approved only if it passes; otherwise leave PendingManualApproval and stop.

The local training path is code-identical to the container path (shared frame
rules, 13 ``SELECTED_FEATURES``, per-fold leakage-free CV), so this package is
byte-for-byte what a pipeline run would have registered.

Run:  AWS_PROFILE=personal uv run python scripts/register_local_model.py
"""

from __future__ import annotations

import argparse
import json
import logging
import tarfile
import tempfile
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from box_office.config import config
from box_office.ml.artifacts import (
    FEATURE_PREPROCESSOR_PKL,
    FEATURE_SCALER_PKL,
    MODEL_PKL,
)
from box_office.ml.model import ModelEvaluator
from box_office.ml.model_registry.aws_model_registry import AWSModelRegistry

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("register_local_model")

ARTIFACT_DIR = Path("artifacts/local")
FRAME_PATH = Path("data/generated/training/train_frame_1980_2026.parquet")

# The three artifacts the SageMaker training container writes into SM_MODEL_DIR
# and that the inference loader verifies by name at the tarball root.
TARBALL_MEMBERS = (MODEL_PKL, FEATURE_PREPROCESSOR_PKL, FEATURE_SCALER_PKL)


def build_model_tarball(artifact_dir: Path, out_path: Path) -> Path:
    """Write ``model.tar.gz`` with the three pkl artifacts at the archive root.

    Layout matches what SageMaker produces from ``SM_MODEL_DIR``: the loader
    looks for ``MODEL_PKL`` / ``FEATURE_PREPROCESSOR_PKL`` / ``FEATURE_SCALER_PKL``
    at the root, so each member's ``arcname`` is just the bare filename.
    """
    for member in TARBALL_MEMBERS:
        if not (artifact_dir / member).exists():
            raise FileNotFoundError(f"missing artifact: {artifact_dir / member}")

    with tarfile.open(out_path, "w:gz") as tar:
        for member in TARBALL_MEMBERS:
            tar.add(artifact_dir / member, arcname=member)
    return out_path


def compute_oof_metrics(cv_results: dict, frame_path: Path) -> dict[str, float]:
    """Pooled OOF metrics via the container's ``evaluate_oof_performance``.

    ``y_train_log = log1p(WORLDWIDE_GROSS)`` reconstructed from the same frame
    the local run trained on; ``ModelEvaluator.evaluate_oof_performance`` then
    inverts log1p with ``expm1`` and computes pooled dollar-space R² — the
    exact number the container writes to ``oof_evaluation.json`` and that the
    promotion gate reads from ``CustomerMetadataProperties['oof_r2']``.
    """
    frame = pd.read_parquet(frame_path)
    y_train_log = pd.Series(np.log1p(frame["WORLDWIDE_GROSS"].astype(float)))
    return ModelEvaluator.evaluate_oof_performance(cv_results, y_train_log)


def build_metrics(cv_results: dict, oof: dict[str, float]) -> dict[str, float]:
    """Metrics dict for the registry (allowlisted in register_model_package)."""
    return {
        "oof_r2": oof["oof_r2"],
        "oof_mae": oof["oof_mae"],
        "oof_rmsle": oof["oof_rmsle"],
        "oof_num_samples": oof["num_oof_samples"],
        "cv_mean_mae": cv_results["mean_cv_mae"],
        "cv_std_mae": cv_results["std_cv_mae"],
        "cv_mean_rmsle": cv_results["mean_cv_rmsle"],
        "cv_std_rmsle": cv_results["std_cv_rmsle"],
        "cv_mean_best_iteration": cv_results["mean_best_iteration"],
    }


def provenance_metadata(metadata: dict) -> dict[str, str]:
    """CustomerMetadataProperties marking this as a local, quota-blocked run."""
    return {
        "trained_on": "local",
        "code_path": "container-identical",
        # CustomerMetadataProperties values must match
        # [\p{L}\p{Z}\p{N}_.:\/=+\-@] — no parentheses, semicolons or commas.
        "provenance_note": (
            "SageMaker training quota-blocked account training quota=0 "
            "registered from artifacts/local via scripts/register_local_model.py"
        ),
        "training_rows": str(metadata.get("training_rows", "")),
        "eval_years": str(metadata.get("eval_years", "")),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--region", default=config.aws.region)
    parser.add_argument(
        "--bucket",
        default="box-office-dev-sagemaker-artifacts-009882533051",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute metrics and build the tarball, but do not touch AWS.",
    )
    args = parser.parse_args()

    cv_results = json.loads((ARTIFACT_DIR / "cv_results.json").read_text())
    metadata = json.loads((ARTIFACT_DIR / "metadata.json").read_text())

    oof = compute_oof_metrics(cv_results, FRAME_PATH)
    metrics = build_metrics(cv_results, oof)
    oof_r2 = oof["oof_r2"]
    threshold = config.model.promotion_threshold
    logger.info("Pooled OOF R2 (dollar space): %.4f", oof_r2)
    logger.info("Promotion threshold: %.2f", threshold)

    with tempfile.TemporaryDirectory() as tmp:
        tarball = build_model_tarball(ARTIFACT_DIR, Path(tmp) / "model.tar.gz")
        logger.info("Built %s (%d bytes)", tarball, tarball.stat().st_size)

        if args.dry_run:
            logger.info("--dry-run: skipping upload and registration")
            return

        timestamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
        key = f"models/local/{timestamp}/model.tar.gz"
        model_data_url = f"s3://{args.bucket}/{key}"

        registry = AWSModelRegistry(region_name=args.region)
        registry.s3_client.upload_file(str(tarball), args.bucket, key)
        logger.info("Uploaded model artifact to %s", model_data_url)

    group_name = AWSModelRegistry.get_model_group_name()
    registry.create_model_package_group(group_name)

    result = registry.register_model_package(
        model_package_group_name=group_name,
        model_data_url=model_data_url,
        framework="XGBOOST",
        model_approval_status="PendingManualApproval",
        metrics=metrics,
        metadata=provenance_metadata(metadata),
    )
    model_package_arn = result["model_package_arn"]
    logger.info("Registered model package: %s", model_package_arn)

    # Pipeline promotion gate. Executing the Prefect task standalone gives it a
    # task-run context (Prefect 3), matching how the registry phase calls it.
    from box_office.orchestration.tasks.training_tasks import (
        validate_model_for_promotion,
    )

    gate = validate_model_for_promotion(model_package_arn=model_package_arn)
    logger.info("Gate result: %s", json.dumps(gate, default=str))

    if gate.get("promote"):
        registry.update_model_approval_status(
            model_package_arn,
            approval_status="Approved",
            approval_description=(
                f"Local v9 model; pooled OOF R2={oof_r2:.4f} >= {threshold} gate. "
                "SageMaker training quota-blocked."
            ),
        )
        logger.info("PROMOTED to Approved: %s", model_package_arn)
    else:
        logger.warning(
            "Gate did NOT pass (oof_r2=%.4f); left PendingManualApproval. Reason: %s",
            oof_r2,
            gate.get("reason") or gate.get("validation_details"),
        )


if __name__ == "__main__":
    main()
