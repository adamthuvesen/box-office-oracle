"""
Model loader for serverless inference API.

Handles model loading from AWS SageMaker Model Registry with caching
and validation optimized for AWS Lambda environment.
"""

import logging
import os
import pickle
import tarfile
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import boto3
import joblib
from botocore.exceptions import BotoCoreError, ClientError, NoCredentialsError

from box_office.ml.artifacts import (
    FEATURE_PREPROCESSOR_PKL,
    FEATURE_SCALER_PKL,
    MODEL_PKL,
)
from box_office.ml.model_registry.aws_model_registry import AWSModelRegistry
from box_office.ml.registry_constants import (
    CURRENT_FEATURE_SCHEMA_VERSION,
    SCHEMA_VERSION_METADATA_KEY,
    FeatureSchemaVersionMismatch,
)
from box_office.utils.aws_helpers import BOTO3_CONFIG
from box_office.utils.safe_tarfile import extractall_data_filter

from .integrity import ArtifactIntegrityError, compute_sha256, verify_artifact

logger = logging.getLogger(__name__)


class RegistryModelInfo:
    """Model information wrapper with to_dict() method."""

    def __init__(self, model_data: dict[str, Any]):
        self.model_id = model_data.get("ModelPackageArn", "unknown")
        self.version = model_data.get("ModelPackageVersion", 1)
        self.status = model_data.get("ModelApprovalStatus", "unknown")
        self.created_at = model_data.get("CreationTime", datetime.now(UTC))
        self.metrics = model_data.get("metrics", {})
        self.framework = model_data.get("framework", "unknown")
        self._raw_data = model_data

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary format expected by prediction engine."""
        return {
            "model_id": self.model_id,
            "version": self.version,
            "status": self.status,
            "created_at": (
                self.created_at.isoformat()
                if hasattr(self.created_at, "isoformat")
                else str(self.created_at)
            ),
            "metrics": self.metrics,
            "framework": self.framework,
        }


class ModelLoadError(Exception):
    """Exception raised when model loading fails."""

    pass


class ModelValidationError(Exception):
    """Exception raised when model validation fails."""

    pass


class ModelLoader:
    def __init__(
        self,
        model_package_group_name: str,
        aws_region: str = "eu-north-1",
        cache_dir: str = "/tmp/models",
        cache_ttl_seconds: int = 3600,
        max_stale_seconds: int = 3600,
    ):
        """
        Initialize ModelLoader with configuration.

        Args:
            model_package_group_name: SageMaker Model Registry group name
            aws_region: AWS region for SageMaker and S3 clients
            cache_dir: Directory for model caching (Lambda /tmp)
            cache_ttl_seconds: Cache time-to-live in seconds
            max_stale_seconds: Once a cached model is older than this, a
                refresh failure drops the cache and forces a reload on the
                next predict — bounds how long a rejected/deleted model
                package can keep being served.
        """
        self.model_package_group_name = model_package_group_name
        self.aws_region = aws_region
        self.cache_dir = Path(cache_dir)
        self.cache_ttl_seconds = cache_ttl_seconds
        self.max_stale_seconds = max_stale_seconds

        try:
            self.s3_client = boto3.client(
                "s3", region_name=aws_region, config=BOTO3_CONFIG
            )
            self.sagemaker_client = boto3.client(
                "sagemaker", region_name=aws_region, config=BOTO3_CONFIG
            )

            self.model_registry = AWSModelRegistry(region_name=aws_region)

        except NoCredentialsError as e:
            logger.error(f"AWS credentials not configured: {e}")
            raise ModelLoadError(f"AWS credentials not configured: {e}") from e
        except Exception as e:
            logger.exception("Failed to initialize AWS clients")
            raise ModelLoadError(f"Failed to initialize AWS clients: {e}") from e

        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self._current_model = None
        self._current_model_info = None
        self._last_load_time = None
        self._extracted_artifacts_cache = {}

        logger.info(f"ModelLoader initialized for group: {model_package_group_name}")

    def load_latest_approved_model(self) -> RegistryModelInfo | None:
        try:
            model_info_dict = self._get_latest_approved_model_info()
            if not model_info_dict:
                logger.warning("No approved model found in registry")
                return None

            model_package_arn = model_info_dict["ModelPackageArn"]
            logger.info(f"Loading model: {model_package_arn}")

            # Download (or reuse from process-local cache) and load.
            # The download path performs SHA256 verification against the
            # Model Package's manifest before any unpickling.
            model_obj = self._download_and_load_model(model_info_dict)
            self._validate_model(model_obj, model_info_dict)

            self._current_model = model_obj
            self._current_model_info = model_info_dict
            self._last_load_time = datetime.now(UTC)

            logger.info(f"Successfully loaded model: {model_package_arn}")
            return RegistryModelInfo(model_info_dict)

        except (
            ModelLoadError,
            ModelValidationError,
            ArtifactIntegrityError,
            FeatureSchemaVersionMismatch,
        ):
            raise
        except (ClientError, BotoCoreError, OSError, pickle.UnpicklingError) as e:
            logger.error(f"Failed to load model: {e}", exc_info=True)
            raise ModelLoadError(f"Unexpected error loading model: {e}") from e
        except Exception as e:
            logger.exception("Unexpected error loading model")
            raise ModelLoadError(f"Unexpected error loading model: {e}") from e

    def get_current_model(self) -> tuple[Any, dict[str, Any]] | None:
        if self._current_model is not None and self._current_model_info is not None:
            return self._current_model, self._current_model_info
        return None

    def is_model_cache_valid(self) -> bool:
        if self._last_load_time is None:
            return False

        cache_age = (datetime.now(UTC) - self._last_load_time).total_seconds()
        return cache_age < self.cache_ttl_seconds

    def refresh_model_if_needed(self) -> bool:
        """Refresh model if cache expired or new version available."""
        try:
            if self.is_model_cache_valid():
                latest_info = self._get_latest_approved_model_info()
                if latest_info and self._current_model_info:
                    current_arn = self._current_model_info.get("ModelPackageArn")
                    latest_arn = latest_info.get("ModelPackageArn")

                    if current_arn == latest_arn:
                        logger.debug(
                            "Current model is still the latest approved version"
                        )
                        return False

            logger.info("Refreshing model due to cache expiry or new version")
            self.load_latest_approved_model()
            return True

        except ModelLoadError as e:
            logger.error(f"Failed to refresh model: {e}")
            # Drop the cache once it crosses max_stale_seconds so a model
            # rejected/deleted upstream stops being served indefinitely.
            if self._current_model is not None and self._last_load_time is not None:
                stale_age = (datetime.now(UTC) - self._last_load_time).total_seconds()
                if stale_age <= self.max_stale_seconds:
                    logger.warning(
                        "Using existing model due to refresh failure "
                        f"(age {stale_age:.0f}s <= max_stale {self.max_stale_seconds}s)"
                    )
                    return False
                logger.error(
                    f"Cached model exceeded max_stale_seconds "
                    f"({stale_age:.0f}s > {self.max_stale_seconds}s); dropping cache"
                )
                self._current_model = None
                self._current_model_info = None
                self._last_load_time = None
            raise ModelLoadError(
                f"Model refresh failed and no fallback available: {e}"
            ) from e

    def _get_latest_approved_model_info(self) -> dict[str, Any] | None:
        try:
            return self.model_registry.get_latest_approved_model(
                self.model_package_group_name
            )

        except ClientError as e:
            error_code = e.response["Error"]["Code"]
            if error_code == "ValidationException":
                logger.error(
                    f"Model package group not found: {self.model_package_group_name}"
                )
                return None
            logger.error(f"AWS API error getting model info: {e}")
            return None
        except BotoCoreError as e:
            logger.error(f"Boto error getting model info: {e}")
            return None

    def _download_and_load_model(self, model_info: dict[str, Any]) -> Any:
        """Download and load model from S3, verifying SHA256 against the
        Model Package's manifest before any unpickling."""
        model_package_arn = model_info["ModelPackageArn"]
        model_data_url = None
        bucket = None

        try:
            cached_model = self._load_from_extracted_cache(model_package_arn)
            if cached_model is not None:
                return cached_model

            package_details = self.sagemaker_client.describe_model_package(
                ModelPackageName=model_package_arn
            )
            model_data_url = self._model_data_url(package_details)

            logger.info(f"Downloading model from: {model_data_url}")
            bucket, key = self._parse_s3_url(model_data_url)

            customer_meta = package_details.get("CustomerMetadataProperties", {}) or {}
            expected_sha256 = self._validated_manifest_sha256(
                model_package_arn, customer_meta
            )

            temp_path = self._download_model_archive(bucket, key)
            try:
                # CRITICAL: verify the tarball SHA256 BEFORE we touch its contents.
                verify_artifact(Path(temp_path), expected_sha256)

                model_obj, extracted_paths = self._extract_and_load_model_with_cache(
                    temp_path, model_package_arn, expected_sha256
                )
                self._extracted_artifacts_cache[model_package_arn] = extracted_paths
                return model_obj
            finally:
                if os.path.exists(temp_path):
                    os.unlink(temp_path)

        except (ArtifactIntegrityError, FeatureSchemaVersionMismatch):
            raise
        except ClientError as e:
            error_code = e.response["Error"]["Code"]
            if error_code == "NoSuchKey":
                raise ModelLoadError(
                    f"Model file not found in S3: {model_data_url}"
                ) from e
            elif error_code == "NoSuchBucket":
                raise ModelLoadError(f"S3 bucket not found: {bucket}") from e
            else:
                raise ModelLoadError(f"S3 error downloading model: {e}") from e
        except (
            BotoCoreError,
            OSError,
            ValueError,
            KeyError,
            pickle.UnpicklingError,
        ) as e:
            logger.error(f"Error downloading model: {e}", exc_info=True)
            raise ModelLoadError(f"Failed to download model: {e}") from e
        except Exception as e:
            logger.exception("Unexpected error downloading model")
            raise ModelLoadError(f"Failed to download model: {e}") from e

    def _load_from_extracted_cache(self, model_package_arn: str) -> Any | None:
        if model_package_arn not in self._extracted_artifacts_cache:
            return None

        logger.info(f"Using cached extracted artifacts for {model_package_arn}")
        cached_paths = self._extracted_artifacts_cache[model_package_arn]
        # The registry manifest hashes the tarball, not the extracted model.pkl.
        # For process-local extracted-cache reuse, verify the model file against
        # the digest captured immediately after the first verified extraction.
        model_sha256 = cached_paths.get("model_sha256")
        if not model_sha256:
            logger.warning(
                "Cached extracted artifacts for %s lack model_sha256; "
                "discarding cache and downloading again",
                model_package_arn,
            )
            self._extracted_artifacts_cache.pop(model_package_arn, None)
            return None

        verify_artifact(Path(cached_paths["model"]), model_sha256)
        return joblib.load(cached_paths["model"])

    def _model_data_url(self, package_details: dict[str, Any]) -> str:
        inference_spec = package_details.get("InferenceSpecification", {})
        containers = inference_spec.get("Containers", [])
        if not containers:
            raise ModelLoadError("No containers found in model package")

        model_data_url = containers[0].get("ModelDataUrl")
        if not model_data_url:
            raise ModelLoadError("No model data URL found in container specification")
        return cast(str, model_data_url)

    def _parse_s3_url(self, model_data_url: str) -> tuple[str, str]:
        if not model_data_url.startswith("s3://"):
            raise ModelLoadError(f"Invalid S3 URL format: {model_data_url}")
        bucket, key = model_data_url[5:].split("/", 1)
        return bucket, key

    def _validated_manifest_sha256(
        self, model_package_arn: str, customer_meta: dict[str, Any]
    ) -> str:
        expected_sha256 = customer_meta.get("sha256")
        if not expected_sha256:
            raise ArtifactIntegrityError(
                f"Model package {model_package_arn} has no 'sha256' in "
                "CustomerMetadataProperties; refusing to load unverified artifact. "
                "Run scripts/backfill_model_manifests.py against this group."
            )

        artifact_schema_version = customer_meta.get(SCHEMA_VERSION_METADATA_KEY)
        if artifact_schema_version != CURRENT_FEATURE_SCHEMA_VERSION:
            raise FeatureSchemaVersionMismatch(
                f"Model package {model_package_arn} has feature_schema_version="
                f"{artifact_schema_version!r}; runtime requires "
                f"{CURRENT_FEATURE_SCHEMA_VERSION!r}. Older artifacts use an "
                f"incompatible feature set (v1 also contains target-leaked "
                f"features) and are not loadable. Retrain and re-register."
            )

        return cast(str, expected_sha256)

    def _download_model_archive(self, bucket: str, key: str) -> str:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".tar.gz") as temp_file:
            temp_path = temp_file.name

        try:
            self.s3_client.download_file(bucket, key, temp_path)
            logger.info(f"Downloaded model to: {temp_path}")
            return temp_path
        except Exception:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
            raise

    def _extract_and_load_model_with_cache(
        self,
        tar_path: str,
        model_package_arn: str,
        expected_sha256: str,
    ) -> tuple[Any, dict[str, str]]:
        """Extract a verified tarball into a SHA256-keyed cache directory and
        return the loaded model plus paths to the cached artifacts.

        Caller MUST have already called ``verify_artifact(tar_path, expected_sha256)``
        — this function trusts the tarball but uses :func:`extractall_data_filter`
        (PEP 706 ``data`` filter on Python 3.12+) as defense-in-depth.

        Single-slot policy for ``/tmp/models``:

        1. Stage extraction into ``<sha256>.tmp`` (removed first if it exists)
           so a mid-extract failure cannot leave a
           half-populated directory that *looks* like a valid SHA-keyed slot.
        2. Atomic-rename to the final ``<sha256>`` only after extraction
           completes successfully.
        3. Reap every sibling SHA-keyed dir that is neither the new slot nor
           the in-flight stage to keep Lambda ``/tmp`` within its 512 MiB limit.
        """
        import shutil

        try:
            final_dir = Path(self.cache_dir) / expected_sha256
            stage_dir = Path(self.cache_dir) / f"{expected_sha256}.tmp"

            # If the final slot already exists, skip extraction entirely. The
            # model file inside is already verified.
            if final_dir.exists() and (final_dir / MODEL_PKL).exists():
                logger.info(f"Reusing cached extract at: {final_dir}")
                self._reap_stale_slots(keep_sha256=expected_sha256)
                extract_dir = final_dir
            else:
                if stage_dir.exists():
                    shutil.rmtree(stage_dir, ignore_errors=True)
                stage_dir.mkdir(parents=True, exist_ok=True)

                with tarfile.open(tar_path, "r:gz") as tar:
                    extractall_data_filter(tar, stage_dir)

                # Atomic rename. If final_dir somehow exists (race with another
                # warm container? unlikely in Lambda, but cheap to handle), drop
                # the stage and use the existing final.
                if final_dir.exists():
                    shutil.rmtree(stage_dir, ignore_errors=True)
                else:
                    stage_dir.rename(final_dir)

                # Now that the new slot is committed, reap every sibling that
                # is neither the new slot nor an in-flight stage of it.
                self._reap_stale_slots(keep_sha256=expected_sha256)

                extract_dir = final_dir

            logger.info(f"Extracted model to: {extract_dir}")

            model_file = extract_dir / MODEL_PKL
            if not model_file.exists():
                alternative_paths = [
                    extract_dir / "model" / MODEL_PKL,
                    extract_dir / "output" / MODEL_PKL,
                ]
                for alt_path in alternative_paths:
                    if alt_path.exists():
                        model_file = alt_path
                        break

            if not model_file.exists():
                raise ModelLoadError(
                    f"{MODEL_PKL} not found in extracted archive at {extract_dir}"
                )

            logger.info(f"Loading model from: {model_file}")
            model_sha256 = compute_sha256(model_file)
            model_obj = joblib.load(model_file)

            preprocessor_file = extract_dir / FEATURE_PREPROCESSOR_PKL
            scaler_file = extract_dir / FEATURE_SCALER_PKL

            if not preprocessor_file.exists():
                raise ModelLoadError(
                    f"{FEATURE_PREPROCESSOR_PKL} not found at {preprocessor_file}"
                )
            if not scaler_file.exists():
                raise ModelLoadError(f"{FEATURE_SCALER_PKL} not found at {scaler_file}")

            # Use the extracted artifacts in place — the SHA256-keyed
            # ``extract_dir`` already gives us a stable cache path. The previous
            # ``shutil.copy2`` step doubled disk writes and added Lambda /tmp
            # pressure for no functional benefit.
            extracted_paths = {
                "model": str(model_file),
                "preprocessor": str(preprocessor_file),
                "scaler": str(scaler_file),
                "extract_dir": str(extract_dir),
                "expected_sha256": expected_sha256,
                "model_sha256": model_sha256,
            }

            logger.info(f"Successfully loaded model of type: {type(model_obj)}")
            return model_obj, extracted_paths

        except ArtifactIntegrityError:
            raise
        except (
            ClientError,
            BotoCoreError,
            OSError,
            tarfile.TarError,
            pickle.UnpicklingError,
            ValueError,
            KeyError,
        ) as e:
            logger.error(f"Error extracting/loading model: {e}", exc_info=True)
            raise ModelLoadError(f"Failed to extract/load model: {e}") from e
        except Exception as e:
            logger.exception("Unexpected error extracting/loading model")
            raise ModelLoadError(f"Failed to extract/load model: {e}") from e

    def _validate_model(self, model_obj: Any, model_info: dict[str, Any]) -> None:
        try:
            if model_obj is None:
                raise ModelValidationError("Model object is None")

            if not callable(getattr(model_obj, "predict", None)):
                raise ModelValidationError(
                    "Loaded model has no callable 'predict' method; it cannot serve inference"
                )

            # Wrappers (e.g. BoxOfficeXGBoostModel) expose their own fitted flag;
            # honour it so an unfitted wrapper is rejected at load rather than on
            # the first prediction. Objects without the attribute default to True.
            if getattr(model_obj, "is_fitted", True) is False:
                raise ModelValidationError(
                    "Loaded model reports is_fitted=False; it was never trained"
                )

            required_fields = ["ModelPackageArn", "ModelPackageStatus"]
            for field in required_fields:
                if field not in model_info:
                    raise ModelValidationError(
                        f"Missing required field in model info: {field}"
                    )

            if model_info.get("ModelPackageStatus") != "Completed":
                raise ModelValidationError(
                    f"Model package status is not 'Completed': {model_info.get('ModelPackageStatus')}"
                )

            model_type = type(model_obj).__name__
            logger.info(f"Validating model of type: {model_type}")

            if "xgboost" in model_type.lower() or "XGB" in model_type:
                self._validate_xgboost_model(model_obj)
            elif hasattr(model_obj, "fit") and hasattr(model_obj, "predict"):
                self._validate_sklearn_model(model_obj)

            logger.info("Model validation completed successfully")

        except ModelValidationError:
            raise
        except (TypeError, AttributeError, ValueError, KeyError) as e:
            logger.error(f"Model validation failed: {e}")
            raise ModelValidationError(f"Unexpected validation error: {e}") from e
        except Exception as e:
            logger.exception("Unexpected model validation failure")
            raise ModelValidationError(f"Unexpected validation error: {e}") from e

    def _validate_xgboost_model(self, model_obj: Any) -> None:
        """Confirm the XGBoost estimator is fitted. ``get_booster()`` raises
        ``NotFittedError`` on an untrained model, so a model that loaded but was
        never fit is rejected here rather than failing on the first request."""
        from xgboost.core import XGBoostError

        get_booster = getattr(model_obj, "get_booster", None)
        if not callable(get_booster):
            return

        try:
            get_booster()
        except XGBoostError as e:
            raise ModelValidationError(f"XGBoost model is not fitted: {e}") from e

    def _validate_sklearn_model(self, model_obj: Any) -> None:
        """Confirm the sklearn-compatible estimator is fitted."""
        from sklearn.exceptions import NotFittedError
        from sklearn.utils.validation import check_is_fitted

        try:
            check_is_fitted(model_obj)
        except NotFittedError as e:
            raise ModelValidationError(f"Model is not fitted: {e}") from e
        except (TypeError, ValueError):
            # check_is_fitted can't introspect every custom estimator; absence
            # of a verdict is not a failure, so don't reject on that alone.
            logger.debug("check_is_fitted could not introspect the model; skipping")

    def _reap_stale_slots(self, keep_sha256: str) -> None:
        """Remove every cache slot under ``cache_dir`` except the one we just
        committed. Single-slot policy: at most one fully-extracted model on
        disk after this returns. The in-flight stage dir for the kept slot is
        also preserved (it'll be renamed away momentarily).
        """
        import shutil

        if not self.cache_dir.exists():
            return
        keep_names = {keep_sha256, f"{keep_sha256}.tmp"}
        for entry in self.cache_dir.iterdir():
            if entry.name in keep_names:
                continue
            try:
                if entry.is_dir():
                    shutil.rmtree(entry, ignore_errors=True)
                else:
                    entry.unlink()
            except OSError as e:
                logger.warning(f"Could not reap stale cache entry {entry}: {e}")

    def clear_cache(self) -> None:
        """Remove all SHA256-keyed extract directories under ``cache_dir``."""
        import shutil

        try:
            for entry in self.cache_dir.iterdir():
                if entry.is_dir():
                    shutil.rmtree(entry, ignore_errors=True)
                elif entry.is_file():
                    entry.unlink()
            self._extracted_artifacts_cache.clear()
            logger.info("Cleared model cache")
        except OSError as e:
            logger.warning(f"Failed to clear cache: {e}")

    def get_cache_info(self) -> dict[str, Any]:
        try:
            cache_dirs = (
                [p for p in self.cache_dir.iterdir() if p.is_dir()]
                if self.cache_dir.exists()
                else []
            )
            total_size = 0
            for entry in cache_dirs:
                for f in entry.rglob("*"):
                    if f.is_file():
                        total_size += f.stat().st_size

            return {
                "cache_dir": str(self.cache_dir),
                "cached_models": len(cache_dirs),
                "total_cache_size_bytes": total_size,
                "cache_ttl_seconds": self.cache_ttl_seconds,
                "current_model_loaded": self._current_model is not None,
                "last_load_time": (
                    self._last_load_time.isoformat() if self._last_load_time else None
                ),
            }
        except OSError as e:
            logger.error(f"Failed to get cache info: {e}")
            return {"error": str(e)}

    def get_latest_approved_model_info(self) -> RegistryModelInfo | None:
        model_info_dict = self._get_latest_approved_model_info()
        return RegistryModelInfo(model_info_dict) if model_info_dict else None

    def get_model_artifacts_paths(self) -> dict[str, str]:
        """Get paths to model artifacts, reusing cached extracted artifacts."""
        if self._current_model is None:
            raise ModelLoadError(
                "No model loaded. Call load_latest_approved_model first."
            )

        try:
            model_package_arn = self._current_model_info["ModelPackageArn"]

            if model_package_arn in self._extracted_artifacts_cache:
                cached_paths = self._extracted_artifacts_cache[model_package_arn]
                logger.info("Using cached extracted artifacts")
                return {
                    "model": cached_paths["model"],
                    "preprocessor": cached_paths["preprocessor"],
                    "scaler": cached_paths["scaler"],
                }

            raise ModelLoadError(
                f"Model artifacts not found in cache for {model_package_arn}"
            )

        except KeyError as e:
            logger.error(f"Failed to get model artifacts paths: {e}")
            raise ModelLoadError(
                f"Failed to get model artifacts paths: missing {e}"
            ) from e
        except ModelLoadError:
            raise
        except Exception as e:
            logger.exception("Unexpected error resolving model artifact paths")
            raise ModelLoadError(f"Failed to get model artifacts paths: {e}") from e
