"""Tests for the pydantic-settings-based configuration."""

import os
import unittest
from unittest.mock import patch

from pydantic import ValidationError

from box_office.config import Settings


# Mirrors the env vars the README documents. The startup test below asserts
# that every entry maps to a defined Settings field — that's our contract
# with users who follow the README to populate `.env`.
DOCUMENTED_ENV_VARS = {
    "AWS_REGION",
    "AWS_DEFAULT_REGION",
    "AWS_S3_BUCKET",
    "S3_BUCKET_NAME",
    "SAGEMAKER_BUCKET",
    "AWS_ACCOUNT_ID",
    "SAGEMAKER_ROLE_ARN",
    "SNOWFLAKE_USER",
    "SNOWFLAKE_PASSWORD",
    "SNOWFLAKE_PRIVATE_KEY_PATH",
    "SNOWFLAKE_PRIVATE_KEY_PASSPHRASE",
    "SNOWFLAKE_ACCOUNT",
    "SNOWFLAKE_WAREHOUSE",
    "SNOWFLAKE_DATABASE",
    "SNOWFLAKE_SCHEMA_RAW",
    "SNOWFLAKE_SCHEMA_STAGING",
    "SNOWFLAKE_SCHEMA_FEATURE_STORE",
    "SNOWFLAKE_SCHEMA_ML_TRAINING",
    "TMDB_START_YEAR",
    "TMDB_END_YEAR",
    "TMDB_MIN_REVENUE",
    "TMDB_PAGE_LIMIT",
}


def _required_env() -> dict:
    return {
        "AWS_S3_BUCKET": "test-bucket",
        "SAGEMAKER_ROLE_ARN": "arn:test",
        "SNOWFLAKE_USER": "test_user",
        "SNOWFLAKE_ACCOUNT": "test_account",
        "SNOWFLAKE_DATABASE": "TEST_DB",
    }


class TestSettingsBasics(unittest.TestCase):
    def test_loads_required_fields(self):
        with patch.dict(os.environ, _required_env(), clear=True):
            s = Settings(_env_file=None)
            self.assertEqual(s.aws.s3_bucket, "test-bucket")
            self.assertEqual(s.snowflake.user, "test_user")
            # Defaults still apply.
            self.assertEqual(s.aws.region, "eu-north-1")
            self.assertEqual(s.snowflake.warehouse, "COMPUTE_WH")
            self.assertEqual(s.snowflake.schemas.staging, "STAGING")

    def test_env_alias_priority(self):
        env = _required_env() | {
            "AWS_REGION": "us-east-1",
            "AWS_DEFAULT_REGION": "eu-west-1",
        }
        with patch.dict(os.environ, env, clear=True):
            s = Settings(_env_file=None)
            # AWS_REGION wins over AWS_DEFAULT_REGION (first in AliasChoices).
            self.assertEqual(s.aws.region, "us-east-1")

    def test_missing_required_raises(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(ValidationError):
                Settings(_env_file=None)

    def test_nested_model_views_are_sane(self):
        with patch.dict(os.environ, _required_env(), clear=True):
            s = Settings(_env_file=None)
            # Static views surface stable defaults.
            self.assertEqual(s.model.cross_validation.cv_folds, 8)
            self.assertEqual(s.model.cross_validation.end_year, 2024)
            self.assertEqual(s.feature_engineering.max_genre_features, 8)
            self.assertEqual(s.sagemaker.instance_type, "ml.m5.large")
            self.assertEqual(s.ingestion.tmdb.start_year, 2024)


class TestEnvCoverage(unittest.TestCase):
    """Every documented env var must map to a Settings field (alias or name)."""

    def test_every_documented_env_var_is_known(self):
        known = set()
        for name, info in Settings.model_fields.items():
            known.add(name)
            alias_obj = info.validation_alias
            if alias_obj is None:
                continue
            choices = getattr(alias_obj, "choices", None)
            if choices is None:
                known.add(str(alias_obj))
            else:
                for c in choices:
                    known.add(str(c))

        missing = DOCUMENTED_ENV_VARS - known
        self.assertEqual(
            missing,
            set(),
            f"Documented env vars not bound to a Settings field: {sorted(missing)}",
        )


if __name__ == "__main__":
    unittest.main()
