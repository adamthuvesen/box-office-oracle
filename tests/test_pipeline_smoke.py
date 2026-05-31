"""Smoke tests for pipeline components.

No module-level ``sys.modules`` mocks for ``snowflake`` / ``sagemaker`` —
they leak across the session and corrupt other tests. Real packages
import cleanly without contacting any service.
"""


class TestPipelineImports:
    def test_ml_pipeline_imports(self):
        from box_office.orchestration.flows.ml_pipeline import (
            ml_pipeline,
            run_ml_pipeline_logic,
            get_logger,
        )

        assert ml_pipeline is not None
        assert run_ml_pipeline_logic is not None
        assert get_logger is not None

    def test_data_tasks_imports(self):
        from box_office.orchestration.tasks.data_tasks import (
            load_staging_box_office_from_snowflake,
            split_data,
            apply_feature_engineering,
            scale_features,
            transform_targets,
            save_artifacts,
            save_dataset_to_snowflake_impl,
        )

        assert load_staging_box_office_from_snowflake is not None
        assert split_data is not None
        assert apply_feature_engineering is not None
        assert scale_features is not None
        assert transform_targets is not None
        assert save_artifacts is not None
        assert save_dataset_to_snowflake_impl is not None

    def test_training_tasks_imports(self):
        from box_office.orchestration.tasks.training_tasks import (
            upload_processed_data_to_s3,
            train_model,
            download_and_analyze_results,
        )

        assert upload_processed_data_to_s3 is not None
        assert train_model is not None
        assert download_and_analyze_results is not None

    def test_metrics_tasks_imports(self):
        from box_office.orchestration.tasks.metrics_tasks import (
            log_pipeline_start_metrics,
            log_data_processing_metrics,
            log_feature_engineering_metrics,
            log_model_training_summary,
            log_pipeline_completion_metrics,
        )

        assert log_pipeline_start_metrics is not None
        assert log_data_processing_metrics is not None
        assert log_feature_engineering_metrics is not None
        assert log_model_training_summary is not None
        assert log_pipeline_completion_metrics is not None


class TestConfigSystem:
    def test_config_singleton_imports(self):
        from box_office.config import config

        assert config is not None
        assert hasattr(config, "aws")
        assert hasattr(config, "snowflake")
        assert hasattr(config, "model")

    def test_config_aws_section(self):
        from box_office.config import config

        assert hasattr(config.aws, "region")
        assert hasattr(config.aws, "sagemaker_role_arn")

        assert config.aws.region is not None
        assert config.aws.sagemaker_role_arn is not None

    def test_config_snowflake_section(self):
        from box_office.config import config

        assert hasattr(config.snowflake, "user")
        assert hasattr(config.snowflake, "account")
        assert hasattr(config.snowflake, "database")
        assert hasattr(config.snowflake, "warehouse")
        assert hasattr(config.snowflake, "schemas")


class TestModelRegistry:
    def test_model_registry_imports(self):
        from box_office.ml.model_registry.aws_model_registry import AWSModelRegistry

        assert AWSModelRegistry is not None

    def test_model_registry_initialization(self):
        from box_office.ml.model_registry.aws_model_registry import AWSModelRegistry

        registry = AWSModelRegistry(region_name="eu-north-1")

        assert registry is not None
        assert hasattr(registry, "sagemaker_client")


class TestFeatureEngineering:
    def test_all_transformers_importable(self):
        from box_office.ml.feature_pipeline import CoreNumericalTransformer
        from box_office.ml.feature_pipeline import TemporalTransformer
        from box_office.ml.feature_pipeline import GenreTransformer
        from box_office.ml.feature_pipeline import IndustryTransformer
        from box_office.ml.feature_pipeline import FinancialTransformer
        from box_office.ml.feature_pipeline import (
            InteractionTransformer,
        )

        assert CoreNumericalTransformer is not None
        assert TemporalTransformer is not None
        assert GenreTransformer is not None
        assert IndustryTransformer is not None
        assert FinancialTransformer is not None
        assert InteractionTransformer is not None

    def test_feature_preprocessor_instantiation(self):
        from box_office.ml.feature_preprocessor import FeaturePreprocessorHigh

        preprocessor = FeaturePreprocessorHigh()

        assert preprocessor is not None
        # Pre-engineered drop + six engineered transformers + raw-column strip.
        assert list(preprocessor.pipeline.named_steps) == [
            "drop_pre_engineered",
            "core",
            "temporal",
            "genre",
            "industry",
            "financial",
            "interactions",
            "select",
        ]


class TestUtilities:
    def test_snowflake_utils_imports(self):
        from box_office.utils.snowflake_connection import (
            create_snowflake_connection,
            enforce_data_types,
            load_private_key_from_file,
        )

        assert create_snowflake_connection is not None
        assert enforce_data_types is not None
        assert load_private_key_from_file is not None


class TestSageMakerIntegration:
    def test_sagemaker_client_imports(self):
        from box_office.sagemaker.sagemaker_train_job import SageMakerClient

        assert SageMakerClient is not None

    def test_sagemaker_client_initialization(self):
        from box_office.sagemaker.sagemaker_train_job import SageMakerClient

        client = SageMakerClient(
            region="eu-north-1", role="arn:aws:iam::123456789:role/test-role"
        )

        assert client is not None
        assert client.region == "eu-north-1"


class TestConsoleScripts:
    def test_main_function_exists(self):
        from box_office.orchestration.flows.ml_pipeline import main

        assert main is not None
        assert callable(main)


class TestPipelineTasks:
    def test_tasks_are_decorated(self):
        """split_data and apply_feature_engineering carry @task wrappers."""
        from box_office.orchestration.tasks.data_tasks import (
            split_data,
            apply_feature_engineering,
        )

        assert hasattr(split_data, "fn") or hasattr(split_data, "__wrapped__")
        assert hasattr(apply_feature_engineering, "fn") or hasattr(
            apply_feature_engineering, "__wrapped__"
        )


class TestPipelineFlow:
    def test_ml_pipeline_is_flow(self):
        from box_office.orchestration.flows.ml_pipeline import ml_pipeline

        assert (
            hasattr(ml_pipeline, "fn")
            or hasattr(ml_pipeline, "__wrapped__")
            or callable(ml_pipeline)
        )

    def test_pipeline_has_parameters(self):
        from box_office.orchestration.flows.ml_pipeline import ml_pipeline
        import inspect

        sig = inspect.signature(ml_pipeline)

        assert "environment" in sig.parameters
        assert "experiment_name" in sig.parameters

        assert sig.parameters["environment"].default == "dev"
        assert sig.parameters["experiment_name"].default == "box-office-predictions"
