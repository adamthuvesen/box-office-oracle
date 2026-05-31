"""Pipeline must not reload training data from Snowflake after the data phase."""

from unittest.mock import patch

import pandas as pd


def test_train_phase_does_not_call_snowflake_reload():
    from box_office.orchestration.phases import data_phase, train_phase

    fake_data = data_phase.DataPhaseResult(
        target_column="WORLDWIDE_GROSS",
        X_train_processed=pd.DataFrame({"RELEASE_YEAR": [2020], "A": [1.0]}),
        X_train_scaled=pd.DataFrame({"A": [1.0]}),
        y_train_log=pd.Series([1.0], name="GROSS_LOG"),
        X_train_shape=(1, 1),
        X_val_shape=(1, 1),
        processor_path="/tmp/p.pkl",
        scaler_path="/tmp/s.pkl",
        save_results={},
        validation_results={},
        feature_metrics={},
        data_metrics={},
        feature_names=["A"],
    )

    with (
        patch.object(
            train_phase,
            "upload_processed_data_to_s3",
            return_value="s3://train",
        ) as upload,
        patch.object(
            train_phase,
            "upload_preprocessing_artifacts_to_s3",
            return_value={},
        ),
        patch.object(train_phase, "train_model", return_value=object()),
        patch.object(
            train_phase,
            "download_and_analyze_results",
            return_value={"job_name": "j", "duration": 0, "model_data_url": None},
        ),
        patch.object(train_phase, "log_model_training_summary"),
        patch.object(
            train_phase.sagemaker_train_job,
            "SageMakerClient",
            return_value=object(),
        ),
    ):
        train_phase.run_train_phase(
            fake_data, logger=type("L", (), {"info": lambda *a, **k: None})()
        )

    # The train phase consumes the in-memory frames it is handed; it must not
    # reload the dataset from Snowflake (there is no reload task to call).
    upload.assert_called_once()


def test_sagemaker_training_frames_restore_unscaled_release_year():
    from box_office.orchestration.phases.data_phase import (
        DataPhaseResult,
        sagemaker_training_frames,
    )

    data = DataPhaseResult(
        target_column="WORLDWIDE_GROSS",
        X_train_processed=pd.DataFrame(
            {"RELEASE_YEAR": [2020, 2021], "A": [10.0, 20.0]}
        ),
        X_train_scaled=pd.DataFrame({"RELEASE_YEAR": [-1.0, 1.0], "A": [-0.5, 0.5]}),
        y_train_log=pd.Series([1.0, 2.0], name="GROSS_LOG"),
        X_train_shape=(2, 2),
        X_val_shape=(0, 2),
        processor_path="/tmp/p.pkl",
        scaler_path="/tmp/s.pkl",
        save_results={},
        validation_results={},
        feature_metrics={},
        data_metrics={},
        feature_names=["RELEASE_YEAR", "A"],
    )

    X_train, y_train = sagemaker_training_frames(data)

    assert X_train["RELEASE_YEAR"].tolist() == [2020, 2021]
    assert y_train["GROSS_LOG"].tolist() == [1.0, 2.0]
