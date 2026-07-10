from datetime import datetime, timedelta

from box_office.orchestration.tasks import metrics_tasks


class CapturingLogger:
    def __init__(self) -> None:
        self.messages: list[str] = []

    def info(self, message: str, *args, **kwargs) -> None:
        self.messages.append(message % args if args else message)

    def warning(self, message: str, *args, **kwargs) -> None:
        self.messages.append(message % args if args else message)


def test_log_pipeline_completion_metrics_builds_summary(monkeypatch):
    logger = CapturingLogger()
    monkeypatch.setattr(metrics_tasks, "get_run_logger", lambda: logger)

    start = datetime.now() - timedelta(minutes=5)
    registry_metrics = {
        "model_registration": {
            "status": "success",
            "aws_result": {
                "status": "success",
                "model_package_arn": "arn:pkg/example/1234567890",
                "approval_status": "PendingManualApproval",
            },
        },
        "model_promotion_validation": {
            "promote": True,
            "validation_details": {"r2_score": 0.82, "min_required": 0.55},
        },
        "aws_promotion": {"status": "success", "promotion_time_seconds": 3.5},
    }

    summary = metrics_tasks.log_pipeline_completion_metrics.fn(
        pipeline_start_metrics={
            "pipeline_start_time": start.isoformat(),
            "pipeline_id": "box-office-ml-test",
        },
        data_metrics={
            "training_samples": 100,
            "validation_samples": 25,
            "target_column": "WORLDWIDE_GROSS",
        },
        feature_metrics={"total_features": 69},
        training_metrics={
            "cv_results": {
                "mean_cv_mae": 10.0,
                "std_cv_mae": 2.0,
                "mean_best_iteration": 42,
                "cv_scores": [1, 2, 3],
            },
            "oof_results": {"oof_r2": 0.82, "oof_mae": 1_000_000},
            "training_time": 120.0,
            "estimated_cost": 0.25,
        },
        model_registry_metrics=registry_metrics,
    )

    assert summary["pipeline_id"] == "box-office-ml-test"
    assert summary["data_summary"]["training_samples"] == 100
    assert summary["feature_summary"]["total_features"] == 69
    assert summary["training_summary"]["training_time"] == 120.0
    assert summary["model_registry_summary"] == registry_metrics
    assert summary["execution_time"]["duration_seconds"] >= 0
    assert "Model registered in AWS Model Registry" in logger.messages
    assert "AWS Model Package promoted to Approved status" in logger.messages


def test_log_pipeline_completion_metrics_without_registry(monkeypatch):
    logger = CapturingLogger()
    monkeypatch.setattr(metrics_tasks, "get_run_logger", lambda: logger)

    summary = metrics_tasks.log_pipeline_completion_metrics.fn(
        pipeline_start_metrics={
            "pipeline_start_time": datetime.now().isoformat(),
            "pipeline_id": "box-office-ml-no-registry",
        },
        data_metrics={
            "training_samples": 1,
            "validation_samples": 0,
            "target_column": "WORLDWIDE_GROSS",
        },
        feature_metrics={"total_features": 1},
        training_metrics={},
    )

    assert summary["model_registry_summary"] is None
    assert "Check S3 output for detailed results" in logger.messages
