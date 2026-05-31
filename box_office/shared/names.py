"""Resource naming helpers shared by training config and inference settings."""


def model_registry_group_name(
    project_name: str = "box-office",
    environment: str = "dev",
) -> str:
    """SageMaker Model Registry package group name."""
    return f"{project_name}-{environment}-box-office-models"
