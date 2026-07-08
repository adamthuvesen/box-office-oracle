"""Configuration management for the serverless inference API."""

from functools import lru_cache

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings

from box_office.shared.names import model_registry_group_name


class Settings(BaseSettings):
    """Application settings with environment variable support."""

    # Environment Configuration
    environment: str = Field(
        default="dev", description="Deployment environment (dev/prod)"
    )
    project_name: str = Field(
        default="box-office", description="Project name used to compose resource names"
    )

    # API Configuration
    api_title: str = Field(
        default="Serverless ML Inference API", description="API title"
    )
    api_version: str = Field(default="1.0.0", description="API version")

    # AWS Configuration
    aws_region: str = Field(default="eu-north-1", description="AWS region")
    aws_account_id: str | None = Field(default=None, description="AWS account ID")

    # Model Registry Configuration
    model_registry_group_name: str | None = Field(
        default=None,
        description="SageMaker Model Registry group name. Composed from project_name and environment when unset.",
    )

    # S3 Configuration
    s3_bucket_name: str = Field(default="", description="S3 bucket for model artifacts")
    s3_model_prefix: str = Field(
        default="models/", description="S3 prefix for model files"
    )

    # Lambda Configuration
    lambda_timeout: int = Field(
        default=30, description="Lambda timeout in seconds", ge=1, le=900
    )
    lambda_memory: int = Field(
        default=3008, description="Lambda memory in MB", ge=128, le=10240
    )

    # Logging Configuration
    log_level: str = Field(default="INFO", description="Logging level")

    # Authentication Configuration
    api_key_header: str = Field(default="X-API-Key", description="API key header name")
    enable_api_key_auth: bool = Field(
        default=True, description="Enable API key authentication"
    )
    api_key: str | None = Field(
        default=None,
        description="API key for authentication (from API_KEY env var)",
    )

    # Performance Configuration
    model_cache_ttl: int = Field(default=3600, description="Model cache TTL in seconds")
    max_request_size: int = Field(
        default=1048576, description="Max request size in bytes (1MB)"
    )
    max_stale_seconds: int = Field(
        default=3600,
        description=(
            "Max seconds a cached model may keep serving after refresh failures. "
            "Once exceeded the loader drops the cache and the next predict re-loads."
        ),
    )

    # CORS Configuration
    cors_origins: list[str] = Field(default=["*"], description="CORS allowed origins")
    cors_methods: list[str] = Field(
        default=["GET", "POST", "OPTIONS"], description="CORS allowed methods"
    )

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": False,
        "extra": "ignore",  # Allow extra fields for forward compatibility
        "protected_namespaces": ("settings_",),  # Only protect settings_ namespace
    }

    @model_validator(mode="after")
    def _compose_model_registry_group_name(self) -> "Settings":
        if not self.model_registry_group_name:
            self.model_registry_group_name = model_registry_group_name(
                project_name=self.project_name,
                environment=self.environment,
            )
        return self


@lru_cache
def get_settings() -> Settings:
    """Get cached application settings instance."""
    return Settings()
