"""
Environment setup utilities for the box office ML pipeline.

Centralizes warning filters and environment variable loading to avoid duplication.
"""

import os
import warnings
from dotenv import load_dotenv


def configure_environment():
    """Suppress noisy third-party warnings and load .env. Call once at entrypoint."""
    warnings.filterwarnings(
        "ignore", message=".*insecure_mode.*", category=DeprecationWarning
    )
    warnings.filterwarnings(
        "ignore", message=".*is_datetime64tz_dtype.*", category=DeprecationWarning
    )
    warnings.filterwarnings(
        "ignore",
        message=".*Attempting to mutate a Context.*",
        category=DeprecationWarning,
    )
    warnings.filterwarnings(
        "ignore", message=".*found in sys.modules.*", category=RuntimeWarning
    )

    os.environ.setdefault("SAGEMAKER_DISABLE_CONFIG", "1")
    os.environ.setdefault("SAGEMAKER_CONFIG_FILE", "/dev/null")

    load_dotenv()
