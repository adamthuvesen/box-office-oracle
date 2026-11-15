"""The logger type the pipeline phases accept: a stdlib logger or Prefect's run logger adapter."""

import logging

type RunLogger = logging.Logger | logging.LoggerAdapter[logging.Logger]
