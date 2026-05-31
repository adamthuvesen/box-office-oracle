"""Batch Snowflake table persistence for the ML pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import pandas as pd

from box_office.config import config


@dataclass
class TableOperation:
    """Track table operation results for batch logging."""

    table_name: str
    schema: str
    success: bool
    error: Optional[str] = None


@dataclass
class TableSaveSpec:
    df: pd.DataFrame
    table_name: str
    schema: str


@dataclass
class TableSaveReport:
    operations: List[TableOperation]
    results: Dict[str, bool]


def save_tables(
    specs: List[TableSaveSpec],
    save_fn: Callable[[pd.DataFrame, str, str], bool],
) -> TableSaveReport:
    """Save multiple tables, collecting per-table success/failure."""
    operations: List[TableOperation] = []
    results: Dict[str, bool] = {}

    for spec in specs:
        try:
            success = save_fn(spec.df, spec.table_name, spec.schema)
            operations.append(
                TableOperation(
                    table_name=spec.table_name,
                    schema=spec.schema,
                    success=success,
                    error=None if success else "Save operation returned False",
                )
            )
            results[spec.table_name] = success
        except (OSError, ValueError, RuntimeError) as e:
            operations.append(
                TableOperation(
                    table_name=spec.table_name,
                    schema=spec.schema,
                    success=False,
                    error=str(e),
                )
            )
            results[spec.table_name] = False

    return TableSaveReport(operations=operations, results=results)


def log_table_operations_summary(operations: List[TableOperation], logger) -> None:
    """Log table operations as a single summary."""
    if not operations:
        logger.info("No table operations to report")
        return

    successful = sum(1 for op in operations if op.success)
    total = len(operations)

    ml_training_ops = [
        op for op in operations if op.schema == config.snowflake.schemas.ml_training
    ]
    feature_store_ops = [
        op for op in operations if op.schema == config.snowflake.schemas.feature_store
    ]

    logger.info(
        f"Snowflake operations completed: {successful}/{total} tables saved successfully"
    )

    if ml_training_ops:
        ml_successful = sum(1 for op in ml_training_ops if op.success)
        logger.info(
            f"  ML_TRAINING schema: {ml_successful}/{len(ml_training_ops)} tables saved"
        )

    if feature_store_ops:
        fs_successful = sum(1 for op in feature_store_ops if op.success)
        logger.info(
            f"  FEATURE_STORE schema: {fs_successful}/{len(feature_store_ops)} tables saved"
        )

    failures = [op for op in operations if not op.success]
    if failures:
        logger.error(f"Failed table operations ({len(failures)}):")
        for failure in failures:
            logger.error(f"{failure.schema}.{failure.table_name}: {failure.error}")
    else:
        logger.info("All table operations completed successfully")
