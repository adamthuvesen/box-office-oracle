"""Validated Snowflake identifier helpers."""

import re

SQL_IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def validate_sql_identifier(name: str, identifier_type: str = "identifier") -> str:
    """Reject strings that are unsafe as Snowflake unquoted identifiers."""
    if not isinstance(name, str) or not SQL_IDENTIFIER_PATTERN.match(name):
        raise ValueError(
            f"Invalid {identifier_type} name: {name!r}. "
            "Must contain only alphanumeric characters and underscores, "
            "and start with a letter or underscore."
        )
    return name


def fully_qualified_name(database: str, schema: str, table: str) -> str:
    """Build a validated ``database.schema.table`` identifier."""
    return ".".join(
        (
            validate_sql_identifier(database, "database"),
            validate_sql_identifier(schema, "schema"),
            validate_sql_identifier(table, "table"),
        )
    )
