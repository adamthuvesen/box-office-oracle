{#
  Overrides dbt's default generate_schema_name behaviour.
  - No custom schema → use target.schema as-is (e.g. DEV_STAGING, PROD_STAGING).
  - Custom schema set → strip whitespace and UPPERCASE it directly, dropping the
    `dbt_<user>_` prefix that dbt would otherwise prepend. This keeps Snowflake
    schema names predictable and avoids stale developer-prefixed schemas in prod.
#}
{% macro generate_schema_name(custom_schema_name, node) -%}
    {%- set default_schema = target.schema -%}
    {%- if custom_schema_name is none -%}
        {{ default_schema }}
    {%- else -%}
        {{ custom_schema_name | trim | upper }}
    {%- endif -%}
{%- endmacro %}
