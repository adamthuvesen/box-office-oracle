-- Snowflake role/permission reconciliation for BOX_OFFICE.
--
-- Run with scripts/apply_snowflake_grants.py (connects as ACCOUNTADMIN).
-- Idempotent: re-running is a no-op. Never drops a table or data.
--
-- Target model (see docs/architecture.md "Security posture"):
--   BOX_OFFICE_LOADER : owns RAW; used only by scripts/load_dataset_to_snowflake.py
--                       (dataset loads). Read-only everywhere else.
--   DBT_RUNNER        : owns STAGING, ML_TRAINING, FEATURE_STORE; read-only on RAW.
--                       The role for `box-office-pipeline` and dbt.
--   ACCOUNTADMIN      : administration only, never in the runtime path.
--
-- The runtime breakage this fixes: objects created by old ACCOUNTADMIN runs
-- (FEATURE_STORE schema + FEATURE_METADATA, RAW schema + tables) were owned by
-- ACCOUNTADMIN, so the least-privilege runtime roles could not replace them.
-- GRANT OWNERSHIP ... COPY CURRENT GRANTS transfers them without dropping the
-- existing read grants.

USE ROLE ACCOUNTADMIN;

-- ---------------------------------------------------------------------------
-- 1. Loader role for RAW dataset loads.
-- ---------------------------------------------------------------------------
CREATE ROLE IF NOT EXISTS BOX_OFFICE_LOADER
    COMMENT = 'Owns BOX_OFFICE.RAW. Used only by scripts/load_dataset_to_snowflake.py for dataset loads.';

CREATE ROLE IF NOT EXISTS DBT_RUNNER
    COMMENT = 'Owns BOX_OFFICE.STAGING, ML_TRAINING, FEATURE_STORE; read-only on RAW. Runs box-office-pipeline and dbt.';

-- Role hierarchy + who can assume the runtime roles.
GRANT ROLE BOX_OFFICE_LOADER TO ROLE SYSADMIN;
GRANT ROLE DBT_RUNNER        TO ROLE SYSADMIN;
GRANT ROLE BOX_OFFICE_LOADER TO USER NEMO;
GRANT ROLE DBT_RUNNER        TO USER NEMO;

-- Compute + database visibility for the loader.
GRANT USAGE ON WAREHOUSE COMPUTE_WH TO ROLE BOX_OFFICE_LOADER;
GRANT USAGE ON DATABASE BOX_OFFICE  TO ROLE BOX_OFFICE_LOADER;

-- ---------------------------------------------------------------------------
-- 2. RAW: hand ownership to the loader, keep DBT_RUNNER read-only.
-- ---------------------------------------------------------------------------
GRANT OWNERSHIP ON SCHEMA BOX_OFFICE.RAW
    TO ROLE BOX_OFFICE_LOADER COPY CURRENT GRANTS;
GRANT OWNERSHIP ON ALL TABLES IN SCHEMA BOX_OFFICE.RAW
    TO ROLE BOX_OFFICE_LOADER COPY CURRENT GRANTS;

-- DBT_RUNNER reads RAW (existing tables + anything the loader creates later).
GRANT USAGE ON SCHEMA BOX_OFFICE.RAW              TO ROLE DBT_RUNNER;
GRANT SELECT ON ALL TABLES IN SCHEMA BOX_OFFICE.RAW    TO ROLE DBT_RUNNER;
GRANT SELECT ON FUTURE TABLES IN SCHEMA BOX_OFFICE.RAW TO ROLE DBT_RUNNER;

-- ---------------------------------------------------------------------------
-- 3. FEATURE_STORE: transfer to DBT_RUNNER (this is the pipeline break).
-- ---------------------------------------------------------------------------
GRANT OWNERSHIP ON SCHEMA BOX_OFFICE.FEATURE_STORE
    TO ROLE DBT_RUNNER COPY CURRENT GRANTS;
GRANT OWNERSHIP ON ALL TABLES IN SCHEMA BOX_OFFICE.FEATURE_STORE
    TO ROLE DBT_RUNNER COPY CURRENT GRANTS;

-- ---------------------------------------------------------------------------
-- 4. STAGING / ML_TRAINING / FEATURE_STORE: DBT_RUNNER owns them, so tables it
--    creates are auto-owned. Future grants are the safety net for any table a
--    different role (e.g. a one-off ACCOUNTADMIN fix) creates in these schemas.
-- ---------------------------------------------------------------------------
GRANT USAGE, CREATE TABLE, CREATE VIEW ON SCHEMA BOX_OFFICE.STAGING       TO ROLE DBT_RUNNER;
GRANT USAGE, CREATE TABLE, CREATE VIEW ON SCHEMA BOX_OFFICE.ML_TRAINING   TO ROLE DBT_RUNNER;
GRANT USAGE, CREATE TABLE, CREATE VIEW ON SCHEMA BOX_OFFICE.FEATURE_STORE TO ROLE DBT_RUNNER;

GRANT ALL PRIVILEGES ON FUTURE TABLES IN SCHEMA BOX_OFFICE.STAGING       TO ROLE DBT_RUNNER;
GRANT ALL PRIVILEGES ON FUTURE TABLES IN SCHEMA BOX_OFFICE.ML_TRAINING   TO ROLE DBT_RUNNER;
GRANT ALL PRIVILEGES ON FUTURE TABLES IN SCHEMA BOX_OFFICE.FEATURE_STORE TO ROLE DBT_RUNNER;

GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA BOX_OFFICE.STAGING       TO ROLE DBT_RUNNER;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA BOX_OFFICE.ML_TRAINING   TO ROLE DBT_RUNNER;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA BOX_OFFICE.FEATURE_STORE TO ROLE DBT_RUNNER;
