# transformations/ — dbt project

Snowflake dbt project that transforms `RAW.BOX_OFFICE_V3` into the
`STAGING.STG_BOX_OFFICE` table consumed by the ML pipeline.

## Connection

The active profile is `transformations.dev` in [profiles.yml](profiles.yml).
All connection fields are env-var driven:

| Variable                       | Required? | Notes                                         |
| ------------------------------ | --------- | --------------------------------------------- |
| `SNOWFLAKE_ACCOUNT`            | yes       |                                               |
| `SNOWFLAKE_USER`               | yes       |                                               |
| `SNOWFLAKE_PRIVATE_KEY_PATH`   | yes       | absolute path; key-pair auth (no password)    |
| `SNOWFLAKE_ROLE`               | yes       | use `DBT_RUNNER` — never `ACCOUNTADMIN`       |
| `SNOWFLAKE_DATABASE`           | yes       |                                               |
| `SNOWFLAKE_WAREHOUSE`          | yes       |                                               |
| `SNOWFLAKE_SCHEMA_STAGING`     | yes       |                                               |
| `DBT_DISABLE_OCSP`             | no        | escape hatch — see below                      |

## OCSP

OCSP certificate revocation checking is **enabled by default**. Disabling it
weakens MITM defense and stops dbt from detecting revoked CAs.

If you hit a local SSL chain error like:

```
SSL: CERTIFICATE_VERIFY_FAILED [...] unable to get local issuer certificate
```

the canonical fix is to refresh your CA bundle:

```bash
uv pip install --upgrade certifi snowflake-connector-python
```

If you need to keep working while you fix the underlying chain, you can
opt out **on your machine only**:

```bash
export DBT_DISABLE_OCSP=true
```

CI environments must leave this variable unset — production-equivalent runs
always perform OCSP checks.
