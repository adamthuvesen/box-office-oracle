# Analysis Directory

Local scratchpad for EDA and modelling experiments. **Nothing here is tracked in git** — notebooks and datasets are all `.gitignore`d. Production code lives in `box_office/`.

## Datasets

```bash
make datasets
```

Pulls the raw box-office snapshot from Snowflake into `analysis/datasets_raw/`. Requires Snowflake credentials in `.env` (see project root `README.md`).

Downstream `datasets_medium / datasets_high / datasets_max` directories are produced by whatever notebooks you keep locally.
