# Analysis Directory

Local scratchpad for EDA and modelling experiments. Notebooks and datasets are
gitignored; production code lives in `box_office/`.

## Datasets

```bash
make datasets
```

Pulls the raw box-office snapshot from Snowflake into `analysis/datasets_raw/`. Requires Snowflake credentials in `.env` (see project root `README.md`).

Downstream `datasets_medium / datasets_high / datasets_max` directories are produced by whatever notebooks you keep locally.
