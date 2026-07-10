# Analysis Directory

Local scratchpad for EDA and modelling experiments. Notebooks and data are
gitignored; production code lives in `box_office/`.

## Contents

- `*.ipynb` — local exploration/training notebooks (not tracked).
- `prior_training_snapshot/` — frozen training data from the previous
  Snowflake-era model. Read by `scripts/evaluate_local_retrain.py` as the
  old-model comparison baseline. Not regenerable — don't delete.

To pull a fresh raw snapshot from Snowflake for notebook work (requires
Snowflake credentials in `.env`, see project root `README.md`):

```bash
uv run box-office-ingest --output analysis/box_office_raw.parquet
```
