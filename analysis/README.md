# Analysis Directory

Local scratchpad for EDA and modelling experiments. Notebooks and data are
gitignored; production code lives in `box_office/`.

## Contents

- `*.ipynb` — local exploration/training notebooks (not tracked).
- `prior_training_snapshot/` — frozen training data from the previous
  Snowflake-era model. Read by `scripts/evaluate_local_retrain.py` as the
  old-model comparison baseline. Not regenerable — don't delete.

To refresh the local TMDB dataset for notebook work:

```bash
uv run box-office-rich-backfill
```
