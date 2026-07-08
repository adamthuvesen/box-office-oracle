# Data Directory

This repo does not commit datasets. CSV and Parquet files under `data/` are
gitignored because the pipeline reads private Snowflake data and writes generated
local outputs.

Use the maintained ingestion CLI for local TMDB discovery and Snowflake loads:

```bash
box-office-ingest --discover-only --start-year 2024 --output data/generated/tmdb/movies_2024.csv
box-office-ingest --start-year 2024 --load-to-snowflake
```

Model training data is built through dbt and the orchestration tasks, then stored
in Snowflake `ML_TRAINING` and uploaded to SageMaker. The v8 feature contract is
pre-release only; generated data files must not include post-release feature
inputs such as vote counts, ratings, popularity scores, ranks, or domestic gross.
