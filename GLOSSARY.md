# Box Office Oracle

Predicts worldwide box-office revenue for movies before release. This file fixes the names the code, the Snowflake tables, the Lambda API, and the web app share.

## Language

**Movie**:
One film in the dataset, keyed by `tmdb_id`. A row of the cleaned 1980-2026 parquet and of `STAGING.STG_BOX_OFFICE`.
_Avoid_: Film, title

**Release**:
The theatrical release of a movie, dated `RELEASE_DATE` (`RELEASE_YEAR` is the year). A movie has one release for the model. Time order of releases decides every train and test split.
_Avoid_: Premiere, launch

**Target**:
Worldwide gross in USD on the log scale, `log1p(gross)`. `to_log_target` and `from_log_target` in `box_office/ml/target.py` are the only converters.
_Avoid_: Label, revenue (as a model column)

**Prediction**:
The model's worldwide gross in USD for one movie, after `from_log_target`. The API returns it as a `PredictionResponse`.
_Avoid_: Forecast, estimate

## Model

**Feature contract**:
The 13 pre-release inputs the model reads, listed in `SELECTED_FEATURES` and versioned by `CURRENT_FEATURE_SCHEMA_VERSION` (now `"9"`, "v9"). The runtime refuses a model with another version. A feature change bumps the version and needs a retrain.
_Avoid_: Schema, feature set

**Model package**:
One registered model in SageMaker Model Registry: the artifact tarball, its SHA256 manifest, and the metadata (feature contract version, data fingerprint). Lambda loads the latest approved package.
_Avoid_: Model artifact, build

**Model version**:
The integer the registry assigns to a model package. `model_version` in the API response reports it. It is not the feature contract version.

**Data fingerprint**:
Row count, latest release date, and a content hash of the training frame (`DataFingerprint`). Stored in the model package metadata and returned by `/model/info`.
_Avoid_: Data version, snapshot

## Evaluation

**Iteration window**:
Eval years 2015-2023 (`START_EVAL_YEAR` to `ITERATION_END_YEAR` in `box_office/eval_windows.py`). All tuning, model selection, and CV score only these years, on the local and the cloud path.
_Avoid_: Dev set, validation set

**Confirmation set**:
Release years 2024-2025 (`CONFIRMATION_YEARS`). It is spent: no tuning or CV scores it, and the cloud split keeps it out of training. Only a deliberate holdout-burning run may touch it.
_Avoid_: Test set, holdout (the code uses "holdout" only for the burning flag)

## Features

**IP tier**:
`IP_TIER`, an ordinal 1-5 for how pre-sold the property was on the release date. 5 means no pre-sold IP and 4 means a nominal one. Built only from facts known before release (`box_office/ip_classification.py`).
_Avoid_: IP score, franchise score

**Collection**:
A TMDB collection: the group of movies TMDB links by `collection_id`. The code uses it as the franchise. `PRIOR_FRANCHISE_GROSS_LOG` and `IS_FRANCHISE_FOLLOWUP` count only strictly earlier releases in the collection.
_Avoid_: Series, franchise (use "franchise" only in feature names)

**Genre vocabulary**:
The 15 genres in `GENRE_VOCABULARY` (`box_office/ml/feature_pipeline/constants.py`). The web once listed 17 (it added History and Music); the web now reads the genres from the generated predict contract.

**Super-genre**:
A named combination of genres, such as `SciFi_Blockbuster`, from `_SUPER_GENRE_RULES`. The model reads it as `SUPER_GENRE_ENCODED`.
_Avoid_: Genre group

## Name drift (code not renamed)

- Gross: `worldwide_gross` (lowercase) and `WORLDWIDE_GROSS` (training frame) name the same quantity. `gross_usd` is the same quantity as a parameter in `target.py`.
- Version: `feature_schema_version` and `FeatureSchemaVersionMismatch` mean the feature contract version. `model_version` means the registry version.
