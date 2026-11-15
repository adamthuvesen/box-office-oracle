# Keep the cloud training path, and share the fit, config, and eval split with the local path

Status: accepted (2026-11-12)

The shipped model comes from the local path (`scripts/train_local.py`, then `scripts/register_local_model.py`). The cloud path (Prefect flow, dbt, SageMaker training, Model Registry) also trains and registers models. We kept the cloud path because it is the part of the project that shows the full MLOps shape. Both paths now call `fit_final_model` in `box_office/ml/model.py`, read hyperparameters from `ModelHyperparameters` in `box_office/config.py`, and take their year windows from `box_office/eval_windows.py`. The cloud path no longer writes the nine `ML_TRAINING` and `FEATURE_STORE` tables, because nothing read them.

## Considered options

- Delete the cloud path and keep only local training. Rejected: the registry, the approval gate, and the Lambda loader all depend on the cloud registration flow, and the project would lose its infrastructure story.
- Keep both paths unchanged. Rejected: two fits, two configs, and two eval splits had already drifted apart (see ADR 0002).

## Consequences

- A change to the fit, the hyperparameters, or the windows lands in one place and reaches both paths.
- The cloud path costs money and needs a custom training image (ADR 0003). Every spend step refuses without the approval flag.
- The grants script still owns the old `ML_TRAINING` and `FEATURE_STORE` schemas, which the pipeline no longer writes.
