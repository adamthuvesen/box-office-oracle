# Local training

A seeded synthetic frame goes through the real training-frame script and the local trainer. The run writes the model artifacts and `metadata.json`, and registering without approval is refused.

Source: scripts/prepare_training_frame.py, scripts/train_local.py, scripts/register_local_model.py, box_office/eval_windows.py
Inventory-id: scripts/prepare_training_frame.py, scripts/train_local.py, scripts/register_local_model.py

## Sub-features

- `train-local` generates the fixture, builds the frame, trains, and checks the artifacts.
- `register-without-approval` runs `register_local_model.py` with neither `--approved` nor `--dry-run`.

## How to get to it (user POV)

- `uv run python scripts/prepare_training_frame.py`, then `uv run python scripts/train_local.py`.
- `uv run python scripts/register_local_model.py` (spends; refuses without `--approved`).

## Driving it with verify

Preconditions: `make verify ARGS=doctor` exits 0.

- **Train.** `make verify ARGS=train-local`. Artifacts `model.pkl`, `feature_preprocessor.pkl`, `feature_scaler.pkl`, `metadata.json`, and `cv_results.json` exist. `data_fingerprint.rows` and `training_rows` equal the fixture's kept rows. `eval_years` and the CV fold years come from `box_office/eval_windows.py`, and no confirmation year is scored. The schema version and selected features equal the code's constants.
- **Register.** The same drive runs `register_local_model.py` without `--approved`. It exits non-zero, prints the refusal token from `box_office/utils/approval.py`, and hits no tripwire.
- **Proof.** `train-local.*-fixture.json`, `-prepare-frame.json`, `-train.json`, `-register-without-approval.json`, and a manifest block.

## Gotchas

- The scripts read and write relative paths, so the drive runs them with the temp copy as cwd.
- 2026 rows and the other designed drops never reach the frame, which is why the fixture has more source rows than kept rows.
