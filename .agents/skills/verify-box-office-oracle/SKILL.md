---
name: verify-box-office-oracle
description: "Drives box-office-oracle end to end (local training, the inference app over ASGI, the web data export, and the spend-refusal gates) on seeded synthetic fixtures in a temp copy. Use to run, launch, drive, or verify box-office-oracle, or to prove a change to it works."
---

# Verify box-office-oracle

`verify` copies the checkout to a temp dir, generates a seeded synthetic movie frame, and runs the real scripts and the real inference app against it. It never calls AWS, Snowflake, dbt, Terraform, Docker, or TMDB. No real movie data is read or committed.

## Launch / Reset

There is no long-lived process. Each run builds its own temp copy, temp HOME, and fixtures, so a run always starts at baseline and `reset` is a new run. Run from the repo root:

- `make verify ARGS=all` runs every drive (about 20 s).
- `make verify` runs the drives whose paths match files changed against `main`, else the smoke drive (`spend-refusal`).
- `make verify ARGS=serve` or `ARGS="serve web-data"` runs named drives.
- `make verify ARGS=list` shows each drive, what it proves, and the paths it covers.
- `make verify ARGS=control` plants one defect in a temp copy and passes only on its expected failure.

`make verify` syncs the locked environment first. Calling `.agents/skills/verify-box-office-oracle/verify` directly needs that environment on `python`.

## Doctor

`make verify ARGS=doctor` checks that the modules the drives need import, that `box_office` imports from the temp copy and not from the checkout, and that `aws`, `snow`, `terraform`, `dbt`, `docker`, and `curl` resolve to the tripwires on `PATH`. Every run repeats these checks before its first drive.

## Drive

Each drive runs `sys.executable` children in the temp copy with a temp HOME, dummy config values, a dead proxy, and the tripwires first on `PATH`. See `features/README.md` for the drive table.

A drive FAILs when any of these hold:

- a tripwire is hit, a child times out, or a child exits with an unexpected code;
- the drive is skipped, exits the process, or finishes without asserting anything;
- the run changes the checkout (tracked, untracked, or ignored paths, HEAD, the index, or `.git/config` and hooks).

The helper does not retry. A failed drive is FAIL, never `flaky`.

## Evidence

Each run writes `.verify/runs/<utc>-<sha7>[-dirty]/`: `manifest.txt` (one block per drive), `summary.json`, and one `<drive>.<n>-<step>.json` per child command (argv, exit, stdout, stderr; values of `KEY`, `TOKEN`, `SECRET`, `PASSWORD` variables redacted). The run prints `PASS|FAIL <drive> <seconds> <detail>` per drive and ends with the evidence path.

- **Control.** `make verify ARGS=control` trains normally, then negates `from_log_target` in the temp copy's `box_office/ml/target.py`, then re-drives `serve`. It passes only when `serve` fails with the `prediction-not-positive` token. A crash, a tripwire hit, a timeout, or a different token fails the control.

## Cleanup

The temp copy, HOME, and tripwires are removed on every exit path. `.verify/runs/` stays and is gitignored. Nothing is written to the checkout, `data/generated/`, `artifacts/`, or `web/data/`; the drives write those paths inside the temp copy only.

## Helper

`verify` is the dispatcher. `harness.py` holds the isolation, `drives.py` the drives, `fixtures.py` the seeded generator, `serve_probe.py` the ASGI client, and `web_snapshot.*` the Zod check.

- `verify` | `verify all` | `verify list` | `verify <drive>...` | `verify control` | `verify doctor`
- `verify exec -- COMMAND [ARGS...]` runs a command inside the isolation for diagnosis.
- Fixture seed: `SEED = 20261112` in `fixtures.py`. Other seeds derive from it (`SEED + 1`, `SEED + 2`).
- Python in this skill is its own type-check project (`pyrightconfig.json`) that `make check` runs.

## Limits

- The stand-in for `ModelLoader` in `serve_probe.py` replaces the S3 and SageMaker registry download. Everything above it is the real app. Loader logic, the tarball integrity check, and the registry stay with `tests/inference/`.
- `box-office-pipeline` runs as `python -m box_office.orchestration.flows.ml_pipeline` so the temp copy's code runs, not the console script installed from the checkout. It starts a temporary Prefect server on loopback, which is why `spend-refusal` takes about 8 s.
- Node is the version on `PATH`, not the pinned 24.22.3. `web/node_modules` is linked into the copy and must exist (`pnpm --dir web install --frozen-lockfile`).
- Model accuracy is not asserted. The fixture is small and synthetic.
