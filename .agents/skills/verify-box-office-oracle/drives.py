"""The four drives. Each returns a one-line detail on success and raises DriveFailure otherwise."""

import json
import shutil
import sys
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from harness import ROOT, SKILL_DIR, DriveFailure, Session

from box_office.eval_windows import CONFIRMATION_YEARS, ITERATION_END_YEAR, START_EVAL_YEAR
from box_office.ml.artifacts import FEATURE_PREPROCESSOR_PKL, FEATURE_SCALER_PKL, MODEL_PKL
from box_office.ml.feature_pipeline.constants import SELECTED_FEATURES
from box_office.ml.feature_schema import CURRENT_FEATURE_SCHEMA_VERSION
from box_office.utils.approval import REFUSAL_TOKEN

PYTHON = sys.executable
ARTIFACT_DIR = Path("artifacts/local")
API_KEY = "verify-api-key"
WEB_DIR = Path("web")
CONTRACT_PATH = WEB_DIR / "src/generated/predict-contract.json"

# Failure tokens the control matches on.
FIXTURE_ROWS = "fixture-row-count-mismatch"
ARTIFACT_MISSING = "artifact-missing"
EVAL_WINDOW = "cv-window-mismatch"
SCHEMA_VERSION = "feature-schema-version-mismatch"
REFUSAL_MISSING = "refusal-missing"
PREDICTION_NOT_POSITIVE = "prediction-not-positive"
AUTH_NOT_ENFORCED = "api-key-not-enforced"
BOUNDS_NOT_ENFORCED = "contract-bounds-not-enforced"
FINGERPRINT_MISSING = "model-info-fingerprint-mismatch"
CONTRACT_STALE = "web-contract-stale"
WEB_SNAPSHOT_INVALID = "web-snapshot-invalid"


@dataclass(frozen=True)
class Drive:
    name: str
    proves: str
    paths: tuple[str, ...]
    run: Callable[[Session], str]


def _load_json(path: Path) -> dict[str, Any]:
    return cast("dict[str, Any]", json.loads(path.read_text()))


def _train(s: Session) -> dict[str, Any]:
    """Fixture, training frame, local training, and the artifact checks. Runs once per session."""
    if "train" in s.memo:
        return cast("dict[str, Any]", s.memo["train"])
    fixture = json.loads(
        s.run(
            "train-local",
            "fixture",
            [PYTHON, str(s.copy / SKILL_DIR.relative_to(ROOT) / "fixtures.py")],
        ).stdout
    )
    s.run("train-local", "prepare-frame", [PYTHON, "scripts/prepare_training_frame.py"])
    s.run("train-local", "train", [PYTHON, "scripts/train_local.py"])

    art = s.copy / ARTIFACT_DIR
    for name in (
        MODEL_PKL,
        FEATURE_PREPROCESSOR_PKL,
        FEATURE_SCALER_PKL,
        "metadata.json",
        "cv_results.json",
    ):
        s.check((art / name).is_file(), ARTIFACT_MISSING, f"{art / name} was not written")
    metadata = _load_json(art / "metadata.json")
    cv_results = _load_json(art / "cv_results.json")

    rows = metadata["data_fingerprint"]["rows"]
    s.check(
        rows == fixture["kept_rows"],
        FIXTURE_ROWS,
        f"fingerprint rows {rows} != fixture kept rows {fixture['kept_rows']}",
    )
    s.check(
        metadata["training_rows"] == fixture["kept_rows"],
        FIXTURE_ROWS,
        "training_rows != fixture kept rows",
    )
    s.check(
        metadata["feature_schema_version"] == CURRENT_FEATURE_SCHEMA_VERSION,
        SCHEMA_VERSION,
        f"artifact has {metadata['feature_schema_version']!r}",
    )
    s.check(
        metadata["selected_features"] == list(SELECTED_FEATURES),
        SCHEMA_VERSION,
        "selected_features drifted",
    )
    expected_years = list(range(START_EVAL_YEAR, ITERATION_END_YEAR + 1))
    fold_years = [fold["eval_year"] for fold in cv_results["fold_results"]]
    s.check(
        fold_years == expected_years,
        EVAL_WINDOW,
        f"CV folds scored {fold_years}, expected {expected_years}",
    )
    s.check(
        not set(fold_years) & set(CONFIRMATION_YEARS), EVAL_WINDOW, "CV scored a confirmation year"
    )
    s.check(
        metadata["eval_years"] == f"{START_EVAL_YEAR}-{ITERATION_END_YEAR}",
        EVAL_WINDOW,
        f"metadata eval_years {metadata['eval_years']!r}",
    )
    result = {"fixture": fixture, "metadata": metadata, "cv_results": cv_results}
    s.memo["train"] = result
    return result


def _assert_refusal(s: Session, drive: str, step: str, argv: list[str]) -> str:
    done = s.run(drive, step, argv, expect_exit=None)
    output = done.stdout + done.stderr
    s.check(done.returncode != 0, REFUSAL_MISSING, f"{step} exited 0 without --approved")
    s.check(
        REFUSAL_TOKEN in output,
        REFUSAL_MISSING,
        f"{step} did not print the refusal token: {output[-300:]}",
    )
    return step


def train_local(s: Session) -> str:
    result = _train(s)
    _assert_refusal(
        s, "train-local", "register-without-approval", [PYTHON, "scripts/register_local_model.py"]
    )
    return f"{result['fixture']['kept_rows']} rows, CV {START_EVAL_YEAR}-{ITERATION_END_YEAR}, register refused"


def serve(s: Session) -> str:
    result = _train(s)
    probe = s.copy / SKILL_DIR.relative_to(ROOT) / "serve_probe.py"
    answers_path = s.scratch / "serve-answers.json"
    s.run(
        "serve",
        "probe",
        [PYTHON, str(probe), str(s.copy / ARTIFACT_DIR), API_KEY, str(answers_path)],
        extra_env={"API_KEY": API_KEY},
    )
    s.check(answers_path.is_file(), ARTIFACT_MISSING, "the probe wrote no answers")
    answers = _load_json(answers_path)
    predict = answers["predict"]
    s.check(
        predict["status"] == 200,
        PREDICTION_NOT_POSITIVE,
        f"predict answered {predict['status']}: {predict['body']}",
    )
    value = predict["body"].get("prediction")
    s.check(
        isinstance(value, int | float) and value > 0,
        PREDICTION_NOT_POSITIVE,
        f"prediction was {value!r}",
    )
    for key in ("missing_key", "wrong_key"):
        s.check(
            answers[key]["status"] == 401,
            AUTH_NOT_ENFORCED,
            f"{key} answered {answers[key]['status']}",
        )
    for key in ("runtime_out_of_bounds", "month_out_of_bounds"):
        s.check(
            answers[key]["status"] == 400,
            BOUNDS_NOT_ENFORCED,
            f"{key} answered {answers[key]['status']}",
        )
    info = answers["model_info"]
    expected = result["metadata"]["data_fingerprint"]
    s.check(
        info["status"] == 200 and info["body"].get("data_fingerprint") == expected,
        FINGERPRINT_MISSING,
        f"/model/info reported {info['body'].get('data_fingerprint')!r}, expected {expected!r}",
    )
    return f"predicted ${value:,.0f}; 401 x2, 400 x2, fingerprint {expected['rows']} rows"


def web_data(s: Session) -> str:
    result = _train(s)
    web = s.copy / WEB_DIR
    contract_before = (s.copy / CONTRACT_PATH).read_bytes()
    s.run("web-data", "generate-contract", [PYTHON, "scripts/generate_web_contract.py"])
    s.check(
        (s.copy / CONTRACT_PATH).read_bytes() == contract_before,
        CONTRACT_STALE,
        f"{CONTRACT_PATH} differs from the Python owners",
    )

    s.run("web-data", "score-all", [PYTHON, "scripts/score_all_movies.py"])
    per_year = s.scratch / "per_year_table.json"
    per_year.write_text(json.dumps(_per_year_rows(result["cv_results"])))
    s.run(
        "web-data",
        "export",
        [PYTHON, "scripts/export_web_data.py", "--per-year-table", str(per_year)],
    )

    for name in ("movies.json", "predictions.json", "model_meta.json"):
        s.check(
            (web / "data" / name).is_file(), ARTIFACT_MISSING, f"web/data/{name} was not written"
        )
    skill_in_copy = s.copy / SKILL_DIR.relative_to(ROOT)
    shutil.copy2(skill_in_copy / "web_snapshot.test.ts", web / "tests/verify-snapshot.test.ts")
    shutil.copy2(skill_in_copy / "web_snapshot.vitest.config.mts", web / "verify.vitest.config.mts")
    (web / "node_modules").symlink_to(ROOT / "web/node_modules")
    vitest = s.run(
        "web-data",
        "validate-snapshot",
        ["node", "node_modules/vitest/vitest.mjs", "run", "--config", "verify.vitest.config.mts"],
        cwd=web,
        extra_env={
            "VERIFY_WEB_ROOT": str(web),
            "VERIFY_EXPECT_ROWS": str(result["fixture"]["source_rows"]),
            "VERIFY_VITE_CACHE": str(s.scratch / "vite-cache"),
        },
        expect_exit=None,
    )
    s.check(vitest.returncode == 0, WEB_SNAPSHOT_INVALID, (vitest.stdout + vitest.stderr)[-800:])
    return f"{result['fixture']['source_rows']} movies and predictions validate against the Zod schemas"


def _per_year_rows(cv_results: dict[str, Any]) -> list[dict[str, Any]]:
    """The backtest table the web schema expects, from this run's own CV folds (no baseline)."""
    return [
        {
            "year": fold["eval_year"],
            "n_movies": fold["val_samples"],
            "baseline_r2_log": 0.0,
            "model_r2_log": fold["model_r2_log"],
            "gain_r2_log": fold["model_r2_log"],
            "baseline_spearman": 0.0,
            "model_spearman": fold["model_spearman"],
            "baseline_r2": 0.0,
            "model_r2": fold["model_r2_dollars"],
            "gain_r2": fold["model_r2_dollars"],
            "model_rmsle": fold["rmsle_score"],
            "model_median_ape": fold["model_median_ape"],
        }
        for fold in cv_results["fold_results"]
    ]


SPEND_ENTRY_POINTS: dict[str, list[str]] = {
    "box-office-pipeline": [PYTHON, "-m", "box_office.orchestration.flows.ml_pipeline"],
    "load_dataset_to_snowflake": [PYTHON, "scripts/load_dataset_to_snowflake.py"],
    "apply_snowflake_grants": [PYTHON, "scripts/apply_snowflake_grants.py"],
    "register_local_model": [PYTHON, "scripts/register_local_model.py"],
    "training-image": ["make", "training-image"],
}


def spend_refusal(s: Session) -> str:
    with ThreadPoolExecutor(max_workers=len(SPEND_ENTRY_POINTS)) as pool:
        futures = [
            pool.submit(_assert_refusal, s, "spend-refusal", name, argv)
            for name, argv in SPEND_ENTRY_POINTS.items()
        ]
        failures = [f.exception() for f in futures if f.exception() is not None]
    if failures:
        first = failures[0]
        raise first if isinstance(first, DriveFailure) else DriveFailure("crash", repr(first))
    return f"{len(SPEND_ENTRY_POINTS)} entry points refused, no tripwire hit"


DRIVES: dict[str, Drive] = {
    drive.name: drive
    for drive in (
        Drive(
            "train-local",
            "synthetic frame -> prepare -> train_local; artifacts, fingerprint rows, CV windows; register refuses",
            (
                "box_office/ml/",
                "box_office/eval_windows.py",
                "box_office/training_frame.py",
                "scripts/train_local.py",
                "scripts/prepare_training_frame.py",
                "scripts/register_local_model.py",
            ),
            train_local,
        ),
        Drive(
            "serve",
            "trained artifacts behind the inference app over ASGI: predict, API key, bounds, /model/info",
            (
                "box_office/inference/",
                "box_office/ml/target.py",
                "box_office/ml/data_fingerprint.py",
            ),
            serve,
        ),
        Drive(
            "web-data",
            "score_all_movies + export_web_data -> web/data validates against the Zod schemas; contract current",
            (
                "scripts/score_all_movies.py",
                "scripts/export_web_data.py",
                "scripts/generate_web_contract.py",
                "web/",
            ),
            web_data,
        ),
        Drive(
            "spend-refusal",
            "each spending entry point refuses without --approved and touches no external tool",
            (
                "box_office/utils/approval.py",
                "box_office/orchestration/",
                "scripts/load_dataset_to_snowflake.py",
                "scripts/apply_snowflake_grants.py",
                "scripts/register_local_model.py",
                "scripts/build_training_image.sh",
                "Makefile",
            ),
            spend_refusal,
        ),
    )
}
