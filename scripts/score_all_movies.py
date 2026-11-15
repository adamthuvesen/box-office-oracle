"""Predictions for every movie in the 1980-2026 dataset, for the web app.

Two sources, clearly labeled:

- Eval-year rows (2015-2023, the v9 iteration window): the fold-clean
  out-of-fold prediction from the leakage-free CV that
  ``scripts/train_local.py`` ran (each movie predicted by a model that never
  saw its release year). ``prediction_kind = "out_of_sample"``.
- Everything else — pre-2015 training rows, 2024-2025 rows (the spent
  confirmation years, honestly labeled: the final all-data model trained on
  them), the quality-dropped rows, and future/2026 rows — scored with the
  final all-data model.
  ``prediction_kind = "in_sample"`` when a real gross exists, ``"no_actuals"``
  when it does not (future releases; the $100M+ gross artifacts).

Requires ``scripts/prepare_training_frame.py`` and ``scripts/train_local.py``
to have run first (train_local must postdate the leakage fix so its
``cv_results.json`` OOF records are fold-clean).

Outputs:

- ``data/generated/training/predictions_all_1980_2026.parquet``
- ``web/data/predictions.json`` keyed by tmdb_id (gitignored, like the rest
  of ``web/data/``)

Run:  uv run python scripts/score_all_movies.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from box_office.inference.app.predictor import ModelInfo, PredictionEngine
from box_office.ml.artifacts import (
    FEATURE_PREPROCESSOR_PKL,
    FEATURE_SCALER_PKL,
    MODEL_PKL,
)
from box_office.ml.metrics_models import CrossValidationResults
from box_office.ml.target import from_log_target
from box_office.training_frame import PREPROCESSOR_INPUT_COLUMNS

FRAME_PATH = Path("data/generated/training/train_frame_1980_2026.parquet")
DROPPED_PATH = Path("data/generated/training/dropped_rows_1980_2026.csv")
ARTIFACT_DIR = Path("artifacts/local")
PARQUET_OUT = Path("data/generated/training/predictions_all_1980_2026.parquet")
WEB_JSON_OUT = Path("web/data/predictions.json")

# Drop reasons whose recorded gross is not a real theatrical actual: future
# releases (placeholder gross) and the documented $100M+ gross artifacts.
UNRELIABLE_GROSS_REASONS = (
    "no_reliable_worldwide_gross",
    "gross_not_final_future_year",
    "gross_over_100m_with_no_documented_budget",
)

OUT_OF_SAMPLE = "out_of_sample"
IN_SAMPLE = "in_sample"
NO_ACTUALS = "no_actuals"


def load_artifacts() -> tuple[PredictionEngine, CrossValidationResults]:
    paths = {
        "model": ARTIFACT_DIR / MODEL_PKL,
        "preprocessor": ARTIFACT_DIR / FEATURE_PREPROCESSOR_PKL,
        "scaler": ARTIFACT_DIR / FEATURE_SCALER_PKL,
        "cv_results": ARTIFACT_DIR / "cv_results.json",
    }
    missing = [str(p) for p in paths.values() if not p.exists()]
    if missing:
        raise SystemExit(
            f"missing artifacts: {missing}. Run `uv run python scripts/train_local.py` first."
        )
    engine = PredictionEngine()
    engine.load_model_artifacts(
        model_path=str(paths["model"]),
        preprocessor_path=str(paths["preprocessor"]),
        scaler_path=str(paths["scaler"]),
        model_info=ModelInfo(model_id="local-tmdb-1980-2026", version=1),
    )
    cv_results = CrossValidationResults.model_validate_json(paths["cv_results"].read_text())
    return engine, cv_results


def final_model_predictions(rows: pd.DataFrame, engine: PredictionEngine) -> np.ndarray:
    """Final-model predictions in USD through the serving engine."""
    pred_dollars = engine.predict_frame(rows[list(PREPROCESSOR_INPUT_COLUMNS)])
    if not np.all(np.isfinite(pred_dollars)):
        bad = int((~np.isfinite(pred_dollars)).sum())
        raise AssertionError(f"{bad} non-finite final-model predictions")
    return pred_dollars


def oof_dollars_by_position(cv_results: CrossValidationResults, n_rows: int) -> dict[int, float]:
    """OOF log-predictions keyed by positional index in the training frame."""
    positions: dict[int, float] = {}
    for record in cv_results.oof_records:
        idx = record.idx
        if not 0 <= idx < n_rows:
            raise AssertionError(
                f"OOF idx {idx} outside the frame (n={n_rows}); cv_results.json "
                "does not match the current training frame — rerun train_local."
            )
        positions[idx] = float(from_log_target(record.pred))
    return positions


def ape_or_none(pred: float, actual: float | None) -> float | None:
    if actual is None:
        return None
    return abs(pred - actual) / max(actual, 1.0)


def build_records(
    frame: pd.DataFrame,
    dropped: pd.DataFrame,
    oof_dollars: dict[int, float],
    engine: PredictionEngine,
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []

    frame_pred_final = final_model_predictions(frame, engine)
    for position, row in enumerate(frame.to_dict(orient="records")):
        oof = oof_dollars.get(position)
        predicted = oof if oof is not None else float(frame_pred_final[position])
        actual = float(row["WORLDWIDE_GROSS"])
        records.append(
            {
                "tmdb_id": int(row["TMDB_ID"]),
                "title": row["TITLE"],
                "release_year": int(row["RELEASE_YEAR"]),
                "predicted_gross": predicted,
                "prediction_kind": OUT_OF_SAMPLE if oof is not None else IN_SAMPLE,
                "actual_gross": actual,
                "ape": ape_or_none(predicted, actual),
            }
        )

    dropped_pred = final_model_predictions(dropped, engine)
    for position, row in enumerate(dropped.to_dict(orient="records")):
        gross_unreliable = any(reason in row["DROP_REASON"] for reason in UNRELIABLE_GROSS_REASONS)
        actual = None if gross_unreliable else float(row["WORLDWIDE_GROSS"])
        predicted = float(dropped_pred[position])
        records.append(
            {
                "tmdb_id": int(row["TMDB_ID"]),
                "title": row["TITLE"],
                "release_year": int(row["RELEASE_YEAR"]),
                "predicted_gross": predicted,
                "prediction_kind": NO_ACTUALS if actual is None else IN_SAMPLE,
                "actual_gross": actual,
                "ape": ape_or_none(predicted, actual),
            }
        )

    result = pd.DataFrame(records)
    duplicated = result["tmdb_id"].duplicated(keep=False)
    if duplicated.any():
        raise AssertionError(
            f"duplicate tmdb_ids would collide in the keyed JSON: "
            f"{sorted(result.loc[duplicated, 'tmdb_id'].unique().tolist())}"
        )
    return result


def write_web_json(result: pd.DataFrame) -> None:
    payload = {
        str(int(row["tmdb_id"])): {
            "predicted_gross": float(row["predicted_gross"]),
            "prediction_kind": row["prediction_kind"],
            "actual_gross": (None if pd.isna(row["actual_gross"]) else float(row["actual_gross"])),
            "ape": None if pd.isna(row["ape"]) else float(row["ape"]),
        }
        for row in result.to_dict(orient="records")
    }
    WEB_JSON_OUT.parent.mkdir(parents=True, exist_ok=True)
    # allow_nan=False: bare NaN would silently break JSON.parse in the web app.
    WEB_JSON_OUT.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    )


def main() -> None:
    for path in (FRAME_PATH, DROPPED_PATH):
        if not path.exists():
            raise SystemExit(f"missing input: {path}. Run scripts/prepare_training_frame.py first.")

    frame = pd.read_parquet(FRAME_PATH)
    dropped = pd.read_csv(DROPPED_PATH)
    engine, cv_results = load_artifacts()
    oof_dollars = oof_dollars_by_position(cv_results, len(frame))

    result = build_records(frame, dropped, oof_dollars, engine)

    PARQUET_OUT.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(PARQUET_OUT, index=False)
    write_web_json(result)

    counts = result["prediction_kind"].value_counts().to_dict()
    print(f"scored {len(result)} movies -> {PARQUET_OUT} and {WEB_JSON_OUT}")
    for kind, count in counts.items():
        print(f"  {kind}: {count}")
    print(
        f"OOF coverage: {len(oof_dollars)} eval-year rows from cv_results.json",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
