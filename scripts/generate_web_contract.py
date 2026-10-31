"""Write the web app's copy of the prediction contract from the Python owners.

The web form and the /api/predict proxy read their bounds, defaults, genre
choices, and IP tiers from ``web/src/generated/predict-contract.json``. This
script writes that file from ``PredictionRequest``, ``PredictionResponse``, and
``GENRE_VOCABULARY``; ``make fix`` runs it, and a ``make check`` test fails when
the committed file drifts.

Run:  uv run python scripts/generate_web_contract.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from box_office.inference.app.predictor import PredictionRequest, PredictionResponse
from box_office.ml.feature_pipeline.constants import GENRE_VOCABULARY

CONTRACT_PATH = Path("web/src/generated/predict-contract.json")

# Form labels for the ordinal ip_tier values (1 = strongest pre-sold IP).
IP_TIER_LABELS: dict[int, str] = {
    5: "Original / no IP",
    4: "Nominal IP (adaptation, minor property)",
    3: "Known source work",
    2: "Strong franchise / brand",
    1: "Top-tier pre-sold IP",
}


def genre_label(token: str) -> str:
    """``science_fiction`` -> ``Science Fiction``; the genre transformer maps it back."""
    return token.replace("_", " ").title()


def build_web_contract() -> dict[str, Any]:
    tier_schema = PredictionRequest.model_json_schema()["properties"]["ip_tier"]
    tier_values = set(range(tier_schema["minimum"], tier_schema["maximum"] + 1))
    if set(IP_TIER_LABELS) != tier_values:
        raise ValueError(f"IP_TIER_LABELS must label exactly the ip_tier values {tier_values}")
    return {
        "predict_request": PredictionRequest.model_json_schema(),
        "predict_response": PredictionResponse.model_json_schema(mode="serialization"),
        "genres": sorted(genre_label(token) for token in GENRE_VOCABULARY),
        "ip_tiers": [{"value": value, "label": label} for value, label in IP_TIER_LABELS.items()],
    }


def render_web_contract() -> str:
    return json.dumps(build_web_contract(), indent=2, sort_keys=True) + "\n"


def main() -> None:
    CONTRACT_PATH.parent.mkdir(parents=True, exist_ok=True)
    CONTRACT_PATH.write_text(render_web_contract())
    print(f"wrote {CONTRACT_PATH}")


if __name__ == "__main__":
    main()
