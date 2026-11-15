"""The web app's contract file matches the Python owners. Fix drift with `make fix`."""

from pathlib import Path

from scripts import generate_web_contract

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_committed_web_contract_is_current():
    committed = (REPO_ROOT / generate_web_contract.CONTRACT_PATH).read_text()
    assert committed == generate_web_contract.render_web_contract(), (
        "web/src/generated/predict-contract.json is stale; run `make fix`."
    )


def test_genre_labels_map_back_to_the_model_vocabulary():
    from box_office.ml.feature_pipeline.constants import GENRE_VOCABULARY
    from box_office.ml.feature_pipeline.transformers.genre import split_genre_field

    labels = generate_web_contract.build_web_contract()["genres"]
    assert sorted(split_genre_field(labels)) == sorted(GENRE_VOCABULARY)
