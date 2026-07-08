"""Inference runtime hygiene: cache TTL bounds, artifact extraction limits, Lambda cleanup."""

from __future__ import annotations

import hashlib
import importlib
import tarfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# --------------------------------------------------------------------------- #
# 2.1 — MAX_LITERAL_EVAL_BYTES bound is enforced
# --------------------------------------------------------------------------- #


class TestLiteralEvalBound:
    def test_in_bounds_input_parses_normally(self):
        from box_office.ml.text_utils import process_text_list

        assert process_text_list("['Action', 'Comedy']") == ["action", "comedy"]

    def test_oversize_input_raises_typed_error(self):
        from box_office.ml.text_utils import (
            MAX_LITERAL_EVAL_BYTES,
            LiteralEvalTooLarge,
            process_text_list,
        )

        oversize = "[" + ", ".join('"a"' for _ in range(MAX_LITERAL_EVAL_BYTES)) + "]"
        with pytest.raises(LiteralEvalTooLarge):
            process_text_list(oversize)


# --------------------------------------------------------------------------- #
# 2.4 — Mangum adapter is built once at module import time
# --------------------------------------------------------------------------- #


class TestMangumHoisting:
    def test_mangum_constructor_called_once_per_import(self):
        """The Mangum adapter must be built at module-scope, not per-call.

        Mangum isn't a dev dependency (only required in the Lambda runtime),
        so we inject a fake ``mangum`` module into ``sys.modules`` before
        importing the FastAPI app and assert the constructor fires once.

        IMPORTANT: this test reloads ``box_office.inference.app.main`` under
        a fake ``mangum`` shim. Other test modules (e.g. ``test_api_contract``)
        import ``app`` at module-load time and hold a reference to the
        ORIGINAL ``app`` object whose route handlers close over the ORIGINAL
        module namespace. If we leave the test-time fresh module installed
        in ``sys.modules``, subsequent ``patch("box_office.inference.app.main.X")``
        calls patch the wrong namespace and the cross-test fixtures break.
        We restore the original module on exit.
        """
        import sys
        import types

        fake_mangum = types.ModuleType("mangum")
        MockMangum = MagicMock(return_value=MagicMock())
        fake_mangum.Mangum = MockMangum

        # Snapshot the ORIGINAL main module (and any cached mangum) so we can
        # put them back exactly as they were after the test runs.
        original_main = sys.modules.get("box_office.inference.app.main")
        original_mangum = sys.modules.get("mangum")

        # Drop cached state so the module-scope code runs under our patch.
        for k in list(sys.modules):
            if k.startswith("box_office.inference.app.main") or k == "mangum":
                del sys.modules[k]
        sys.modules["mangum"] = fake_mangum
        try:
            importlib.import_module("box_office.inference.app.main")
            # Importing again must NOT re-run module-scope code.
            importlib.import_module("box_office.inference.app.main")
            assert MockMangum.call_count == 1
        finally:
            # Pop everything we touched, then restore originals (or leave
            # absent if they never existed).
            sys.modules.pop("mangum", None)
            sys.modules.pop("box_office.inference.app.main", None)
            if original_main is not None:
                sys.modules["box_office.inference.app.main"] = original_main
            if original_mangum is not None:
                sys.modules["mangum"] = original_mangum


# --------------------------------------------------------------------------- #
# 2.6 — single-slot /tmp/models cleanup
# --------------------------------------------------------------------------- #


def _make_tarball(dest: Path, *, with_files: dict[str, bytes]) -> Path:
    """Build a `model.tar.gz` containing the given members."""
    src = dest.parent / "_src"
    src.mkdir(parents=True, exist_ok=True)
    for name, payload in with_files.items():
        (src / name).write_bytes(payload)
    with tarfile.open(dest, "w:gz") as tar:
        for name in with_files:
            tar.add(src / name, arcname=name)
    return dest


def _build_loader(cache_dir: Path):
    from box_office.inference.app.model_loader import ModelLoader

    loader = ModelLoader.__new__(ModelLoader)
    loader.cache_dir = cache_dir
    loader._extracted_artifacts_cache = {}
    loader.cache_dir.mkdir(parents=True, exist_ok=True)
    return loader


class TestSingleSlotCacheCleanup:
    def _minimal_pickle_bytes(self) -> bytes:
        import pickle

        return pickle.dumps({"weights": [1, 2, 3]})

    def _make_payload(self) -> dict[str, bytes]:
        # Minimal artifact set the loader expects.
        m = self._minimal_pickle_bytes()
        return {
            "model.pkl": m,
            "feature_preprocessor.pkl": m,
            "feature_scaler.pkl": m,
        }

    def test_new_arn_load_reaps_sibling_slot(self, tmp_path):
        loader = _build_loader(tmp_path / "cache")

        # Pre-existing sibling slot with an unrelated SHA.
        prev_sha = "0" * 64
        prev_dir = loader.cache_dir / prev_sha
        prev_dir.mkdir(parents=True)
        (prev_dir / "model.pkl").write_bytes(b"old")

        # New extract, real SHA.
        tar_dir = tmp_path / "tars"
        tar_dir.mkdir()
        tar_path = _make_tarball(
            tar_dir / "new.tar.gz", with_files=self._make_payload()
        )
        new_sha = "1" * 64

        loader._extract_and_load_model_with_cache(
            str(tar_path),
            model_package_arn="arn:aws:sagemaker:eu-north-1:123:model-package/foo/1",
            expected_sha256=new_sha,
        )

        assert (loader.cache_dir / new_sha / "model.pkl").exists()
        assert not prev_dir.exists(), "sibling slot should be reaped"

    def test_failed_mid_extract_preserves_sibling_slot(self, tmp_path):
        loader = _build_loader(tmp_path / "cache")

        prev_sha = "0" * 64
        prev_dir = loader.cache_dir / prev_sha
        prev_dir.mkdir(parents=True)
        (prev_dir / "model.pkl").write_bytes(b"old")

        new_sha = "1" * 64
        # Point at a non-existent tar so tarfile.open raises before any
        # rename happens. The sibling slot must still be intact.
        from box_office.inference.app.model_loader import ModelLoadError

        with pytest.raises(ModelLoadError):
            loader._extract_and_load_model_with_cache(
                "/tmp/does-not-exist.tar.gz",
                model_package_arn="arn:aws:sagemaker:eu-north-1:123:model-package/foo/2",
                expected_sha256=new_sha,
            )

        assert prev_dir.exists(), "sibling slot must NOT be reaped on failure"
        assert (prev_dir / "model.pkl").exists()
        # No final slot for the failed sha.
        assert not (loader.cache_dir / new_sha).exists()

    def test_extracted_cache_verifies_model_file_hash_not_tarball_hash(self, tmp_path):
        loader = _build_loader(tmp_path / "cache")
        arn = "arn:aws:sagemaker:eu-north-1:123:model-package/foo/3"
        model_file = loader.cache_dir / "model.pkl"
        model_bytes = self._minimal_pickle_bytes()
        model_file.write_bytes(model_bytes)
        model_sha = hashlib.sha256(model_bytes).hexdigest()

        loader._extracted_artifacts_cache[arn] = {
            "model": str(model_file),
            "preprocessor": str(model_file),
            "scaler": str(model_file),
            "extract_dir": str(loader.cache_dir),
            "expected_sha256": "0" * 64,
            "model_sha256": model_sha,
        }

        loaded = loader._download_and_load_model({"ModelPackageArn": arn})

        assert loaded == {"weights": [1, 2, 3]}


# --------------------------------------------------------------------------- #
# 2.8 — genres with apostrophes parse correctly
# --------------------------------------------------------------------------- #


class TestGenresWithApostrophes:
    def test_apostrophe_genre_normalizes_to_list(self):
        from box_office.ml.text_utils import process_text_list

        assert process_text_list(["Children's", "Comedy"]) == ["children's", "comedy"]
