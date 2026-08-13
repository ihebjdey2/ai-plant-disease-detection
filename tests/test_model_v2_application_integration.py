from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from app.services import prediction_service
from app.taxonomy import CLASS_NAMES
from config import Config, DEFAULT_MODEL_PATH


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FROZEN_SELECTION_PATH = (
    PROJECT_ROOT / "training/config/model-v2-final-selection.json"
)
EXPECTED_MODEL_PATH = PROJECT_ROOT / "models/agri-diagnose-v2-exp-a.keras"
EXPECTED_MODEL_SHA256 = (
    "bba4d044bcafbbee6dcd9f604e9c3f10c42f2531f17f21769b991540e36b8ca0"
)


def sha256_file(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


@pytest.fixture(autouse=True)
def reset_cached_model():
    """Keep mocked and real model loads isolated from every other test."""
    cached_get_model = prediction_service.get_model
    cached_get_model.cache_clear()
    yield
    cached_get_model.cache_clear()


def test_repository_default_model_path_is_frozen_model_v2():
    assert DEFAULT_MODEL_PATH == EXPECTED_MODEL_PATH
    if Config.MODEL_PATH_IS_DEFAULT:
        assert Config.MODEL_PATH == DEFAULT_MODEL_PATH


def test_model_path_environment_override_is_preserved(tmp_path):
    custom_model = tmp_path / "custom-model.keras"
    environment = os.environ.copy()
    environment["MODEL_PATH"] = str(custom_model)
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import json; "
                "from config import Config; "
                "print(json.dumps({"
                "'model_path': str(Config.MODEL_PATH), "
                "'is_default': Config.MODEL_PATH_IS_DEFAULT"
                "}))"
            ),
        ],
        cwd=PROJECT_ROOT,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(result.stdout)
    assert Path(payload["model_path"]) == custom_model
    assert payload["is_default"] is False


def test_empty_model_path_environment_value_keeps_default_integrity_lock():
    environment = os.environ.copy()
    environment["MODEL_PATH"] = ""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import json; "
                "from config import Config; "
                "print(json.dumps({"
                "'model_path': str(Config.MODEL_PATH), "
                "'is_default': Config.MODEL_PATH_IS_DEFAULT"
                "}))"
            ),
        ],
        cwd=PROJECT_ROOT,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(result.stdout)
    assert Path(payload["model_path"]) == DEFAULT_MODEL_PATH
    assert payload["is_default"] is True


def test_frozen_metadata_and_committed_artifact_sha_are_identical():
    selection = json.loads(FROZEN_SELECTION_PATH.read_text(encoding="utf-8"))

    assert selection["selected_model_sha256"] == EXPECTED_MODEL_SHA256
    assert DEFAULT_MODEL_PATH.is_file(), "The frozen Model V2 artifact is missing."
    assert sha256_file(DEFAULT_MODEL_PATH) == EXPECTED_MODEL_SHA256


def test_default_hash_mismatch_fails_before_model_loading(tmp_path, monkeypatch):
    invalid_model = tmp_path / "invalid-default.keras"
    invalid_model.write_bytes(b"not the frozen model")

    monkeypatch.setattr(Config, "MODEL_PATH", invalid_model)
    monkeypatch.setattr(Config, "MODEL_PATH_IS_DEFAULT", True)

    def reject_model_load(*_args, **_kwargs):
        raise AssertionError("A model with an invalid SHA must not be loaded.")

    monkeypatch.setattr(prediction_service, "_load_model_file", reject_model_load)

    with pytest.raises(prediction_service.PredictionError, match="SHA-256"):
        prediction_service.get_model()


def test_custom_model_override_skips_frozen_sha_requirement(tmp_path, monkeypatch):
    custom_model = tmp_path / "custom-model.keras"
    loaded_model = SimpleNamespace(
        input_shape=(None, 224, 224, 3),
        output_shape=(None, len(CLASS_NAMES)),
    )
    loader_calls: list[tuple[Path, bool]] = []

    monkeypatch.setattr(Config, "MODEL_PATH", custom_model)
    monkeypatch.setattr(Config, "MODEL_PATH_IS_DEFAULT", False)

    def reject_integrity_check(_path):
        raise AssertionError("A custom MODEL_PATH must not use the frozen SHA lock.")

    def fake_loader(path, *, use_frozen_windows_compat):
        loader_calls.append((Path(path), use_frozen_windows_compat))
        return loaded_model

    monkeypatch.setattr(
        prediction_service, "verify_frozen_model_integrity", reject_integrity_check
    )
    monkeypatch.setattr(prediction_service, "_load_model_file", fake_loader)

    assert prediction_service.get_model() is loaded_model
    assert loader_calls == [(custom_model, False)]


def test_explicit_frozen_path_keeps_sha_lock_and_windows_compat(monkeypatch):
    loaded_model = SimpleNamespace(
        input_shape=(None, 224, 224, 3),
        output_shape=(None, len(CLASS_NAMES)),
    )
    loader_calls: list[tuple[Path, bool]] = []

    monkeypatch.setattr(Config, "MODEL_PATH", DEFAULT_MODEL_PATH)
    monkeypatch.setattr(Config, "MODEL_PATH_IS_DEFAULT", False)
    integrity_calls: list[Path] = []

    def fake_integrity_check(path):
        integrity_calls.append(Path(path))

    def fake_loader(path, *, use_frozen_windows_compat):
        loader_calls.append((Path(path), use_frozen_windows_compat))
        return loaded_model

    monkeypatch.setattr(
        prediction_service, "verify_frozen_model_integrity", fake_integrity_check
    )
    monkeypatch.setattr(prediction_service, "_load_model_file", fake_loader)

    assert prediction_service.get_model() is loaded_model
    assert integrity_calls == [DEFAULT_MODEL_PATH]
    assert loader_calls == [(DEFAULT_MODEL_PATH, True)]


def test_default_model_integrity_and_load_run_once_per_process(monkeypatch):
    loaded_model = SimpleNamespace(
        input_shape=(None, 224, 224, 3),
        output_shape=(None, len(CLASS_NAMES)),
    )
    integrity_calls: list[Path] = []
    loader_calls: list[tuple[Path, bool]] = []

    monkeypatch.setattr(Config, "MODEL_PATH", DEFAULT_MODEL_PATH)
    monkeypatch.setattr(Config, "MODEL_PATH_IS_DEFAULT", True)

    def fake_integrity_check(path):
        integrity_calls.append(Path(path))

    def fake_loader(path, *, use_frozen_windows_compat):
        loader_calls.append((Path(path), use_frozen_windows_compat))
        return loaded_model

    monkeypatch.setattr(
        prediction_service, "verify_frozen_model_integrity", fake_integrity_check
    )
    monkeypatch.setattr(prediction_service, "_load_model_file", fake_loader)

    assert prediction_service.get_model() is loaded_model
    assert prediction_service.get_model() is loaded_model
    assert integrity_calls == [DEFAULT_MODEL_PATH]
    assert loader_calls == [(DEFAULT_MODEL_PATH, True)]


def test_model_loader_disables_keras_compilation(tmp_path, monkeypatch):
    custom_model = tmp_path / "custom-model.keras"
    loaded_model = object()
    calls: list[tuple[Path, bool]] = []

    def fake_keras_loader(path, *, compile):
        calls.append((Path(path), compile))
        return loaded_model

    monkeypatch.setattr(prediction_service, "load_model", fake_keras_loader)

    result = prediction_service._load_model_file(
        custom_model, use_frozen_windows_compat=False
    )

    assert result is loaded_model
    assert calls == [(custom_model, False)]


def test_windows_keras_archive_compatibility_uses_posix_join_and_restores_it(
    tmp_path, monkeypatch
):
    import tensorflow.compat.v2 as tensorflow_compat_v2

    model_path = tmp_path / "frozen-model.keras"
    loaded_model = object()
    original_join = tensorflow_compat_v2.io.gfile.join
    calls: list[tuple[Path, bool, str]] = []

    def fake_keras_loader(path, *, compile):
        calls.append(
            (
                Path(path),
                compile,
                tensorflow_compat_v2.io.gfile.join("model\\layers", "vars"),
            )
        )
        assert (
            tensorflow_compat_v2.io.gfile.join
            is prediction_service._keras_archive_posix_join
        )
        return loaded_model

    monkeypatch.setattr(prediction_service, "os", SimpleNamespace(name="nt"))
    monkeypatch.setattr(prediction_service, "load_model", fake_keras_loader)

    result = prediction_service._load_model_file(
        model_path, use_frozen_windows_compat=True
    )

    assert result is loaded_model
    assert calls == [(model_path, False, "model/layers/vars")]
    assert tensorflow_compat_v2.io.gfile.join is original_join


def test_windows_keras_archive_compatibility_restores_join_after_failure(
    tmp_path, monkeypatch
):
    import tensorflow.compat.v2 as tensorflow_compat_v2

    model_path = tmp_path / "frozen-model.keras"
    original_join = tensorflow_compat_v2.io.gfile.join

    def failing_keras_loader(_path, *, compile):
        assert compile is False
        assert (
            tensorflow_compat_v2.io.gfile.join
            is prediction_service._keras_archive_posix_join
        )
        raise ValueError("synthetic loader failure")

    monkeypatch.setattr(prediction_service, "os", SimpleNamespace(name="nt"))
    monkeypatch.setattr(prediction_service, "load_model", failing_keras_loader)

    with pytest.raises(ValueError, match="synthetic loader failure"):
        prediction_service._load_model_file(
            model_path, use_frozen_windows_compat=True
        )

    assert tensorflow_compat_v2.io.gfile.join is original_join


def test_deployed_preprocessing_is_rgb_224_float32_and_divided_by_255(
    tmp_path, monkeypatch
):
    image_path = tmp_path / "rgba-leaf.png"
    Image.new("RGBA", (5, 3), color=(255, 128, 0, 17)).save(image_path)
    captured: dict[str, np.ndarray] = {}

    class CapturingModel:
        def predict(self, batch, verbose=0):
            captured["batch"] = batch
            assert verbose == 0
            scores = np.zeros((1, len(CLASS_NAMES)), dtype=np.float32)
            scores[0, 3] = 1.0
            return scores

    monkeypatch.setattr(prediction_service, "get_model", lambda: CapturingModel())

    result = prediction_service.predict(image_path)

    batch = captured["batch"]
    assert batch.shape == (1, 224, 224, 3)
    assert batch.dtype == np.float32
    np.testing.assert_allclose(
        batch[0, 0, 0], np.asarray([1.0, 128.0 / 255.0, 0.0], dtype=np.float32)
    )
    assert result["prediction"]["class_index"] == 3
    assert result["prediction"]["status"] == "healthy"


def test_frozen_model_v2_loads_and_produces_finite_39_value_output(monkeypatch):
    assert DEFAULT_MODEL_PATH.is_file(), "The frozen Model V2 artifact is missing."
    monkeypatch.setattr(Config, "MODEL_PATH", DEFAULT_MODEL_PATH)
    monkeypatch.setattr(Config, "MODEL_PATH_IS_DEFAULT", True)

    model = prediction_service.get_model()

    assert tuple(model.input_shape) == (None, 224, 224, 3)
    assert tuple(model.output_shape) == (None, len(CLASS_NAMES))
    output = np.asarray(
        model(np.zeros((1, 224, 224, 3), dtype=np.float32), training=False)
    )
    assert output.shape == (1, len(CLASS_NAMES))
    assert np.isfinite(output).all()
