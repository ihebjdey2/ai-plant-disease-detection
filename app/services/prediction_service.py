from __future__ import annotations

import hashlib
import json
import logging
import os
import posixpath
from functools import lru_cache
from pathlib import Path
from threading import Lock

import numpy as np
from PIL import Image, UnidentifiedImageError
from tensorflow.keras.models import load_model

from config import Config, DEFAULT_MODEL_PATH
from app.taxonomy import CLASS_NAMES

logger = logging.getLogger(__name__)

FROZEN_MODEL_SHA256 = (
    "bba4d044bcafbbee6dcd9f604e9c3f10c42f2531f17f21769b991540e36b8ca0"
)
MODEL_INPUT_SHAPE = (224, 224, 3)
_KERAS_LOAD_LOCK = Lock()


class PredictionError(Exception):
    pass


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as model_file:
        for chunk in iter(lambda: model_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_frozen_model_integrity(path: Path) -> str:
    """Verify the frozen default artifact against immutable selection metadata."""
    try:
        selection = json.loads(
            Path(Config.MODEL_SELECTION_PATH).read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise PredictionError(
            "Frozen Model V2 selection metadata could not be read."
        ) from exc

    metadata_sha = selection.get("selected_model_sha256")
    if metadata_sha != FROZEN_MODEL_SHA256:
        raise PredictionError(
            "Frozen Model V2 selection metadata has an unexpected SHA-256."
        )

    try:
        actual_sha = _sha256_file(path)
    except OSError as exc:
        raise PredictionError("The configured model file could not be read.") from exc
    if actual_sha != metadata_sha:
        raise PredictionError(
            "Frozen Model V2 integrity verification failed: SHA-256 mismatch."
        )
    return actual_sha


def _is_frozen_artifact_path(path: Path) -> bool:
    return path.resolve() == DEFAULT_MODEL_PATH.resolve()


def _keras_archive_posix_join(*parts: object) -> str:
    """Join paths inside a Keras archive independently of the host separator."""
    return posixpath.join(*(str(part).replace("\\", "/") for part in parts))


def _load_model_file(path: Path, *, use_frozen_windows_compat: bool):
    """Load a model, including the Keras 2.15 Windows archive-path workaround.

    Keras 2.15 joins nested HDF5 paths with the host separator. A `.keras`
    archive written on Linux therefore needs POSIX joins while it is read on
    Windows. The override is process-local, lock-protected, and always restored.
    The model archive itself is never modified.
    """
    if (
        os.name != "nt"
        or not use_frozen_windows_compat
        or path.suffix.lower() != ".keras"
    ):
        return load_model(path, compile=False)

    import tensorflow.compat.v2 as tensorflow_compat_v2

    with _KERAS_LOAD_LOCK:
        original_join = tensorflow_compat_v2.io.gfile.join
        try:
            tensorflow_compat_v2.io.gfile.join = _keras_archive_posix_join
            return load_model(path, compile=False)
        finally:
            tensorflow_compat_v2.io.gfile.join = original_join


def _validate_model_contract(model) -> None:
    try:
        input_shape = tuple(model.input_shape)
        output_shape = tuple(model.output_shape)
    except (AttributeError, TypeError) as exc:
        raise PredictionError(
            "The configured model has invalid input/output shapes."
        ) from exc

    if len(input_shape) != 4 or input_shape[-3:] != MODEL_INPUT_SHAPE:
        raise PredictionError(
            f"Model input shape {input_shape} is incompatible with "
            f"{MODEL_INPUT_SHAPE}."
        )
    if len(output_shape) != 2 or output_shape[-1] != len(CLASS_NAMES):
        raise PredictionError(
            f"Model has output shape {output_shape}, but {len(CLASS_NAMES)} "
            "labels are configured."
        )


def determine_status(
    class_index: int,
    disease: str,
    confidence: float,
    threshold: float | None = None,
) -> str:
    """Classify a model result using one shared, explicit precedence order."""
    confidence_threshold = (
        Config.PREDICTION_CONFIDENCE_THRESHOLD if threshold is None else threshold
    )
    if class_index == 4:
        return "no_leaf"
    if confidence < confidence_threshold:
        return "uncertain"
    if "healthy" in disease.lower():
        return "healthy"
    return "diseased"


@lru_cache(maxsize=1)
def get_model():
    model_path = Path(Config.MODEL_PATH)
    is_frozen_artifact = _is_frozen_artifact_path(model_path)
    if Config.MODEL_PATH_IS_DEFAULT or is_frozen_artifact:
        verify_frozen_model_integrity(model_path)
    try:
        model = _load_model_file(
            model_path,
            use_frozen_windows_compat=is_frozen_artifact,
        )
    except PredictionError:
        raise
    except (OSError, ValueError, TypeError) as exc:
        logger.exception("Unable to load configured model from %s", model_path)
        raise PredictionError(
            "The configured plant-disease model could not be loaded."
        ) from exc
    _validate_model_contract(model)
    return model


def validate_image(path: Path) -> None:
    try:
        with Image.open(path) as img:
            img.verify()
    except (UnidentifiedImageError, OSError) as exc:
        raise PredictionError("The uploaded file is not a valid image.") from exc


def predict(path: Path) -> dict:
    validate_image(path)
    with Image.open(path) as img:
        image_array = (
            np.asarray(
                img.convert("RGB").resize((224, 224)), dtype=np.float32
            )
            / 255.0
        )
    scores = get_model().predict(np.expand_dims(image_array, axis=0), verbose=0)[0]
    top_indices = np.argsort(scores)[::-1][:3]
    top = [
        {
            "class_index": int(i),
            "disease": CLASS_NAMES[i],
            "confidence": round(float(scores[i]) * 100, 2),
        }
        for i in top_indices
    ]
    primary = top[0]
    primary["is_background"] = primary["class_index"] == 4
    primary["status"] = determine_status(
        primary["class_index"], primary["disease"], primary["confidence"]
    )
    primary["uncertain"] = primary["status"] == "uncertain"
    if primary["is_background"]:
        logger.info(
            "Background/no-leaf class predicted with confidence=%s",
            primary["confidence"],
        )
    return {"prediction": primary, "top_predictions": top}
