from __future__ import annotations

from functools import lru_cache
import logging
from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError
from tensorflow.keras.models import load_model

from config import Config

# Recovered from the companion PlantVillage project in the local workspace.
# Its 39-class `idx_to_classes` mapping matches this model's output count and the
# alphabetical training-generator ordering. Index 4 is the non-leaf background class.
CLASS_NAMES = [
 "Apple Apple scab","Apple Black rot","Apple Cedar apple rust","Apple healthy","Background without leaves","Blueberry healthy","Cherry Powdery mildew","Cherry healthy","Corn Cercospora leaf spot","Corn Common rust","Corn Northern Leaf Blight","Corn healthy","Grape Black rot","Grape Esca","Grape Leaf blight","Grape healthy","Orange Huanglongbing","Peach Bacterial spot","Peach healthy","Bell pepper Bacterial spot","Bell pepper healthy","Potato Early blight","Potato Late blight","Potato healthy","Raspberry healthy","Soybean healthy","Squash Powdery mildew","Strawberry Leaf scorch","Strawberry healthy","Tomato Bacterial spot","Tomato Early blight","Tomato Late blight","Tomato Leaf Mold","Tomato Septoria leaf spot","Tomato Spider mites","Tomato Target Spot","Tomato Yellow Leaf Curl Virus","Tomato mosaic virus","Tomato healthy"
]
logger = logging.getLogger(__name__)

class PredictionError(Exception): pass


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
    model = load_model(Config.MODEL_PATH, compile=False)
    if model.output_shape[-1] != len(CLASS_NAMES):
        raise PredictionError(f"Model has {model.output_shape[-1]} outputs but {len(CLASS_NAMES)} labels are configured.")
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
        image_array = np.asarray(img.convert("RGB").resize((224, 224)), dtype=np.float32) / 255.0
    scores = get_model().predict(np.expand_dims(image_array, axis=0), verbose=0)[0]
    top_indices = np.argsort(scores)[::-1][:3]
    top = [{"class_index": int(i), "disease": CLASS_NAMES[i], "confidence": round(float(scores[i]) * 100, 2)} for i in top_indices]
    primary = top[0]
    primary["is_background"] = primary["class_index"] == 4
    primary["status"] = determine_status(
        primary["class_index"], primary["disease"], primary["confidence"]
    )
    primary["uncertain"] = primary["status"] == "uncertain"
    if primary["is_background"]: logger.info("Background/no-leaf class predicted with confidence=%s", primary["confidence"])
    return {"prediction": primary, "top_predictions": top}
