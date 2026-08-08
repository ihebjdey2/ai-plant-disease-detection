from __future__ import annotations

from functools import lru_cache
import logging
from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError
from tensorflow.keras.models import load_model

from config import Config

# The model exposes 39 outputs while the repository only supplies 38 verified labels.
# The final output is intentionally not guessed.
CLASS_NAMES = [
 "Apple Apple scab","Apple Black rot","Apple Cedar apple rust","Apple healthy","Blueberry healthy","Cherry Powdery mildew","Cherry healthy","Corn Cercospora leaf spot","Corn Common rust","Corn Northern Leaf Blight","Corn healthy","Grape Black rot","Grape Esca","Grape Leaf blight","Grape healthy","Orange Huanglongbing","Peach Bacterial spot","Peach healthy","Bell pepper Bacterial spot","Bell pepper healthy","Potato Early blight","Potato Late blight","Potato healthy","Raspberry healthy","Soybean healthy","Squash Powdery mildew","Strawberry Leaf scorch","Strawberry healthy","Tomato Bacterial spot","Tomato Early blight","Tomato Late blight","Tomato Leaf Mold","Tomato Septoria leaf spot","Tomato Spider mites","Tomato Target Spot","Tomato Yellow Leaf Curl Virus","Tomato mosaic virus","Tomato healthy"
]
UNKNOWN_CLASS = "UNKNOWN_CLASS_38"
logger = logging.getLogger(__name__)

class PredictionError(Exception): pass

@lru_cache(maxsize=1)
def get_model():
    model = load_model(Config.MODEL_PATH, compile=False)
    if model.output_shape[-1] != len(CLASS_NAMES):
        logger.warning("Model output count %s differs from %s verified class labels; index 38 is unresolved.", model.output_shape[-1], len(CLASS_NAMES))
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
    top = [{"class_index": int(i), "disease": CLASS_NAMES[i] if i < len(CLASS_NAMES) else "Unknown plant disease class", "internal_label": None if i < len(CLASS_NAMES) else UNKNOWN_CLASS, "confidence": round(float(scores[i]) * 100, 2)} for i in top_indices]
    primary = top[0]
    primary["uncertain"] = primary["confidence"] < Config.PREDICTION_CONFIDENCE_THRESHOLD or primary["class_index"] >= len(CLASS_NAMES)
    if primary["class_index"] >= len(CLASS_NAMES): logger.warning("Unresolved model output predicted: index=%s confidence=%s", primary["class_index"], primary["confidence"])
    return {"prediction": primary, "top_predictions": top}
