"""Cautious, presentation-ready disease guidance.

This content is decision support only and is not an agricultural diagnosis.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=1)
def _metadata() -> dict[int, dict]:
    path = Path(__file__).resolve().parent.parent / "data" / "disease_metadata.json"
    return {item["class_index"]: item for item in json.loads(path.read_text(encoding="utf-8"))}


def get_disease_info(label: str, class_index: int | None = None) -> dict:
    if label == "Background without leaves":
        return {
            "plant_name": "Unknown",
            "disease_name": "No leaf detected",
            "description": "The model detected a background or non-leaf image rather than a plant leaf.",
            "symptoms": [], "causes": [],
            "treatment": ["Upload a clear, well-lit photo of a single plant leaf."],
            "prevention": ["Keep the leaf centered and avoid busy backgrounds."],
            "disclaimer": "No disease assessment was made for this image.",
        }
    if class_index is not None and class_index in _metadata():
        item = _metadata()[class_index]
        return {
            "plant_name": label.split(" ", 1)[0], "disease_name": item["disease_name"],
            "description": item["description"], "symptoms": [], "causes": [],
            "treatment": [item["recommended_steps"]], "prevention": [],
            "reference_image_url": item["reference_image_url"],
            "disclaimer": "General guidance from the referenced dataset; consult local agricultural expertise.",
        }
    plant, _, condition = label.partition(" ")
    healthy = "healthy" in label.lower()
    if healthy:
        return {
            "plant_name": plant or "Plant",
            "disease_name": "Healthy plant",
            "description": "The image is classified as healthy by the model.",
            "symptoms": [],
            "causes": [],
            "treatment": ["Continue regular observation and crop care."],
            "prevention": ["Use clean tools, appropriate irrigation, and routine field checks."],
            "disclaimer": "Model output is decision support, not a confirmed agricultural diagnosis.",
        }
    return {
        "plant_name": plant or "Plant",
        "disease_name": condition or label,
        "description": "The model detected a pattern associated with this condition.",
        "symptoms": ["Compare the leaf with reliable local extension-service guidance."],
        "causes": ["Causes vary by crop, climate, and growing conditions."],
        "treatment": ["Isolate or remove severely affected material where appropriate.", "Consult a qualified local agronomist before applying treatment products."],
        "prevention": ["Monitor plants regularly.", "Improve airflow, sanitation, and irrigation practices appropriate to the crop."],
        "disclaimer": "Model output is decision support, not a confirmed agricultural diagnosis.",
    }
