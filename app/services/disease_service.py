"""Cautious, presentation-ready disease guidance.

This content is decision support only and is not an agricultural diagnosis.
"""

from __future__ import annotations


def get_disease_info(label: str) -> dict:
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
