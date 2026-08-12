from __future__ import annotations

from pathlib import Path
from typing import Mapping

import tensorflow as tf

from training.data_pipeline import TrainingPolicyError, load_policy
from training.experiment_a import (
    callback_policy,
    compile_phase1,
    compile_phase2,
    configure_phase2,
    parameter_audit,
)
from training.taxonomy import CLASS_NAMES


EXPERIMENT_NAME = "agri-diagnose-v2-exp-b"
BASELINE_EXPERIMENT_NAME = "agri-diagnose-v2-exp-a"
CLASS_WEIGHTS = None
POLICY_PATH = (
    Path(__file__).resolve().parent / "config" / "model-v2-experiment-b-policy.json"
)

BASELINE_POLICY_SHA256_LF = (
    "16c16e56819aa96df972f33fb29317fd82fd84e2c9945bf8d4d974c85f682f11"
)
TRAIN_MANIFEST_SHA256_LF = (
    "957d4acb4c097116099c57446733b3d70088bf083e7869aadd11e26caf70a915"
)
VALIDATION_MANIFEST_SHA256_LF = (
    "9c10de69e935324ee325667fab2902b372a144a722c4fc793d3b4f1afe01767e"
)

APPROVED_AUGMENTATION = {
    "brightness_factor": 0.15,
    "contrast_factor": 0.15,
    "enabled_for": ["TRAIN"],
    "fill_mode": "reflect",
    "horizontal_flip": True,
    "output_clip": [0.0, 1.0],
    "rotation_degrees": 20.0,
    "translation_fraction": 0.12,
    "vertical_flip": True,
    "zoom_range": [0.85, 1.15],
}

FORBIDDEN_AUGMENTATION_METHODS = (
    "hue or saturation augmentation",
    "blur",
    "sensor noise",
    "shear",
    "perspective transforms",
    "random crops",
    "MixUp",
    "CutMix",
    "synthetic images",
    "background replacement",
)


def validate_experiment_b_policy(policy: Mapping[str, object]) -> None:
    """Fail closed unless the dedicated Experiment B policy is exactly approved."""
    experiment = policy.get("experiment")
    baseline = policy.get("baseline")
    augmentation = policy.get("augmentation")
    if not isinstance(experiment, Mapping) or (
        experiment.get("name") != EXPERIMENT_NAME
        or experiment.get("baseline_experiment") != BASELINE_EXPERIMENT_NAME
        or experiment.get("primary_variable") != "TRAIN_ONLY_AUGMENTATION_POLICY"
    ):
        raise TrainingPolicyError("Experiment B identity or hypothesis metadata changed.")
    if not isinstance(baseline, Mapping) or (
        baseline.get("experiment_name") != BASELINE_EXPERIMENT_NAME
        or baseline.get("policy_sha256_lf") != BASELINE_POLICY_SHA256_LF
    ):
        raise TrainingPolicyError("Experiment A baseline lock changed.")
    if policy.get("class_weights", object()) is not None or CLASS_WEIGHTS is not None:
        raise TrainingPolicyError("Experiment B class weights must remain disabled.")
    if not isinstance(augmentation, Mapping):
        raise TrainingPolicyError("Experiment B augmentation policy is unavailable.")
    mismatches = {
        key: {"expected": expected, "actual": augmentation.get(key)}
        for key, expected in APPROVED_AUGMENTATION.items()
        if augmentation.get(key) != expected
    }
    if mismatches:
        raise TrainingPolicyError(
            f"Experiment B augmentation policy changed: {mismatches}"
        )
    if tuple(augmentation.get("forbidden_methods", ())) != (
        FORBIDDEN_AUGMENTATION_METHODS
    ):
        raise TrainingPolicyError("Experiment B forbidden augmentation list changed.")


def load_experiment_b_policy() -> dict[str, object]:
    policy = load_policy(POLICY_PATH)
    validate_experiment_b_policy(policy)
    return policy


def build_model(
    policy: Mapping[str, object], *, weights: str | None = "imagenet"
):
    """Build the A-equivalent architecture with an Experiment B identifier."""
    validate_experiment_b_policy(policy)
    input_policy = policy["input"]
    architecture = policy["architecture"]
    inputs = tf.keras.Input(
        shape=(
            int(input_policy["height"]),
            int(input_policy["width"]),
            int(input_policy["channels"]),
        ),
        name="leaf_image",
    )
    backbone = tf.keras.applications.MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False,
        weights=weights,
    )
    backbone.trainable = False
    features = backbone(inputs, training=False)
    pooled = tf.keras.layers.GlobalAveragePooling2D(
        name="global_average_pooling"
    )(features)
    regularized = tf.keras.layers.Dropout(
        float(architecture["dropout"]), name="classifier_dropout"
    )(pooled)
    outputs = tf.keras.layers.Dense(
        len(CLASS_NAMES), activation="softmax", name="disease_probabilities"
    )(regularized)
    model = tf.keras.Model(inputs, outputs, name="agri_diagnose_v2_exp_b")
    if model.output_shape != (None, len(CLASS_NAMES)):
        raise RuntimeError(f"Unexpected Model V2 output shape: {model.output_shape}")
    return model, backbone


__all__ = [
    "APPROVED_AUGMENTATION",
    "BASELINE_EXPERIMENT_NAME",
    "BASELINE_POLICY_SHA256_LF",
    "CLASS_WEIGHTS",
    "EXPERIMENT_NAME",
    "POLICY_PATH",
    "TRAIN_MANIFEST_SHA256_LF",
    "VALIDATION_MANIFEST_SHA256_LF",
    "build_model",
    "callback_policy",
    "compile_phase1",
    "compile_phase2",
    "configure_phase2",
    "load_experiment_b_policy",
    "parameter_audit",
    "validate_experiment_b_policy",
]
