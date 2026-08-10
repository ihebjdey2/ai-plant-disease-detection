from __future__ import annotations

from pathlib import Path
from typing import Mapping

import tensorflow as tf

from app.taxonomy import CLASS_NAMES
from training.metrics import MacroF1


EXPERIMENT_NAME = "agri-diagnose-v2-exp-a"
CLASS_WEIGHTS = None


def build_model(policy: Mapping[str, object], *, weights: str = "imagenet"):
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
    pooled = tf.keras.layers.GlobalAveragePooling2D(name="global_average_pooling")(features)
    regularized = tf.keras.layers.Dropout(
        float(architecture["dropout"]), name="classifier_dropout"
    )(pooled)
    outputs = tf.keras.layers.Dense(
        len(CLASS_NAMES), activation="softmax", name="disease_probabilities"
    )(regularized)
    model = tf.keras.Model(inputs, outputs, name="agri_diagnose_v2_exp_a")
    if model.output_shape != (None, len(CLASS_NAMES)):
        raise RuntimeError(f"Unexpected Model V2 output shape: {model.output_shape}")
    return model, backbone


def compile_phase1(model, policy: Mapping[str, object]) -> None:
    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=float(policy["phase1"]["learning_rate"])
        ),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy"), MacroF1()],
    )


def configure_phase2(backbone, policy: Mapping[str, object]) -> dict[str, object]:
    start_name = str(policy["phase2"]["fine_tune_from_layer_name"])
    names = [layer.name for layer in backbone.layers]
    if start_name not in names:
        raise RuntimeError(f"Fine-tuning boundary is unavailable: {start_name}")
    start_index = names.index(start_name)
    backbone.trainable = True
    for index, layer in enumerate(backbone.layers):
        layer.trainable = index >= start_index and not isinstance(
            layer, tf.keras.layers.BatchNormalization
        )
    trainable = [layer for layer in backbone.layers if layer.trainable]
    batch_norm = [
        layer
        for layer in backbone.layers
        if isinstance(layer, tf.keras.layers.BatchNormalization)
    ]
    return {
        "total_backbone_layers": len(backbone.layers),
        "fine_tune_boundary_index": start_index,
        "first_trainable_backbone_layer": trainable[0].name if trainable else None,
        "trainable_backbone_layer_count": len(trainable),
        "frozen_backbone_layer_count": len(backbone.layers) - len(trainable),
        "batch_normalization_layer_count": len(batch_norm),
        "frozen_batch_normalization_count": sum(not layer.trainable for layer in batch_norm),
    }


def compile_phase2(model, policy: Mapping[str, object]) -> None:
    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=float(policy["phase2"]["learning_rate"])
        ),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy"), MacroF1()],
    )


def parameter_audit(model) -> dict[str, int]:
    count = tf.keras.backend.count_params
    trainable = sum(count(weight) for weight in model.trainable_weights)
    non_trainable = sum(count(weight) for weight in model.non_trainable_weights)
    return {
        "total_parameters": int(trainable + non_trainable),
        "trainable_parameters": int(trainable),
        "non_trainable_parameters": int(non_trainable),
    }


def callback_policy(policy: Mapping[str, object], checkpoint_path: Path) -> dict[str, object]:
    """Return inspectable callback metadata; callback creation occurs at execution time."""
    callbacks = policy["callbacks"]
    return {
        "early_stopping": dict(callbacks["early_stopping"]),
        "reduce_lr_on_plateau": dict(callbacks["reduce_lr_on_plateau"]),
        "model_checkpoint": {
            **dict(callbacks["model_checkpoint"]),
            "path": checkpoint_path.as_posix(),
        },
    }
