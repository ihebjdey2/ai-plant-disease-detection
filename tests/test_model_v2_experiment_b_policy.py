from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest
import tensorflow as tf

from training.data_pipeline import TrainingPolicyError, load_policy
from training.experiment_a import build_model as build_experiment_a_model
from training.experiment_b import (
    APPROVED_AUGMENTATION,
    BASELINE_EXPERIMENT_NAME,
    BASELINE_POLICY_SHA256_LF,
    CLASS_WEIGHTS,
    EXPERIMENT_NAME,
    POLICY_PATH,
    TRAIN_MANIFEST_SHA256_LF,
    VALIDATION_MANIFEST_SHA256_LF,
    build_model,
    configure_phase2,
    load_experiment_b_policy,
    validate_experiment_b_policy,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_A_POLICY = (
    PROJECT_ROOT / "training/config/model-v2-training-policy.json"
)
TRAIN_MANIFEST = PROJECT_ROOT / "training/datasets/manifests/dataset-v2-train.csv"
VALIDATION_MANIFEST = (
    PROJECT_ROOT / "training/datasets/manifests/dataset-v2-validation.csv"
)

CONTROLLED_POLICY_KEYS = (
    "architecture",
    "batch_size",
    "callbacks",
    "class_weight_policy",
    "experiment_seed",
    "input",
    "locked_test_policy",
    "loss",
    "optimizer",
    "phase1",
    "phase2",
    "selection_metrics",
)


def sha256_with_lf(path: Path) -> str:
    content = path.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return hashlib.sha256(content).hexdigest()


def test_experiment_a_policy_and_development_manifests_remain_locked():
    assert sha256_with_lf(EXPERIMENT_A_POLICY) == BASELINE_POLICY_SHA256_LF
    assert sha256_with_lf(TRAIN_MANIFEST) == TRAIN_MANIFEST_SHA256_LF
    assert sha256_with_lf(VALIDATION_MANIFEST) == VALIDATION_MANIFEST_SHA256_LF


def test_experiment_b_policy_identity_and_baseline_are_explicit():
    policy = load_experiment_b_policy()

    assert POLICY_PATH.name == "model-v2-experiment-b-policy.json"
    assert policy["experiment"] == {
        "baseline_experiment": BASELINE_EXPERIMENT_NAME,
        "hypothesis": (
            "A moderate increase in TRAIN-only geometric and photometric "
            "augmentation will improve robustness to real-world image conditions "
            "while preserving overall VALIDATION performance."
        ),
        "name": EXPERIMENT_NAME,
        "primary_variable": "TRAIN_ONLY_AUGMENTATION_POLICY",
    }
    assert policy["baseline"]["policy_sha256_lf"] == BASELINE_POLICY_SHA256_LF
    assert policy["baseline"]["train_manifest"]["sha256_lf"] == (
        TRAIN_MANIFEST_SHA256_LF
    )
    assert policy["baseline"]["validation_manifest"]["sha256_lf"] == (
        VALIDATION_MANIFEST_SHA256_LF
    )
    assert policy["class_weights"] is None
    assert CLASS_WEIGHTS is None


def test_experiment_b_differs_scientifically_only_in_approved_augmentation():
    experiment_a = load_policy(EXPERIMENT_A_POLICY)
    experiment_b = load_experiment_b_policy()

    for key in CONTROLLED_POLICY_KEYS:
        assert experiment_b[key] == experiment_a[key], key

    for key, expected in APPROVED_AUGMENTATION.items():
        assert experiment_b["augmentation"][key] == expected

    assert experiment_a["augmentation"]["rotation_degrees"] == 15.0
    assert experiment_b["augmentation"]["rotation_degrees"] == 20.0
    assert experiment_a["augmentation"]["translation_fraction"] == 0.1
    assert experiment_b["augmentation"]["translation_fraction"] == 0.12
    assert experiment_a["augmentation"]["zoom_range"] == [0.9, 1.1]
    assert experiment_b["augmentation"]["zoom_range"] == [0.85, 1.15]
    assert experiment_a["augmentation"]["brightness_factor"] == 0.1
    assert experiment_b["augmentation"]["brightness_factor"] == 0.15
    assert experiment_a["augmentation"]["contrast_factor"] == 0.1
    assert experiment_b["augmentation"]["contrast_factor"] == 0.15
    assert experiment_b["augmentation"]["horizontal_flip"] is True
    assert experiment_b["augmentation"]["vertical_flip"] is True
    assert experiment_b["augmentation"]["enabled_for"] == ["TRAIN"]
    assert experiment_b["augmentation"]["fill_mode"] == "reflect"
    assert experiment_b["augmentation"]["output_clip"] == [0.0, 1.0]


def test_experiment_b_policy_lists_every_forbidden_augmentation_family():
    policy = load_experiment_b_policy()
    forbidden = set(policy["augmentation"]["forbidden_methods"])

    assert forbidden == {
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
    }


def test_experiment_b_policy_validation_fails_closed_on_unapproved_change():
    policy = load_experiment_b_policy()
    changed = copy.deepcopy(policy)
    changed["augmentation"]["rotation_degrees"] = 25.0

    with pytest.raises(TrainingPolicyError, match="augmentation policy changed"):
        validate_experiment_b_policy(changed)

    weighted = copy.deepcopy(policy)
    weighted["class_weights"] = "recommended_moderate"
    with pytest.raises(TrainingPolicyError, match="class weights"):
        validate_experiment_b_policy(weighted)


def test_experiment_b_model_matches_a_architecture_but_has_separate_identity():
    experiment_a_policy = load_policy(EXPERIMENT_A_POLICY)
    experiment_b_policy = load_experiment_b_policy()

    tf.keras.backend.clear_session()
    model_a, backbone_a = build_experiment_a_model(
        experiment_a_policy, weights=None
    )
    audit_a = {
        "input_shape": model_a.input_shape,
        "output_shape": model_a.output_shape,
        "parameter_count": model_a.count_params(),
        "layer_types": [type(layer).__name__ for layer in model_a.layers],
        "backbone_layer_types": [
            type(layer).__name__ for layer in backbone_a.layers
        ],
    }

    tf.keras.backend.clear_session()
    model_b, backbone_b = build_model(experiment_b_policy, weights=None)
    audit_b = {
        "input_shape": model_b.input_shape,
        "output_shape": model_b.output_shape,
        "parameter_count": model_b.count_params(),
        "layer_types": [type(layer).__name__ for layer in model_b.layers],
        "backbone_layer_types": [
            type(layer).__name__ for layer in backbone_b.layers
        ],
    }

    assert audit_b == audit_a
    assert model_a.name == "agri_diagnose_v2_exp_a"
    assert model_b.name == "agri_diagnose_v2_exp_b"
    assert backbone_b.trainable is False

    phase2 = configure_phase2(backbone_b, experiment_b_policy)
    assert phase2["first_trainable_backbone_layer"] == "block_13_expand"
    assert phase2["frozen_batch_normalization_count"] == 52


def test_policy_file_is_valid_json_and_uses_train_only_augmentation():
    raw = json.loads(POLICY_PATH.read_text(encoding="utf-8"))
    validated = load_policy(POLICY_PATH)

    assert raw == validated
    assert validated["augmentation"]["enabled_for"] == ["TRAIN"]
    assert validated["controlled_variables"]["selection_partition"] == (
        "VALIDATION_ONLY"
    )
