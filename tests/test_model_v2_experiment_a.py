from __future__ import annotations

import numpy as np
import pytest
import tensorflow as tf
from sklearn.metrics import f1_score

from scripts.run_model_v2_experiment_a import validate_preflight_metadata
from training.data_pipeline import load_policy
from training.experiment_a import (
    CLASS_WEIGHTS,
    build_model,
    callback_policy,
    compile_phase1,
    configure_phase2,
)
from training.metrics import MacroF1


def metric_value(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    metric = MacroF1()
    probabilities = tf.one_hot(y_pred, depth=39, dtype=tf.float32)
    midpoint = max(1, len(y_true) // 2)
    metric.update_state(y_true[:midpoint], probabilities[:midpoint])
    metric.update_state(y_true[midpoint:], probabilities[midpoint:])
    return float(metric.result().numpy())


def assert_matches_sklearn(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    expected = f1_score(y_true, y_pred, average="macro")
    assert metric_value(y_true, y_pred) == pytest.approx(expected, abs=1e-6)


def test_macro_f1_perfect_39_class_classification():
    labels = np.arange(39, dtype=np.int32)
    assert_matches_sklearn(labels, labels)
    assert metric_value(labels, labels) == pytest.approx(1.0)


def test_macro_f1_fully_wrong_classification():
    labels = np.arange(39, dtype=np.int32)
    predictions = (labels + 1) % 39
    assert_matches_sklearn(labels, predictions)
    assert metric_value(labels, predictions) == pytest.approx(0.0)


def test_macro_f1_imbalanced_and_rare_classes():
    labels = np.repeat(np.arange(39, dtype=np.int32), np.arange(1, 40))
    predictions = labels.copy()
    predictions[::7] = (predictions[::7] + 3) % 39
    assert_matches_sklearn(labels, predictions)


def test_macro_f1_missing_predicted_class():
    labels = np.repeat(np.arange(39, dtype=np.int32), 3)
    predictions = labels.copy()
    predictions[predictions == 38] = 0
    assert_matches_sklearn(labels, predictions)


def test_macro_f1_multiclass_mixed_errors():
    labels = np.tile(np.arange(39, dtype=np.int32), 5)
    predictions = labels.copy()
    predictions[::4] = (predictions[::4] + 11) % 39
    assert_matches_sklearn(labels, predictions)


def test_macro_f1_reset_state():
    labels = np.arange(39, dtype=np.int32)
    metric = MacroF1()
    metric.update_state(labels, tf.one_hot(labels, 39))
    assert float(metric.result().numpy()) == pytest.approx(1.0)
    metric.reset_state()
    assert float(metric.result().numpy()) == 0.0
    assert np.count_nonzero(metric.confusion_matrix.numpy()) == 0


def test_phase1_model_has_frozen_backbone_and_39_outputs():
    policy = load_policy()
    model, backbone = build_model(policy, weights=None)
    compile_phase1(model, policy)
    assert backbone.trainable is False
    assert model.output_shape == (None, 39)
    assert any(
        metric.name == "macro_f1" for metric in model.compiled_metrics._user_metrics
    )


def test_phase2_starts_at_block_13_and_freezes_batchnorm():
    policy = load_policy()
    _, backbone = build_model(policy, weights=None)
    audit = configure_phase2(backbone, policy)
    assert audit["first_trainable_backbone_layer"] == "block_13_expand"
    assert audit["fine_tune_boundary_index"] == 116
    assert audit["trainable_backbone_layer_count"] == 25
    assert audit["frozen_batch_normalization_count"] == 52
    assert all(
        not layer.trainable
        for layer in backbone.layers
        if isinstance(layer, tf.keras.layers.BatchNormalization)
    )


def test_experiment_a_has_no_class_weights_and_macro_f1_checkpoint(tmp_path):
    policy = load_policy()
    callbacks = callback_policy(policy, tmp_path / "phase1-best.keras")
    assert CLASS_WEIGHTS is None
    assert callbacks["model_checkpoint"]["monitor"] == "val_macro_f1"
    assert callbacks["model_checkpoint"]["mode"] == "max"


def test_preflight_metadata_validation_rejects_test_access():
    payload = {
        "experiment_name": "agri-diagnose-v2-exp-a",
        "status": "GPU_NOT_AVAILABLE_FOR_TENSORFLOW",
        "training_authorized": False,
        "training_performed": False,
        "model_fit_called": False,
        "class_weights": None,
        "train_manifest": {},
        "validation_manifest": {},
        "macro_f1_validation": {},
        "phase1_model_audit": {},
        "phase2_model_audit": {},
        "internal_test": {"loaded": False, "evaluated": False},
        "plantdoc_test": {"loaded": False, "evaluated": False},
        "candidate_model_created": False,
    }
    validate_preflight_metadata(payload)
    payload["internal_test"]["loaded"] = True
    with pytest.raises(RuntimeError, match="internal TEST access"):
        validate_preflight_metadata(payload)
