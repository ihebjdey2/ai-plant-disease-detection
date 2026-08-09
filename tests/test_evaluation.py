from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from app.services.prediction_service import CLASS_NAMES
from scripts.evaluate_model import (
    DATASET_DIRECTORY_TO_CLASS,
    EvaluationError,
    audit_dataset,
    calculate_metrics,
    load_image_for_evaluation,
    validate_audit,
    validate_mapping,
    validate_model_output_count,
)


def save_test_image(path, color=(255, 128, 0)):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (6, 4), color=color).save(path)


def test_evaluation_mapping_exactly_matches_deployed_order():
    validate_mapping()

    assert len(DATASET_DIRECTORY_TO_CLASS) == 39
    assert list(DATASET_DIRECTORY_TO_CLASS.values()) == CLASS_NAMES
    assert DATASET_DIRECTORY_TO_CLASS["Background_without_leaves"] == CLASS_NAMES[4]
    assert "Pepper__bell___healthy" not in DATASET_DIRECTORY_TO_CLASS


def test_evaluation_preprocessing_matches_deployed_pipeline(tmp_path):
    image_path = tmp_path / "leaf.png"
    save_test_image(image_path)

    image = load_image_for_evaluation(image_path)

    assert image.shape == (224, 224, 3)
    assert image.dtype == np.float32
    np.testing.assert_allclose(image[0, 0], [1.0, 128.0 / 255.0, 0.0])


def test_dataset_audit_counts_valid_corrupt_and_missing_classes(tmp_path):
    save_test_image(tmp_path / "Apple___healthy" / "apple.jpg")
    save_test_image(tmp_path / "Background_without_leaves" / "background.webp")
    (tmp_path / "Apple___healthy" / "corrupt.png").write_bytes(b"not an image")

    audit = audit_dataset(tmp_path)

    assert audit.candidate_image_count == 3
    assert audit.valid_image_count == 2
    assert audit.classes_found == 2
    assert audit.minimum_samples_per_class == 1
    assert audit.maximum_samples_per_class == 1
    assert audit.images_per_class["Apple healthy"] == 1
    assert len(audit.corrupted_images) == 1
    assert len(audit.missing_classes) == 37
    with pytest.raises(EvaluationError, match="requires all 39 classes"):
        validate_audit(audit)
    validate_audit(audit, allow_subset=True)


def test_unknown_or_ambiguous_directory_is_rejected(tmp_path):
    save_test_image(tmp_path / "Pepper__bell___healthy" / "leaf.jpg")

    audit = audit_dataset(tmp_path)

    assert audit.unexpected_classes == ["Pepper__bell___healthy"]
    with pytest.raises(EvaluationError, match="Unknown class directories"):
        validate_audit(audit, allow_subset=True)


def test_corruption_cannot_leave_a_mapped_class_empty(tmp_path):
    directory = tmp_path / "Tomato___healthy"
    directory.mkdir()
    (directory / "corrupt.jpg").write_bytes(b"broken")

    audit = audit_dataset(tmp_path)

    with pytest.raises(EvaluationError, match="contain no valid images"):
        validate_audit(audit, allow_subset=True)


def test_model_must_have_exactly_39_outputs():
    assert validate_model_output_count(SimpleNamespace(output_shape=(None, 39))) == 39
    with pytest.raises(EvaluationError, match="requires exactly 39"):
        validate_model_output_count(SimpleNamespace(output_shape=(None, 38)))


def test_metrics_keep_confidence_analysis_separate_from_model_classes():
    y_true = np.asarray([3, 31, 4], dtype=np.int64)
    scores = np.zeros((3, 39), dtype=np.float32)
    scores[0, 3] = 0.90
    scores[1, 31] = 0.55
    scores[2, 4] = 0.80

    metrics, matrix = calculate_metrics(y_true, scores, confidence_threshold=60.0)

    assert metrics["accuracy"] == 1.0
    assert metrics["macro"]["f1"] == 1.0
    assert metrics["metric_scope"]["class_count"] == 3
    assert len(metrics["per_class"]) == 39
    assert metrics["confidence_analysis"]["uncertain_prediction_count"] == 1
    assert metrics["confidence_analysis"]["accuracy_at_or_above_threshold"] == 1.0
    assert metrics["background_class"]["precision"] == 1.0
    assert metrics["background_class"]["recall"] == 1.0
    assert matrix.shape == (39, 39)


def test_subset_metrics_do_not_fabricate_missing_background_results():
    y_true = np.asarray([0, 0, 21], dtype=np.int64)
    scores = np.zeros((3, 39), dtype=np.float32)
    scores[0, 0] = 0.9
    scores[1, 0] = 0.8
    scores[2, 21] = 0.7

    metrics, _ = calculate_metrics(y_true, scores, confidence_threshold=60.0)

    assert metrics["macro"]["f1"] == 1.0
    assert metrics["metric_scope"]["classes"] == [
        "Apple Apple scab",
        "Potato Early blight",
    ]
    assert metrics["background_class_evaluated"] is False
    assert metrics["background_class"] is None


def test_low_confidence_background_prediction_is_no_leaf_not_uncertain():
    y_true = np.asarray([0], dtype=np.int64)
    scores = np.zeros((1, 39), dtype=np.float32)
    scores[0, 4] = 0.5

    metrics, _ = calculate_metrics(y_true, scores, confidence_threshold=60.0)

    confidence = metrics["confidence_analysis"]
    assert confidence["below_threshold_count"] == 1
    assert confidence["no_leaf_prediction_count"] == 1
    assert confidence["no_leaf_below_threshold_count"] == 1
    assert confidence["uncertain_prediction_count"] == 0
