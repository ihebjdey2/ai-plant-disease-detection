from __future__ import annotations

import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SELECTION_PATH = (
    PROJECT_ROOT / "training/config/model-v2-final-selection.json"
)
DOCUMENTATION_PATH = PROJECT_ROOT / "docs/model-v2-final-evaluation.md"
SELECTED_MODEL_SHA256 = (
    "bba4d044bcafbbee6dcd9f604e9c3f10c42f2531f17f21769b991540e36b8ca0"
)
INTERNAL_TEST_MANIFEST_SHA256 = (
    "f0df59c42268163d485feea0e54dd7780aa56fe08a7984ae7869a09c604a9151"
)
PLANTDOC_SOURCE_REVISION = "5467f6012d78d1c446145d5f582da6096f852ae8"


def load_selection() -> dict[str, object]:
    return json.loads(SELECTION_PATH.read_text(encoding="utf-8"))


def test_frozen_selection_identity_and_partition_are_exact():
    selection = load_selection()

    assert selection["schema"] == "agridiagnose.model-v2-final-selection"
    assert selection["version"] == 1
    assert selection["status"] == "FROZEN"
    assert selection["deployment_status"] == "NOT_DEPLOYED"
    assert selection["selected_experiment"] == "agri-diagnose-v2-exp-a"
    assert selection["rejected_experiment"] == "agri-diagnose-v2-exp-b"
    assert selection["selection_partition"] == "VALIDATION"
    assert selection["primary_selection_metric"] == "overall_macro_f1"
    assert selection["selected_model_sha256"] == SELECTED_MODEL_SHA256


def test_final_internal_test_metadata_is_locked():
    internal = load_selection()["internal_test"]

    assert internal["split"] == "TEST"
    assert internal["image_count"] == 7344
    assert internal["class_count"] == 39
    assert internal["plantdoc_test_contamination"] is False
    assert internal["manifest_sha256"] == INTERNAL_TEST_MANIFEST_SHA256
    assert internal["model_sha256"] == SELECTED_MODEL_SHA256
    assert internal["metrics"] == {
        "accuracy": 0.9543845315904139,
        "macro_f1": 0.9592043738824634,
        "weighted_f1": 0.954331728063991,
        "keras_loss": 0.15016824007034302,
    }
    assert internal["confidence"]["below_threshold_percent"] == (
        2.1377995642701526
    )


def test_external_plantdoc_metadata_is_partial_and_locked():
    plantdoc = load_selection()["external_plantdoc_test"]

    assert plantdoc["partial_benchmark"] is True
    assert plantdoc["source_revision"] == PLANTDOC_SOURCE_REVISION
    assert plantdoc["source_split"] == "TEST"
    assert plantdoc["official_test"] == {
        "total_image_count": 236,
        "original_label_count": 27,
        "corrupted_image_count": 0,
    }
    assert plantdoc["mapping"]["matched_label_count"] == 12
    assert plantdoc["mapping"]["excluded_label_count"] == 15
    assert plantdoc["prepared_subset"]["image_count"] == 99
    assert plantdoc["prepared_subset"]["evaluated_class_count"] == 12
    assert plantdoc["prepared_subset"]["exact_duplicates_removed"] == 0
    assert plantdoc["metrics"] == {
        "accuracy": 0.41414141414141414,
        "macro_f1": 0.45789013014076957,
        "weighted_f1": 0.4335358321873086,
    }
    assert plantdoc["confidence"]["below_threshold_percent"] == (
        34.34343434343434
    )
    assert plantdoc["experiment_b_test_evaluated"] is False


def test_validation_selection_values_and_bootstrap_are_frozen():
    validation = load_selection()["validation_results"]

    assert validation["experiment_a"]["overall_macro_f1"] == (
        0.9601076689427503
    )
    assert validation["experiment_b"]["overall_macro_f1"] == (
        0.9588616764404703
    )
    bootstrap = validation["paired_real_world_bootstrap"]
    assert bootstrap["confidence_interval_lower"] == -0.07292016725729139
    assert bootstrap["confidence_interval_upper"] == 0.016322318122805997
    assert bootstrap["resamples"] == 10000


def test_test_policy_preserves_selection_chronology_and_freeze():
    policy = load_selection()["test_policy"]

    assert policy == {
        "selection_completed_before_test": True,
        "internal_test_used_for_selection": False,
        "external_test_used_for_selection": False,
        "post_test_tuning_allowed": False,
        "experiment_b_test_evaluated": False,
    }


def test_documentation_distinguishes_internal_and_external_results():
    documentation = DOCUMENTATION_PATH.read_text(encoding="utf-8")

    assert "Model V2 is **frozen**" in documentation
    assert "95.44%" in documentation
    assert "41.41%" in documentation
    assert "99-image / 12-class partial external benchmark" in documentation
    assert "not a field-accuracy claim" in documentation
    assert "Experiment B was never evaluated on either TEST" in documentation
