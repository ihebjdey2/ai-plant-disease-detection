from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.run_kaggle_model_v2_experiment_b import (
    KAGGLE_ARCHIVE_BASE,
    KAGGLE_CANDIDATE_DIR,
    KAGGLE_RESULTS_DIR,
    assemble_preflight_payload,
    require_training_authorization,
    run_validation_comparison,
)
from training.experiment_b import APPROVED_AUGMENTATION, EXPERIMENT_NAME
from training.kaggle_experiment_b import (
    EXPECTED_TAXONOMY_SHA256,
    EXPECTED_TRAIN_MANIFEST_SHA256,
    EXPECTED_VALIDATION_MANIFEST_SHA256,
    RESULT_FILENAMES,
    augmentation_audit,
    build_execution_config_b,
    ensure_experiment_b_output_paths,
    load_execution_config_b,
    load_experiment_b_policy,
    manifest_hashes,
    package_results_b,
    taxonomy_audit,
)
from training.kaggle_runtime import SOURCE_ROOT_KEYS


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def source_roots() -> dict[str, str]:
    return {
        key: f"/kaggle/input/{key}-source/{key}-data"
        for key in SOURCE_ROOT_KEYS
    }


def write_config(path: Path, *, start_training: bool) -> Path:
    payload = build_execution_config_b(
        source_roots(), start_training=start_training
    )
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def runtime_payload() -> dict[str, object]:
    return {
        "python_version": "3.11.15",
        "tensorflow_version": "2.15.0",
        "keras_version": "2.15.0",
        "numpy_version": "1.26.4",
        "tensorflow_built_with_cuda": True,
        "tensorflow_gpu_devices": ["/physical_device:GPU:0"],
        "gpu_smoke_test_passed": True,
        "gpu_smoke_device": "/job:localhost/device:GPU:0",
    }


def data_preflight() -> dict[str, object]:
    return {
        "train": {
            "expected": 58_857,
            "resolved": 58_857,
            "missing": 0,
            "unreadable": 0,
        },
        "validation": {
            "expected": 7_362,
            "resolved": 7_362,
            "missing": 0,
            "unreadable": 0,
        },
        "train_class_coverage": 39,
        "validation_class_coverage": 39,
        "manifest_hashes": {
            "train": EXPECTED_TRAIN_MANIFEST_SHA256,
            "validation": EXPECTED_VALIDATION_MANIFEST_SHA256,
        },
        "taxonomy_audit": taxonomy_audit(),
        "policy_audit": {
            "primary_variable": "TRAIN_ONLY_AUGMENTATION_POLICY",
            "class_weights": None,
        },
        "augmentation_audit": {
            "validation_augmentation_enabled": False,
            "values": APPROVED_AUGMENTATION,
        },
        "internal_test_loaded": False,
        "plantdoc_test_loaded": False,
        "training_performed": False,
    }


def phase_audits() -> tuple[dict[str, object], dict[str, object]]:
    return (
        {
            "input_shape": [None, 224, 224, 3],
            "output_shape": [None, 39],
            "backbone_trainable": False,
            "initialization": "imagenet",
            "production_model_loaded": False,
            "total_parameters": 2_307_943,
            "trainable_parameters": 49_959,
            "non_trainable_parameters": 2_257_984,
        },
        {
            "first_trainable_backbone_layer": "block_13_expand",
            "fine_tune_boundary_index": 116,
            "total_backbone_layers": 154,
            "trainable_backbone_layer_count": 25,
            "frozen_backbone_layer_count": 129,
            "batch_normalization_layer_count": 52,
            "frozen_batch_normalization_count": 52,
            "total_parameters": 2_307_943,
            "trainable_parameters": 1_713_319,
            "non_trainable_parameters": 594_624,
        },
    )


def test_b_execution_config_is_separate_locked_and_disabled_by_default(tmp_path):
    payload = build_execution_config_b(source_roots())
    assert payload["experiment"] == EXPERIMENT_NAME
    assert payload["batch_size"] == 32
    assert payload["start_training"] is False
    assert payload["interrupted_phase_action"] == "fail"
    assert "restart_interrupted_phase" not in payload
    assert payload["internal_test_loaded"] is False
    assert payload["plantdoc_test_loaded"] is False
    with pytest.raises(ValueError, match="batch size is locked to 32"):
        build_execution_config_b(source_roots(), batch_size=16)

    path = tmp_path / "config.json"
    path.write_text(json.dumps({**payload, "experiment": "agri-diagnose-v2-exp-a"}))
    with pytest.raises(RuntimeError, match="identity mismatch"):
        load_execution_config_b(path)


@pytest.mark.parametrize(
    ("config_authorized", "cli_authorized", "passes"),
    [(False, False, False), (False, True, False), (True, False, False), (True, True, True)],
)
def test_training_requires_both_authorizations(
    tmp_path, config_authorized, cli_authorized, passes
):
    path = write_config(
        tmp_path / f"config-{config_authorized}-{cli_authorized}.json",
        start_training=config_authorized,
    )
    if passes:
        assert require_training_authorization(
            path, authorize_training=cli_authorized
        )["start_training"] is True
    else:
        with pytest.raises(RuntimeError, match="TRAINING_DISABLED_BY_USER"):
            require_training_authorization(path, authorize_training=cli_authorized)


def test_b_source_config_rejects_both_test_families():
    for marker in ("internal-test", "internal_test", "plantdoc-test", "plantdoc_test"):
        roots = source_roots()
        roots["plantdoc_train"] = f"/kaggle/input/{marker}/data"
        with pytest.raises(ValueError, match="TEST-like"):
            build_execution_config_b(roots)


def test_b_output_paths_cannot_target_experiment_a_or_wrong_identity(tmp_path):
    ensure_experiment_b_output_paths(
        KAGGLE_CANDIDATE_DIR, KAGGLE_RESULTS_DIR, KAGGLE_ARCHIVE_BASE
    )
    with pytest.raises(RuntimeError, match="Experiment A artifacts"):
        ensure_experiment_b_output_paths(
            tmp_path / "models/agri-diagnose-v2-exp-a",
            tmp_path / "agridiagnose-exp-b-results",
            tmp_path / "agridiagnose-exp-b-results",
        )
    with pytest.raises(RuntimeError, match="candidate directory identity"):
        ensure_experiment_b_output_paths(
            tmp_path / "models/candidates/wrong-name",
            tmp_path / "agridiagnose-exp-b-results",
            tmp_path / "agridiagnose-exp-b-results",
        )


def test_b_manifest_hashes_match_finalized_a_portably():
    assert manifest_hashes(PROJECT_ROOT) == {
        "train": EXPECTED_TRAIN_MANIFEST_SHA256,
        "validation": EXPECTED_VALIDATION_MANIFEST_SHA256,
    }


def test_b_taxonomy_is_the_shared_locked_experiment_a_order():
    assert taxonomy_audit() == {
        "class_count": 39,
        "class_names_sha256": EXPECTED_TAXONOMY_SHA256,
        "background_class_index": 4,
        "background_class_name": "Background without leaves",
        "shared_with_experiment_a": True,
    }


def test_b_effective_augmentation_has_exact_approved_layers_and_values():
    policy, _ = load_experiment_b_policy(PROJECT_ROOT)
    audit = augmentation_audit(policy)
    assert audit["layer_order"] == [
        "RandomFlip",
        "RandomRotation",
        "RandomTranslation",
        "RandomZoom",
        "RandomBrightness",
        "RandomContrast",
    ]
    assert audit["layer_seeds"] == list(range(20260810, 20260816))
    assert audit["validation_augmentation_enabled"] is False
    assert audit["values"] == APPROVED_AUGMENTATION
    assert audit["output_clipping"] == [0.0, 1.0]


def test_b_preflight_payload_is_training_free_and_test_locked():
    phase1, phase2 = phase_audits()
    payload = assemble_preflight_payload(
        runtime=runtime_payload(),
        data=data_preflight(),
        phase1=phase1,
        phase2=phase2,
        batch_size=32,
    )
    assert payload["status"] == "KAGGLE_TF215_GPU_EXPERIMENT_B_PREFLIGHT_PASSED"
    assert payload["training_performed"] is False
    assert payload["start_training"] is False
    assert payload["class_weights"] is None
    assert payload["internal_test_loaded"] is False
    assert payload["plantdoc_test_loaded"] is False


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (("train", "resolved", 58_856), "TRAIN preflight"),
        (("validation", "unreadable", 1), "VALIDATION preflight"),
        ((None, "train_class_coverage", 38), "class coverage"),
        (("augmentation_audit", "validation_augmentation_enabled", True), "augmentation"),
        ((None, "internal_test_loaded", True), "safety flag"),
    ],
)
def test_b_preflight_fails_closed_on_invariant_mutation(mutation, message):
    data = data_preflight()
    section, key, value = mutation
    if section is None:
        data[key] = value
    else:
        data[section][key] = value
    phase1, phase2 = phase_audits()
    with pytest.raises(RuntimeError, match=message):
        assemble_preflight_payload(
            runtime=runtime_payload(),
            data=data,
            phase1=phase1,
            phase2=phase2,
            batch_size=32,
        )


def test_b_package_requires_every_b_artifact_and_never_an_a_model(tmp_path):
    results = tmp_path / "agridiagnose-exp-b-results"
    results.mkdir()
    for name in RESULT_FILENAMES:
        (results / name).write_text("test", encoding="utf-8")
    archive = package_results_b(results, tmp_path / "agridiagnose-exp-b-results")
    assert archive.is_file()
    assert "exp-b" in archive.name
    assert not any("exp-a" in name for name in RESULT_FILENAMES)
    (results / "validation-metrics.json").unlink()
    with pytest.raises(RuntimeError, match="incomplete"):
        package_results_b(results, tmp_path / "missing")


def test_b_runner_forces_agg_and_keeps_fit_behind_training_function():
    source = (
        PROJECT_ROOT / "scripts/run_kaggle_model_v2_experiment_b.py"
    ).read_text(encoding="utf-8")
    assert 'os.environ["MPLBACKEND"] = "Agg"' in source
    assert source.index('os.environ["MPLBACKEND"] = "Agg"') < source.index(
        "from scripts.run_kaggle_model_v2_experiment_a import"
    )
    assert source.count(".fit(") == 4
    assert source.count("**fit_epoch_arguments(") == 4
    assert source.index("require_training_authorization(") < source.index(
        "phase1_fit = model.fit("
    )
    training_source = source[source.index("def run_training(") :]
    assert training_source.index(
        "preflight = assemble_preflight_payload("
    ) < training_source.index("phase1_fit = model.fit(")
    assert callable(run_validation_comparison)
