from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from training.data_pipeline import TrainingPolicyError
from training.kaggle_experiment_a import (
    EXPECTED_INTERNAL_TEST_SHA256,
    EXPECTED_TRAIN_COUNT,
    EXPECTED_VALIDATION_COUNT,
    approved_stack_status,
    kaggle_source_roots,
    load_train_validation,
    require_approved_stack,
    require_fresh_or_explicit_restart,
    require_kaggle_gpu,
    select_candidate,
    sha256_csv_with_canonical_crlf,
    verify_internal_test_lock,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = PROJECT_ROOT / "notebooks/kaggle_model_v2_experiment_a.ipynb"


def exact_audit() -> dict[str, object]:
    return {
        "python_version": "3.11.9",
        "tensorflow_version": "2.15.0",
        "keras_version": "2.15.0",
        "numpy_version": "1.26.4",
        "tensorflow_built_with_cuda": True,
        "tensorflow_gpu_devices": ["/physical_device:GPU:0"],
    }


def test_kaggle_expected_counts_and_locked_hash_are_fixed():
    assert EXPECTED_TRAIN_COUNT == 58_857
    assert EXPECTED_VALIDATION_COUNT == 7_362
    assert EXPECTED_INTERNAL_TEST_SHA256 == (
        "f0df59c42268163d485feea0e54dd7780aa56fe08a7984ae7869a09c604a9151"
    )


def test_kaggle_source_root_mapping_uses_existing_directories(tmp_path):
    config = {}
    for name in ("historical", "pldd_up", "seasonal_corn", "plantdoc_train", "banu_deb"):
        path = tmp_path / name
        path.mkdir()
        config[name] = path
    roots = kaggle_source_roots(config)
    assert set(roots) == {
        "Historical Mendeley 39-class source",
        "PLDD-UP",
        "Seasonal Corn Leaf Disease Dataset",
        "PlantDoc",
        "Potato Leaf Disease Dataset",
    }


def test_kaggle_source_root_rejects_locked_test_marker(tmp_path):
    config = {}
    for name in ("historical", "pldd_up", "seasonal_corn", "plantdoc_train", "banu_deb"):
        path = tmp_path / name
        path.mkdir()
        config[name] = path
    forbidden = tmp_path / "plantdoc-test"
    forbidden.mkdir()
    config["plantdoc_train"] = forbidden
    with pytest.raises(TrainingPolicyError, match="Locked TEST-like"):
        kaggle_source_roots(config)


def test_gpu_gate_never_falls_back_to_cpu():
    with pytest.raises(RuntimeError, match="KAGGLE_GPU_NOT_AVAILABLE"):
        require_kaggle_gpu(
            {"tensorflow_built_with_cuda": False, "tensorflow_gpu_devices": []}
        )
    require_kaggle_gpu(exact_audit())


def test_tensorflow_215_stack_gate_is_explicit():
    assert approved_stack_status(exact_audit())["approved_stack_exact"] is True
    require_approved_stack(exact_audit())
    incompatible = {**exact_audit(), "python_version": "3.12.1"}
    with pytest.raises(RuntimeError, match="KAGGLE_TF215_RUNTIME_INCOMPATIBLE"):
        require_approved_stack(incompatible)
    changed = {**exact_audit(), "tensorflow_version": "2.18.0"}
    with pytest.raises(RuntimeError, match="KAGGLE_APPROVED_STACK_REQUIRED"):
        require_approved_stack(changed)


def test_kaggle_loader_never_loads_internal_test():
    train, validation = load_train_validation(PROJECT_ROOT)
    assert len(train) == EXPECTED_TRAIN_COUNT
    assert len(validation) == EXPECTED_VALIDATION_COUNT
    assert {row.split for row in train} == {"TRAIN"}
    assert {row.split for row in validation} == {"VALIDATION"}
    assert verify_internal_test_lock(PROJECT_ROOT) == EXPECTED_INTERNAL_TEST_SHA256


def test_internal_test_hash_is_identical_for_crlf_and_lf(tmp_path):
    source = (
        PROJECT_ROOT / "training/datasets/manifests/dataset-v2-test.csv"
    ).read_bytes()
    lf = source.replace(b"\r\n", b"\n")
    crlf = lf.replace(b"\n", b"\r\n")
    lf_path = tmp_path / "test-lf.csv"
    crlf_path = tmp_path / "test-crlf.csv"
    lf_path.write_bytes(lf)
    crlf_path.write_bytes(crlf)
    assert sha256_csv_with_canonical_crlf(crlf_path) == EXPECTED_INTERNAL_TEST_SHA256
    assert sha256_csv_with_canonical_crlf(lf_path) == EXPECTED_INTERNAL_TEST_SHA256


def test_internal_test_hash_rejects_actual_record_modification(tmp_path):
    source = (
        PROJECT_ROOT / "training/datasets/manifests/dataset-v2-test.csv"
    ).read_bytes()
    modified = source.replace(b"FINAL_INTERNAL_TEST", b"FINAL_INTERNAL_TESX", 1)
    assert modified != source
    manifest_dir = tmp_path / "training/datasets/manifests"
    manifest_dir.mkdir(parents=True)
    modified_path = manifest_dir / "dataset-v2-test.csv"
    modified_path.write_bytes(modified)
    assert sha256_csv_with_canonical_crlf(modified_path) != EXPECTED_INTERNAL_TEST_SHA256
    with pytest.raises(TrainingPolicyError, match="hash changed"):
        verify_internal_test_lock(tmp_path)


def test_candidate_selection_uses_validation_tie_break_order():
    candidates = [
        {
            "name": "phase1",
            "partition": "VALIDATION",
            "val_macro_f1": 0.8,
            "val_loss": 0.5,
            "macro_recall": 0.79,
            "epoch": 4,
        },
        {
            "name": "phase2",
            "partition": "VALIDATION",
            "val_macro_f1": 0.8,
            "val_loss": 0.4,
            "macro_recall": 0.78,
            "epoch": 8,
        },
    ]
    assert select_candidate(candidates)["name"] == "phase2"
    with pytest.raises(TrainingPolicyError, match="VALIDATION-only"):
        select_candidate([{**candidates[0], "partition": "TEST"}])


def test_interrupted_phase_requires_explicit_restart(tmp_path):
    (tmp_path / "phase1-best.keras").touch()
    with pytest.raises(RuntimeError, match="INTERRUPTED_PHASE1_DETECTED"):
        require_fresh_or_explicit_restart(
            tmp_path, "phase1", restart_interrupted_phase=False
        )
    assert (
        require_fresh_or_explicit_restart(
            tmp_path, "phase1", restart_interrupted_phase=True
        )
        == "restarted"
    )


def test_notebook_is_valid_and_training_is_manual():
    notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    assert notebook["nbformat"] == 4
    code = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
    assert code
    first = "".join(code[0]["source"])
    all_code = "\n".join("".join(cell["source"]) for cell in code)
    assert "system_tf.config.list_physical_devices('GPU')" in first
    assert "System TensorFlow:" in first
    assert "START_TRAINING = False" in all_code
    assert "dataset-v2-test.csv" not in all_code
    assert "plant_disease_model.h5" not in all_code
    assert "/mnt/c/" not in all_code
    assert ".fit(" not in all_code
    assert "bootstrap_kaggle_tf215_runtime.py" in all_code
    assert "sys.executable, '-I'" in all_code
    assert "ISOLATED_ENV" in all_code
    assert "'PYTHONPATH', 'PYTHONHOME', 'VIRTUAL_ENV'" in all_code
    assert "PYTHONNOUSERSITE" in all_code
    assert all_code.count("env=ISOLATED_ENV") == 4
    assert "TF215_PYTHON" in all_code
    assert "'verify-runtime'" in all_code
    assert "'preflight'" in all_code
    assert "'--authorize-training'" in all_code
    revision = re.search(r"APPROVED_CODE_REVISION = '([0-9a-f]{40})'", all_code)
    assert revision is not None
    assert revision.group(1) == "7be0deb3c7983f240df2a4cb8dda886258e40dad"
