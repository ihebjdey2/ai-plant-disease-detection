from __future__ import annotations

import builtins
import json
import sys
import zipfile
from pathlib import Path

import pandas as pd
import pytest
import tensorflow as tf
from sklearn.metrics import confusion_matrix

from scripts.run_kaggle_model_v2_experiment_a import finalize_existing
import training.kaggle_experiment_a as kaggle_experiment
from training.kaggle_experiment_a import headless_pyplot
from training.kaggle_runtime import SOURCE_ROOT_KEYS, build_execution_config
from training.taxonomy import CLASS_NAMES


EXPECTED_FINAL_ARTIFACTS = {
    "validation-confusion-matrix.png",
    "learning-curve-loss.png",
    "learning-curve-accuracy.png",
    "learning-curve-macro-f1.png",
    "environment.json",
    "experiment.json",
    "preflight.json",
    "model-v2-exp-a-summary.json",
    "model-v2-exp-a-report.md",
}


def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def write_history(path: Path, rows: list[tuple[object, ...]]) -> None:
    header = (
        "epoch,learning_rate,loss,accuracy,macro_f1,val_loss,val_accuracy,"
        "val_macro_f1,duration_seconds\n"
    )
    content = header + "".join(",".join(map(str, row)) + "\n" for row in rows)
    path.write_text(content, encoding="utf-8")


@pytest.fixture
def completed_experiment(tmp_path):
    results_dir = tmp_path / "results"
    candidate_dir = tmp_path / "candidates"
    results_dir.mkdir()
    candidate_dir.mkdir()

    phase1 = candidate_dir / "phase1-best.keras"
    phase2 = candidate_dir / "phase2-best.keras"
    selected = results_dir / "agri-diagnose-v2-exp-a.keras"
    phase1.write_bytes(b"completed-phase-1-model")
    phase2.write_bytes(b"completed-selected-phase-2-model")
    selected.write_bytes(phase2.read_bytes())

    history_rows = [
        (0, 0.001, 0.4, 0.85, 0.84, 0.3, 0.88, 0.87, 10),
        (1, 0.001, 0.2, 0.93, 0.92, 0.18, 0.94, 0.93, 10),
    ]
    write_history(results_dir / "phase1-history.csv", history_rows)
    write_history(
        results_dir / "phase2-history.csv",
        [
            (0, 0.00001, 0.15, 0.95, 0.94, 0.14, 0.95, 0.95, 10),
            (1, 0.00001, 0.08, 0.98, 0.97, 0.12, 0.96, 0.96, 10),
        ],
    )

    true_indices = [0, 1, 1, 2, 2, 2]
    predicted_indices = [0, 1, 2, 2, 2, 1]
    metrics = {
        "loss": 0.15,
        "accuracy": 4 / 6,
        "overall_validation": {
            "image_count": 6,
            "accuracy": 4 / 6,
            "macro_precision": 0.7,
            "macro_recall": 0.7,
            "macro_f1": 0.7,
        },
        "real_world_validation": None,
        "true_indices": true_indices,
        "predicted_indices": predicted_indices,
    }
    write_json(results_dir / "validation-metrics.json", metrics)
    matrix = confusion_matrix(
        true_indices, predicted_indices, labels=list(range(len(CLASS_NAMES)))
    )
    pd.DataFrame(matrix, index=CLASS_NAMES, columns=CLASS_NAMES).to_csv(
        results_dir / "validation-confusion-matrix.csv"
    )

    runtime = {
        "status": "TF215_GPU_RUNTIME_VALIDATED",
        "python_version": "3.11.15",
        "tensorflow_version": "2.15.0",
        "keras_version": "2.15.0",
        "numpy_version": "1.26.4",
        "tensorflow_built_with_cuda": True,
        "tensorflow_gpu_devices": ["/physical_device:GPU:0"],
        "gpu_smoke_test_passed": True,
        "gpu_smoke_device": "/job:localhost/device:GPU:0",
        "training_performed": False,
        "internal_test_loaded": False,
        "plantdoc_test_loaded": False,
    }
    write_json(results_dir / "environment-runtime.json", runtime)

    data_preflight = {
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
        "internal_test_manifest_sha256": "locked-test-manifest-hash",
        "internal_test_loaded": False,
        "plantdoc_test_loaded": False,
        "training_performed": False,
    }
    preflight_report = tmp_path / "experiment-a-preflight.json"
    write_json(
        preflight_report,
        {
            "status": "KAGGLE_TF215_GPU_PREFLIGHT_PASSED",
            "preflight": data_preflight,
            "training_performed": False,
            "internal_test_loaded": False,
            "plantdoc_test_loaded": False,
        },
    )

    roots = {
        key: f"/kaggle/input/{key}-source/{key}-data"
        for key in SOURCE_ROOT_KEYS
    }
    config = build_execution_config(roots, start_training=True)
    config_path = tmp_path / "experiment-a-config.json"
    write_json(config_path, config)
    return {
        "results_dir": results_dir,
        "candidate_dir": candidate_dir,
        "selected": selected,
        "phase1": phase1,
        "phase2": phase2,
        "preflight_report": preflight_report,
        "config": config_path,
        "archive_base": tmp_path / "agridiagnose-exp-a-final",
    }


def test_headless_plotting_uses_agg_without_matplotlib_inline(monkeypatch, tmp_path):
    monkeypatch.setenv("MPLBACKEND", "module://matplotlib_inline.backend_inline")
    monkeypatch.delitem(sys.modules, "matplotlib_inline", raising=False)
    original_import = builtins.__import__

    def reject_inline(name, *args, **kwargs):
        if name.startswith("matplotlib_inline"):
            raise AssertionError("matplotlib_inline must not be imported")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", reject_inline)
    pyplot = headless_pyplot()
    figure, axis = pyplot.subplots()
    axis.plot([0, 1], [0, 1])
    output = tmp_path / "headless.png"
    figure.savefig(output)
    pyplot.close(figure)
    assert pyplot.get_backend().casefold() == "agg"
    assert output.is_file()


def test_finalize_existing_generates_reports_without_fit_or_model_changes(
    completed_experiment, monkeypatch
):
    def reject_neural_network_work(*args, **kwargs):
        raise AssertionError(
            "training, model loading, and inference are forbidden during finalization"
        )

    def reject_dataset_access(*args, **kwargs):
        raise AssertionError("datasets are forbidden during finalization")

    monkeypatch.setattr(tf.keras.Model, "fit", reject_neural_network_work)
    monkeypatch.setattr(tf.keras.Model, "evaluate", reject_neural_network_work)
    monkeypatch.setattr(tf.keras.Model, "predict", reject_neural_network_work)
    monkeypatch.setattr(tf.keras.models, "load_model", reject_neural_network_work)
    monkeypatch.setattr(
        kaggle_experiment, "build_kaggle_datasets", reject_dataset_access
    )
    monkeypatch.setattr(kaggle_experiment, "run_full_preflight", reject_dataset_access)
    monkeypatch.setattr(kaggle_experiment, "load_train_validation", reject_dataset_access)
    monkeypatch.setattr(kaggle_experiment, "verify_internal_test_lock", reject_dataset_access)
    model_paths = [
        completed_experiment["selected"],
        completed_experiment["phase1"],
        completed_experiment["phase2"],
    ]
    before = {
        path: (path.read_bytes(), path.stat().st_mtime_ns) for path in model_paths
    }
    archive = finalize_existing(
        completed_experiment["config"],
        completed_experiment["preflight_report"],
        results_dir=completed_experiment["results_dir"],
        candidate_dir=completed_experiment["candidate_dir"],
        archive_base=completed_experiment["archive_base"],
    )
    after = {
        path: (path.read_bytes(), path.stat().st_mtime_ns) for path in model_paths
    }
    assert after == before
    assert archive.is_file()
    for name in EXPECTED_FINAL_ARTIFACTS:
        assert (completed_experiment["results_dir"] / name).is_file()
    with zipfile.ZipFile(archive) as package:
        assert EXPECTED_FINAL_ARTIFACTS <= set(package.namelist())

    environment = json.loads(
        (completed_experiment["results_dir"] / "environment.json").read_text()
    )
    experiment = json.loads(
        (completed_experiment["results_dir"] / "experiment.json").read_text()
    )
    summary = json.loads(
        (
            completed_experiment["results_dir"] / "model-v2-exp-a-summary.json"
        ).read_text()
    )
    assert environment["training_performed"] is True
    assert environment["retraining_performed"] is False
    assert experiment["selected_phase"] == "phase2"
    assert experiment["internal_test_loaded"] is False
    assert experiment["plantdoc_test_loaded"] is False
    assert experiment["retraining_performed"] is False
    assert summary["test_sets_evaluated"] is False


def test_finalize_existing_fails_closed_when_artifact_is_missing(
    completed_experiment,
):
    (completed_experiment["results_dir"] / "validation-metrics.json").unlink()
    with pytest.raises(RuntimeError, match="Missing existing VALIDATION metrics"):
        finalize_existing(
            completed_experiment["config"],
            completed_experiment["preflight_report"],
            results_dir=completed_experiment["results_dir"],
            candidate_dir=completed_experiment["candidate_dir"],
            archive_base=completed_experiment["archive_base"],
        )


def test_finalize_existing_rejects_test_like_source(completed_experiment):
    payload = json.loads(completed_experiment["config"].read_text())
    payload["source_roots"]["plantdoc_train"] = (
        "/kaggle/input/plantdoc-test/plantdoc-train"
    )
    write_json(completed_experiment["config"], payload)
    with pytest.raises(ValueError, match="TEST-like source root is forbidden"):
        finalize_existing(
            completed_experiment["config"],
            completed_experiment["preflight_report"],
            results_dir=completed_experiment["results_dir"],
            candidate_dir=completed_experiment["candidate_dir"],
            archive_base=completed_experiment["archive_base"],
        )


def test_finalize_existing_rejects_preflight_with_test_access(completed_experiment):
    payload = json.loads(completed_experiment["preflight_report"].read_text())
    payload["preflight"]["internal_test_loaded"] = True
    write_json(completed_experiment["preflight_report"], payload)
    with pytest.raises(RuntimeError, match="INTERNAL TEST lock"):
        finalize_existing(
            completed_experiment["config"],
            completed_experiment["preflight_report"],
            results_dir=completed_experiment["results_dir"],
            candidate_dir=completed_experiment["candidate_dir"],
            archive_base=completed_experiment["archive_base"],
        )
