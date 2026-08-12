from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

import scripts.run_kaggle_model_v2_experiment_b as runner
from training.experiment_b_resume import HISTORY_FIELDS, ResumeSafetyError


class PhaseOneFitObserved(RuntimeError):
    """Stop the orchestration test immediately after observing the resumed fit."""


def _experiment_paths(tmp_path: Path) -> tuple[Path, Path, Path]:
    candidate_dir = (
        tmp_path / "models/candidates/agri-diagnose-v2-exp-b"
    )
    results_dir = tmp_path / "agridiagnose-exp-b-results"
    candidate_dir.mkdir(parents=True)
    results_dir.mkdir(parents=True)
    return candidate_dir, results_dir, results_dir


def _write_phase1_history(path: Path, epochs: int = 5) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=HISTORY_FIELDS)
        writer.writeheader()
        for epoch in range(1, epochs + 1):
            writer.writerow(
                {
                    "epoch": epoch,
                    "learning_rate": 0.001,
                    "loss": 1.0 / epoch,
                    "accuracy": 0.50 + epoch * 0.01,
                    "macro_f1": 0.45 + epoch * 0.01,
                    "val_loss": 1.1 / epoch,
                    "val_accuracy": 0.48 + epoch * 0.01,
                    "val_macro_f1": 0.44 + epoch * 0.01,
                    "duration_seconds": 10.0,
                }
            )


def _authorized_config(action: str) -> dict[str, object]:
    return {
        "experiment": "agri-diagnose-v2-exp-b",
        "source_roots": {"approved": "/kaggle/input/approved"},
        "batch_size": 32,
        "start_training": True,
        "interrupted_phase_action": action,
        "internal_test_loaded": False,
        "plantdoc_test_loaded": False,
    }


def _patch_paths(monkeypatch, tmp_path: Path) -> tuple[Path, Path]:
    candidate_dir, results_dir, archive_base = _experiment_paths(tmp_path)
    monkeypatch.setattr(runner, "KAGGLE_CANDIDATE_DIR", candidate_dir)
    monkeypatch.setattr(runner, "KAGGLE_RESULTS_DIR", results_dir)
    monkeypatch.setattr(runner, "KAGGLE_ARCHIVE_BASE", archive_base)
    monkeypatch.setattr(
        runner, "ensure_experiment_b_output_paths", lambda *args, **kwargs: None
    )
    return candidate_dir, results_dir


def test_run_training_resumes_phase1_at_epoch_6_without_rebuilding_or_recompiling(
    tmp_path, monkeypatch
):
    candidate_dir, results_dir = _patch_paths(monkeypatch, tmp_path)
    checkpoint = candidate_dir / "phase1-best.keras"
    checkpoint.write_bytes(b"preserved-complete-keras-checkpoint")
    _write_phase1_history(results_dir / "phase1-history.csv", epochs=5)
    (results_dir / "preflight.json").write_text("{}", encoding="utf-8")

    policy = {"phase1": {"max_epochs": 10}, "phase2": {"max_epochs": 20}}
    monkeypatch.setattr(
        runner,
        "require_training_authorization",
        lambda *args, **kwargs: _authorized_config("resume"),
    )
    monkeypatch.setattr(
        runner, "load_experiment_b_policy", lambda *args: (policy, {})
    )
    monkeypatch.setattr(runner, "kaggle_source_roots", lambda roots: {})
    monkeypatch.setattr(runner, "verify_runtime", lambda output: {"runtime": True})
    monkeypatch.setattr(
        runner, "run_full_preflight_b", lambda *args, **kwargs: {"data": True}
    )
    monkeypatch.setattr(
        runner,
        "validate_resume_provenance",
        lambda **kwargs: {"status": "validated"},
    )
    train_dataset = object()
    validation_dataset = object()
    monkeypatch.setattr(
        runner,
        "build_kaggle_datasets_b",
        lambda *args, **kwargs: (
            train_dataset,
            validation_dataset,
            [],
            [],
        ),
    )
    monkeypatch.setattr(runner, "write_safe_json", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        runner, "build_resume_callbacks", lambda *args, **kwargs: ["resume-callback"]
    )

    forbidden_calls: list[str] = []

    def forbidden_build(*args, **kwargs):
        forbidden_calls.append("build_model")
        raise AssertionError("A resumed Phase 1 must not rebuild the model.")

    def forbidden_compile(*args, **kwargs):
        forbidden_calls.append("compile_phase1")
        raise AssertionError("A restored Phase 1 checkpoint must not be recompiled.")

    monkeypatch.setattr(runner, "build_model", forbidden_build)
    monkeypatch.setattr(runner, "compile_phase1", forbidden_compile)

    fit_calls: list[tuple[object, dict[str, object]]] = []

    class RestoredModel:
        def fit(self, dataset, **kwargs):
            fit_calls.append((dataset, kwargs))
            raise PhaseOneFitObserved

    monkeypatch.setattr(
        runner,
        "load_resume_checkpoint",
        lambda *args, **kwargs: (
            RestoredModel(),
            {"checkpoint": "compatible"},
            {"sha256": "preserved"},
        ),
    )

    with pytest.raises(PhaseOneFitObserved):
        runner.run_training(tmp_path / "config.json", authorize_training=True)

    assert forbidden_calls == []
    assert len(fit_calls) == 1
    dataset, kwargs = fit_calls[0]
    assert dataset is train_dataset
    assert kwargs["validation_data"] is validation_dataset
    assert kwargs["initial_epoch"] == 5
    assert kwargs["epochs"] == 10
    assert kwargs["callbacks"] == ["resume-callback"]
    assert kwargs["class_weight"] is None


def test_orphan_phase2_artifacts_fail_before_any_fit_or_runtime_work(
    tmp_path, monkeypatch
):
    candidate_dir, _ = _patch_paths(monkeypatch, tmp_path)
    (candidate_dir / "phase2-best.keras").write_bytes(b"orphan-phase2")

    policy = {"phase1": {"max_epochs": 10}, "phase2": {"max_epochs": 20}}
    monkeypatch.setattr(
        runner,
        "require_training_authorization",
        lambda *args, **kwargs: _authorized_config("fail"),
    )
    monkeypatch.setattr(
        runner, "load_experiment_b_policy", lambda *args: (policy, {})
    )

    downstream_calls: list[str] = []

    def forbidden_downstream(name):
        def fail(*args, **kwargs):
            downstream_calls.append(name)
            raise AssertionError(f"{name} must not run for orphan Phase 2 artifacts.")

        return fail

    monkeypatch.setattr(runner, "verify_runtime", forbidden_downstream("runtime"))
    monkeypatch.setattr(
        runner, "build_kaggle_datasets_b", forbidden_downstream("datasets")
    )
    monkeypatch.setattr(runner, "build_model", forbidden_downstream("build_model"))
    monkeypatch.setattr(
        runner, "load_resume_checkpoint", forbidden_downstream("checkpoint_load")
    )

    with pytest.raises(
        ResumeSafetyError,
        match="Phase 2 artifacts exist before Phase 1 is proven complete",
    ):
        runner.run_training(tmp_path / "config.json", authorize_training=True)

    assert downstream_calls == []
