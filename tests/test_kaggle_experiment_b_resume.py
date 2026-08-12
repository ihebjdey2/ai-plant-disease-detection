from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest
import tensorflow as tf

import training.experiment_b_resume as resume_module
from training.experiment_b import load_experiment_b_policy
from training.experiment_b_resume import (
    EXPECTED_POLICY_SHA256_LF,
    EXPECTED_TAXONOMY_SHA256,
    EXPECTED_TRAIN_SHA256_LF,
    EXPECTED_VALIDATION_SHA256_LF,
    HISTORY_FIELDS,
    ResumeSafetyError,
    ResumablePersistentHistory,
    archive_restart_artifacts,
    build_resume_callbacks,
    fit_epoch_arguments,
    load_resume_checkpoint,
    plan_phase,
    read_phase_history,
    replay_plateau_state,
    validate_checkpoint_audit,
    validate_completion_marker,
    validate_resume_provenance,
    write_completion_marker,
)
from training.kaggle_experiment_b import (
    build_execution_config_b,
    load_execution_config_b,
)
from training.kaggle_runtime import SOURCE_ROOT_KEYS, build_execution_config


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def source_roots() -> dict[str, str]:
    return {key: f"/kaggle/input/{key}/data" for key in SOURCE_ROOT_KEYS}


def history_rows(count: int, *, best_macro_epoch: int | None = None):
    best_macro_epoch = best_macro_epoch or count
    rows = []
    for epoch in range(1, count + 1):
        macro = 0.5 + epoch * 0.01
        if best_macro_epoch != count and epoch == best_macro_epoch:
            macro = 0.99
        rows.append(
            {
                "epoch": epoch,
                "learning_rate": 0.001,
                "loss": 1.0 / epoch,
                "accuracy": 0.5 + epoch * 0.01,
                "macro_f1": 0.49 + epoch * 0.01,
                "val_loss": 1.1 / epoch,
                "val_accuracy": 0.48 + epoch * 0.01,
                "val_macro_f1": macro,
                "duration_seconds": 10.0,
            }
        )
    return rows


def write_history(path: Path, rows) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=HISTORY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return path


def make_phase_paths(tmp_path: Path, phase: str):
    candidate = tmp_path / "models/candidates/agri-diagnose-v2-exp-b"
    results = tmp_path / "agridiagnose-exp-b-results"
    candidate.mkdir(parents=True)
    results.mkdir()
    return candidate / f"{phase}-best.keras", results / f"{phase}-history.csv"


def runtime_payload() -> dict[str, object]:
    return {
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


def data_payload() -> dict[str, object]:
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
            "train": EXPECTED_TRAIN_SHA256_LF,
            "validation": EXPECTED_VALIDATION_SHA256_LF,
        },
        "taxonomy_audit": {
            "class_count": 39,
            "class_names_sha256": EXPECTED_TAXONOMY_SHA256,
            "background_class_index": 4,
            "background_class_name": "Background without leaves",
            "shared_with_experiment_a": True,
        },
        "policy_audit": {
            "baseline": "agri-diagnose-v2-exp-a",
            "primary_variable": "TRAIN_ONLY_AUGMENTATION_POLICY",
            "class_weights": None,
        },
        "augmentation_audit": {
            "enabled_for": ["TRAIN"],
            "validation_augmentation_enabled": False,
            "values": {"rotation_degrees": 20.0},
        },
        "training_performed": False,
        "internal_test_loaded": False,
        "plantdoc_test_loaded": False,
    }


def preflight_payload(data: dict[str, object]) -> dict[str, object]:
    runtime = runtime_payload()
    return {
        "status": "KAGGLE_TF215_GPU_EXPERIMENT_B_PREFLIGHT_PASSED",
        "experiment": "agri-diagnose-v2-exp-b",
        "batch_size": 32,
        "class_weights": None,
        "training_performed": False,
        "internal_test_loaded": False,
        "plantdoc_test_loaded": False,
        "runtime": runtime,
        "preflight": data,
        "phase1_model_audit": {
            "input_shape": [None, 224, 224, 3],
            "output_shape": [None, 39],
            "backbone_trainable": False,
            "initialization": "imagenet",
            "production_model_loaded": False,
            "total_parameters": 2_307_943,
            "trainable_parameters": 49_959,
            "non_trainable_parameters": 2_257_984,
        },
        "phase2_model_audit": {
            "total_backbone_layers": 154,
            "fine_tune_boundary_index": 116,
            "first_trainable_backbone_layer": "block_13_expand",
            "trainable_backbone_layer_count": 25,
            "frozen_backbone_layer_count": 129,
            "batch_normalization_layer_count": 52,
            "frozen_batch_normalization_count": 52,
            "total_parameters": 2_307_943,
            "trainable_parameters": 1_713_319,
            "non_trainable_parameters": 594_624,
        },
    }


def test_phase1_epochs_1_to_5_resume_at_epoch_6_of_10(tmp_path):
    checkpoint, history = make_phase_paths(tmp_path, "phase1")
    checkpoint.write_bytes(b"complete-keras-checkpoint")
    write_history(history, history_rows(5))

    plan = plan_phase(
        action="resume",
        phase="phase1",
        checkpoint_path=checkpoint,
        history_path=history,
        max_epochs=10,
        allow_completed=True,
        allow_fresh_during_resume=False,
    )

    assert plan.mode == "resumed"
    assert plan.initial_epoch == 5
    assert plan.history.completed_epochs == 5
    assert plan.history.best_macro_f1_epoch == 5
    assert fit_epoch_arguments(plan) == {"initial_epoch": 5, "epochs": 10}


def test_resumable_history_appends_without_duplicating_epochs_1_to_5(tmp_path):
    _, history = make_phase_paths(tmp_path, "phase1")
    original = history_rows(5)
    write_history(history, original)
    audit = read_phase_history(history, max_epochs=10)
    callback = ResumablePersistentHistory(audit)

    class Optimizer:
        learning_rate = tf.Variable(0.001, dtype=tf.float32)

    class Model:
        optimizer = Optimizer()

    callback.set_model(Model())
    callback.on_epoch_begin(5)
    callback.on_epoch_end(
        5,
        {
            "loss": 0.1,
            "accuracy": 0.9,
            "macro_f1": 0.89,
            "val_loss": 0.11,
            "val_accuracy": 0.88,
            "val_macro_f1": 0.87,
        },
    )
    with history.open("r", encoding="utf-8", newline="") as handle:
        persisted = list(csv.DictReader(handle))
    assert [int(row["epoch"]) for row in persisted] == [1, 2, 3, 4, 5, 6]
    for before, after in zip(original, persisted[:5]):
        assert {key: str(value) for key, value in before.items()} == after


def test_resume_callbacks_restore_history_plateau_and_checkpoint_threshold(tmp_path):
    checkpoint, history_path = make_phase_paths(tmp_path, "phase1")
    checkpoint.write_bytes(b"preserved-selected-checkpoint")
    write_history(history_path, history_rows(5))
    history = read_phase_history(history_path, max_epochs=10)
    callbacks = build_resume_callbacks(
        load_experiment_b_policy(),
        checkpoint_path=checkpoint,
        history=history,
    )
    assert [callback.__class__.__name__ for callback in callbacks] == [
        "ResumablePersistentHistory",
        "ResumableEarlyStopping",
        "ResumableReduceLROnPlateau",
        "ModelCheckpoint",
    ]

    model = tf.keras.Sequential(
        [tf.keras.layers.Input((1,)), tf.keras.layers.Dense(1)], name="callback-test"
    )
    model.compile(optimizer=tf.keras.optimizers.Adam(0.001), loss="mse")
    for callback in callbacks:
        callback.set_model(model)
        callback.on_train_begin({})

    early, plateau, selected = callbacks[1:]
    assert early.best == pytest.approx(history.best_val_loss)
    assert early.wait == 0
    assert early.best_weights is not None
    replayed = replay_plateau_state(history, load_experiment_b_policy())
    assert plateau.best == pytest.approx(replayed.best)
    assert plateau.wait == replayed.wait
    assert selected.best == pytest.approx(history.best_macro_f1)

    before = checkpoint.read_bytes()
    selected.on_epoch_end(5, {"val_macro_f1": history.best_macro_f1 - 0.01})
    assert checkpoint.read_bytes() == before


def test_resume_fails_when_best_macro_epoch_is_not_last(tmp_path):
    checkpoint, history = make_phase_paths(tmp_path, "phase1")
    checkpoint.write_bytes(b"checkpoint")
    write_history(history, history_rows(5, best_macro_epoch=4))
    with pytest.raises(ResumeSafetyError, match="not the last completed"):
        plan_phase(
            action="resume",
            phase="phase1",
            checkpoint_path=checkpoint,
            history_path=history,
            max_epochs=10,
            allow_completed=True,
            allow_fresh_during_resume=False,
        )


def test_resume_fails_when_checkpoint_is_absent(tmp_path):
    checkpoint, history = make_phase_paths(tmp_path, "phase1")
    write_history(history, history_rows(5))
    with pytest.raises(ResumeSafetyError, match="incomplete"):
        plan_phase(
            action="resume",
            phase="phase1",
            checkpoint_path=checkpoint,
            history_path=history,
            max_epochs=10,
            allow_completed=True,
            allow_fresh_during_resume=False,
        )


def test_resume_fails_for_non_contiguous_history(tmp_path):
    checkpoint, history = make_phase_paths(tmp_path, "phase1")
    checkpoint.write_bytes(b"checkpoint")
    rows = history_rows(5)
    rows[3]["epoch"] = 5
    write_history(history, rows)
    with pytest.raises(ResumeSafetyError, match="contiguous"):
        read_phase_history(history, max_epochs=10)


def test_resume_fails_when_history_already_reached_max_epochs(tmp_path):
    checkpoint, history = make_phase_paths(tmp_path, "phase2")
    checkpoint.write_bytes(b"checkpoint")
    rows = history_rows(20)
    for row in rows:
        row["learning_rate"] = 2e-5
    write_history(history, rows)
    with pytest.raises(ResumeSafetyError, match="max_epochs=20"):
        plan_phase(
            action="resume",
            phase="phase2",
            checkpoint_path=checkpoint,
            history_path=history,
            max_epochs=20,
            allow_completed=False,
            allow_fresh_during_resume=True,
        )


def test_tied_macro_f1_keeps_earlier_checkpoint_and_blocks_resume(tmp_path):
    checkpoint, history = make_phase_paths(tmp_path, "phase1")
    checkpoint.write_bytes(b"checkpoint")
    rows = history_rows(5)
    rows[3]["val_macro_f1"] = rows[4]["val_macro_f1"]
    write_history(history, rows)
    audit = read_phase_history(history, max_epochs=10)
    assert audit.best_macro_f1_epoch == 4
    with pytest.raises(ResumeSafetyError, match="not the last completed"):
        plan_phase(
            action="resume",
            phase="phase1",
            checkpoint_path=checkpoint,
            history_path=history,
            max_epochs=10,
            allow_completed=True,
            allow_fresh_during_resume=False,
        )


@pytest.mark.parametrize("bad_value", ["nan", "inf", "", "not-a-number"])
def test_resume_rejects_malformed_metric_values(tmp_path, bad_value):
    _, history = make_phase_paths(tmp_path, "phase1")
    rows = history_rows(5)
    rows[2]["val_macro_f1"] = bad_value
    write_history(history, rows)
    with pytest.raises(ResumeSafetyError, match="history"):
        read_phase_history(history, max_epochs=10)


def test_resume_never_falls_back_to_epoch_zero(tmp_path):
    checkpoint, history = make_phase_paths(tmp_path, "phase1")
    with pytest.raises(ResumeSafetyError, match="no compatible artifacts"):
        plan_phase(
            action="resume",
            phase="phase1",
            checkpoint_path=checkpoint,
            history_path=history,
            max_epochs=10,
            allow_completed=True,
            allow_fresh_during_resume=False,
        )


def test_phase2_resume_uses_the_same_next_epoch_rule(tmp_path):
    checkpoint, history = make_phase_paths(tmp_path, "phase2")
    checkpoint.write_bytes(b"checkpoint")
    rows = history_rows(3)
    for row in rows:
        row["learning_rate"] = 2e-5
    write_history(history, rows)
    plan = plan_phase(
        action="resume",
        phase="phase2",
        checkpoint_path=checkpoint,
        history_path=history,
        max_epochs=20,
        allow_completed=False,
        allow_fresh_during_resume=True,
    )
    assert fit_epoch_arguments(plan) == {"initial_epoch": 3, "epochs": 20}


def test_phase2_starts_fresh_only_after_caller_allows_it(tmp_path):
    checkpoint, history = make_phase_paths(tmp_path, "phase2")
    plan = plan_phase(
        action="resume",
        phase="phase2",
        checkpoint_path=checkpoint,
        history_path=history,
        max_epochs=20,
        allow_completed=False,
        allow_fresh_during_resume=True,
    )
    assert plan.mode == "fresh"
    assert fit_epoch_arguments(plan) == {"epochs": 20}


def test_fresh_and_explicit_restart_behavior_remain_distinct(tmp_path):
    checkpoint, history = make_phase_paths(tmp_path, "phase1")
    fresh = plan_phase(
        action="fail",
        phase="phase1",
        checkpoint_path=checkpoint,
        history_path=history,
        max_epochs=10,
        allow_completed=True,
        allow_fresh_during_resume=False,
    )
    assert fresh.mode == "fresh"
    assert fit_epoch_arguments(fresh) == {"epochs": 10}
    checkpoint.write_bytes(b"old")
    restarted = plan_phase(
        action="restart",
        phase="phase1",
        checkpoint_path=checkpoint,
        history_path=history,
        max_epochs=10,
        allow_completed=True,
        allow_fresh_during_resume=False,
    )
    assert restarted.mode == "restarted"
    assert fit_epoch_arguments(restarted) == {"epochs": 10}


def test_explicit_restart_archives_existing_lineage(tmp_path):
    candidate = tmp_path / "candidate" / "phase1-best.keras"
    history = tmp_path / "results" / "phase1-history.csv"
    marker = tmp_path / "results" / "phase1-complete.json"
    candidate.parent.mkdir()
    history.parent.mkdir()
    candidate.write_bytes(b"checkpoint")
    history.write_text("history", encoding="utf-8")
    marker.write_text("marker", encoding="utf-8")

    checkpoint_archive = archive_restart_artifacts((candidate,))
    result_archive = archive_restart_artifacts((history, marker))

    assert checkpoint_archive is not None
    assert result_archive is not None
    assert not candidate.exists() and not history.exists() and not marker.exists()
    assert (checkpoint_archive / candidate.name).read_bytes() == b"checkpoint"
    assert (result_archive / history.name).read_text(encoding="utf-8") == "history"
    assert (result_archive / marker.name).read_text(encoding="utf-8") == "marker"


def valid_checkpoint_audit(history, phase: str) -> dict[str, object]:
    policy = load_experiment_b_policy()
    learning_rate = replay_plateau_state(history, policy).learning_rate
    phase1 = phase == "phase1"
    trainable_shapes = [[1280, 39], [39]] if phase1 else [[index + 1] for index in range(15)]
    optimizer_shapes = [[]] + [shape for shape in trainable_shapes for _ in range(2)]
    return {
        "model_name": "agri_diagnose_v2_exp_b",
        "input_shape": [None, 224, 224, 3],
        "output_shape": [None, 39],
        "total_parameters": 2_307_943,
        "trainable_parameters": 49_959 if phase1 else 1_713_319,
        "non_trainable_parameters": 2_257_984 if phase1 else 594_624,
        "backbone_trainable": not phase1,
        "total_backbone_layers": 154,
        "first_trainable_backbone_layer": None if phase1 else "block_13_expand",
        "trainable_backbone_layer_count": 0 if phase1 else 25,
        "frozen_backbone_layer_count": 154 if phase1 else 129,
        "batch_normalization_layer_count": 52,
        "frozen_batch_normalization_count": 52,
        "optimizer_class": "Adam",
        "optimizer_iterations": history.best_macro_f1_epoch * 1840,
        "trainable_variable_count": len(trainable_shapes),
        "trainable_variable_shapes": trainable_shapes,
        "optimizer_variable_count": len(optimizer_shapes),
        "optimizer_variable_shapes": optimizer_shapes,
        "optimizer_learning_rate": learning_rate,
        "optimizer_beta_1": 0.9,
        "optimizer_beta_2": 0.999,
        "optimizer_epsilon": 1e-7,
        "loss_class": "SparseCategoricalCrossentropy",
    }


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("model_name", "wrong"),
        ("output_shape", [None, 38]),
        ("optimizer_class", "SGD"),
        ("optimizer_iterations", 0),
        ("optimizer_variable_count", 3),
    ],
)
def test_checkpoint_identity_mismatches_fail_closed(tmp_path, field, value):
    _, history_path = make_phase_paths(tmp_path, "phase1")
    write_history(history_path, history_rows(5))
    history = read_phase_history(history_path, max_epochs=10)
    audit = valid_checkpoint_audit(history, "phase1")
    audit[field] = value
    with pytest.raises(ResumeSafetyError, match="Checkpoint"):
        validate_checkpoint_audit(
            audit,
            phase="phase1",
            history=history,
            policy=load_experiment_b_policy(),
        )


def test_checkpoint_requires_every_adam_slot_shape(tmp_path):
    _, history_path = make_phase_paths(tmp_path, "phase1")
    write_history(history_path, history_rows(5))
    history = read_phase_history(history_path, max_epochs=10)
    audit = valid_checkpoint_audit(history, "phase1")
    audit["optimizer_variable_shapes"] = audit["optimizer_variable_shapes"][:-1]
    audit["optimizer_variable_count"] -= 1
    with pytest.raises(ResumeSafetyError, match="optimizer slots"):
        validate_checkpoint_audit(
            audit,
            phase="phase1",
            history=history,
            policy=load_experiment_b_policy(),
        )


def test_checkpoint_load_is_read_only_and_restores_complete_model(tmp_path, monkeypatch):
    checkpoint, history_path = make_phase_paths(tmp_path, "phase1")
    checkpoint.write_bytes(b"complete-keras-state")
    write_history(history_path, history_rows(5))
    history = read_phase_history(history_path, max_epochs=10)
    before = (checkpoint.read_bytes(), checkpoint.stat().st_mtime_ns)
    sentinel_model = object()
    monkeypatch.setattr(
        resume_module,
        "checkpoint_model_audit",
        lambda model, phase: valid_checkpoint_audit(history, phase),
    )
    calls = []

    def loader(path, **kwargs):
        calls.append((path, kwargs))
        return sentinel_model

    model, _, _ = load_resume_checkpoint(
        checkpoint,
        phase="phase1",
        history=history,
        policy=load_experiment_b_policy(),
        loader=loader,
    )
    assert model is sentinel_model
    assert calls[0][1]["compile"] is True
    assert calls[0][1]["safe_mode"] is True
    assert "MacroF1" in calls[0][1]["custom_objects"]
    assert (checkpoint.read_bytes(), checkpoint.stat().st_mtime_ns) == before


def test_corrupted_checkpoint_load_fails_without_fallback(tmp_path):
    checkpoint, history_path = make_phase_paths(tmp_path, "phase1")
    checkpoint.write_bytes(b"not-a-keras-archive")
    write_history(history_path, history_rows(5))
    history = read_phase_history(history_path, max_epochs=10)

    def broken_loader(*args, **kwargs):
        raise ValueError("corrupt")

    with pytest.raises(ResumeSafetyError, match="Could not restore"):
        load_resume_checkpoint(
            checkpoint,
            phase="phase1",
            history=history,
            policy=load_experiment_b_policy(),
            loader=broken_loader,
        )


def test_resume_provenance_validates_policy_manifests_taxonomy_and_test_locks(tmp_path):
    project = tmp_path / "project"
    policy_target = project / "training/config/model-v2-experiment-b-policy.json"
    policy_target.parent.mkdir(parents=True)
    policy_target.write_bytes(
        (PROJECT_ROOT / "training/config/model-v2-experiment-b-policy.json").read_bytes()
    )
    data = data_payload()
    preflight = tmp_path / "preflight.json"
    runtime = tmp_path / "environment-runtime.json"
    preflight.write_text(json.dumps(preflight_payload(data)), encoding="utf-8")
    runtime.write_text(json.dumps(runtime_payload()), encoding="utf-8")
    result = validate_resume_provenance(
        project_root=project,
        preflight_path=preflight,
        runtime_path=runtime,
        current_data=data,
        current_runtime=runtime_payload(),
    )
    assert result["policy_sha256_lf"] == EXPECTED_POLICY_SHA256_LF
    assert result["internal_test_loaded"] is False
    assert result["plantdoc_test_loaded"] is False


@pytest.mark.parametrize(
    ("location", "key", "value", "message"),
    [
        ("top", "experiment", "wrong", "identity"),
        ("data", "manifest_hashes", {"train": "wrong", "validation": "wrong"}, "manifest"),
        ("data", "taxonomy_audit", {}, "taxonomy"),
        ("top", "internal_test_loaded", True, "internal_test_loaded"),
        ("runtime", "plantdoc_test_loaded", True, "plantdoc_test_loaded"),
    ],
)
def test_resume_provenance_mismatch_fails_closed(
    tmp_path, location, key, value, message
):
    project = tmp_path / "project"
    policy_target = project / "training/config/model-v2-experiment-b-policy.json"
    policy_target.parent.mkdir(parents=True)
    policy_target.write_bytes(
        (PROJECT_ROOT / "training/config/model-v2-experiment-b-policy.json").read_bytes()
    )
    data = data_payload()
    stored = preflight_payload(data)
    runtime_data = runtime_payload()
    if location == "top":
        stored[key] = value
    elif location == "data":
        stored["preflight"][key] = value
    else:
        runtime_data[key] = value
    preflight = tmp_path / "preflight.json"
    runtime = tmp_path / "environment-runtime.json"
    preflight.write_text(json.dumps(stored), encoding="utf-8")
    runtime.write_text(json.dumps(runtime_data), encoding="utf-8")
    with pytest.raises(ResumeSafetyError, match=message):
        validate_resume_provenance(
            project_root=project,
            preflight_path=preflight,
            runtime_path=runtime,
            current_data=data,
            current_runtime=runtime_payload(),
        )


def test_resume_provenance_rejects_embedded_runtime_mismatch(tmp_path):
    project = tmp_path / "project"
    policy_target = project / "training/config/model-v2-experiment-b-policy.json"
    policy_target.parent.mkdir(parents=True)
    policy_target.write_bytes(
        (PROJECT_ROOT / "training/config/model-v2-experiment-b-policy.json").read_bytes()
    )
    data = data_payload()
    stored = preflight_payload(data)
    stored["runtime"]["gpu_smoke_device"] = "/job:localhost/device:GPU:1"
    preflight = tmp_path / "preflight.json"
    runtime = tmp_path / "environment-runtime.json"
    preflight.write_text(json.dumps(stored), encoding="utf-8")
    runtime.write_text(json.dumps(runtime_payload()), encoding="utf-8")

    with pytest.raises(ResumeSafetyError, match="Embedded preflight runtime"):
        validate_resume_provenance(
            project_root=project,
            preflight_path=preflight,
            runtime_path=runtime,
            current_data=data,
            current_runtime=runtime_payload(),
        )


def test_resume_provenance_rejects_changed_policy_bytes(tmp_path):
    project = tmp_path / "project"
    policy_target = project / "training/config/model-v2-experiment-b-policy.json"
    policy_target.parent.mkdir(parents=True)
    policy_target.write_bytes(
        (PROJECT_ROOT / "training/config/model-v2-experiment-b-policy.json").read_bytes()
        + b"\n"
    )
    data = data_payload()
    preflight = tmp_path / "preflight.json"
    runtime = tmp_path / "environment-runtime.json"
    preflight.write_text(json.dumps(preflight_payload(data)), encoding="utf-8")
    runtime.write_text(json.dumps(runtime_payload()), encoding="utf-8")
    with pytest.raises(ResumeSafetyError, match="policy hash changed"):
        validate_resume_provenance(
            project_root=project,
            preflight_path=preflight,
            runtime_path=runtime,
            current_data=data,
            current_runtime=runtime_payload(),
        )


def test_completion_marker_records_and_locks_terminal_reason(tmp_path):
    checkpoint, history_path = make_phase_paths(tmp_path, "phase1")
    checkpoint.write_bytes(b"selected-checkpoint")
    write_history(history_path, history_rows(7))
    history = read_phase_history(history_path, max_epochs=10)
    marker = history_path.with_name("phase1-complete.json")

    write_completion_marker(
        marker, phase="phase1", history=history, checkpoint=checkpoint
    )
    payload = json.loads(marker.read_text(encoding="utf-8"))
    assert payload["completion_reason"] == "early_stopping_returned_from_fit"
    assert payload["selected_macro_f1_epoch"] == 7
    validate_completion_marker(
        marker, phase="phase1", history=history, checkpoint=checkpoint
    )

    payload["completion_reason"] = "interrupted"
    marker.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ResumeSafetyError, match="completion marker"):
        validate_completion_marker(
            marker, phase="phase1", history=history, checkpoint=checkpoint
        )


def test_b_config_uses_explicit_fail_resume_restart_enum(tmp_path):
    for action in ("fail", "resume", "restart"):
        payload = build_execution_config_b(
            source_roots(), interrupted_phase_action=action
        )
        assert payload["interrupted_phase_action"] == action
        assert "restart_interrupted_phase" not in payload
        path = tmp_path / f"{action}.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        assert load_execution_config_b(path)["interrupted_phase_action"] == action
    with pytest.raises(ValueError, match="fail, resume, or restart"):
        build_execution_config_b(source_roots(), interrupted_phase_action="RESUME")


def test_legacy_b_config_maps_exact_boolean_without_changing_a(tmp_path):
    legacy = build_execution_config(
        source_roots(), restart_interrupted_phase=True
    )
    legacy["experiment"] = "agri-diagnose-v2-exp-b"
    path = tmp_path / "legacy.json"
    path.write_text(json.dumps(legacy), encoding="utf-8")
    assert load_execution_config_b(path)["interrupted_phase_action"] == "restart"
    assert build_execution_config(source_roots())["restart_interrupted_phase"] is False
    legacy["restart_interrupted_phase"] = "false"
    path.write_text(json.dumps(legacy), encoding="utf-8")
    with pytest.raises(RuntimeError, match="must be boolean"):
        load_execution_config_b(path)
