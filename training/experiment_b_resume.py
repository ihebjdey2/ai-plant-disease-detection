"""Fail-closed, exact-enough resume primitives for Model V2 Experiment B.

Resume restores the complete Keras checkpoint (including Adam state), callback
counters that can be reconstructed from the persisted CSV, and absolute epoch
numbering. TensorFlow dataset/augmentation/Dropout RNG streams are not stored by
the historical run, so this deliberately does not claim bit-identical replay.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Sequence

import numpy as np
import tensorflow as tf

from training.experiment_a import callback_policy, parameter_audit
from training.experiment_b import EXPERIMENT_NAME
from training.kaggle_runtime import validate_runtime_payload
from training.metrics import MacroF1


HISTORY_FIELDS = (
    "epoch",
    "learning_rate",
    "loss",
    "accuracy",
    "macro_f1",
    "val_loss",
    "val_accuracy",
    "val_macro_f1",
    "duration_seconds",
)
EXPECTED_POLICY_SHA256_LF = (
    "029d3599f75daf9224680e0aa58ae66696fd08c08708ccea73057819ba928007"
)
EXPECTED_TRAIN_SHA256_LF = (
    "957d4acb4c097116099c57446733b3d70088bf083e7869aadd11e26caf70a915"
)
EXPECTED_VALIDATION_SHA256_LF = (
    "9c10de69e935324ee325667fab2902b372a144a722c4fc793d3b4f1afe01767e"
)
EXPECTED_TAXONOMY_SHA256 = (
    "2207c34ff2673bde7f36c53938cf5e6d97ca0652f21ef087be15680851ae87da"
)
EXPECTED_IMAGES_PER_EPOCH = 58_857
EXPECTED_BATCH_SIZE = 32
STEPS_PER_EPOCH = math.ceil(EXPECTED_IMAGES_PER_EPOCH / EXPECTED_BATCH_SIZE)
SAFETY_FLAGS = ("internal_test_loaded", "plantdoc_test_loaded")
RUNTIME_IDENTITY_FIELDS = (
    "python_version",
    "tensorflow_version",
    "keras_version",
    "numpy_version",
    "tensorflow_built_with_cuda",
    "tensorflow_gpu_devices",
    "gpu_smoke_test_passed",
    "gpu_smoke_device",
)


class ResumeSafetyError(RuntimeError):
    """Raised when exact-enough continuation cannot be proved safe."""


@dataclass(frozen=True)
class HistoryAudit:
    path: Path
    rows: tuple[dict[str, str], ...]
    numeric_rows: tuple[dict[str, float], ...]
    completed_epochs: int
    best_macro_f1_epoch: int
    best_macro_f1: float
    best_val_loss_epoch: int
    best_val_loss: float


@dataclass(frozen=True)
class PhasePlan:
    phase: str
    mode: str
    initial_epoch: int
    max_epochs: int
    checkpoint_path: Path
    history_path: Path
    completion_marker_path: Path
    history: HistoryAudit | None = None


@dataclass(frozen=True)
class PlateauState:
    best: float
    wait: int
    cooldown_counter: int
    learning_rate: float


def fit_epoch_arguments(plan: PhasePlan) -> dict[str, int]:
    arguments = {"epochs": int(plan.max_epochs)}
    if plan.mode == "resumed":
        if plan.initial_epoch <= 0:
            raise ResumeSafetyError("Resume cannot fall back to initial_epoch=0.")
        arguments["initial_epoch"] = int(plan.initial_epoch)
    return arguments


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_with_canonical_lf(path: Path) -> str:
    content = Path(path).read_bytes()
    canonical = content.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return hashlib.sha256(canonical).hexdigest()


def file_signature(path: Path) -> dict[str, int | str]:
    candidate = Path(path)
    if not candidate.is_file() or candidate.stat().st_size <= 0:
        raise ResumeSafetyError(f"Required artifact is missing or empty: {candidate.name}")
    stat = candidate.stat()
    return {
        "sha256": sha256_file(candidate),
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def archive_restart_artifacts(paths: Sequence[Path]) -> Path | None:
    """Move an explicitly restarted lineage aside before any new phase writes."""

    existing = [Path(path) for path in paths if Path(path).is_file()]
    if not existing:
        return None
    parent = existing[0].parent
    archive = parent / f"restart-archive-{int(time.time())}"
    suffix = 0
    while archive.exists():
        suffix += 1
        archive = parent / f"restart-archive-{int(time.time())}-{suffix}"
    archive.mkdir(parents=True)
    for source in existing:
        if source.parent != parent:
            destination = archive / source.parent.name / source.name
            destination.parent.mkdir(parents=True, exist_ok=True)
        else:
            destination = archive / source.name
        source.replace(destination)
    return archive


def _finite_number(value: str, field: str, row_number: int) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ResumeSafetyError(
            f"Malformed history {field} at CSV row {row_number}."
        ) from exc
    if not math.isfinite(number):
        raise ResumeSafetyError(
            f"Non-finite history {field} at CSV row {row_number}."
        )
    return number


def read_phase_history(path: Path, *, max_epochs: int) -> HistoryAudit:
    history_path = Path(path)
    if not history_path.is_file():
        raise ResumeSafetyError(f"Resume history is missing: {history_path.name}")
    try:
        with history_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if tuple(reader.fieldnames or ()) != HISTORY_FIELDS:
                raise ResumeSafetyError(
                    f"Malformed history header in {history_path.name}."
                )
            rows = [dict(row) for row in reader]
    except OSError as exc:
        raise ResumeSafetyError(
            f"Resume history cannot be read: {history_path.name}"
        ) from exc
    if not rows:
        raise ResumeSafetyError(f"Resume history is empty: {history_path.name}")

    numeric_rows: list[dict[str, float]] = []
    epochs: list[int] = []
    for row_number, row in enumerate(rows, start=2):
        raw_epoch = row.get("epoch", "")
        try:
            epoch = int(raw_epoch)
        except (TypeError, ValueError) as exc:
            raise ResumeSafetyError(
                f"Malformed history epoch at CSV row {row_number}."
            ) from exc
        if str(epoch) != str(raw_epoch).strip() or epoch <= 0:
            raise ResumeSafetyError(
                f"Malformed history epoch at CSV row {row_number}."
            )
        epochs.append(epoch)
        numeric = {"epoch": float(epoch)}
        for field in HISTORY_FIELDS[1:]:
            numeric[field] = _finite_number(row.get(field, ""), field, row_number)
        numeric_rows.append(numeric)

    expected_epochs = list(range(1, len(rows) + 1))
    if epochs != expected_epochs:
        raise ResumeSafetyError(
            f"History epochs must be contiguous 1..N in {history_path.name}; "
            f"found {epochs}."
        )
    if len(rows) > int(max_epochs):
        raise ResumeSafetyError(
            f"History exceeds the configured {max_epochs} epochs in {history_path.name}."
        )

    macro_values = [row["val_macro_f1"] for row in numeric_rows]
    loss_values = [row["val_loss"] for row in numeric_rows]
    best_macro_offset = int(np.argmax(macro_values))
    best_loss_offset = int(np.argmin(loss_values))
    return HistoryAudit(
        path=history_path,
        rows=tuple(rows),
        numeric_rows=tuple(numeric_rows),
        completed_epochs=len(rows),
        best_macro_f1_epoch=best_macro_offset + 1,
        best_macro_f1=float(macro_values[best_macro_offset]),
        best_val_loss_epoch=best_loss_offset + 1,
        best_val_loss=float(loss_values[best_loss_offset]),
    )


def validate_interrupted_history(audit: HistoryAudit, *, max_epochs: int) -> None:
    if audit.completed_epochs >= int(max_epochs):
        raise ResumeSafetyError(
            f"{audit.path.name} already reached max_epochs={max_epochs}; resume is forbidden."
        )
    if audit.best_macro_f1_epoch != audit.completed_epochs:
        raise ResumeSafetyError(
            "The checkpoint-selected val_macro_f1 epoch is not the last completed epoch."
        )
    # EarlyStopping restores the best val_loss weights. Historical weights from
    # an earlier val_loss epoch are unavailable, so exact-enough callback replay
    # is provable only when the loaded last-epoch checkpoint also owns this best.
    if audit.best_val_loss_epoch != audit.completed_epochs:
        raise ResumeSafetyError(
            "The best val_loss epoch is not the last completed epoch; "
            "EarlyStopping state cannot be restored safely."
        )


def plan_phase(
    *,
    action: str,
    phase: str,
    checkpoint_path: Path,
    history_path: Path,
    max_epochs: int,
    allow_completed: bool,
    allow_fresh_during_resume: bool,
) -> PhasePlan:
    if action not in {"fail", "resume", "restart"}:
        raise ResumeSafetyError(f"Unsupported interrupted phase action: {action!r}")
    checkpoint = Path(checkpoint_path)
    history = Path(history_path)
    marker = history.with_name(f"{phase}-complete.json")
    checkpoint_exists = checkpoint.is_file()
    history_exists = history.is_file()
    any_artifact = checkpoint_exists or history_exists or marker.is_file()

    if action == "fail":
        if any_artifact:
            raise ResumeSafetyError(
                f"INTERRUPTED_{phase.upper()}_DETECTED: action is fail."
            )
        return PhasePlan(phase, "fresh", 0, max_epochs, checkpoint, history, marker)

    if action == "restart":
        return PhasePlan(
            phase,
            "restarted" if any_artifact else "fresh",
            0,
            max_epochs,
            checkpoint,
            history,
            marker,
        )

    if not checkpoint_exists and not history_exists and not marker.is_file():
        if allow_fresh_during_resume:
            return PhasePlan(phase, "fresh", 0, max_epochs, checkpoint, history, marker)
        raise ResumeSafetyError(
            f"Resume requested for {phase}, but no compatible artifacts exist."
        )
    if not checkpoint_exists or not history_exists:
        raise ResumeSafetyError(
            f"Resume artifacts for {phase} are incomplete; history and checkpoint are required."
        )

    audit = read_phase_history(history, max_epochs=max_epochs)
    if marker.is_file():
        if not allow_completed:
            raise ResumeSafetyError(f"{phase} is already marked complete; resume is forbidden.")
        validate_completion_marker(marker, phase=phase, history=audit, checkpoint=checkpoint)
        return PhasePlan(
            phase,
            "completed",
            audit.completed_epochs,
            max_epochs,
            checkpoint,
            history,
            marker,
            audit,
        )
    if audit.completed_epochs == max_epochs and allow_completed:
        return PhasePlan(
            phase,
            "completed",
            audit.completed_epochs,
            max_epochs,
            checkpoint,
            history,
            marker,
            audit,
        )
    validate_interrupted_history(audit, max_epochs=max_epochs)
    return PhasePlan(
        phase,
        "resumed",
        audit.completed_epochs,
        max_epochs,
        checkpoint,
        history,
        marker,
        audit,
    )


def replay_plateau_state(
    audit: HistoryAudit,
    policy: Mapping[str, object],
    *,
    through_epoch: int | None = None,
) -> PlateauState:
    config = policy["callbacks"]["reduce_lr_on_plateau"]
    monitor = str(config["monitor"])
    if monitor != "val_loss":
        raise ResumeSafetyError("Experiment B ReduceLROnPlateau monitor changed.")
    factor = float(config["factor"])
    patience = int(config["patience"])
    min_lr = float(config["min_lr"])
    min_delta = float(config.get("min_delta", 1e-4))
    cooldown = int(config.get("cooldown", 0))
    phase = "phase1" if audit.path.name.startswith("phase1") else "phase2"
    learning_rate = float(policy[phase]["learning_rate"])
    best = math.inf
    wait = 0
    cooldown_counter = 0
    limit = int(through_epoch or audit.completed_epochs)

    for row in audit.numeric_rows[:limit]:
        row_lr = float(row["learning_rate"])
        if not math.isclose(row_lr, learning_rate, rel_tol=1e-5, abs_tol=1e-10):
            raise ResumeSafetyError(
                f"Persisted learning rate is inconsistent at epoch {int(row['epoch'])}."
            )
        current = float(row[monitor])
        if cooldown_counter > 0:
            cooldown_counter -= 1
            wait = 0
        if current < best - min_delta:
            best = current
            wait = 0
        elif cooldown_counter <= 0:
            wait += 1
            if wait >= patience and learning_rate > np.float32(min_lr):
                learning_rate = max(learning_rate * factor, min_lr)
                cooldown_counter = cooldown
                wait = 0
    return PlateauState(best, wait, cooldown_counter, learning_rate)


class ResumablePersistentHistory(tf.keras.callbacks.Callback):
    """Append absolute epochs atomically while preserving existing CSV rows."""

    def __init__(self, audit: HistoryAudit):
        super().__init__()
        self.path = audit.path
        self.rows = [dict(row) for row in audit.rows]
        self._started = 0.0

    def on_epoch_begin(self, epoch, logs=None):
        del epoch, logs
        self._started = time.perf_counter()

    def on_epoch_end(self, epoch, logs=None):
        expected_epoch = len(self.rows) + 1
        actual_epoch = int(epoch) + 1
        if actual_epoch != expected_epoch:
            raise ResumeSafetyError(
                f"History append expected epoch {expected_epoch}, got {actual_epoch}."
            )
        values = dict(logs or {})
        required_logs = set(HISTORY_FIELDS) - {
            "epoch",
            "learning_rate",
            "duration_seconds",
        }
        if not required_logs.issubset(values):
            raise ResumeSafetyError(
                f"History append is missing metrics: {sorted(required_logs - set(values))}"
            )
        learning_rate = float(
            tf.keras.backend.get_value(self.model.optimizer.learning_rate)
        )
        row: dict[str, object] = {
            "epoch": actual_epoch,
            "learning_rate": learning_rate,
            "duration_seconds": time.perf_counter() - self._started,
        }
        for key in required_logs:
            value = float(values[key])
            if not math.isfinite(value):
                raise ResumeSafetyError(f"History append metric {key} is non-finite.")
            row[key] = value
        prospective = [*self.rows, row]
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        try:
            with temporary.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=HISTORY_FIELDS)
                writer.writeheader()
                writer.writerows(prospective)
            temporary.replace(self.path)
        except OSError as exc:
            raise ResumeSafetyError(
                f"Could not append {self.path.name} atomically."
            ) from exc
        self.rows = prospective


class ResumableEarlyStopping(tf.keras.callbacks.EarlyStopping):
    def __init__(self, audit: HistoryAudit, **kwargs):
        self._audit = audit
        super().__init__(**kwargs)

    def on_train_begin(self, logs=None):
        super().on_train_begin(logs)
        self.best = self._audit.best_val_loss
        self.best_epoch = self._audit.best_val_loss_epoch - 1
        best_offset = self._audit.best_val_loss_epoch - 1
        self.wait = self._audit.completed_epochs - best_offset - 1
        self.best_weights = self.model.get_weights() if self.restore_best_weights else None


class ResumableReduceLROnPlateau(tf.keras.callbacks.ReduceLROnPlateau):
    def __init__(self, state: PlateauState, **kwargs):
        self._resume_state = state
        super().__init__(**kwargs)

    def on_train_begin(self, logs=None):
        super().on_train_begin(logs)
        self.best = self._resume_state.best
        self.wait = self._resume_state.wait
        self.cooldown_counter = self._resume_state.cooldown_counter


def build_resume_callbacks(
    policy: Mapping[str, object],
    *,
    checkpoint_path: Path,
    history: HistoryAudit,
) -> list[tf.keras.callbacks.Callback]:
    metadata = callback_policy(policy, checkpoint_path)
    early = dict(metadata["early_stopping"])
    reduce = dict(metadata["reduce_lr_on_plateau"])
    checkpoint = dict(metadata["model_checkpoint"])
    plateau = replay_plateau_state(history, policy)
    return [
        ResumablePersistentHistory(history),
        ResumableEarlyStopping(history, **early),
        ResumableReduceLROnPlateau(plateau, **reduce),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoint_path),
            monitor=checkpoint["monitor"],
            mode=checkpoint["mode"],
            save_best_only=checkpoint["save_best_only"],
            initial_value_threshold=history.best_macro_f1,
        ),
    ]


def _optimizer_variables(optimizer) -> list[object]:
    variables = optimizer.variables
    return list(variables() if callable(variables) else variables)


def checkpoint_model_audit(model, *, phase: str) -> dict[str, object]:
    nested = [
        layer
        for layer in model.layers
        if isinstance(layer, tf.keras.Model) and "mobilenetv2" in layer.name
    ]
    if len(nested) != 1:
        raise ResumeSafetyError("Checkpoint MobileNetV2 identity is incompatible.")
    backbone = nested[0]
    trainable = [layer for layer in backbone.layers if layer.trainable]
    batch_norm = [
        layer
        for layer in backbone.layers
        if isinstance(layer, tf.keras.layers.BatchNormalization)
    ]
    optimizer = model.optimizer
    if optimizer is None:
        raise ResumeSafetyError("Checkpoint optimizer state is missing.")
    optimizer_config = optimizer.get_config()
    learning_rate = float(tf.keras.backend.get_value(optimizer.learning_rate))
    trainable_variable_shapes = [list(variable.shape) for variable in model.trainable_variables]
    optimizer_variables = _optimizer_variables(optimizer)
    return {
        "phase": phase,
        "model_name": model.name,
        "input_shape": list(model.input_shape),
        "output_shape": list(model.output_shape),
        **parameter_audit(model),
        "backbone_trainable": bool(backbone.trainable),
        "total_backbone_layers": len(backbone.layers),
        "first_trainable_backbone_layer": trainable[0].name if trainable else None,
        "trainable_backbone_layer_count": len(trainable),
        "frozen_backbone_layer_count": len(backbone.layers) - len(trainable),
        "batch_normalization_layer_count": len(batch_norm),
        "frozen_batch_normalization_count": sum(
            not layer.trainable for layer in batch_norm
        ),
        "optimizer_class": optimizer.__class__.__name__,
        "optimizer_iterations": int(tf.keras.backend.get_value(optimizer.iterations)),
        "trainable_variable_count": len(model.trainable_variables),
        "trainable_variable_shapes": trainable_variable_shapes,
        "optimizer_variable_count": len(optimizer_variables),
        "optimizer_variable_shapes": [list(variable.shape) for variable in optimizer_variables],
        "optimizer_learning_rate": learning_rate,
        "optimizer_beta_1": float(optimizer_config.get("beta_1", math.nan)),
        "optimizer_beta_2": float(optimizer_config.get("beta_2", math.nan)),
        "optimizer_epsilon": float(optimizer_config.get("epsilon", math.nan)),
        "loss_class": model.loss.__class__.__name__ if model.loss is not None else None,
    }


def validate_checkpoint_audit(
    audit: Mapping[str, object],
    *,
    phase: str,
    history: HistoryAudit,
    policy: Mapping[str, object],
) -> None:
    expected_parameters = {
        "phase1": (2_307_943, 49_959, 2_257_984),
        "phase2": (2_307_943, 1_713_319, 594_624),
    }[phase]
    expected_trainable_variables = {"phase1": 2, "phase2": 15}[phase]
    common = (
        audit.get("model_name") == "agri_diagnose_v2_exp_b",
        audit.get("input_shape") == [None, 224, 224, 3],
        audit.get("output_shape") == [None, 39],
        (
            audit.get("total_parameters"),
            audit.get("trainable_parameters"),
            audit.get("non_trainable_parameters"),
        )
        == expected_parameters,
        audit.get("total_backbone_layers") == 154,
        audit.get("batch_normalization_layer_count") == 52,
        audit.get("optimizer_class") == "Adam",
        audit.get("trainable_variable_count") == expected_trainable_variables,
        audit.get("loss_class") == "SparseCategoricalCrossentropy",
        math.isclose(float(audit.get("optimizer_beta_1", math.nan)), 0.9),
        math.isclose(float(audit.get("optimizer_beta_2", math.nan)), 0.999),
        math.isclose(float(audit.get("optimizer_epsilon", math.nan)), 1e-7),
    )
    if not all(common):
        raise ResumeSafetyError("Checkpoint model/optimizer identity is incompatible.")

    trainable_shapes = audit.get("trainable_variable_shapes")
    optimizer_shapes = audit.get("optimizer_variable_shapes")
    if not isinstance(trainable_shapes, list) or not isinstance(optimizer_shapes, list):
        raise ResumeSafetyError("Checkpoint Adam slot-shape audit is missing.")
    expected_optimizer_count = 1 + (2 * expected_trainable_variables)
    expected_slot_shapes = sorted(
        tuple(shape) for shape in trainable_shapes for _ in range(2)
    )
    observed_slot_shapes = sorted(tuple(shape) for shape in optimizer_shapes[1:])
    if (
        audit.get("optimizer_variable_count") != expected_optimizer_count
        or len(optimizer_shapes) != expected_optimizer_count
        or optimizer_shapes[0] != []
        or observed_slot_shapes != expected_slot_shapes
    ):
        raise ResumeSafetyError("Checkpoint Adam optimizer slots are incomplete.")
    if phase == "phase1":
        phase_compatible = (
            audit.get("backbone_trainable") is False
            and audit.get("first_trainable_backbone_layer") is None
            and audit.get("trainable_backbone_layer_count") == 0
            and audit.get("frozen_backbone_layer_count") == 154
            and audit.get("frozen_batch_normalization_count") == 52
        )
    else:
        phase_compatible = (
            audit.get("backbone_trainable") is True
            and audit.get("first_trainable_backbone_layer") == "block_13_expand"
            and audit.get("trainable_backbone_layer_count") == 25
            and audit.get("frozen_backbone_layer_count") == 129
            and audit.get("frozen_batch_normalization_count") == 52
        )
    if not phase_compatible:
        raise ResumeSafetyError(f"Checkpoint {phase} trainability is incompatible.")

    expected_iterations = history.best_macro_f1_epoch * STEPS_PER_EPOCH
    if audit.get("optimizer_iterations") != expected_iterations:
        raise ResumeSafetyError(
            "Checkpoint optimizer iteration count does not match the selected epoch."
        )
    expected_plateau = replay_plateau_state(
        history, policy, through_epoch=history.best_macro_f1_epoch
    )
    if not math.isclose(
        float(audit.get("optimizer_learning_rate", math.nan)),
        expected_plateau.learning_rate,
        rel_tol=1e-5,
        abs_tol=1e-10,
    ):
        raise ResumeSafetyError("Checkpoint optimizer learning rate is incompatible.")


def load_resume_checkpoint(
    checkpoint_path: Path,
    *,
    phase: str,
    history: HistoryAudit,
    policy: Mapping[str, object],
    loader: Callable[..., object] | None = None,
):
    path = Path(checkpoint_path)
    if path.name != f"{phase}-best.keras" or "agri-diagnose-v2-exp-b" not in (
        path.resolve().as_posix().casefold()
    ):
        raise ResumeSafetyError("Checkpoint path does not belong to Experiment B.")
    before = file_signature(path)
    load = loader or tf.keras.models.load_model
    try:
        model = load(
            path,
            custom_objects={"MacroF1": MacroF1},
            compile=True,
            safe_mode=True,
        )
    except Exception as exc:
        raise ResumeSafetyError(
            f"Could not restore the complete {phase} Keras checkpoint."
        ) from exc
    if file_signature(path) != before:
        raise ResumeSafetyError("Checkpoint changed while it was being validated.")
    audit = checkpoint_model_audit(model, phase=phase)
    validate_checkpoint_audit(audit, phase=phase, history=history, policy=policy)
    return model, audit, before


def _read_json(path: Path, label: str) -> dict[str, object]:
    candidate = Path(path)
    if not candidate.is_file():
        raise ResumeSafetyError(f"Existing {label} is missing: {candidate.name}")
    try:
        payload = json.loads(candidate.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ResumeSafetyError(f"Existing {label} is invalid.") from exc
    if not isinstance(payload, dict):
        raise ResumeSafetyError(f"Existing {label} must be a JSON object.")
    return payload


def _require_test_locks(payload: Mapping[str, object], label: str) -> None:
    for flag in SAFETY_FLAGS:
        if payload.get(flag) is not False:
            raise ResumeSafetyError(f"{label} must explicitly set {flag}=false.")
    if "test_sets_evaluated" in payload and payload.get("test_sets_evaluated") is not False:
        raise ResumeSafetyError(f"{label} violated test_sets_evaluated=false.")


def validate_resume_provenance(
    *,
    project_root: Path,
    preflight_path: Path,
    runtime_path: Path,
    current_data: Mapping[str, object],
    current_runtime: Mapping[str, object],
) -> dict[str, object]:
    root = Path(project_root)
    policy_path = root / "training/config/model-v2-experiment-b-policy.json"
    policy_sha = sha256_with_canonical_lf(policy_path)
    if policy_sha != EXPECTED_POLICY_SHA256_LF:
        raise ResumeSafetyError("Experiment B policy hash changed.")

    stored = _read_json(preflight_path, "Experiment B preflight")
    stored_runtime = _read_json(runtime_path, "Experiment B runtime audit")
    if (
        stored.get("status") != "KAGGLE_TF215_GPU_EXPERIMENT_B_PREFLIGHT_PASSED"
        or stored.get("experiment") != EXPERIMENT_NAME
        or stored.get("batch_size") != EXPECTED_BATCH_SIZE
        or stored.get("class_weights") is not None
        or stored.get("training_performed") is not False
    ):
        raise ResumeSafetyError("Existing preflight identity is incompatible.")
    _require_test_locks(stored, "preflight")
    _require_test_locks(stored_runtime, "runtime audit")
    validate_runtime_payload(stored_runtime)
    validate_runtime_payload(current_runtime)
    for key in RUNTIME_IDENTITY_FIELDS:
        if stored_runtime.get(key) != current_runtime.get(key):
            raise ResumeSafetyError(f"Runtime {key} changed since the interrupted run.")

    data = stored.get("preflight")
    if not isinstance(data, Mapping):
        raise ResumeSafetyError("Existing preflight data audit is missing.")
    _require_test_locks(data, "preflight data")
    _require_test_locks(current_data, "current preflight data")
    if data.get("training_performed") is not False or current_data.get(
        "training_performed"
    ) is not False:
        raise ResumeSafetyError("Preflight training safety flag is incompatible.")

    expected_manifests = {
        "train": EXPECTED_TRAIN_SHA256_LF,
        "validation": EXPECTED_VALIDATION_SHA256_LF,
    }
    if data.get("manifest_hashes") != expected_manifests or current_data.get(
        "manifest_hashes"
    ) != expected_manifests:
        raise ResumeSafetyError("TRAIN or VALIDATION manifest hash changed.")
    expected_taxonomy = {
        "class_count": 39,
        "class_names_sha256": EXPECTED_TAXONOMY_SHA256,
        "background_class_index": 4,
        "background_class_name": "Background without leaves",
        "shared_with_experiment_a": True,
    }
    if data.get("taxonomy_audit") != expected_taxonomy or current_data.get(
        "taxonomy_audit"
    ) != expected_taxonomy:
        raise ResumeSafetyError("Experiment B taxonomy changed.")
    for key in ("policy_audit", "augmentation_audit"):
        if data.get(key) != current_data.get(key):
            raise ResumeSafetyError(f"Experiment B {key} changed.")
    expected_partition_counts = {"train": 58_857, "validation": 7_362}
    for partition, expected_count in expected_partition_counts.items():
        for label, payload in (("stored", data), ("current", current_data)):
            audit = payload.get(partition)
            if not isinstance(audit, Mapping) or any(
                audit.get(key) != expected
                for key, expected in (
                    ("expected", expected_count),
                    ("resolved", expected_count),
                    ("missing", 0),
                    ("unreadable", 0),
                )
            ):
                raise ResumeSafetyError(
                    f"{label} {partition.upper()} preflight counts are incompatible."
                )
        if data.get(f"{partition}_class_coverage") != 39 or current_data.get(
            f"{partition}_class_coverage"
        ) != 39:
            raise ResumeSafetyError(
                f"{partition.upper()} class coverage changed from 39 classes."
            )

    phase1 = stored.get("phase1_model_audit")
    phase2 = stored.get("phase2_model_audit")
    expected_phase1 = {
        "input_shape": [None, 224, 224, 3],
        "output_shape": [None, 39],
        "backbone_trainable": False,
        "initialization": "imagenet",
        "production_model_loaded": False,
        "total_parameters": 2_307_943,
        "trainable_parameters": 49_959,
        "non_trainable_parameters": 2_257_984,
    }
    expected_phase2 = {
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
    }
    if not isinstance(phase1, Mapping) or any(
        phase1.get(key) != value for key, value in expected_phase1.items()
    ):
        raise ResumeSafetyError("Stored Phase 1 model identity is incompatible.")
    if not isinstance(phase2, Mapping) or any(
        phase2.get(key) != value for key, value in expected_phase2.items()
    ):
        raise ResumeSafetyError("Stored Phase 2 model identity is incompatible.")

    embedded_runtime = stored.get("runtime")
    if not isinstance(embedded_runtime, Mapping):
        raise ResumeSafetyError("Existing preflight runtime audit is missing.")
    _require_test_locks(embedded_runtime, "embedded runtime")
    validate_runtime_payload(embedded_runtime)
    for key in RUNTIME_IDENTITY_FIELDS:
        if embedded_runtime.get(key) != stored_runtime.get(key):
            raise ResumeSafetyError(
                f"Embedded preflight runtime {key} does not match the runtime audit."
            )
    return {
        "status": "EXPERIMENT_B_RESUME_PROVENANCE_VALIDATED",
        "experiment": EXPERIMENT_NAME,
        "policy_sha256_lf": policy_sha,
        "manifest_hashes": expected_manifests,
        "taxonomy_sha256": EXPECTED_TAXONOMY_SHA256,
        "legacy_policy_proof": (
            "semantic preflight identity plus immutable current policy hash; "
            "the historical preflight did not record the raw B policy hash"
        ),
        "internal_test_loaded": False,
        "plantdoc_test_loaded": False,
        "test_sets_evaluated": False,
    }


def write_completion_marker(
    path: Path,
    *,
    phase: str,
    history: HistoryAudit,
    checkpoint: Path,
) -> None:
    max_epochs = {"phase1": 10, "phase2": 20}.get(phase)
    if max_epochs is None:
        raise ResumeSafetyError(f"Unknown completion-marker phase: {phase!r}")
    completion_reason = (
        "max_epochs_reached"
        if history.completed_epochs == max_epochs
        else "early_stopping_returned_from_fit"
    )
    payload = {
        "experiment": EXPERIMENT_NAME,
        "phase": phase,
        "completed_epochs": history.completed_epochs,
        "completion_reason": completion_reason,
        "selected_macro_f1_epoch": history.best_macro_f1_epoch,
        "history_sha256": sha256_file(history.path),
        "checkpoint_sha256": sha256_file(checkpoint),
        "policy_sha256_lf": EXPECTED_POLICY_SHA256_LF,
        "train_manifest_sha256_lf": EXPECTED_TRAIN_SHA256_LF,
        "validation_manifest_sha256_lf": EXPECTED_VALIDATION_SHA256_LF,
        "taxonomy_sha256": EXPECTED_TAXONOMY_SHA256,
        "internal_test_loaded": False,
        "plantdoc_test_loaded": False,
        "test_sets_evaluated": False,
    }
    target = Path(path)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(target)


def validate_completion_marker(
    path: Path,
    *,
    phase: str,
    history: HistoryAudit,
    checkpoint: Path,
) -> None:
    payload = _read_json(path, f"{phase} completion marker")
    _require_test_locks(payload, f"{phase} completion marker")
    max_epochs = {"phase1": 10, "phase2": 20}.get(phase)
    if max_epochs is None:
        raise ResumeSafetyError(f"Unknown completion-marker phase: {phase!r}")
    expected = {
        "experiment": EXPERIMENT_NAME,
        "phase": phase,
        "completed_epochs": history.completed_epochs,
        "completion_reason": (
            "max_epochs_reached"
            if history.completed_epochs == max_epochs
            else "early_stopping_returned_from_fit"
        ),
        "selected_macro_f1_epoch": history.best_macro_f1_epoch,
        "history_sha256": sha256_file(history.path),
        "checkpoint_sha256": sha256_file(checkpoint),
        "policy_sha256_lf": EXPECTED_POLICY_SHA256_LF,
        "train_manifest_sha256_lf": EXPECTED_TRAIN_SHA256_LF,
        "validation_manifest_sha256_lf": EXPECTED_VALIDATION_SHA256_LF,
        "taxonomy_sha256": EXPECTED_TAXONOMY_SHA256,
    }
    if any(payload.get(key) != value for key, value in expected.items()):
        raise ResumeSafetyError(f"{phase} completion marker is incompatible.")


__all__ = [
    "EXPECTED_POLICY_SHA256_LF",
    "HISTORY_FIELDS",
    "HistoryAudit",
    "PhasePlan",
    "ResumeSafetyError",
    "ResumablePersistentHistory",
    "archive_restart_artifacts",
    "build_resume_callbacks",
    "checkpoint_model_audit",
    "file_signature",
    "fit_epoch_arguments",
    "load_resume_checkpoint",
    "plan_phase",
    "read_phase_history",
    "replay_plateau_state",
    "validate_checkpoint_audit",
    "validate_completion_marker",
    "validate_interrupted_history",
    "validate_resume_provenance",
    "write_completion_marker",
]
