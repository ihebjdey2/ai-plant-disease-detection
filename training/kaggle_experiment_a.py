from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import tensorflow as tf
import keras
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)

from training.taxonomy import CLASS_NAMES
from scripts.run_model_v2_experiment_a import (
    audit_paths,
    validate_macro_f1,
    validate_tensor_pipeline,
)
from training.data_pipeline import (
    ManifestRecord,
    TrainingPolicyError,
    build_tf_dataset,
    load_development_manifest,
    load_local_path_aliases,
    load_policy,
    set_experiment_seeds,
)
from training.experiment_a import callback_policy


EXPERIMENT_NAME = "agri-diagnose-v2-exp-a"
EXPECTED_TRAIN_COUNT = 58_857
EXPECTED_VALIDATION_COUNT = 7_362
EXPECTED_INTERNAL_TEST_SHA256 = (
    "f0df59c42268163d485feea0e54dd7780aa56fe08a7984ae7869a09c604a9151"
)
SOURCE_ROOT_KEYS = {
    "historical": "Historical Mendeley 39-class source",
    "pldd_up": "PLDD-UP",
    "seasonal_corn": "Seasonal Corn Leaf Disease Dataset",
    "plantdoc_train": "PlantDoc",
    "banu_deb": "Potato Leaf Disease Dataset",
}
FORBIDDEN_INPUT_MARKERS = ("plantdoc-test", "plantdoc_test", "internal-test")
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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_csv_with_canonical_crlf(path: Path) -> str:
    """Hash CSV bytes after changing line endings only to canonical CRLF."""
    content = Path(path).read_bytes()
    without_crlf = content.replace(b"\r\n", b"\n")
    canonical = without_crlf.replace(b"\n", b"\r\n")
    return hashlib.sha256(canonical).hexdigest()


def write_json(path: Path, payload: object) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def headless_pyplot():
    """Return pyplot with a deterministic non-interactive backend."""
    os.environ["MPLBACKEND"] = "Agg"
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    return plt


def runtime_audit() -> dict[str, object]:
    gpus = tf.config.list_physical_devices("GPU")
    command = shutil.which("nvidia-smi")
    nvidia_smi = (
        subprocess.run(
            [
                command,
                "--query-gpu=name,driver_version,memory.total,memory.free",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        if command
        else None
    )
    gpu_rows = []
    if nvidia_smi is not None and nvidia_smi.returncode == 0:
        for row in nvidia_smi.stdout.splitlines():
            values = [value.strip() for value in row.split(",")]
            if len(values) == 4:
                gpu_rows.append(
                    {
                        "name": values[0],
                        "driver_version": values[1],
                        "memory_total_mib": int(values[2]),
                        "memory_free_mib": int(values[3]),
                    }
                )
    return {
        "os": platform.platform(),
        "python_version": platform.python_version(),
        "tensorflow_version": tf.__version__,
        "keras_version": keras.__version__,
        "numpy_version": np.__version__,
        "tensorflow_built_with_cuda": bool(tf.test.is_built_with_cuda()),
        "tensorflow_gpu_devices": [device.name for device in gpus],
        "nvidia_smi_available": nvidia_smi is not None and nvidia_smi.returncode == 0,
        "nvidia_smi_error": (
            None
            if nvidia_smi is not None and nvidia_smi.returncode == 0
            else (nvidia_smi.stderr.strip()[:500] if nvidia_smi is not None else "not found")
        ),
        "gpus": gpu_rows,
    }


def require_kaggle_gpu(audit: Mapping[str, object]) -> None:
    if not audit.get("tensorflow_built_with_cuda") or not audit.get(
        "tensorflow_gpu_devices"
    ):
        raise RuntimeError(
            "KAGGLE_GPU_NOT_AVAILABLE: open Kaggle Notebook Settings and set "
            "Accelerator to GPU, then restart and rerun the runtime audit."
        )


def approved_stack_status(audit: Mapping[str, object]) -> dict[str, object]:
    python_parts = tuple(int(part) for part in str(audit["python_version"]).split(".")[:2])
    python_compatible = (3, 9) <= python_parts < (3, 12)
    exact = (
        python_parts == (3, 11)
        and str(audit["tensorflow_version"]).startswith("2.15.")
        and str(audit.get("keras_version") or "").startswith("2.15.")
        and str(audit["numpy_version"]) == "1.26.4"
    )
    return {
        "python_tf215_compatible": python_compatible,
        "approved_stack_exact": exact,
        "requires_manual_pinned_install": python_compatible and not exact,
    }


def require_approved_stack(audit: Mapping[str, object]) -> None:
    status = approved_stack_status(audit)
    if not status["python_tf215_compatible"]:
        raise RuntimeError(
            "KAGGLE_TF215_RUNTIME_INCOMPATIBLE: this Python runtime cannot safely "
            "run the approved TensorFlow 2.15 stack."
        )
    if not status["approved_stack_exact"]:
        raise RuntimeError(
            "KAGGLE_APPROVED_STACK_REQUIRED: inspect the audit, install only "
            "tensorflow==2.15.0 keras==2.15.0 numpy==1.26.4, restart the session, "
            "and rerun both gates."
        )


def configure_gpu_memory_growth() -> list[str]:
    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        raise RuntimeError("KAGGLE_GPU_NOT_AVAILABLE")
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as exc:
            raise RuntimeError(
                "GPU memory growth must be configured before TensorFlow initializes the GPU."
            ) from exc
    return [gpu.name for gpu in gpus]


def kaggle_source_roots(config: Mapping[str, str | Path]) -> dict[str, Path]:
    missing_keys = sorted(set(SOURCE_ROOT_KEYS) - set(config))
    extra_keys = sorted(set(config) - set(SOURCE_ROOT_KEYS))
    if missing_keys or extra_keys:
        raise TrainingPolicyError(
            f"Kaggle source-root keys mismatch; missing={missing_keys}, extra={extra_keys}."
        )
    roots: dict[str, Path] = {}
    for short_name, source_name in SOURCE_ROOT_KEYS.items():
        root = Path(config[short_name]).expanduser().resolve()
        normalized = root.as_posix().casefold()
        if any(marker in normalized for marker in FORBIDDEN_INPUT_MARKERS):
            raise TrainingPolicyError(
                f"Locked TEST-like Kaggle input is forbidden for {short_name}."
            )
        if not root.is_dir():
            raise TrainingPolicyError(f"Kaggle source root is unavailable: {short_name}")
        roots[source_name] = root
    return roots


def load_train_validation(project_root: Path) -> tuple[list[ManifestRecord], list[ManifestRecord]]:
    project_root = Path(project_root)
    train_path = project_root / "training/datasets/manifests/dataset-v2-train.csv"
    validation_path = (
        project_root / "training/datasets/manifests/dataset-v2-validation.csv"
    )
    if "test" in train_path.name.casefold() or "test" in validation_path.name.casefold():
        raise TrainingPolicyError("Locked TEST manifest path supplied to Kaggle workflow.")
    train = load_development_manifest(train_path, training=True)
    validation = load_development_manifest(validation_path, training=False)
    if len(train) != EXPECTED_TRAIN_COUNT or len(validation) != EXPECTED_VALIDATION_COUNT:
        raise TrainingPolicyError("Authoritative Kaggle TRAIN/VALIDATION count mismatch.")
    if {row.target_index for row in train} != set(range(39)):
        raise TrainingPolicyError("Kaggle TRAIN coverage must remain 39/39.")
    if {row.target_index for row in validation} != set(range(39)):
        raise TrainingPolicyError("Kaggle VALIDATION coverage must remain 39/39.")
    return train, validation


def verify_internal_test_lock(project_root: Path) -> str:
    path = Path(project_root) / "training/datasets/manifests/dataset-v2-test.csv"
    digest = sha256_csv_with_canonical_crlf(path)
    if digest != EXPECTED_INTERNAL_TEST_SHA256:
        raise TrainingPolicyError("Locked INTERNAL TEST manifest hash changed.")
    return digest


def run_full_preflight(
    project_root: Path,
    roots: Mapping[str, Path],
    *,
    verification_workers_note: bool = True,
) -> dict[str, object]:
    del verification_workers_note
    project_root = Path(project_root)
    policy = load_policy(
        project_root / "training/config/model-v2-training-policy.json"
    )
    seed = set_experiment_seeds(policy)
    train, validation = load_train_validation(project_root)
    aliases = load_local_path_aliases(
        {
            "PlantDoc": project_root / "training/datasets/manifests/plantdoc-train.csv",
            "Potato Leaf Disease Dataset": project_root
            / "training/datasets/manifests/potato-banu-deb-originals.csv",
        }
    )
    train_audit = audit_paths(train, roots, aliases)
    validation_audit = audit_paths(validation, roots, aliases)
    for name, audit in (("TRAIN", train_audit), ("VALIDATION", validation_audit)):
        if (
            audit["missing"]
            or audit["unreadable"]
            or audit["resolved"] != audit["expected"]
        ):
            raise TrainingPolicyError(f"Kaggle {name} preflight failed: {audit}")
    tensor_audit = validate_tensor_pipeline(
        train, validation, policy, roots, aliases
    )
    macro_f1 = validate_macro_f1()
    return {
        "seed": seed,
        "train": train_audit,
        "validation": validation_audit,
        "train_class_coverage": len({row.target_index for row in train}),
        "validation_class_coverage": len({row.target_index for row in validation}),
        "tensor_pipeline": tensor_audit,
        "macro_f1": macro_f1,
        "internal_test_manifest_sha256": verify_internal_test_lock(project_root),
        "internal_test_loaded": False,
        "plantdoc_test_loaded": False,
        "training_performed": False,
    }


def build_kaggle_datasets(
    project_root: Path,
    roots: Mapping[str, Path],
    *,
    batch_size: int = 32,
):
    project_root = Path(project_root)
    policy = load_policy(project_root / "training/config/model-v2-training-policy.json")
    train, validation = load_train_validation(project_root)
    aliases = load_local_path_aliases(
        {
            "PlantDoc": project_root / "training/datasets/manifests/plantdoc-train.csv",
            "Potato Leaf Disease Dataset": project_root
            / "training/datasets/manifests/potato-banu-deb-originals.csv",
        }
    )
    return (
        build_tf_dataset(
            train,
            policy,
            roots,
            training=True,
            batch_size=batch_size,
            local_path_aliases=aliases,
        ),
        build_tf_dataset(
            validation,
            policy,
            roots,
            training=False,
            batch_size=batch_size,
            local_path_aliases=aliases,
        ),
        train,
        validation,
    )


class PersistentHistory(tf.keras.callbacks.Callback):
    """Atomically persist one phase history row after every completed epoch."""

    def __init__(self, path: Path):
        super().__init__()
        self.path = Path(path)
        self.rows: list[dict[str, object]] = []
        self._started = 0.0

    def on_epoch_begin(self, epoch, logs=None):
        del epoch, logs
        self._started = time.perf_counter()

    def on_epoch_end(self, epoch, logs=None):
        values = dict(logs or {})
        learning_rate = float(tf.keras.backend.get_value(self.model.optimizer.learning_rate))
        row = {
            "epoch": int(epoch) + 1,
            "learning_rate": learning_rate,
            "duration_seconds": time.perf_counter() - self._started,
        }
        row.update(
            {
                key: float(values[key]) if key in values else None
                for key in HISTORY_FIELDS
                if key not in {"epoch", "learning_rate", "duration_seconds"}
            }
        )
        self.rows.append(row)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=HISTORY_FIELDS)
            writer.writeheader()
            writer.writerows(self.rows)
        temporary.replace(self.path)


def build_phase_callbacks(
    policy: Mapping[str, object],
    *,
    checkpoint_path: Path,
    history_path: Path,
) -> list[tf.keras.callbacks.Callback]:
    metadata = callback_policy(policy, checkpoint_path)
    early = metadata["early_stopping"]
    reduce = metadata["reduce_lr_on_plateau"]
    checkpoint = metadata["model_checkpoint"]
    return [
        PersistentHistory(history_path),
        tf.keras.callbacks.EarlyStopping(**early),
        tf.keras.callbacks.ReduceLROnPlateau(**reduce),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoint_path),
            monitor=checkpoint["monitor"],
            mode=checkpoint["mode"],
            save_best_only=checkpoint["save_best_only"],
        ),
    ]


def detect_existing_phase_artifacts(output_dir: Path) -> dict[str, list[str]]:
    output_dir = Path(output_dir)
    return {
        phase: sorted(path.name for path in output_dir.glob(f"{phase}-*"))
        for phase in ("phase1", "phase2")
    }


def require_fresh_or_explicit_restart(
    output_dir: Path,
    phase: str,
    *,
    restart_interrupted_phase: bool,
    history_path: Path | None = None,
) -> str:
    existing = detect_existing_phase_artifacts(output_dir).get(phase, [])
    if history_path is not None and Path(history_path).is_file():
        existing.append(Path(history_path).name)
    if existing and not restart_interrupted_phase:
        raise RuntimeError(
            f"INTERRUPTED_{phase.upper()}_DETECTED: exact resume is not guaranteed. "
            "Download existing artifacts, then explicitly approve a phase restart."
        )
    return "restarted" if existing else "fresh"


def validation_report(
    records: Sequence[ManifestRecord], score_matrix: np.ndarray
) -> dict[str, object]:
    if not records or {record.split for record in records} != {"VALIDATION"}:
        raise TrainingPolicyError("Candidate evaluation accepts VALIDATION records only.")
    scores = np.asarray(score_matrix)
    if scores.shape != (len(records), len(CLASS_NAMES)):
        raise TrainingPolicyError("Candidate score matrix does not match 39-class VALIDATION.")
    true = np.asarray([record.target_index for record in records], dtype=np.int32)
    predicted = np.argmax(scores, axis=1)

    def calculate(indices: np.ndarray, labels: Sequence[int]) -> dict[str, object]:
        y_true = true[indices]
        y_pred = predicted[indices]
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, labels=labels, zero_division=0
        )
        macro = precision_recall_fscore_support(
            y_true, y_pred, labels=labels, average="macro", zero_division=0
        )
        weighted = precision_recall_fscore_support(
            y_true, y_pred, labels=labels, average="weighted", zero_division=0
        )
        return {
            "image_count": int(len(indices)),
            "supported_class_count": len(set(int(value) for value in y_true)),
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "macro_precision": float(macro[0]),
            "macro_recall": float(macro[1]),
            "macro_f1": float(macro[2]),
            "weighted_precision": float(weighted[0]),
            "weighted_recall": float(weighted[1]),
            "weighted_f1": float(weighted[2]),
            "per_class": [
                {
                    "target_index": int(index),
                    "target_class": CLASS_NAMES[index],
                    "precision": float(precision[offset]),
                    "recall": float(recall[offset]),
                    "f1": float(f1[offset]),
                    "support": int(support[offset]),
                }
                for offset, index in enumerate(labels)
            ],
        }

    overall = np.arange(len(records), dtype=np.int32)
    real_world = np.asarray(
        [index for index, row in enumerate(records) if row.source_domain == "REAL_WORLD"],
        dtype=np.int32,
    )
    supported_real = sorted(set(int(true[index]) for index in real_world))
    return {
        "overall_validation": calculate(overall, list(range(39))),
        "real_world_validation": (
            calculate(real_world, supported_real) if len(real_world) else None
        ),
        "true_indices": true.tolist(),
        "predicted_indices": predicted.tolist(),
    }


def select_candidate(candidates: Sequence[Mapping[str, object]]) -> Mapping[str, object]:
    if not candidates:
        raise ValueError("At least one validation-only candidate is required.")
    for candidate in candidates:
        if candidate.get("partition") != "VALIDATION":
            raise TrainingPolicyError("Candidate selection is VALIDATION-only.")
    return max(
        candidates,
        key=lambda row: (
            float(row["val_macro_f1"]),
            -float(row["val_loss"]),
            float(row["macro_recall"]),
            -int(row.get("selection_epoch", row["epoch"])),
        ),
    )


def best_history_row(path: Path) -> dict[str, object]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise RuntimeError(f"No completed epochs were persisted in {Path(path).name}.")
    best = max(
        rows,
        key=lambda row: (
            float(row["val_macro_f1"]),
            -float(row["val_loss"]),
            -int(row["epoch"]),
        ),
    )
    return {
        key: int(value) if key == "epoch" else float(value)
        for key, value in best.items()
        if value not in {None, ""}
    }


def plot_learning_curves(
    phase1_history: Path, phase2_history: Path, output_dir: Path
) -> list[Path]:
    import pandas as pd

    plt = headless_pyplot()

    phase1 = pd.read_csv(phase1_history)
    phase2 = pd.read_csv(phase2_history)
    phase2 = phase2.copy()
    phase2["epoch"] = phase2["epoch"] + int(phase1["epoch"].max())
    combined = pd.concat([phase1, phase2], ignore_index=True)
    boundary = int(phase1["epoch"].max()) + 0.5
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    for metric, validation_metric, filename in (
        ("loss", "val_loss", "learning-curve-loss.png"),
        ("accuracy", "val_accuracy", "learning-curve-accuracy.png"),
        ("macro_f1", "val_macro_f1", "learning-curve-macro-f1.png"),
    ):
        figure, axis = plt.subplots(figsize=(9, 5))
        axis.plot(combined["epoch"], combined[metric], label=f"TRAIN {metric}")
        axis.plot(
            combined["epoch"], combined[validation_metric], label=f"VALIDATION {metric}"
        )
        axis.axvline(boundary, color="black", linestyle="--", label="Phase boundary")
        axis.set_xlabel("Epoch")
        axis.set_ylabel(metric)
        axis.legend()
        axis.grid(alpha=0.2)
        figure.tight_layout()
        output = output_dir / filename
        figure.savefig(output, dpi=160)
        plt.close(figure)
        outputs.append(output)
    return outputs


def save_confusion_artifacts(
    report: Mapping[str, object],
    output_dir: Path,
    *,
    preserve_existing_csv: bool = False,
) -> tuple[Path, Path]:
    import pandas as pd

    plt = headless_pyplot()

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    matrix = confusion_matrix(
        report["true_indices"], report["predicted_indices"], labels=list(range(39))
    )
    csv_path = output_dir / "validation-confusion-matrix.csv"
    png_path = output_dir / "validation-confusion-matrix.png"
    matrix_frame = pd.DataFrame(matrix, index=CLASS_NAMES, columns=CLASS_NAMES)
    if preserve_existing_csv:
        if not csv_path.is_file():
            raise RuntimeError(
                "Existing validation confusion-matrix CSV is required for recovery."
            )
        existing = pd.read_csv(csv_path, index_col=0)
        if (
            list(existing.index) != list(CLASS_NAMES)
            or list(existing.columns) != list(CLASS_NAMES)
            or not np.array_equal(existing.to_numpy(), matrix)
        ):
            raise RuntimeError(
                "Existing validation confusion-matrix CSV does not match "
                "validation-metrics.json."
            )
    else:
        matrix_frame.to_csv(csv_path)
    figure, axis = plt.subplots(figsize=(18, 16))
    image = axis.imshow(matrix, cmap="Blues")
    axis.set_title("Experiment A - VALIDATION confusion matrix")
    axis.set_xlabel("Predicted")
    axis.set_ylabel("True")
    axis.set_xticks(range(39), CLASS_NAMES, rotation=90, fontsize=6)
    axis.set_yticks(range(39), CLASS_NAMES, fontsize=6)
    figure.colorbar(image, ax=axis)
    figure.tight_layout()
    figure.savefig(png_path, dpi=160)
    plt.close(figure)
    return csv_path, png_path


def major_confusion_pairs(
    report: Mapping[str, object], *, limit: int = 15
) -> list[dict[str, object]]:
    matrix = confusion_matrix(
        report["true_indices"], report["predicted_indices"], labels=list(range(39))
    )
    pairs: list[dict[str, object]] = []
    for true_index in range(39):
        for predicted_index in range(39):
            count = int(matrix[true_index, predicted_index])
            if true_index != predicted_index and count:
                pairs.append(
                    {
                        "true_index": true_index,
                        "true_class": CLASS_NAMES[true_index],
                        "predicted_index": predicted_index,
                        "predicted_class": CLASS_NAMES[predicted_index],
                        "count": count,
                    }
                )
    return sorted(pairs, key=lambda row: (-int(row["count"]), row["true_index"]))[
        :limit
    ]


def package_results(results_dir: Path, archive_base: Path) -> Path:
    results_dir = Path(results_dir)
    required = {
        "agri-diagnose-v2-exp-a.keras",
        "environment.json",
        "experiment.json",
        "phase1-history.csv",
        "phase2-history.csv",
        "validation-metrics.json",
        "validation-confusion-matrix.csv",
        "validation-confusion-matrix.png",
        "learning-curve-loss.png",
        "learning-curve-accuracy.png",
        "learning-curve-macro-f1.png",
        "preflight.json",
        "model-v2-exp-a-summary.json",
        "model-v2-exp-a-report.md",
    }
    missing = sorted(name for name in required if not (results_dir / name).is_file())
    if missing:
        raise RuntimeError(f"Kaggle result package is incomplete: {missing}")
    archive = shutil.make_archive(str(archive_base), "zip", root_dir=results_dir)
    return Path(archive)
