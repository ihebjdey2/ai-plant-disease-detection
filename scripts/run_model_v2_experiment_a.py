from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Mapping, Sequence

import keras
import numpy as np
import sklearn
import tensorflow as tf
from PIL import Image, UnidentifiedImageError
from sklearn.metrics import f1_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training.taxonomy import CLASS_NAMES  # noqa: E402
from training.data_pipeline import (  # noqa: E402
    DEFAULT_POLICY_PATH,
    ManifestRecord,
    TrainingPolicyError,
    build_tf_dataset,
    configured_source_roots,
    load_development_manifest,
    load_local_path_aliases,
    load_policy,
    resolve_record_path,
    set_experiment_seeds,
)
from training.experiment_a import (  # noqa: E402
    CLASS_WEIGHTS,
    EXPERIMENT_NAME,
    build_model,
    callback_policy,
    compile_phase1,
    compile_phase2,
    configure_phase2,
    parameter_audit,
)
from training.metrics import MacroF1  # noqa: E402


TRAIN_MANIFEST = PROJECT_ROOT / "training/datasets/manifests/dataset-v2-train.csv"
VALIDATION_MANIFEST = (
    PROJECT_ROOT / "training/datasets/manifests/dataset-v2-validation.csv"
)
TEST_MANIFEST = PROJECT_ROOT / "training/datasets/manifests/dataset-v2-test.csv"
TEST_LOCK = PROJECT_ROOT / "training/datasets/reports/dataset-v2-test-lock.json"
RUN_DIR = PROJECT_ROOT / "training/runs/agri-diagnose-v2-exp-a"
EXPECTED_TEST_SHA256 = "f0df59c42268163d485feea0e54dd7780aa56fe08a7984ae7869a09c604a9151"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def git_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def cpu_model() -> str | None:
    if platform.system() == "Windows":
        try:
            import winreg

            with winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE,
                r"HARDWARE\DESCRIPTION\System\CentralProcessor\0",
            ) as key:
                return str(winreg.QueryValueEx(key, "ProcessorNameString")[0]).strip()
        except OSError:
            pass
    return platform.processor() or None


def total_ram_bytes() -> int | None:
    if platform.system() == "Windows":
        class MemoryStatus(ctypes.Structure):
            _fields_ = [
                ("length", ctypes.c_ulong),
                ("memory_load", ctypes.c_ulong),
                ("total_physical", ctypes.c_ulonglong),
                ("available_physical", ctypes.c_ulonglong),
                ("total_page_file", ctypes.c_ulonglong),
                ("available_page_file", ctypes.c_ulonglong),
                ("total_virtual", ctypes.c_ulonglong),
                ("available_virtual", ctypes.c_ulonglong),
                ("available_extended_virtual", ctypes.c_ulonglong),
            ]

        status = MemoryStatus()
        status.length = ctypes.sizeof(MemoryStatus)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return int(status.total_physical)
    try:
        page_size = os.sysconf("SC_PAGE_SIZE")
        pages = os.sysconf("SC_PHYS_PAGES")
        return int(page_size * pages)
    except (AttributeError, ValueError, OSError):
        return None


def collect_environment(seed: int) -> dict[str, object]:
    build = tf.sysconfig.get_build_info()
    physical = tf.config.list_physical_devices()
    gpus = tf.config.list_physical_devices("GPU")
    determinism_enabled = False
    determinism_error = None
    try:
        tf.config.experimental.enable_op_determinism()
        determinism_enabled = True
    except (AttributeError, RuntimeError) as exc:
        determinism_error = type(exc).__name__
    return {
        "experiment_name": EXPERIMENT_NAME,
        "experiment_seed": seed,
        "git_base_commit": git_commit(),
        "operating_system": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
        },
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
        },
        "packages": {
            "tensorflow": tf.__version__,
            "keras": keras.__version__,
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "tensorflow_build": {
            "built_with_cuda": bool(tf.test.is_built_with_cuda()),
            "cuda_version": build.get("cuda_version"),
            "cudnn_version": build.get("cudnn_version"),
        },
        "physical_devices": [
            {"name": device.name, "device_type": device.device_type}
            for device in physical
        ],
        "gpu": {
            "detected": bool(gpus),
            "devices": [device.name for device in gpus],
            "model": None,
            "memory_bytes": None,
        },
        "cpu": {
            "model": cpu_model(),
            "logical_processor_count": os.cpu_count(),
        },
        "ram_bytes": total_ram_bytes(),
        "determinism": {
            "requested": True,
            "tensorflow_op_determinism_enabled": determinism_enabled,
            "enable_error_type": determinism_error,
            "bit_identical_gpu_training_guaranteed": False,
        },
        "gpu_gate_status": (
            "GPU_AVAILABLE" if gpus else "GPU_NOT_AVAILABLE_FOR_TENSORFLOW"
        ),
        "contains_private_paths": False,
        "contains_secrets": False,
    }


def audit_paths(
    records: Sequence[ManifestRecord],
    roots: Mapping[str, Path],
    aliases: Mapping[str, Mapping[str, str]],
) -> dict[str, object]:
    missing: list[str] = []
    unreadable: list[str] = []
    resolved_records: list[tuple[ManifestRecord, Path]] = []
    started = time.perf_counter()
    for record in records:
        try:
            path = resolve_record_path(record, roots, aliases)
        except TrainingPolicyError:
            missing.append(record.composition_record_id)
            continue
        if not path.is_file():
            missing.append(record.composition_record_id)
            continue
        resolved_records.append((record, path))

    def verify(item: tuple[ManifestRecord, Path]) -> tuple[str, bool]:
        record, path = item
        try:
            with Image.open(path) as image:
                image.verify()
        except (UnidentifiedImageError, OSError, ValueError):
            return record.composition_record_id, False
        return record.composition_record_id, True

    workers = min(8, os.cpu_count() or 1)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        verification = list(executor.map(verify, resolved_records, chunksize=64))
    unreadable.extend(record_id for record_id, readable in verification if not readable)
    readable_ids = {record_id for record_id, readable in verification if readable}
    source_counts = Counter(
        record.source_dataset
        for record, _ in resolved_records
        if record.composition_record_id in readable_ids
    )
    resolved = len(readable_ids)
    return {
        "expected": len(records),
        "resolved": resolved,
        "missing": len(missing),
        "unreadable": len(unreadable),
        "missing_record_ids_preview": missing[:20],
        "unreadable_record_ids_preview": unreadable[:20],
        "resolved_by_source": dict(sorted(source_counts.items())),
        "verification_workers": workers,
        "audit_duration_seconds": round(time.perf_counter() - started, 6),
    }


def validate_tensor_pipeline(
    train_records: Sequence[ManifestRecord],
    validation_records: Sequence[ManifestRecord],
    policy: Mapping[str, object],
    roots: Mapping[str, Path],
    aliases: Mapping[str, Mapping[str, str]],
) -> dict[str, object]:
    train_sample = sorted(train_records, key=lambda row: row.composition_record_id)[:32]
    validation_sample = sorted(
        validation_records, key=lambda row: row.composition_record_id
    )[:32]
    train_dataset = build_tf_dataset(
        train_sample,
        policy,
        roots,
        training=True,
        batch_size=32,
        local_path_aliases=aliases,
    )
    validation_dataset = build_tf_dataset(
        validation_sample,
        policy,
        roots,
        training=False,
        batch_size=32,
        local_path_aliases=aliases,
    )

    def inspect(dataset) -> dict[str, object]:
        images, labels = next(iter(dataset))
        return {
            "shape": list(images.shape),
            "dtype": images.dtype.name,
            "minimum": round(float(tf.reduce_min(images)), 8),
            "maximum": round(float(tf.reduce_max(images)), 8),
            "minimum_label": int(tf.reduce_min(labels)),
            "maximum_label": int(tf.reduce_max(labels)),
        }

    return {
        "train": {**inspect(train_dataset), "augmentation_enabled": True},
        "validation": {
            **inspect(validation_dataset),
            "augmentation_enabled": False,
            "shuffle_enabled": False,
        },
    }


def validate_macro_f1() -> dict[str, object]:
    y_true = np.repeat(np.arange(39, dtype=np.int32), np.arange(1, 40))
    y_pred = y_true.copy()
    y_pred[::11] = (y_pred[::11] + 7) % 39
    probabilities = tf.one_hot(y_pred, depth=39, dtype=tf.float32)
    metric = MacroF1()
    midpoint = len(y_true) // 2
    metric.update_state(y_true[:midpoint], probabilities[:midpoint])
    metric.update_state(y_true[midpoint:], probabilities[midpoint:])
    tensorflow_value = float(metric.result().numpy())
    sklearn_value = float(f1_score(y_true, y_pred, average="macro"))
    difference = abs(tensorflow_value - sklearn_value)
    if difference > 1e-6:
        raise RuntimeError(f"MacroF1 validation failed: difference={difference}")
    metric.reset_state()
    reset_value = float(metric.result().numpy())
    if reset_value != 0.0:
        raise RuntimeError("MacroF1 reset_state validation failed.")
    return {
        "tensorflow_macro_f1": round(tensorflow_value, 10),
        "sklearn_macro_f1": round(sklearn_value, 10),
        "absolute_difference": round(difference, 12),
        "tolerance": 1e-6,
        "reset_state_result": reset_value,
        "passed": True,
    }


def validate_preflight_metadata(payload: Mapping[str, object]) -> None:
    required = {
        "experiment_name",
        "status",
        "training_authorized",
        "training_performed",
        "model_fit_called",
        "class_weights",
        "train_manifest",
        "validation_manifest",
        "macro_f1_validation",
        "phase1_model_audit",
        "phase2_model_audit",
        "internal_test",
        "plantdoc_test",
        "candidate_model_created",
    }
    missing = required - set(payload)
    if missing:
        raise RuntimeError(f"Experiment A preflight metadata is incomplete: {sorted(missing)}")
    if payload["training_performed"] or payload["model_fit_called"]:
        raise RuntimeError("Preflight metadata cannot claim training execution.")
    if payload["class_weights"] is not None:
        raise RuntimeError("Experiment A metadata must record class_weights=null.")
    if payload["internal_test"]["loaded"] or payload["internal_test"]["evaluated"]:
        raise RuntimeError("Preflight metadata indicates forbidden internal TEST access.")
    if payload["plantdoc_test"]["loaded"] or payload["plantdoc_test"]["evaluated"]:
        raise RuntimeError("Preflight metadata indicates forbidden PlantDoc TEST access.")


def run_preflight(policy_path: Path) -> dict[str, object]:
    policy = load_policy(policy_path)
    seed = set_experiment_seeds(policy)
    environment = collect_environment(seed)
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    write_json(RUN_DIR / "environment.json", environment)

    train_records = load_development_manifest(TRAIN_MANIFEST, training=True)
    validation_records = load_development_manifest(
        VALIDATION_MANIFEST, training=False
    )
    if len(train_records) != 58_857 or len(validation_records) != 7_362:
        raise RuntimeError("Authoritative TRAIN/VALIDATION count mismatch.")
    if {record.target_index for record in train_records} != set(range(39)):
        raise RuntimeError("TRAIN class coverage is not 39/39.")
    if {record.target_index for record in validation_records} != set(range(39)):
        raise RuntimeError("VALIDATION class coverage is not 39/39.")
    roots = configured_source_roots()
    required_sources = {record.source_dataset for record in train_records}
    missing_roots = sorted(required_sources - set(roots))
    if missing_roots:
        raise RuntimeError(f"Required local source roots are not configured: {missing_roots}")
    aliases = load_local_path_aliases()
    train_audit = audit_paths(train_records, roots, aliases)
    validation_audit = audit_paths(validation_records, roots, aliases)
    for name, audit in (("TRAIN", train_audit), ("VALIDATION", validation_audit)):
        if audit["missing"] or audit["unreadable"] or audit["resolved"] != audit["expected"]:
            raise RuntimeError(f"{name} path preflight failed: {audit}")

    tensor_audit = validate_tensor_pipeline(
        train_records, validation_records, policy, roots, aliases
    )
    macro_f1_audit = validate_macro_f1()
    model, backbone = build_model(policy, weights="imagenet")
    compile_phase1(model, policy)
    phase1_audit = {
        **parameter_audit(model),
        "backbone_trainable": backbone.trainable,
        "output_shape": [None, 39],
        "learning_rate": float(policy["phase1"]["learning_rate"]),
        "metric_names": ["accuracy", "macro_f1"],
    }
    if backbone.trainable:
        raise RuntimeError("Phase 1 backbone unexpectedly trainable.")
    phase2_layers = configure_phase2(backbone, policy)
    compile_phase2(model, policy)
    phase2_audit = {
        **parameter_audit(model),
        **phase2_layers,
        "learning_rate": float(policy["phase2"]["learning_rate"]),
        "new_optimizer_created": True,
        "metric_names": ["accuracy", "macro_f1"],
    }
    if phase2_layers["frozen_batch_normalization_count"] != 52:
        raise RuntimeError("Phase 2 BatchNormalization freeze audit failed.")
    callback_audit = callback_policy(
        policy, Path("models/candidates/agri-diagnose-v2-exp-a/phase1-best.keras")
    )
    if callback_audit["model_checkpoint"]["monitor"] != "val_macro_f1":
        raise RuntimeError("Checkpoint monitor must remain val_macro_f1.")
    if CLASS_WEIGHTS is not None:
        raise RuntimeError("Experiment A must not define class weights.")

    test_hash = file_sha256(TEST_MANIFEST)
    lock = json.loads(TEST_LOCK.read_text(encoding="utf-8"))
    if test_hash != EXPECTED_TEST_SHA256 or lock["test_manifest_sha256"] != test_hash:
        raise RuntimeError("Locked internal TEST manifest hash changed.")
    result = {
        "experiment_name": EXPERIMENT_NAME,
        "status": environment["gpu_gate_status"],
        "training_authorized": bool(environment["gpu"]["detected"]),
        "training_performed": False,
        "model_fit_called": False,
        "class_weights": None,
        "policy_version": policy["policy_version"],
        "seed": seed,
        "batch_size": int(policy["batch_size"]["default"]),
        "train_manifest": {
            "sha256": file_sha256(TRAIN_MANIFEST),
            "class_coverage": 39,
            **train_audit,
        },
        "validation_manifest": {
            "sha256": file_sha256(VALIDATION_MANIFEST),
            "class_coverage": 39,
            **validation_audit,
        },
        "tensor_pipeline": tensor_audit,
        "macro_f1_validation": macro_f1_audit,
        "phase1_model_audit": phase1_audit,
        "phase2_model_audit": phase2_audit,
        "callback_policy": callback_audit,
        "internal_test": {
            "manifest_sha256": test_hash,
            "hash_unchanged": True,
            "loaded": False,
            "evaluated": False,
        },
        "plantdoc_test": {"loaded": False, "evaluated": False},
        "confidence_threshold_changed": False,
        "production_model_sha256": file_sha256(PROJECT_ROOT / "plant_disease_model.h5"),
        "candidate_model_created": False,
        "stop_reason": (
            None
            if environment["gpu"]["detected"]
            else "GPU_NOT_AVAILABLE_FOR_TENSORFLOW"
        ),
    }
    validate_preflight_metadata(result)
    write_json(RUN_DIR / "preflight.json", result)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the safe Experiment A preflight without neural-network training."
    )
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY_PATH)
    args = parser.parse_args()
    if not args.preflight_only:
        parser.error(
            "This environment-gated command requires --preflight-only; training is not automatic."
        )
    return args


def main() -> int:
    result = run_preflight(parse_args().policy)
    print(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
