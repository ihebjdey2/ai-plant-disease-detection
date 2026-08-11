from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import shutil
import sys
from pathlib import Path
from typing import Mapping


# Kaggle's notebook kernel may export its inline backend to this subprocess.
# Experiment reporting is file-only, so always select a deterministic headless backend.
os.environ["MPLBACKEND"] = "Agg"


PROJECT_ROOT = Path(__file__).resolve().parents[1]
KAGGLE_RESULTS_DIR = Path("/kaggle/working/agridiagnose-exp-a-results")
KAGGLE_CANDIDATE_DIR = Path(
    "/kaggle/working/models/candidates/agri-diagnose-v2-exp-a"
)
KAGGLE_ARCHIVE_BASE = Path("/kaggle/working/agridiagnose-exp-a-results")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training.kaggle_runtime import (  # noqa: E402
    load_execution_config,
    validate_runtime_payload,
)


def write_safe_json(path: Path | None, payload: object) -> None:
    if path is None:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def verify_runtime(output: Path | None = None) -> dict[str, object]:
    try:
        import tensorflow as tf

        from training.kaggle_experiment_a import (
            configure_gpu_memory_growth,
            require_approved_stack,
            require_kaggle_gpu,
            runtime_audit,
        )

        audit = runtime_audit()
        require_kaggle_gpu(audit)
        require_approved_stack(audit)
        devices = configure_gpu_memory_growth()
        tf.config.set_soft_device_placement(False)
        with tf.device("/GPU:0"):
            left = tf.reshape(tf.range(1024, dtype=tf.float32), (32, 32))
            right = tf.eye(32, dtype=tf.float32)
            result = tf.matmul(left, right)
        smoke_sum = float(tf.reduce_sum(result).numpy())
        smoke_device = result.device
        audit.update(
            {
                "status": "TF215_GPU_RUNTIME_VALIDATED",
                "gpu_memory_growth_devices": devices,
                "gpu_smoke_test_passed": "GPU" in smoke_device.upper(),
                "gpu_smoke_device": smoke_device,
                "gpu_smoke_sum": smoke_sum,
                "training_performed": False,
                "internal_test_loaded": False,
                "plantdoc_test_loaded": False,
            }
        )
        validate_runtime_payload(audit)
        write_safe_json(output, audit)
        print("Python:", audit["python_version"])
        print("TensorFlow:", audit["tensorflow_version"])
        print("Keras:", audit["keras_version"])
        print("NumPy:", audit["numpy_version"])
        print("Built with CUDA:", audit["tensorflow_built_with_cuda"])
        print("GPUs:", audit["tensorflow_gpu_devices"])
        print("GPU smoke device:", smoke_device)
        print(json.dumps(audit, indent=2, sort_keys=True))
        return audit
    except Exception as exc:
        failure = {
            "status": "KAGGLE_TF215_GPU_RUNTIME_FAILED",
            "error_type": type(exc).__name__,
            "error": str(exc)[:500],
            "training_performed": False,
            "internal_test_loaded": False,
            "plantdoc_test_loaded": False,
        }
        write_safe_json(output, failure)
        print(json.dumps(failure, indent=2), file=sys.stderr)
        raise RuntimeError("KAGGLE_TF215_GPU_RUNTIME_FAILED") from exc


def run_preflight(config_path: Path, output: Path) -> dict[str, object]:
    runtime = verify_runtime(output.with_name("tf215-gpu-runtime.json"))
    config = load_execution_config(config_path, allow_training=False)
    from training.data_pipeline import load_policy
    from training.experiment_a import (
        CLASS_WEIGHTS,
        build_model,
        compile_phase1,
        compile_phase2,
        configure_phase2,
        parameter_audit,
    )
    from training.kaggle_experiment_a import (
        kaggle_source_roots,
        run_full_preflight,
    )

    roots = kaggle_source_roots(config["source_roots"])
    preflight = run_full_preflight(PROJECT_ROOT, roots)
    policy = load_policy(PROJECT_ROOT / "training/config/model-v2-training-policy.json")
    model, backbone = build_model(policy, weights="imagenet")
    compile_phase1(model, policy)
    phase1 = {
        **parameter_audit(model),
        "input_shape": list(model.input_shape),
        "output_shape": list(model.output_shape),
        "backbone_trainable": backbone.trainable,
        "initialization": "imagenet",
        "production_model_loaded": False,
    }
    if model.input_shape != (None, 224, 224, 3) or model.output_shape != (None, 39):
        raise RuntimeError("Experiment A model shape audit failed.")
    if backbone.trainable or CLASS_WEIGHTS is not None:
        raise RuntimeError("Experiment A Phase 1 freeze/class-weight policy failed.")
    phase2_layers = configure_phase2(backbone, policy)
    compile_phase2(model, policy)
    phase2 = {**parameter_audit(model), **phase2_layers}
    if phase2_layers["first_trainable_backbone_layer"] != "block_13_expand":
        raise RuntimeError("Experiment A Phase 2 boundary audit failed.")
    if phase2_layers["frozen_batch_normalization_count"] != 52:
        raise RuntimeError("Experiment A BatchNormalization freeze audit failed.")

    payload = {
        "status": "KAGGLE_TF215_GPU_PREFLIGHT_PASSED",
        "runtime": runtime,
        "preflight": preflight,
        "phase1_model_audit": phase1,
        "phase2_model_audit": phase2,
        "start_training": False,
        "training_performed": False,
        "internal_test_loaded": False,
        "plantdoc_test_loaded": False,
    }
    write_safe_json(output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return payload


def evaluate_checkpoint(
    name,
    checkpoint,
    best_row,
    selection_epoch,
    validation_dataset,
    validation_records,
):
    import tensorflow as tf

    from training.kaggle_experiment_a import validation_report
    from training.metrics import MacroF1

    model = tf.keras.models.load_model(
        checkpoint, custom_objects={"MacroF1": MacroF1}
    )
    values = model.evaluate(validation_dataset, return_dict=True, verbose=1)
    scores = model.predict(validation_dataset, verbose=1)
    report = validation_report(validation_records, scores)
    return {
        "name": name,
        "partition": "VALIDATION",
        "checkpoint": str(checkpoint),
        "epoch": int(best_row["epoch"]),
        "selection_epoch": int(selection_epoch),
        "val_macro_f1": float(best_row["val_macro_f1"]),
        "val_loss": float(values["loss"]),
        "val_accuracy": float(values["accuracy"]),
        "macro_recall": report["overall_validation"]["macro_recall"],
        "report": report,
    }


def read_json_mapping(path: Path, label: str) -> dict[str, object]:
    path = Path(path)
    if not path.is_file():
        raise RuntimeError(f"Missing existing {label}: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Invalid existing {label}: {path}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"Existing {label} must contain a JSON object: {path}")
    return payload


def require_existing_file(path: Path, label: str) -> Path:
    path = Path(path)
    if not path.is_file() or path.stat().st_size <= 0:
        raise RuntimeError(f"Missing existing {label}: {path}")
    return path


def history_epoch_count(path: Path) -> int:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        count = sum(1 for _ in csv.DictReader(handle))
    if count <= 0:
        raise RuntimeError(f"No completed epochs were persisted in {Path(path).name}.")
    return count


def validate_existing_preflight(payload: Mapping[str, object]) -> dict[str, object]:
    if payload.get("status") != "KAGGLE_TF215_GPU_PREFLIGHT_PASSED":
        raise RuntimeError("Existing Kaggle preflight did not pass successfully.")
    preflight = payload.get("preflight")
    if not isinstance(preflight, dict):
        raise RuntimeError("Existing Kaggle preflight payload is incomplete.")
    for layer_name, layer in (("preflight report", payload), ("data preflight", preflight)):
        if layer.get("internal_test_loaded") is not False:
            raise RuntimeError(f"{layer_name} does not preserve the INTERNAL TEST lock.")
        if layer.get("plantdoc_test_loaded") is not False:
            raise RuntimeError(f"{layer_name} does not preserve the PlantDoc TEST lock.")
    if payload.get("training_performed") is not False:
        raise RuntimeError("The saved preflight must precede neural-network training.")
    for partition, expected in (("train", 58_857), ("validation", 7_362)):
        audit = preflight.get(partition)
        if not isinstance(audit, dict) or any(
            (
                audit.get("expected") != expected,
                audit.get("resolved") != expected,
                audit.get("missing") != 0,
                audit.get("unreadable") != 0,
            )
        ):
            raise RuntimeError(f"Existing {partition.upper()} preflight is invalid.")
    if (
        preflight.get("train_class_coverage") != 39
        or preflight.get("validation_class_coverage") != 39
    ):
        raise RuntimeError("Existing TRAIN/VALIDATION class coverage is invalid.")
    if not isinstance(preflight.get("internal_test_manifest_sha256"), str):
        raise RuntimeError("Existing INTERNAL TEST manifest lock metadata is missing.")
    return preflight


def validate_existing_validation_metrics(
    payload: Mapping[str, object],
) -> dict[str, object]:
    required = {
        "loss",
        "accuracy",
        "overall_validation",
        "real_world_validation",
        "true_indices",
        "predicted_indices",
    }
    missing = sorted(required - set(payload))
    if missing:
        raise RuntimeError(f"Existing validation metrics are incomplete: {missing}")
    true_indices = payload["true_indices"]
    predicted_indices = payload["predicted_indices"]
    if (
        not isinstance(true_indices, list)
        or not isinstance(predicted_indices, list)
        or not true_indices
        or len(true_indices) != len(predicted_indices)
    ):
        raise RuntimeError("Existing VALIDATION prediction indices are invalid.")
    if any(
        not isinstance(value, int) or value < 0 or value >= 39
        for value in [*true_indices, *predicted_indices]
    ):
        raise RuntimeError("Existing VALIDATION prediction indices are out of range.")
    overall = payload["overall_validation"]
    if not isinstance(overall, dict) or overall.get("image_count") != len(true_indices):
        raise RuntimeError("Existing VALIDATION metric counts are inconsistent.")
    return dict(payload)


def model_signature(path: Path, sha256_file) -> tuple[str, int, int]:
    stat = Path(path).stat()
    return sha256_file(path), stat.st_size, stat.st_mtime_ns


def finalize_existing(
    config_path: Path,
    preflight_report_path: Path,
    *,
    results_dir: Path = KAGGLE_RESULTS_DIR,
    candidate_dir: Path = KAGGLE_CANDIDATE_DIR,
    archive_base: Path = KAGGLE_ARCHIVE_BASE,
) -> Path:
    """Finish VALIDATION-only reporting from completed Experiment A artifacts."""
    config = load_execution_config(config_path, allow_training=True)
    if (
        config["internal_test_loaded"] is not False
        or config["plantdoc_test_loaded"] is not False
    ):
        raise RuntimeError("Locked TEST datasets are forbidden during finalization.")

    from training.kaggle_experiment_a import (
        EXPERIMENT_NAME,
        best_history_row,
        major_confusion_pairs,
        package_results,
        plot_learning_curves,
        save_confusion_artifacts,
        sha256_file,
        write_json,
    )

    results_dir = Path(results_dir)
    candidate_dir = Path(candidate_dir)
    selected_path = require_existing_file(
        results_dir / "agri-diagnose-v2-exp-a.keras", "selected candidate model"
    )
    phase1_checkpoint = require_existing_file(
        candidate_dir / "phase1-best.keras", "Phase 1 checkpoint"
    )
    phase2_checkpoint = require_existing_file(
        candidate_dir / "phase2-best.keras", "Phase 2 checkpoint"
    )
    phase1_history = require_existing_file(
        results_dir / "phase1-history.csv", "Phase 1 history"
    )
    phase2_history = require_existing_file(
        results_dir / "phase2-history.csv", "Phase 2 history"
    )
    require_existing_file(
        results_dir / "validation-confusion-matrix.csv",
        "VALIDATION confusion-matrix CSV",
    )
    metrics = validate_existing_validation_metrics(
        read_json_mapping(
            results_dir / "validation-metrics.json", "VALIDATION metrics"
        )
    )
    runtime = read_json_mapping(
        results_dir / "environment-runtime.json", "training runtime audit"
    )
    validate_runtime_payload(runtime)
    preflight = validate_existing_preflight(
        read_json_mapping(preflight_report_path, "Kaggle preflight report")
    )

    model_paths = {
        "selected": selected_path,
        "phase1": phase1_checkpoint,
        "phase2": phase2_checkpoint,
    }
    signatures_before = {
        name: model_signature(path, sha256_file)
        for name, path in model_paths.items()
    }
    selected_hash = signatures_before["selected"][0]
    matching_phases = [
        phase
        for phase in ("phase1", "phase2")
        if signatures_before[phase][0] == selected_hash
    ]
    if len(matching_phases) != 1:
        raise RuntimeError(
            "Selected candidate must match exactly one existing phase checkpoint."
        )
    selected_phase = matching_phases[0]
    phase1_best = best_history_row(phase1_history)
    phase2_best = best_history_row(phase2_history)
    selected_best = phase1_best if selected_phase == "phase1" else phase2_best
    selection_epoch = int(selected_best["epoch"])
    if selected_phase == "phase2":
        selection_epoch += history_epoch_count(phase1_history)

    save_confusion_artifacts(
        metrics, results_dir, preserve_existing_csv=True
    )
    plot_learning_curves(phase1_history, phase2_history, results_dir)
    environment = {
        **runtime,
        "status": "EXPERIMENT_A_EXISTING_ARTIFACTS_FINALIZED",
        "os_environment": "Kaggle isolated Python 3.11 subprocess",
        "batch_size": int(config["batch_size"]),
        "training_performed": True,
        "retraining_performed": False,
    }
    experiment = {
        "experiment": EXPERIMENT_NAME,
        "selected_phase": selected_phase,
        "selected_epoch": int(selected_best["epoch"]),
        "selection_epoch": selection_epoch,
        "candidate_sha256": selected_hash,
        "candidate_size_bytes": selected_path.stat().st_size,
        "train_manifest_sha256": sha256_file(
            PROJECT_ROOT / "training/datasets/manifests/dataset-v2-train.csv"
        ),
        "validation_manifest_sha256": sha256_file(
            PROJECT_ROOT / "training/datasets/manifests/dataset-v2-validation.csv"
        ),
        "policy_sha256": sha256_file(
            PROJECT_ROOT / "training/config/model-v2-training-policy.json"
        ),
        "internal_test_manifest_sha256": preflight[
            "internal_test_manifest_sha256"
        ],
        "internal_test_loaded": False,
        "plantdoc_test_loaded": False,
        "training_performed": True,
        "retraining_performed": False,
        "finalization_mode": "finalize-existing",
        "phase1_run_mode": "completed_before_recovery",
        "phase2_run_mode": "completed_before_recovery",
    }
    summary = {
        "selected_phase": selected_phase,
        "selected_epoch": int(selected_best["epoch"]),
        "validation": metrics["overall_validation"],
        "real_world_validation": metrics["real_world_validation"],
        "major_confusion_pairs": major_confusion_pairs(metrics),
        "test_sets_evaluated": False,
        "training_performed": True,
        "retraining_performed": False,
    }
    write_json(results_dir / "environment.json", environment)
    write_json(results_dir / "experiment.json", experiment)
    write_json(results_dir / "preflight.json", preflight)
    write_json(results_dir / "model-v2-exp-a-summary.json", summary)
    (results_dir / "model-v2-exp-a-report.md").write_text(
        "# Model V2 Experiment A\n\n"
        "Existing completed training finalized without retraining.\n\n"
        "Candidate selection and metrics use VALIDATION only. INTERNAL TEST and "
        "PlantDoc TEST remain locked and unevaluated.\n",
        encoding="utf-8",
    )
    archive = package_results(results_dir, Path(archive_base))

    signatures_after = {
        name: model_signature(path, sha256_file)
        for name, path in model_paths.items()
    }
    if signatures_after != signatures_before:
        raise RuntimeError("Recovery modified an existing trained model artifact.")
    recovery = {
        "status": "EXPERIMENT_A_EXISTING_ARTIFACTS_FINALIZED",
        "archive": str(archive),
        "selected_phase": selected_phase,
        "training_performed": True,
        "retraining_performed": False,
        "internal_test_loaded": False,
        "plantdoc_test_loaded": False,
        "test_sets_evaluated": False,
    }
    print(json.dumps(recovery, indent=2, sort_keys=True))
    return archive


def run_training(config_path: Path, *, authorize_training: bool) -> Path:
    if not authorize_training:
        raise RuntimeError("TRAINING_DISABLED_BY_USER")
    config = load_execution_config(config_path, allow_training=True)
    if config["start_training"] is not True:
        raise RuntimeError("TRAINING_DISABLED_BY_USER")
    runtime = verify_runtime(
        Path("/kaggle/working/agridiagnose-exp-a-results/environment-runtime.json")
    )

    import keras
    import tensorflow as tf

    from training.data_pipeline import load_policy
    from training.experiment_a import (
        CLASS_WEIGHTS,
        build_model,
        compile_phase1,
        compile_phase2,
        configure_phase2,
    )
    from training.kaggle_experiment_a import (
        EXPERIMENT_NAME,
        best_history_row,
        build_kaggle_datasets,
        build_phase_callbacks,
        kaggle_source_roots,
        major_confusion_pairs,
        package_results,
        plot_learning_curves,
        require_fresh_or_explicit_restart,
        run_full_preflight,
        save_confusion_artifacts,
        select_candidate,
        sha256_file,
        write_json,
    )
    from training.metrics import MacroF1

    if CLASS_WEIGHTS is not None:
        raise RuntimeError("Experiment A class weights must remain disabled.")
    roots = kaggle_source_roots(config["source_roots"])
    preflight = run_full_preflight(PROJECT_ROOT, roots)
    policy = load_policy(PROJECT_ROOT / "training/config/model-v2-training-policy.json")
    batch_size = int(config["batch_size"])
    train_dataset, validation_dataset, _, validation_records = build_kaggle_datasets(
        PROJECT_ROOT, roots, batch_size=batch_size
    )
    candidate_dir = KAGGLE_CANDIDATE_DIR
    results_dir = KAGGLE_RESULTS_DIR
    candidate_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    restart = bool(config["restart_interrupted_phase"])

    model, _ = build_model(policy, weights="imagenet")
    compile_phase1(model, policy)
    phase1_checkpoint = candidate_dir / "phase1-best.keras"
    phase1_history = results_dir / "phase1-history.csv"
    phase1_mode = require_fresh_or_explicit_restart(
        candidate_dir,
        "phase1",
        restart_interrupted_phase=restart,
        history_path=phase1_history,
    )
    phase1_fit = model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=int(policy["phase1"]["max_epochs"]),
        callbacks=build_phase_callbacks(
            policy,
            checkpoint_path=phase1_checkpoint,
            history_path=phase1_history,
        ),
        class_weight=None,
        verbose=1,
    )
    phase1_best = best_history_row(phase1_history)

    phase2_history = results_dir / "phase2-history.csv"
    phase2_mode = require_fresh_or_explicit_restart(
        candidate_dir,
        "phase2",
        restart_interrupted_phase=restart,
        history_path=phase2_history,
    )
    phase2_model = tf.keras.models.load_model(
        phase1_checkpoint, custom_objects={"MacroF1": MacroF1}
    )
    phase2_backbone = next(
        layer
        for layer in phase2_model.layers
        if isinstance(layer, tf.keras.Model) and "mobilenetv2" in layer.name
    )
    trainability = configure_phase2(phase2_backbone, policy)
    if trainability["frozen_batch_normalization_count"] != 52:
        raise RuntimeError("Phase 2 BatchNormalization policy changed.")
    compile_phase2(phase2_model, policy)
    phase2_checkpoint = candidate_dir / "phase2-best.keras"
    phase2_fit = phase2_model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=int(policy["phase2"]["max_epochs"]),
        callbacks=build_phase_callbacks(
            policy,
            checkpoint_path=phase2_checkpoint,
            history_path=phase2_history,
        ),
        class_weight=None,
        verbose=1,
    )
    del phase2_fit
    phase2_best = best_history_row(phase2_history)
    phase1_result = evaluate_checkpoint(
        "phase1",
        phase1_checkpoint,
        phase1_best,
        phase1_best["epoch"],
        validation_dataset,
        validation_records,
    )
    phase2_result = evaluate_checkpoint(
        "phase2",
        phase2_checkpoint,
        phase2_best,
        len(phase1_fit.epoch) + phase2_best["epoch"],
        validation_dataset,
        validation_records,
    )
    selected = select_candidate([phase1_result, phase2_result])
    selected_path = results_dir / "agri-diagnose-v2-exp-a.keras"
    shutil.copy2(Path(selected["checkpoint"]), selected_path)
    metrics = selected["report"]
    metrics["loss"] = selected["val_loss"]
    metrics["accuracy"] = selected["val_accuracy"]
    write_json(results_dir / "validation-metrics.json", metrics)
    save_confusion_artifacts(metrics, results_dir)
    plot_learning_curves(phase1_history, phase2_history, results_dir)
    environment = {
        **runtime,
        "os_environment": "Kaggle isolated Python 3.11 subprocess",
        "keras_version": keras.__version__,
        "batch_size": batch_size,
        "training_performed": True,
    }
    experiment = {
        "experiment": EXPERIMENT_NAME,
        "selected_phase": selected["name"],
        "selected_epoch": selected["epoch"],
        "selection_epoch": selected["selection_epoch"],
        "candidate_sha256": sha256_file(selected_path),
        "candidate_size_bytes": selected_path.stat().st_size,
        "train_manifest_sha256": sha256_file(
            PROJECT_ROOT / "training/datasets/manifests/dataset-v2-train.csv"
        ),
        "validation_manifest_sha256": sha256_file(
            PROJECT_ROOT / "training/datasets/manifests/dataset-v2-validation.csv"
        ),
        "policy_sha256": sha256_file(
            PROJECT_ROOT / "training/config/model-v2-training-policy.json"
        ),
        "internal_test_manifest_sha256": preflight[
            "internal_test_manifest_sha256"
        ],
        "internal_test_loaded": False,
        "plantdoc_test_loaded": False,
        "training_performed": True,
        "phase1_run_mode": phase1_mode,
        "phase2_run_mode": phase2_mode,
    }
    summary = {
        "selected_phase": selected["name"],
        "selected_epoch": selected["epoch"],
        "validation": metrics["overall_validation"],
        "real_world_validation": metrics["real_world_validation"],
        "major_confusion_pairs": major_confusion_pairs(metrics),
        "test_sets_evaluated": False,
    }
    write_json(results_dir / "environment.json", environment)
    write_json(results_dir / "experiment.json", experiment)
    write_json(results_dir / "preflight.json", preflight)
    write_json(results_dir / "model-v2-exp-a-summary.json", summary)
    (results_dir / "model-v2-exp-a-report.md").write_text(
        "# Model V2 Experiment A\n\nVALIDATION-only selection. Both TEST sets remain locked.\n",
        encoding="utf-8",
    )
    archive = package_results(
        results_dir, KAGGLE_ARCHIVE_BASE
    )
    print("Experiment A archive:", archive)
    return archive


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run AgriDiagnose Experiment A in isolated Kaggle Python 3.11."
    )
    subparsers = parser.add_subparsers(dest="action", required=True)
    verify = subparsers.add_parser("verify-runtime")
    verify.add_argument("--output", type=Path)
    preflight = subparsers.add_parser("preflight")
    preflight.add_argument("--config", type=Path, required=True)
    preflight.add_argument("--output", type=Path, required=True)
    train = subparsers.add_parser("train")
    train.add_argument("--config", type=Path, required=True)
    train.add_argument("--authorize-training", action="store_true")
    finalize = subparsers.add_parser("finalize-existing")
    finalize.add_argument("--config", type=Path, required=True)
    finalize.add_argument("--preflight-report", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.action == "verify-runtime":
        verify_runtime(args.output)
    elif args.action == "preflight":
        run_preflight(args.config, args.output)
    elif args.action == "train":
        run_training(args.config, authorize_training=args.authorize_training)
    elif args.action == "finalize-existing":
        finalize_existing(args.config, args.preflight_report)
    else:  # pragma: no cover - argparse restricts actions before dispatch.
        raise RuntimeError(f"Unsupported action: {args.action}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
