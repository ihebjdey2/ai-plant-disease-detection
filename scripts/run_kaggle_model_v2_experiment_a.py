from __future__ import annotations

import argparse
import json
import platform
import shutil
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
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
    candidate_dir = Path(
        "/kaggle/working/models/candidates/agri-diagnose-v2-exp-a"
    )
    results_dir = Path("/kaggle/working/agridiagnose-exp-a-results")
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
        results_dir, Path("/kaggle/working/agridiagnose-exp-a-results")
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
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.action == "verify-runtime":
        verify_runtime(args.output)
    elif args.action == "preflight":
        run_preflight(args.config, args.output)
    else:
        run_training(args.config, authorize_training=args.authorize_training)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
