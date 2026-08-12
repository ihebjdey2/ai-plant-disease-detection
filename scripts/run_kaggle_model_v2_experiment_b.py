from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Mapping


os.environ["MPLBACKEND"] = "Agg"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
KAGGLE_RESULTS_DIR = Path("/kaggle/working/agridiagnose-exp-b-results")
KAGGLE_CANDIDATE_DIR = Path(
    "/kaggle/working/models/candidates/agri-diagnose-v2-exp-b"
)
KAGGLE_ARCHIVE_BASE = Path("/kaggle/working/agridiagnose-exp-b-results")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_kaggle_model_v2_experiment_a import (  # noqa: E402
    evaluate_checkpoint,
    verify_runtime,
    write_safe_json,
)
from training.experiment_b import (  # noqa: E402
    APPROVED_AUGMENTATION,
    CLASS_WEIGHTS,
    EXPERIMENT_NAME,
    POLICY_PATH,
    build_model,
    compile_phase1,
    compile_phase2,
    configure_phase2,
    parameter_audit,
)
from training.kaggle_experiment_b import (  # noqa: E402
    EXPECTED_TRAIN_MANIFEST_SHA256,
    EXPECTED_TRAIN_COUNT,
    EXPECTED_TAXONOMY_SHA256,
    EXPECTED_VALIDATION_MANIFEST_SHA256,
    EXPECTED_VALIDATION_COUNT,
    best_history_row,
    build_kaggle_datasets_b,
    build_phase_callbacks,
    ensure_experiment_b_output_paths,
    kaggle_source_roots,
    load_execution_config_b,
    load_experiment_b_policy,
    major_confusion_pairs,
    package_results_b,
    plot_learning_curves,
    require_fresh_or_explicit_restart,
    run_full_preflight_b,
    save_confusion_artifacts_b,
    select_candidate,
    sha256_file,
    write_json,
)
from training.kaggle_runtime import validate_runtime_payload  # noqa: E402
from training.validation_comparison import (  # noqa: E402
    BOOTSTRAP_REPETITIONS,
    write_validation_comparison,
)


def require_training_authorization(
    config_path: Path, *, authorize_training: bool
) -> dict[str, object]:
    if not authorize_training:
        raise RuntimeError("TRAINING_DISABLED_BY_USER")
    config = load_execution_config_b(config_path, allow_training=True)
    if config["start_training"] is not True:
        raise RuntimeError("TRAINING_DISABLED_BY_USER")
    return config


def assemble_preflight_payload(
    *,
    runtime: Mapping[str, object],
    data: Mapping[str, object],
    phase1: Mapping[str, object],
    phase2: Mapping[str, object],
    batch_size: int,
) -> dict[str, object]:
    validate_runtime_payload(runtime)
    if batch_size != 32:
        raise RuntimeError("Experiment B batch size must remain 32.")
    for partition, expected in (
        ("train", EXPECTED_TRAIN_COUNT),
        ("validation", EXPECTED_VALIDATION_COUNT),
    ):
        audit = data.get(partition)
        if not isinstance(audit, Mapping) or any(
            (
                audit.get("expected") != expected,
                audit.get("resolved") != expected,
                audit.get("missing") != 0,
                audit.get("unreadable") != 0,
            )
        ):
            raise RuntimeError(f"Experiment B {partition.upper()} preflight failed.")
    if (
        data.get("train_class_coverage") != 39
        or data.get("validation_class_coverage") != 39
    ):
        raise RuntimeError("Experiment B class coverage must remain 39/39.")
    manifest_hashes = data.get("manifest_hashes")
    if not isinstance(manifest_hashes, Mapping) or set(manifest_hashes) != {
        "train",
        "validation",
    }:
        raise RuntimeError("Experiment B manifest hash audit is missing.")
    if dict(manifest_hashes) != {
        "train": EXPECTED_TRAIN_MANIFEST_SHA256,
        "validation": EXPECTED_VALIDATION_MANIFEST_SHA256,
    }:
        raise RuntimeError("Experiment B manifest hashes differ from Experiment A.")
    taxonomy = data.get("taxonomy_audit")
    if not isinstance(taxonomy, Mapping) or dict(taxonomy) != {
        "class_count": 39,
        "class_names_sha256": EXPECTED_TAXONOMY_SHA256,
        "background_class_index": 4,
        "background_class_name": "Background without leaves",
        "shared_with_experiment_a": True,
    }:
        raise RuntimeError("Experiment B taxonomy differs from Experiment A.")
    augmentation = data.get("augmentation_audit")
    if (
        not isinstance(augmentation, Mapping)
        or augmentation.get("validation_augmentation_enabled") is not False
    ):
        raise RuntimeError("Experiment B VALIDATION augmentation must remain disabled.")
    if augmentation.get("values") != APPROVED_AUGMENTATION:
        raise RuntimeError("Experiment B augmentation values changed.")
    policy_audit = data.get("policy_audit")
    if (
        not isinstance(policy_audit, Mapping)
        or policy_audit.get("class_weights") is not None
        or policy_audit.get("primary_variable")
        != "TRAIN_ONLY_AUGMENTATION_POLICY"
    ):
        raise RuntimeError("Experiment B controlled-variable audit failed.")
    if phase1.get("input_shape") != [None, 224, 224, 3]:
        raise RuntimeError("Experiment B input shape changed.")
    if phase1.get("output_shape") != [None, 39]:
        raise RuntimeError("Experiment B output shape changed.")
    if phase1.get("backbone_trainable") is not False:
        raise RuntimeError("Experiment B Phase 1 backbone must remain frozen.")
    if (
        phase1.get("initialization") != "imagenet"
        or phase1.get("production_model_loaded") is not False
        or phase1.get("total_parameters") != 2_307_943
        or phase1.get("trainable_parameters") != 49_959
        or phase1.get("non_trainable_parameters") != 2_257_984
    ):
        raise RuntimeError("Experiment B Phase 1 architecture changed.")
    if phase2.get("first_trainable_backbone_layer") != "block_13_expand":
        raise RuntimeError("Experiment B Phase 2 boundary changed.")
    if (
        phase2.get("fine_tune_boundary_index") != 116
        or phase2.get("total_backbone_layers") != 154
        or phase2.get("trainable_backbone_layer_count") != 25
        or phase2.get("frozen_backbone_layer_count") != 129
        or phase2.get("batch_normalization_layer_count") != 52
        or phase2.get("frozen_batch_normalization_count") != 52
        or phase2.get("total_parameters") != 2_307_943
        or phase2.get("trainable_parameters") != 1_713_319
        or phase2.get("non_trainable_parameters") != 594_624
    ):
        raise RuntimeError("Experiment B BatchNormalization policy changed.")
    if CLASS_WEIGHTS is not None:
        raise RuntimeError("Experiment B class weights must remain disabled.")
    if (
        data.get("internal_test_loaded") is not False
        or data.get("plantdoc_test_loaded") is not False
        or data.get("training_performed") is not False
    ):
        raise RuntimeError("Experiment B preflight violated a locked safety flag.")
    return {
        "status": "KAGGLE_TF215_GPU_EXPERIMENT_B_PREFLIGHT_PASSED",
        "experiment": EXPERIMENT_NAME,
        "runtime": dict(runtime),
        "preflight": dict(data),
        "phase1_model_audit": dict(phase1),
        "phase2_model_audit": dict(phase2),
        "batch_size": 32,
        "class_weights": None,
        "start_training": False,
        "training_performed": False,
        "internal_test_loaded": False,
        "plantdoc_test_loaded": False,
    }


def build_model_preflight_audits(
    policy: Mapping[str, object],
) -> tuple[dict[str, object], dict[str, object]]:
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
    phase2_layers = configure_phase2(backbone, policy)
    compile_phase2(model, policy)
    phase2 = {**parameter_audit(model), **phase2_layers}
    return phase1, phase2


def run_preflight(config_path: Path, output: Path) -> dict[str, object]:
    runtime = verify_runtime(output.with_name("tf215-gpu-runtime.json"))
    config = load_execution_config_b(config_path, allow_training=False)
    roots = kaggle_source_roots(config["source_roots"])
    data = run_full_preflight_b(PROJECT_ROOT, roots)
    policy, _ = load_experiment_b_policy(PROJECT_ROOT)
    phase1, phase2 = build_model_preflight_audits(policy)
    payload = assemble_preflight_payload(
        runtime=runtime,
        data=data,
        phase1=phase1,
        phase2=phase2,
        batch_size=int(config["batch_size"]),
    )
    write_safe_json(output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return payload


def run_training(config_path: Path, *, authorize_training: bool) -> Path:
    config = require_training_authorization(
        config_path, authorize_training=authorize_training
    )
    ensure_experiment_b_output_paths(
        KAGGLE_CANDIDATE_DIR, KAGGLE_RESULTS_DIR, KAGGLE_ARCHIVE_BASE
    )
    runtime = verify_runtime(KAGGLE_RESULTS_DIR / "environment-runtime.json")

    import keras
    import tensorflow as tf

    from training.metrics import MacroF1

    if CLASS_WEIGHTS is not None:
        raise RuntimeError("Experiment B class weights must remain disabled.")
    roots = kaggle_source_roots(config["source_roots"])
    preflight_data = run_full_preflight_b(PROJECT_ROOT, roots)
    policy, _ = load_experiment_b_policy(PROJECT_ROOT)
    batch_size = int(config["batch_size"])
    phase1_audit, phase2_audit = build_model_preflight_audits(policy)
    preflight = assemble_preflight_payload(
        runtime=runtime,
        data=preflight_data,
        phase1=phase1_audit,
        phase2=phase2_audit,
        batch_size=batch_size,
    )
    write_safe_json(KAGGLE_RESULTS_DIR / "preflight.json", preflight)
    train_dataset, validation_dataset, _, validation_records = (
        build_kaggle_datasets_b(PROJECT_ROOT, roots, batch_size=batch_size)
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
    phase2_audit = configure_phase2(phase2_backbone, policy)
    if phase2_audit["first_trainable_backbone_layer"] != "block_13_expand":
        raise RuntimeError("Experiment B Phase 2 boundary changed.")
    if phase2_audit["frozen_batch_normalization_count"] != 52:
        raise RuntimeError("Experiment B BatchNormalization policy changed.")
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
    selected_path = results_dir / "agri-diagnose-v2-exp-b.keras"
    shutil.copy2(Path(selected["checkpoint"]), selected_path)
    metrics = selected["report"]
    metrics["loss"] = selected["val_loss"]
    metrics["accuracy"] = selected["val_accuracy"]
    write_json(results_dir / "validation-metrics.json", metrics)
    save_confusion_artifacts_b(metrics, results_dir)
    plot_learning_curves(phase1_history, phase2_history, results_dir)

    environment = {
        **runtime,
        "experiment": EXPERIMENT_NAME,
        "os_environment": "Kaggle isolated Python 3.11 subprocess",
        "keras_version": keras.__version__,
        "batch_size": batch_size,
        "training_performed": True,
        "internal_test_loaded": False,
        "plantdoc_test_loaded": False,
    }
    experiment = {
        "experiment": EXPERIMENT_NAME,
        "baseline_experiment": "agri-diagnose-v2-exp-a",
        "primary_variable": "TRAIN_ONLY_AUGMENTATION_POLICY",
        "selected_phase": selected["name"],
        "selected_epoch": selected["epoch"],
        "selection_epoch": selected["selection_epoch"],
        "selection_partition": "VALIDATION",
        "selection_metric": "validation_macro_f1",
        "candidate_sha256": sha256_file(selected_path),
        "candidate_size_bytes": selected_path.stat().st_size,
        "train_manifest_sha256": preflight_data["manifest_hashes"]["train"],
        "validation_manifest_sha256": preflight_data["manifest_hashes"]["validation"],
        "policy_sha256": sha256_file(POLICY_PATH),
        "class_weights": None,
        "internal_test_manifest_sha256": preflight_data[
            "internal_test_manifest_sha256"
        ],
        "internal_test_loaded": False,
        "plantdoc_test_loaded": False,
        "test_sets_evaluated": False,
        "training_performed": True,
        "phase1_run_mode": phase1_mode,
        "phase2_run_mode": phase2_mode,
    }
    summary = {
        "experiment": EXPERIMENT_NAME,
        "selected_phase": selected["name"],
        "selected_epoch": selected["epoch"],
        "validation": metrics["overall_validation"],
        "real_world_validation": metrics["real_world_validation"],
        "major_confusion_pairs": major_confusion_pairs(metrics),
        "test_sets_evaluated": False,
        "internal_test_loaded": False,
        "plantdoc_test_loaded": False,
    }
    write_json(results_dir / "environment.json", environment)
    write_json(results_dir / "experiment.json", experiment)
    write_json(results_dir / "preflight.json", preflight)
    write_json(results_dir / "model-v2-exp-b-summary.json", summary)
    (results_dir / "model-v2-exp-b-report.md").write_text(
        "# Model V2 Experiment B\n\n"
        "TRAIN-only augmentation is the sole primary change from Experiment A.\n\n"
        "Candidate selection uses overall VALIDATION Macro-F1 only. INTERNAL TEST "
        "and PlantDoc TEST remain locked and unevaluated.\n",
        encoding="utf-8",
    )
    archive = package_results_b(results_dir, KAGGLE_ARCHIVE_BASE)
    print("Experiment B archive:", archive)
    return archive


def run_validation_comparison(
    experiment_a_dir: Path,
    experiment_b_dir: Path,
    validation_manifest: Path,
    output_dir: Path,
    *,
    bootstrap_repetitions: int = BOOTSTRAP_REPETITIONS,
) -> dict[str, Path]:
    outputs = write_validation_comparison(
        experiment_a_dir,
        experiment_b_dir,
        validation_manifest,
        output_dir,
        bootstrap_repetitions=bootstrap_repetitions,
    )
    print(
        json.dumps(
            {name: path.as_posix() for name, path in outputs.items()},
            indent=2,
            sort_keys=True,
        )
    )
    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run AgriDiagnose Experiment B in isolated Kaggle Python 3.11."
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
    compare = subparsers.add_parser(
        "compare-validation",
        help="Compare finalized A/B VALIDATION artifacts without model inference.",
    )
    compare.add_argument("--experiment-a-dir", type=Path, required=True)
    compare.add_argument("--experiment-b-dir", type=Path, required=True)
    compare.add_argument("--validation-manifest", type=Path, required=True)
    compare.add_argument("--output-dir", type=Path, required=True)
    compare.add_argument(
        "--bootstrap-repetitions", type=int, default=BOOTSTRAP_REPETITIONS
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.action == "verify-runtime":
        verify_runtime(args.output)
    elif args.action == "preflight":
        run_preflight(args.config, args.output)
    elif args.action == "train":
        run_training(args.config, authorize_training=args.authorize_training)
    elif args.action == "compare-validation":
        run_validation_comparison(
            args.experiment_a_dir,
            args.experiment_b_dir,
            args.validation_manifest,
            args.output_dir,
            bootstrap_repetitions=args.bootstrap_repetitions,
        )
    else:  # pragma: no cover
        raise RuntimeError(f"Unsupported action: {args.action}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
