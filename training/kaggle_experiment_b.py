from __future__ import annotations

import hashlib
import json
import math
import shutil
from pathlib import Path
from typing import Mapping

import numpy as np
from sklearn.metrics import confusion_matrix

from scripts.run_model_v2_experiment_a import (
    audit_paths,
    validate_macro_f1,
    validate_tensor_pipeline,
)
from training.data_pipeline import (
    TrainingPolicyError,
    build_augmentation,
    build_tf_dataset,
    load_local_path_aliases,
    load_policy,
    set_experiment_seeds,
)
from training.experiment_b import (
    APPROVED_AUGMENTATION,
    BASELINE_EXPERIMENT_NAME,
    CLASS_WEIGHTS,
    EXPERIMENT_NAME,
    POLICY_PATH,
    validate_experiment_b_policy as validate_policy_identity,
)
from training.kaggle_experiment_a import (
    EXPECTED_TRAIN_COUNT,
    EXPECTED_VALIDATION_COUNT,
    SOURCE_ROOT_KEYS,
    best_history_row,
    build_phase_callbacks,
    configure_gpu_memory_growth,
    headless_pyplot,
    kaggle_source_roots,
    load_train_validation,
    major_confusion_pairs,
    plot_learning_curves,
    require_approved_stack,
    require_fresh_or_explicit_restart,
    require_kaggle_gpu,
    runtime_audit,
    select_candidate,
    sha256_file,
    validation_report,
    verify_internal_test_lock,
    write_json,
)
from training.kaggle_runtime import build_execution_config
from training.taxonomy import CLASS_NAMES


BASELINE_POLICY_PATH = Path("training/config/model-v2-training-policy.json")
BASELINE_POLICY_CANONICAL_LF_SHA256 = (
    "16c16e56819aa96df972f33fb29317fd82fd84e2c9945bf8d4d974c85f682f11"
)
EXPECTED_TRAIN_MANIFEST_SHA256 = (
    "957d4acb4c097116099c57446733b3d70088bf083e7869aadd11e26caf70a915"
)
EXPECTED_VALIDATION_MANIFEST_SHA256 = (
    "9c10de69e935324ee325667fab2902b372a144a722c4fc793d3b4f1afe01767e"
)
EXPECTED_TAXONOMY_SHA256 = (
    "2207c34ff2673bde7f36c53938cf5e6d97ca0652f21ef087be15680851ae87da"
)
CONTROLLED_POLICY_SECTIONS = (
    "input",
    "architecture",
    "batch_size",
    "callbacks",
    "phase1",
    "phase2",
    "optimizer",
    "loss",
    "selection_metrics",
    "locked_test_policy",
    "experiment_seed",
)
FORBIDDEN_A_OUTPUT_MARKERS = (
    "agri-diagnose-v2-exp-a",
    "agridiagnose-exp-a-results",
)
RESULT_FILENAMES = {
    "agri-diagnose-v2-exp-b.keras",
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
    "model-v2-exp-b-summary.json",
    "model-v2-exp-b-report.md",
}


def sha256_with_canonical_lf(path: Path) -> str:
    content = Path(path).read_bytes().replace(b"\r\n", b"\n")
    return hashlib.sha256(content).hexdigest()


def manifest_hashes(project_root: Path) -> dict[str, str]:
    root = Path(project_root)
    values = {
        "train": sha256_with_canonical_lf(
            root / "training/datasets/manifests/dataset-v2-train.csv"
        ),
        "validation": sha256_with_canonical_lf(
            root / "training/datasets/manifests/dataset-v2-validation.csv"
        ),
    }
    expected = {
        "train": EXPECTED_TRAIN_MANIFEST_SHA256,
        "validation": EXPECTED_VALIDATION_MANIFEST_SHA256,
    }
    if values != expected:
        raise TrainingPolicyError(
            f"Experiment B manifest hashes differ from Experiment A: {values}"
        )
    return values


def taxonomy_audit() -> dict[str, object]:
    serialized = json.dumps(
        CLASS_NAMES, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    digest = hashlib.sha256(serialized).hexdigest()
    if (
        len(CLASS_NAMES) != 39
        or CLASS_NAMES[4] != "Background without leaves"
        or digest != EXPECTED_TAXONOMY_SHA256
    ):
        raise TrainingPolicyError("Experiment B taxonomy differs from Experiment A.")
    return {
        "class_count": 39,
        "class_names_sha256": digest,
        "background_class_index": 4,
        "background_class_name": CLASS_NAMES[4],
        "shared_with_experiment_a": True,
    }


def validate_experiment_b_policy(
    policy: Mapping[str, object], baseline: Mapping[str, object]
) -> dict[str, object]:
    validate_policy_identity(policy)
    experiment = policy.get("experiment")
    if not isinstance(experiment, Mapping):  # guarded above, kept for type narrowing
        raise TrainingPolicyError("Experiment B policy identity is missing.")
    if policy.get("class_weights") is not None or CLASS_WEIGHTS is not None:
        raise TrainingPolicyError("Experiment B class weights must remain disabled.")
    if int(policy.get("experiment_seed", -1)) != 20260810:
        raise TrainingPolicyError("Experiment B seed must remain 20260810.")
    changed = [
        section
        for section in CONTROLLED_POLICY_SECTIONS
        if policy.get(section) != baseline.get(section)
    ]
    if changed:
        raise TrainingPolicyError(
            f"Experiment B controlled policy sections changed: {changed}"
        )
    augmentation = policy.get("augmentation")
    allowed_augmentation_keys = set(APPROVED_AUGMENTATION) | {"forbidden_methods"}
    if (
        not isinstance(augmentation, Mapping)
        or set(augmentation) != allowed_augmentation_keys
        or any(
            augmentation.get(key) != value
            for key, value in APPROVED_AUGMENTATION.items()
        )
    ):
        raise TrainingPolicyError(
            "Experiment B augmentation does not match the approved policy."
        )
    return {
        "baseline": BASELINE_EXPERIMENT_NAME,
        "primary_variable": "TRAIN_ONLY_AUGMENTATION_POLICY",
        "controlled_sections_equal": list(CONTROLLED_POLICY_SECTIONS),
        "class_weights": None,
        "augmentation": dict(APPROVED_AUGMENTATION),
    }


def load_experiment_b_policy(project_root: Path) -> tuple[dict, dict[str, object]]:
    root = Path(project_root)
    baseline_path = root / BASELINE_POLICY_PATH
    if sha256_with_canonical_lf(baseline_path) != BASELINE_POLICY_CANONICAL_LF_SHA256:
        raise TrainingPolicyError("Finalized Experiment A policy bytes changed.")
    baseline = load_policy(baseline_path)
    policy = load_policy(POLICY_PATH)
    return policy, validate_experiment_b_policy(policy, baseline)


def augmentation_audit(policy: Mapping[str, object]) -> dict[str, object]:
    augmentation = policy.get("augmentation")
    if (
        not isinstance(augmentation, Mapping)
        or set(augmentation) != set(APPROVED_AUGMENTATION) | {"forbidden_methods"}
        or any(
            augmentation.get(key) != value
            for key, value in APPROVED_AUGMENTATION.items()
        )
    ):
        raise TrainingPolicyError("Experiment B augmentation audit failed.")
    augmenter = build_augmentation(policy)
    layer_names = [layer.__class__.__name__ for layer in augmenter.layers]
    expected_layers = [
        "RandomFlip",
        "RandomRotation",
        "RandomTranslation",
        "RandomZoom",
        "RandomBrightness",
        "RandomContrast",
    ]
    if layer_names != expected_layers:
        raise TrainingPolicyError(
            f"Experiment B added or removed an augmentation family: {layer_names}"
        )
    configs = [layer.get_config() for layer in augmenter.layers]
    expected_zoom = (-0.15, 0.15)
    effective_checks = (
        configs[0].get("mode") == "horizontal_and_vertical",
        math.isclose(float(configs[1].get("factor", -1)), 20.0 / 360.0),
        configs[1].get("fill_mode") == "reflect",
        math.isclose(float(configs[2].get("height_factor", -1)), 0.12),
        math.isclose(float(configs[2].get("width_factor", -1)), 0.12),
        configs[2].get("fill_mode") == "reflect",
        all(
            math.isclose(float(actual), expected)
            for actual, expected in zip(
                configs[3].get("height_factor", ()), expected_zoom
            )
        ),
        all(
            math.isclose(float(actual), expected)
            for actual, expected in zip(
                configs[3].get("width_factor", ()), expected_zoom
            )
        ),
        configs[3].get("fill_mode") == "reflect",
        all(
            math.isclose(float(actual), expected)
            for actual, expected in zip(
                configs[4].get("factor", ()), (-0.15, 0.15)
            )
        ),
        list(configs[4].get("value_range", ())) == [0.0, 1.0],
        math.isclose(float(configs[5].get("factor", -1)), 0.15),
    )
    if not all(effective_checks):
        raise TrainingPolicyError("Effective Experiment B augmentation changed.")
    return {
        "enabled_for": ["TRAIN"],
        "validation_augmentation_enabled": False,
        "layer_order": layer_names,
        "layer_seeds": [config.get("seed") for config in configs],
        "values": dict(APPROVED_AUGMENTATION),
        "output_clipping": [0.0, 1.0],
    }


def build_execution_config_b(
    source_roots: Mapping[str, str | Path],
    *,
    batch_size: int = 32,
    start_training: bool = False,
    restart_interrupted_phase: bool = False,
) -> dict[str, object]:
    if batch_size != 32:
        raise ValueError("Experiment B batch size is locked to 32.")
    payload = build_execution_config(
        source_roots,
        batch_size=batch_size,
        start_training=start_training,
        restart_interrupted_phase=restart_interrupted_phase,
    )
    payload["experiment"] = EXPERIMENT_NAME
    return payload


def load_execution_config_b(
    path: Path, *, allow_training: bool = False
) -> dict[str, object]:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError("Experiment B execution config is invalid.") from exc
    if payload.get("experiment") != EXPERIMENT_NAME:
        raise RuntimeError("Experiment B execution config identity mismatch.")
    validated = build_execution_config_b(
        payload.get("source_roots", {}),
        batch_size=int(payload.get("batch_size", 0)),
        start_training=bool(payload.get("start_training", False)),
        restart_interrupted_phase=bool(
            payload.get("restart_interrupted_phase", False)
        ),
    )
    if validated["start_training"] and not allow_training:
        raise RuntimeError("TRAINING_DISABLED_BY_USER")
    return validated


def ensure_experiment_b_output_paths(
    candidate_dir: Path, results_dir: Path, archive_base: Path
) -> None:
    paths = [
        Path(candidate_dir).expanduser().resolve(),
        Path(results_dir).expanduser().resolve(),
        Path(archive_base).expanduser().resolve(),
    ]
    normalized = [path.as_posix().casefold() for path in paths]
    for value in normalized:
        if any(marker in value for marker in FORBIDDEN_A_OUTPUT_MARKERS):
            raise RuntimeError("Experiment B output path targets Experiment A artifacts.")
    if "agri-diagnose-v2-exp-b" not in normalized[0]:
        raise RuntimeError("Experiment B candidate directory identity is invalid.")
    if "agridiagnose-exp-b-results" not in normalized[1]:
        raise RuntimeError("Experiment B results directory identity is invalid.")
    if "agridiagnose-exp-b-results" not in normalized[2]:
        raise RuntimeError("Experiment B archive identity is invalid.")


def run_full_preflight_b(
    project_root: Path, roots: Mapping[str, Path]
) -> dict[str, object]:
    root = Path(project_root)
    policy, policy_audit = load_experiment_b_policy(root)
    seed = set_experiment_seeds(policy)
    train, validation = load_train_validation(root)
    aliases = load_local_path_aliases(
        {
            "PlantDoc": root / "training/datasets/manifests/plantdoc-train.csv",
            "Potato Leaf Disease Dataset": root
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
            raise TrainingPolicyError(f"Experiment B {name} preflight failed: {audit}")
    tensor_audit = validate_tensor_pipeline(
        train, validation, policy, roots, aliases
    )
    return {
        "experiment": EXPERIMENT_NAME,
        "baseline": BASELINE_EXPERIMENT_NAME,
        "seed": seed,
        "train": train_audit,
        "validation": validation_audit,
        "train_class_coverage": len({row.target_index for row in train}),
        "validation_class_coverage": len({row.target_index for row in validation}),
        "manifest_hashes": manifest_hashes(root),
        "taxonomy_audit": taxonomy_audit(),
        "policy_audit": policy_audit,
        "augmentation_audit": augmentation_audit(policy),
        "tensor_pipeline": tensor_audit,
        "macro_f1": validate_macro_f1(),
        "internal_test_manifest_sha256": verify_internal_test_lock(root),
        "internal_test_loaded": False,
        "plantdoc_test_loaded": False,
        "training_performed": False,
    }


def build_kaggle_datasets_b(
    project_root: Path,
    roots: Mapping[str, Path],
    *,
    batch_size: int = 32,
):
    root = Path(project_root)
    policy, _ = load_experiment_b_policy(root)
    train, validation = load_train_validation(root)
    aliases = load_local_path_aliases(
        {
            "PlantDoc": root / "training/datasets/manifests/plantdoc-train.csv",
            "Potato Leaf Disease Dataset": root
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


def save_confusion_artifacts_b(
    report: Mapping[str, object], output_dir: Path
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
    pd.DataFrame(matrix, index=CLASS_NAMES, columns=CLASS_NAMES).to_csv(csv_path)
    figure, axis = plt.subplots(figsize=(18, 16))
    image = axis.imshow(matrix, cmap="Blues")
    axis.set_title("Experiment B - VALIDATION confusion matrix")
    axis.set_xlabel("Predicted")
    axis.set_ylabel("True")
    axis.set_xticks(range(39), CLASS_NAMES, rotation=90, fontsize=6)
    axis.set_yticks(range(39), CLASS_NAMES, fontsize=6)
    figure.colorbar(image, ax=axis)
    figure.tight_layout()
    figure.savefig(png_path, dpi=160)
    plt.close(figure)
    return csv_path, png_path


def package_results_b(results_dir: Path, archive_base: Path) -> Path:
    results_dir = Path(results_dir)
    missing = sorted(
        name for name in RESULT_FILENAMES if not (results_dir / name).is_file()
    )
    if missing:
        raise RuntimeError(f"Experiment B result package is incomplete: {missing}")
    return Path(shutil.make_archive(str(archive_base), "zip", root_dir=results_dir))


__all__ = [
    "EXPECTED_TRAIN_COUNT",
    "EXPECTED_VALIDATION_COUNT",
    "EXPECTED_TAXONOMY_SHA256",
    "SOURCE_ROOT_KEYS",
    "best_history_row",
    "build_execution_config_b",
    "build_kaggle_datasets_b",
    "build_phase_callbacks",
    "configure_gpu_memory_growth",
    "ensure_experiment_b_output_paths",
    "kaggle_source_roots",
    "load_execution_config_b",
    "major_confusion_pairs",
    "manifest_hashes",
    "package_results_b",
    "plot_learning_curves",
    "require_approved_stack",
    "require_fresh_or_explicit_restart",
    "require_kaggle_gpu",
    "run_full_preflight_b",
    "runtime_audit",
    "save_confusion_artifacts_b",
    "select_candidate",
    "sha256_file",
    "taxonomy_audit",
    "validation_report",
    "write_json",
]
