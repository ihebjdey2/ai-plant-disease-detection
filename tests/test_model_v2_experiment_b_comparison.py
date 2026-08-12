from __future__ import annotations

import csv
import inspect
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pytest

import training.validation_comparison as comparison_module
from training.taxonomy import CLASS_NAMES
from training.validation_comparison import (
    BOOTSTRAP_SEED,
    COMPARISON_JSON,
    COMPARISON_MARKDOWN,
    CONFUSION_CSV,
    EXPERIMENT_A,
    EXPERIMENT_B,
    EXPECTED_REAL_WORLD_CLASS_INDICES,
    EXPECTED_VALIDATION_COUNT,
    PER_CLASS_CSV,
    ValidationComparisonError,
    build_validation_comparison,
    paired_class_aware_bootstrap,
    sha256_file,
    sha256_with_canonical_lf,
    write_validation_comparison,
)


REAL_WORLD_COUNTS = {
    0: 8,
    8: 95,
    9: 13,
    11: 103,
    12: 6,
    21: 476,
    22: 611,
    23: 438,
    26: 13,
    29: 10,
    30: 7,
    31: 9,
    32: 9,
    33: 13,
    37: 5,
}


def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def create_manifest(path: Path) -> tuple[list[dict[str, str]], list[int], list[int]]:
    rows: list[dict[str, str]] = []

    def add(target: int, domain: str) -> None:
        number = len(rows)
        rows.append(
            {
                "composition_record_id": f"validation_{number:05d}",
                "source_domain": domain,
                "target_index": str(target),
                "target_class": CLASS_NAMES[target],
                "split": "VALIDATION",
                "evaluation_role": "MODEL_DEVELOPMENT_VALIDATION",
            }
        )

    for target, count in REAL_WORLD_COUNTS.items():
        for _ in range(count):
            add(target, "REAL_WORLD")
    remaining = EXPECTED_VALIDATION_COUNT - len(rows)
    for index in range(remaining):
        add(index % len(CLASS_NAMES), "HISTORICAL_CONTROLLED")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    true = [int(row["target_index"]) for row in rows]
    real = [index for index, row in enumerate(rows) if row["source_domain"] == "REAL_WORLD"]
    return rows, true, real


def predictions_for(true: list[int]) -> tuple[list[int], list[int]]:
    positions: defaultdict[int, list[int]] = defaultdict(list)
    for index, target in enumerate(true):
        positions[target].append(index)
    predicted_a = list(true)
    predicted_b = list(true)

    def confuse(predictions: list[int], target: int, predicted: int, count: int) -> None:
        for index in positions[target][:count]:
            predictions[index] = predicted

    confuse(predicted_a, 21, 22, 20)
    confuse(predicted_b, 21, 22, 8)
    confuse(predicted_a, 22, 21, 10)
    confuse(predicted_b, 22, 21, 14)
    confuse(predicted_a, 30, 31, 10)
    confuse(predicted_b, 30, 31, 4)
    confuse(predicted_a, 32, 30, 8)
    confuse(predicted_b, 32, 30, 3)
    confuse(predicted_a, 29, 30, 1)
    confuse(predicted_b, 29, 30, 2)
    return predicted_a, predicted_b


def metric_payload(
    true: list[int], predicted: list[int], real_indices: list[int], loss: float
) -> dict[str, object]:
    true_array = np.asarray(true, dtype=np.int32)
    predicted_array = np.asarray(predicted, dtype=np.int32)
    overall = comparison_module._metric_block(
        true_array,
        predicted_array,
        np.arange(len(true), dtype=np.int32),
        list(range(39)),
    )
    real = np.asarray(real_indices, dtype=np.int32)
    real_labels = sorted(set(int(true_array[index]) for index in real))
    real_metrics = comparison_module._metric_block(
        true_array, predicted_array, real, real_labels
    )
    return {
        "loss": loss,
        "accuracy": overall["accuracy"],
        "overall_validation": overall,
        "real_world_validation": real_metrics,
        "true_indices": true,
        "predicted_indices": predicted,
    }


def create_artifacts(
    root: Path,
    experiment_name: str,
    manifest_hash: str,
    metrics: dict[str, object],
) -> None:
    root.mkdir()
    suffix = "a" if experiment_name == EXPERIMENT_A else "b"
    write_json(root / "validation-metrics.json", metrics)
    write_json(
        root / "experiment.json",
        {
            "experiment": experiment_name,
            "validation_manifest_sha256": manifest_hash,
            "internal_test_loaded": False,
            "plantdoc_test_loaded": False,
        },
    )
    write_json(
        root / "preflight.json",
        {"internal_test_loaded": False, "plantdoc_test_loaded": False},
    )
    write_json(
        root / f"model-v2-exp-{suffix}-summary.json",
        {"test_sets_evaluated": False},
    )
    (root / f"agri-diagnose-v2-exp-{suffix}.keras").write_bytes(
        f"untouched-{suffix}-model".encode()
    )


@pytest.fixture
def paired_artifacts(tmp_path):
    manifest = tmp_path / "dataset-v2-validation.csv"
    _, true, real = create_manifest(manifest)
    predicted_a, predicted_b = predictions_for(true)
    manifest_hash = sha256_with_canonical_lf(manifest)
    a_root = tmp_path / "experiment-a"
    b_root = tmp_path / "experiment-b"
    create_artifacts(
        a_root, EXPERIMENT_A, manifest_hash, metric_payload(true, predicted_a, real, 0.15)
    )
    create_artifacts(
        b_root, EXPERIMENT_B, manifest_hash, metric_payload(true, predicted_b, real, 0.13)
    )
    return {"manifest": manifest, "a": a_root, "b": b_root, "output": b_root / "comparison"}


def test_validation_comparison_generates_requested_outputs_without_touching_models(
    paired_artifacts,
):
    model_a = paired_artifacts["a"] / "agri-diagnose-v2-exp-a.keras"
    model_b = paired_artifacts["b"] / "agri-diagnose-v2-exp-b.keras"
    before = {path: (path.read_bytes(), path.stat().st_mtime_ns) for path in (model_a, model_b)}
    outputs = write_validation_comparison(
        paired_artifacts["a"],
        paired_artifacts["b"],
        paired_artifacts["manifest"],
        paired_artifacts["output"],
        bootstrap_repetitions=12,
    )
    assert {path.name for path in outputs.values()} == {
        COMPARISON_JSON,
        COMPARISON_MARKDOWN,
        PER_CLASS_CSV,
        CONFUSION_CSV,
    }
    assert all(path.is_file() for path in outputs.values())
    assert before == {
        path: (path.read_bytes(), path.stat().st_mtime_ns) for path in (model_a, model_b)
    }
    report = json.loads(outputs["json"].read_text(encoding="utf-8"))
    assert report["overall_validation"]["delta_b_minus_a"]["loss"] == pytest.approx(-0.02)
    assert report["real_world_validation"]["delta_b_minus_a"]["macro_f1"] > 0
    assert report["tomato_aggregate"]["real_world_validation"]["experiment_a"]["image_count"] == 53
    potato = report["potato_early_late_bidirectional_confusion"]["overall_validation"]
    assert potato["delta"]["early_to_late_count"] == -12
    assert potato["delta"]["late_to_early_count"] == 4
    assert report["paired_class_aware_bootstrap"]["real_world_validation"]["seed"] == BOOTSTRAP_SEED
    assert report["paired_class_aware_bootstrap"]["real_world_validation"]["repetitions"] == 12
    assert report["safety"] == {
        "partition": "VALIDATION_ONLY",
        "internal_test_loaded": False,
        "plantdoc_test_loaded": False,
        "test_sets_evaluated": False,
        "models_loaded": False,
        "images_loaded": False,
        "inference_performed": False,
    }


def test_bootstrap_is_paired_class_aware_and_deterministic(paired_artifacts):
    first, _, _ = build_validation_comparison(
        paired_artifacts["a"], paired_artifacts["b"], paired_artifacts["manifest"], bootstrap_repetitions=9
    )
    second, _, _ = build_validation_comparison(
        paired_artifacts["a"], paired_artifacts["b"], paired_artifacts["manifest"], bootstrap_repetitions=9
    )
    assert first["paired_class_aware_bootstrap"] == second["paired_class_aware_bootstrap"]
    real = first["paired_class_aware_bootstrap"]["real_world_validation"]
    assert sum(real["class_support"].values()) == 1_816
    assert len(real["class_support"]) == len(EXPECTED_REAL_WORLD_CLASS_INDICES)


@pytest.mark.parametrize(
    ("artifact", "key", "message"),
    [
        ("experiment.json", "internal_test_loaded", "internal_test_loaded"),
        ("preflight.json", "plantdoc_test_loaded", "plantdoc_test_loaded"),
        ("model-v2-exp-b-summary.json", "test_sets_evaluated", "test_sets_evaluated"),
    ],
)
def test_comparison_rejects_any_test_access_flag(
    paired_artifacts, artifact, key, message
):
    path = paired_artifacts["b"] / artifact
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload[key] = True
    write_json(path, payload)
    with pytest.raises(ValidationComparisonError, match=message):
        build_validation_comparison(
            paired_artifacts["a"],
            paired_artifacts["b"],
            paired_artifacts["manifest"],
            bootstrap_repetitions=1,
        )


def test_comparison_rejects_test_manifest_without_reading_it(paired_artifacts):
    forbidden = paired_artifacts["manifest"].with_name("dataset-v2-test.csv")
    forbidden.write_bytes(paired_artifacts["manifest"].read_bytes())
    with pytest.raises(ValidationComparisonError, match="TEST-like"):
        build_validation_comparison(
            paired_artifacts["a"], paired_artifacts["b"], forbidden, bootstrap_repetitions=1
        )


def test_validation_manifest_hash_is_portable_between_lf_and_crlf(tmp_path):
    manifest = tmp_path / "dataset-v2-validation.csv"
    create_manifest(manifest)
    lf = manifest.read_bytes().replace(b"\r\n", b"\n")
    manifest.write_bytes(lf)
    expected = sha256_with_canonical_lf(manifest)
    manifest.write_bytes(lf.replace(b"\n", b"\r\n"))
    assert sha256_with_canonical_lf(manifest) == expected


def test_comparison_fails_closed_on_prediction_manifest_mismatch(paired_artifacts):
    path = paired_artifacts["b"] / "validation-metrics.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["true_indices"][0] = 1
    write_json(path, payload)
    with pytest.raises(ValidationComparisonError, match="manifest"):
        build_validation_comparison(
            paired_artifacts["a"],
            paired_artifacts["b"],
            paired_artifacts["manifest"],
            bootstrap_repetitions=1,
        )


def test_comparison_fails_closed_on_inconsistent_stored_metrics(paired_artifacts):
    path = paired_artifacts["a"] / "validation-metrics.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["overall_validation"]["macro_f1"] = 0.123
    write_json(path, payload)
    with pytest.raises(ValidationComparisonError, match="inconsistent"):
        build_validation_comparison(
            paired_artifacts["a"],
            paired_artifacts["b"],
            paired_artifacts["manifest"],
            bootstrap_repetitions=1,
        )


def test_comparison_cannot_write_inside_immutable_experiment_a(paired_artifacts):
    with pytest.raises(ValidationComparisonError, match="must not modify Experiment A"):
        write_validation_comparison(
            paired_artifacts["a"],
            paired_artifacts["b"],
            paired_artifacts["manifest"],
            paired_artifacts["a"] / "comparison",
            bootstrap_repetitions=1,
        )


def test_bootstrap_seed_is_locked():
    values = np.asarray([0, 0, 1, 1], dtype=np.int32)
    with pytest.raises(ValidationComparisonError, match="seed is locked"):
        paired_class_aware_bootstrap(
            values,
            values,
            values,
            np.arange(4, dtype=np.int32),
            repetitions=1,
            seed=1,
        )


def test_comparison_module_has_no_tensorflow_model_or_image_dependency():
    source = inspect.getsource(comparison_module)
    assert "import tensorflow" not in source
    assert "load_model" not in source
    assert ".fit(" not in source
    assert ".predict(" not in source
    assert "PIL" not in source
