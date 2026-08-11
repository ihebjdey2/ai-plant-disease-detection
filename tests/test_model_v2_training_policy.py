from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from training.taxonomy import CLASS_NAMES
from scripts.prepare_model_v2_training import (
    analyze_class_weights,
    run_dry_run,
)
from training.data_pipeline import (
    DEFAULT_POLICY_PATH,
    ManifestRecord,
    TrainingPolicyError,
    augmentation_enabled,
    build_tf_dataset,
    compute_validation_metrics,
    load_development_manifest,
    load_policy,
    preprocess_image,
    resolve_record_path,
    validate_policy,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_FIELDS = [
    "composition_record_id",
    "source_domain",
    "source_dataset",
    "source_path",
    "target_index",
    "target_class",
    "split",
    "evaluation_role",
]


def manifest_row(index: int, split: str, number: int = 0) -> dict[str, str]:
    role = {
        "TRAIN": "MODEL_TRAINING",
        "VALIDATION": "MODEL_DEVELOPMENT_VALIDATION",
        "TEST": "FINAL_INTERNAL_TEST",
    }[split]
    return {
        "composition_record_id": f"comp-{split.lower()}-{index}-{number}",
        "source_domain": "HISTORICAL_CONTROLLED",
        "source_dataset": "Historical Mendeley 39-class source",
        "source_path": f"images/{split.lower()}-{index}-{number}.png",
        "target_index": str(index),
        "target_class": CLASS_NAMES[index],
        "split": split,
        "evaluation_role": role,
    }


def write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def records_for_weights(counts: list[int]) -> list[ManifestRecord]:
    return [
        ManifestRecord(
            composition_record_id=f"comp-{index}-{number}",
            source_domain="HISTORICAL_CONTROLLED",
            source_dataset="Historical Mendeley 39-class source",
            source_path=f"images/{index}-{number}.png",
            target_index=index,
            target_class=CLASS_NAMES[index],
            split="TRAIN",
            evaluation_role="MODEL_TRAINING",
        )
        for index, count in enumerate(counts)
        for number in range(count)
    ]


def test_authoritative_train_counts_are_loaded_correctly():
    records = load_development_manifest(
        PROJECT_ROOT / "training/datasets/manifests/dataset-v2-train.csv",
        training=True,
    )
    counts = {index: 0 for index in range(39)}
    for record in records:
        counts[record.target_index] += 1
    assert len(records) == 58_857
    assert counts[2] == 220
    assert counts[22] == 5_685
    assert set(counts) == set(range(39))


def test_class_weights_accept_train_only():
    train = records_for_weights([1] * 39)
    rows, stats = analyze_class_weights(train)
    assert len(rows) == 39
    assert stats["train_count"] == 39
    validation = [
        ManifestRecord(**{**record.__dict__, "split": "VALIDATION"}) for record in train
    ]
    with pytest.raises(TrainingPolicyError, match="TRAIN records only"):
        analyze_class_weights(validation)


def test_recommended_class_weights_are_clipped():
    rows, _ = analyze_class_weights(
        records_for_weights([10_000, *([1] * 38)]),
        clip_minimum=1.0,
        clip_maximum=2.0,
    )
    weights = [float(row["clipped_recommended_weight"]) for row in rows]
    assert min(weights) == 1.0
    assert max(weights) == 2.0


def test_target_indices_remain_zero_through_38():
    records = records_for_weights([2] * 39)
    assert {record.target_index for record in records} == set(range(39))
    rows, _ = analyze_class_weights(records)
    assert [row["target_index"] for row in rows] == list(range(39))


def test_augmentation_is_enabled_only_for_train():
    assert augmentation_enabled(training=True, split="TRAIN") is True
    assert augmentation_enabled(training=False, split="TRAIN") is False
    assert augmentation_enabled(training=False, split="VALIDATION") is False
    with pytest.raises(TrainingPolicyError, match="forbidden"):
        augmentation_enabled(training=False, split="TEST")


def test_validation_manifest_has_no_random_augmentation(tmp_path):
    manifest = tmp_path / "validation.csv"
    write_manifest(manifest, [manifest_row(index, "VALIDATION") for index in range(39)])
    records = load_development_manifest(manifest, training=False)
    assert {record.split for record in records} == {"VALIDATION"}
    assert not augmentation_enabled(training=False, split="VALIDATION")


def test_internal_test_development_guard(tmp_path):
    manifest = tmp_path / "test.csv"
    write_manifest(manifest, [manifest_row(index, "TEST") for index in range(39)])
    with pytest.raises(TrainingPolicyError, match="Internal TEST is frozen"):
        load_development_manifest(manifest, training=False)
    with pytest.raises(TrainingPolicyError, match="Internal TEST is frozen"):
        load_development_manifest(manifest, training=True)


def test_plantdoc_test_development_guard(tmp_path):
    manifest = tmp_path / "plantdoc.csv"
    row = manifest_row(0, "TRAIN")
    row["source_dataset"] = "PlantDoc"
    row["source_path"] = "test/Apple Scab Leaf/image.jpg"
    write_manifest(manifest, [row])
    with pytest.raises(TrainingPolicyError, match="PlantDoc TEST is frozen"):
        load_development_manifest(manifest, training=True)


def test_preprocessing_is_rgb_float32_224_and_divided_by_255(tmp_path):
    path = tmp_path / "sample.png"
    pixels = np.zeros((10, 12, 3), dtype=np.uint8)
    pixels[..., 0] = 255
    pixels[..., 1] = 128
    Image.fromarray(pixels, mode="RGB").save(path)
    image = preprocess_image(path, load_policy())
    assert image.shape == (224, 224, 3)
    assert image.dtype == np.float32
    assert float(image.min()) >= 0.0
    assert float(image.max()) <= 1.0
    assert np.isclose(image[0, 0, 0], 1.0)
    assert np.isclose(image[0, 0, 1], 128.0 / 255.0)


def test_validation_tf_pipeline_has_deterministic_shape_and_range(tmp_path):
    image_path = tmp_path / "images" / "validation.png"
    image_path.parent.mkdir()
    Image.new("RGB", (20, 14), (255, 64, 0)).save(image_path)
    record = ManifestRecord(
        composition_record_id="comp-validation",
        source_domain="REAL_WORLD",
        source_dataset="PlantDoc",
        source_path="images/validation.png",
        target_index=0,
        target_class=CLASS_NAMES[0],
        split="VALIDATION",
        evaluation_role="MODEL_DEVELOPMENT_VALIDATION",
    )
    dataset = build_tf_dataset(
        [record],
        load_policy(),
        {"PlantDoc": tmp_path},
        training=False,
        batch_size=1,
    )
    images, labels = next(iter(dataset))
    assert tuple(images.shape) == (1, 224, 224, 3)
    assert images.dtype.name == "float32"
    assert float(np.min(images.numpy())) >= 0.0
    assert float(np.max(images.numpy())) <= 1.0
    assert labels.numpy().tolist() == [0]


def test_train_tf_augmentation_remains_in_zero_one_range(tmp_path):
    image_path = tmp_path / "images" / "train.png"
    image_path.parent.mkdir()
    Image.new("RGB", (20, 14), (255, 255, 255)).save(image_path)
    record = ManifestRecord(
        composition_record_id="comp-train",
        source_domain="REAL_WORLD",
        source_dataset="PlantDoc",
        source_path="images/train.png",
        target_index=0,
        target_class=CLASS_NAMES[0],
        split="TRAIN",
        evaluation_role="MODEL_TRAINING",
    )
    dataset = build_tf_dataset(
        [record],
        load_policy(),
        {"PlantDoc": tmp_path},
        training=True,
        batch_size=1,
    )
    images, _ = next(iter(dataset))
    assert tuple(images.shape) == (1, 224, 224, 3)
    assert float(np.min(images.numpy())) >= 0.0
    assert float(np.max(images.numpy())) <= 1.0


def test_validation_metrics_report_real_world_supported_slice_only():
    records = [
        ManifestRecord(
            composition_record_id=f"comp-validation-{index}",
            source_domain=domain,
            source_dataset="PlantDoc" if domain == "REAL_WORLD" else "Historical",
            source_path=f"images/{index}.png",
            target_index=index,
            target_class=CLASS_NAMES[index],
            split="VALIDATION",
            evaluation_role="MODEL_DEVELOPMENT_VALIDATION",
        )
        for index, domain in enumerate(
            ["HISTORICAL_CONTROLLED", "REAL_WORLD", "REAL_WORLD"]
        )
    ]
    scores = np.zeros((3, 39), dtype=np.float32)
    scores[np.arange(3), np.arange(3)] = 1.0
    metrics = compute_validation_metrics(records, scores)
    assert metrics["overall_validation"]["macro_f1"] == 1.0
    assert metrics["real_world_validation"]["supported_class_indices"] == [1, 2]
    assert metrics["real_world_validation"]["image_count"] == 2


def test_policy_config_validation_rejects_incompatible_scaling():
    policy = load_policy(DEFAULT_POLICY_PATH)
    changed = json.loads(json.dumps(policy))
    changed["input"]["scaling"] = "mobilenet_v2.preprocess_input"
    with pytest.raises(TrainingPolicyError, match="pixel / 255.0"):
        validate_policy(changed)


def test_path_resolution_rejects_manifest_escape(tmp_path):
    record = ManifestRecord(
        composition_record_id="comp-escape",
        source_domain="HISTORICAL_CONTROLLED",
        source_dataset="Historical Mendeley 39-class source",
        source_path="../secret.png",
        target_index=0,
        target_class=CLASS_NAMES[0],
        split="TRAIN",
        evaluation_role="MODEL_TRAINING",
    )
    with pytest.raises(TrainingPolicyError, match="escapes"):
        resolve_record_path(
            record, {"Historical Mendeley 39-class source": tmp_path / "raw"}
        )


def dry_run_args(tmp_path: Path) -> argparse.Namespace:
    return argparse.Namespace(
        policy=DEFAULT_POLICY_PATH,
        train_manifest=tmp_path / "train.csv",
        validation_manifest=tmp_path / "validation.csv",
        weight_report=tmp_path / "weights.csv",
        policy_summary=tmp_path / "summary.json",
        sample_size=4,
        no_write_reports=False,
    )


def test_dry_run_is_deterministic_and_checks_local_sample(tmp_path):
    args = dry_run_args(tmp_path)
    train_rows = [manifest_row(index, "TRAIN") for index in range(39)]
    validation_rows = [manifest_row(index, "VALIDATION") for index in range(39)]
    write_manifest(args.train_manifest, train_rows)
    write_manifest(args.validation_manifest, validation_rows)
    images = tmp_path / "raw" / "images"
    images.mkdir(parents=True)
    for row in [*train_rows, *validation_rows]:
        value = int(row["target_index"]) * 5
        Image.new("RGB", (16, 12), (value, 20, 40)).save(
            tmp_path / "raw" / row["source_path"]
        )
    overrides = {"Historical Mendeley 39-class source": tmp_path / "raw"}

    first = run_dry_run(args, source_root_overrides=overrides)
    first_weights = args.weight_report.read_bytes()
    first_summary = args.policy_summary.read_bytes()
    second = run_dry_run(args, source_root_overrides=overrides)

    assert first == second
    assert args.weight_report.read_bytes() == first_weights
    assert args.policy_summary.read_bytes() == first_summary
    assert first["sample_audit"]["sample_preprocessing_status"] == "PASSED"
    assert first["sample_audit"]["inspected_image_count"] == 4
    assert all(
        sample["shape"] == [224, 224, 3]
        for sample in first["sample_audit"]["samples"]
    )
    assert first["internal_test_loaded"] is False
    assert first["training_performed"] is False
