import csv
import hashlib
import io
import zipfile
from pathlib import Path

import pytest
from PIL import Image

from scripts.audit_dataset_v2_sources import (
    DatasetAuditError,
    ZipSource,
    apply_candidate_decisions,
    audit_index,
    compact_perceptual_report,
    exact_duplicate_groups,
    extract_plantdoc_train,
    extract_potato_originals,
    extract_verified_original_archives,
    load_mapping,
    perceptual_duplicate_pairs,
    safe_component,
    write_csv,
)


def image_bytes(color=(20, 100, 30), image_format="JPEG"):
    buffer = io.BytesIO()
    Image.new("RGB", (12, 8), color=color).save(buffer, format=image_format)
    return buffer.getvalue()


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_safe_component_replaces_windows_invalid_characters():
    assert safe_component("Tomato leaf? 1.jpg") == "Tomato leaf_ 1.jpg"
    assert safe_component("CON") == "_CON"
    assert safe_component("...") == "unnamed"


def test_plantdoc_train_materialization_preserves_source(tmp_path):
    source = tmp_path / "plantdoc"
    source_image = source / "train" / "Tomato leaf" / "leaf.jpg"
    source_image.parent.mkdir(parents=True)
    source_image.write_bytes(image_bytes())
    source_hash = sha256(source_image)
    destination = tmp_path / "raw" / "plantdoc-train"

    index = extract_plantdoc_train(source, destination)

    assert index["source_image_count"] == 1
    assert index["materialized_image_count"] == 1
    assert index["corrupted"] == []
    local_file = destination / index["files"][0]["local_file"]
    assert local_file.is_file()
    assert sha256(local_file) == source_hash
    assert sha256(source_image) == source_hash


def test_potato_extraction_keeps_only_originals_and_preserves_archive(tmp_path):
    archive_path = tmp_path / "potato.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("Dataset/Healthy/orig_1.jpg", image_bytes())
        archive.writestr("Dataset/Healthy/aug_1.jpg", image_bytes((40, 120, 40)))
    archive_hash = sha256(archive_path)
    destination = tmp_path / "raw" / "potato"

    index = extract_potato_originals(archive_path, destination, archive_hash)

    assert index["archive_file_count"] == 2
    assert index["original_named_count"] == 1
    assert index["augmented_named_count"] == 1
    assert index["materialized_image_count"] == 1
    assert len(list(destination.rglob("*.jpg"))) == 1
    assert sha256(archive_path) == archive_hash


def test_potato_extraction_rejects_unsafe_paths_before_writing(tmp_path):
    archive_path = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("../outside/orig_1.jpg", image_bytes())
    destination = tmp_path / "raw" / "potato"

    with pytest.raises(DatasetAuditError, match="unsafe paths"):
        extract_potato_originals(archive_path, destination, sha256(archive_path))

    assert not destination.exists()
    assert not (tmp_path / "outside").exists()


def test_audit_and_duplicate_detectors_report_without_removing_images(tmp_path):
    root = tmp_path / "raw"
    first = root / "Healthy" / "one.jpg"
    second = root / "Healthy" / "two.jpg"
    first.parent.mkdir(parents=True)
    payload = image_bytes()
    first.write_bytes(payload)
    second.write_bytes(payload)
    index = {
        "files": [
            {
                "source_label": "Healthy",
                "source_path": "Dataset/Healthy/orig_1.jpg",
                "local_file": "Healthy/one.jpg",
                "source_sha256": hashlib.sha256(payload).hexdigest(),
            },
            {
                "source_label": "Healthy",
                "source_path": "Dataset/Healthy/orig_2.jpg",
                "local_file": "Healthy/two.jpg",
                "source_sha256": hashlib.sha256(payload).hexdigest(),
            },
        ]
    }
    mapping = {
        "Healthy": {
            "status": "MATCHED",
            "target_class": "Potato healthy",
            "reason": "Exact healthy-potato match.",
        }
    }

    records = audit_index("Potato", "training_candidate", root, index, mapping)
    groups = exact_duplicate_groups(records)
    perceptual = perceptual_duplicate_pairs(records, maximum_distance=4)

    assert len(records) == 2
    assert len(groups) == 1
    assert groups[0]["member_count"] == 2
    assert perceptual == []  # Exact duplicates are reported only in the SHA-256 audit.
    assert first.is_file() and second.is_file()


def test_verified_original_archive_preserves_paths_and_source(tmp_path):
    archive_path = tmp_path / "seasonal.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr(
            "Final corn dataset/Healthy/leaf.jpg", image_bytes((20, 120, 20))
        )
        archive.writestr(
            "Final corn dataset/Common_rust/rust.png",
            image_bytes((100, 40, 20), "PNG"),
        )
    archive_hash = sha256(archive_path)
    destination = tmp_path / "raw" / "seasonal_corn"

    index = extract_verified_original_archives(
        dataset_id="test_seasonal",
        source_url="https://example.test/dataset",
        source_version="1",
        archives=[ZipSource(archive_path, archive_hash, "official-file-id")],
        destination=destination,
        expected_class_counts={"Common_rust": 1, "Healthy": 1},
        layout="seasonal_corn",
    )

    assert index["valid_image_count"] == 2
    assert index["augmented_image_count"] == 0
    assert index["corrupted"] == []
    assert (destination / "Final corn dataset" / "Healthy" / "leaf.jpg").is_file()
    assert sha256(archive_path) == archive_hash


def test_verified_archive_rejects_unexpected_class_counts_before_extracting(tmp_path):
    archive_path = tmp_path / "pldd.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("EB/early-blight.jpg", image_bytes())
    destination = tmp_path / "raw" / "pldd"

    with pytest.raises(DatasetAuditError, match="class counts differ"):
        extract_verified_original_archives(
            dataset_id="test_pldd",
            source_url="https://example.test/dataset",
            source_version="1",
            archives=[ZipSource(archive_path, sha256(archive_path), "id", "EB")],
            destination=destination,
            expected_class_counts={"EB": 2},
            layout="pldd_up",
        )

    assert not destination.exists()


def duplicate_record(dataset, role, path, dhash, digest, target="Potato healthy"):
    return {
        "dataset": dataset,
        "source_version": "1",
        "role": role,
        "source_label": target,
        "mapping_status": "MATCHED",
        "target_class": target,
        "source_path": path,
        "local_file": path if role == "training_candidate" else "",
        "sha256": digest,
        "dhash": f"{dhash:016x}",
        "format": "JPEG",
        "mode": "RGB",
        "width": 12,
        "height": 8,
        "bytes": 100,
        "original_or_augmented": "original",
        "candidate_status": (
            "APPROVED_CANDIDATE"
            if role == "training_candidate"
            else "LOCKED_BENCHMARK"
        ),
    }


def test_indexed_perceptual_audit_finds_cross_role_neighbour():
    records = [
        duplicate_record("Train", "training_candidate", "train/a.jpg", 0, "a"),
        duplicate_record("Test", "locked_test", "test/a.jpg", 0b1111, "b"),
        duplicate_record("Train", "training_candidate", "train/far.jpg", 0xFFFF, "c"),
    ]

    pairs = perceptual_duplicate_pairs(records, maximum_distance=4)

    assert len(pairs) == 1
    assert pairs[0]["hamming_distance"] == 4
    assert pairs[0]["touches_locked_test"] is True


def test_candidate_decisions_exclude_exact_leakage_and_queue_perceptual_review():
    exact_train = duplicate_record(
        "Train", "training_candidate", "train/exact.jpg", 0, "same"
    )
    exact_test = duplicate_record("Test", "locked_test", "test/exact.jpg", 0, "same")
    near_train = duplicate_record(
        "Train", "training_candidate", "train/near.jpg", 0b1, "near-train"
    )
    near_test = duplicate_record(
        "Test", "locked_test", "test/near.jpg", 0b11, "near-test"
    )
    records = [exact_train, exact_test, near_train, near_test]
    exact_groups = exact_duplicate_groups(records)
    perceptual_pairs = perceptual_duplicate_pairs(records, maximum_distance=1)

    apply_candidate_decisions(records, exact_groups, perceptual_pairs)

    assert exact_train["candidate_status"] == "EXCLUDE_FROM_TRAINING"
    assert near_train["candidate_status"] == "NEEDS_MANUAL_REVIEW"


def test_new_repository_mappings_are_valid():
    corn = load_mapping(Path("training/datasets/mappings/seasonal-corn.json"))
    pldd = load_mapping(Path("training/datasets/mappings/pldd-up.json"))

    assert corn["Common_rust"]["target_class"] == "Corn Common rust"
    assert corn["Bacterial Leaf Streak"]["status"] == "NOT_SUPPORTED"
    assert pldd["EB"]["target_class"] == "Potato Early blight"
    assert pldd["LB"]["target_class"] == "Potato Late blight"


def test_perceptual_compaction_keeps_critical_pairs_and_aggregate_counts(tmp_path):
    fields = [
        "first_dataset",
        "first_role",
        "first_label",
        "first_target_class",
        "first_path",
        "second_dataset",
        "second_role",
        "second_label",
        "second_target_class",
        "second_path",
        "hamming_distance",
        "same_target_class",
        "touches_locked_test",
    ]
    base = {
        "first_dataset": "PLDD-UP",
        "first_role": "training_candidate",
        "first_label": "EB",
        "first_target_class": "Potato Early blight",
        "second_dataset": "PLDD-UP",
        "second_role": "training_candidate",
        "second_label": "EB",
        "second_target_class": "Potato Early blight",
        "hamming_distance": "1",
        "same_target_class": "True",
        "touches_locked_test": "False",
    }
    rows = [
        {**base, "first_path": f"EB/{index}.jpg", "second_path": f"EB/{index + 1}.jpg"}
        for index in range(3)
    ]
    rows.append(
        {
            **base,
            "first_path": "EB/cross.jpg",
            "second_dataset": "PlantDoc",
            "second_path": "train/cross.jpg",
        }
    )
    full = tmp_path / "full.csv"
    tracked = tmp_path / "tracked.csv"
    aggregate = tmp_path / "aggregate.csv"
    write_csv(full, rows, fields)

    result = compact_perceptual_report(
        full, tracked, aggregate, fields, sample_limit_per_group=1
    )

    with tracked.open(encoding="utf-8", newline="") as handle:
        tracked_rows = list(csv.DictReader(handle))
    assert result["full_candidate_pair_count"] == 4
    assert result["aggregate_group_count"] == 2
    assert result["tracked_review_row_count"] == 2
    assert {row["selection_reason"] for row in tracked_rows} == {
        "CROSS_DATASET",
        "DETERMINISTIC_GROUP_SAMPLE",
    }
