import json
from pathlib import Path

import pytest

from app.taxonomy import CLASS_NAMES
from scripts.audit_historical_39class_source import (
    HistoricalAuditError,
    apply_perceptual_policy,
    audit_cross_exact,
    audit_internal_exact_duplicates,
    clean_historical_records,
    coverage_counts,
    load_explicit_mapping,
    stable_record_id,
    validate_full_mapping,
)


MAPPING_PATH = Path("training/datasets/mappings/historical-mendeley-39.json")


def record(path, digest, target_index=0, status="INCLUDE_CANDIDATE"):
    return {
        "record_id": stable_record_id(path),
        "dataset": "Historical Mendeley 39-class source",
        "source_version": "1",
        "source_path": path,
        "source_label": f"source-{target_index}",
        "target_index": target_index,
        "target_class": CLASS_NAMES[target_index],
        "sha256": digest,
        "dhash": "0" * 16,
        "phash": "0" * 16,
        "format": "JPEG",
        "mode": "RGB",
        "width": 32,
        "height": 32,
        "bytes": 100,
        "integrity_status": "VALID",
        "candidate_status": status,
        "candidate_reason": "",
        "canonical_record_id": "",
        "exact_duplicate_group_id": "",
        "perceptual_group_id": "",
        "exact_overlap_dataset_v2": "false",
        "perceptual_overlap_dataset_v2": "false",
        "benchmark_leakage": "false",
    }


def test_explicit_mapping_covers_all_39_deployed_classes():
    mapping = load_explicit_mapping(MAPPING_PATH)
    validate_full_mapping(mapping.keys(), mapping)

    assert len(mapping) == 39
    assert {entry["target_index"] for entry in mapping.values()} == set(range(39))
    assert mapping["Background_without_leaves"]["target_index"] == 4


def test_missing_source_class_is_rejected():
    mapping = load_explicit_mapping(MAPPING_PATH)
    observed = list(mapping)[:-1]

    with pytest.raises(HistoricalAuditError, match="Source/mapping labels differ"):
        validate_full_mapping(observed, mapping)


def test_duplicate_target_mapping_is_rejected(tmp_path):
    payload = json.loads(MAPPING_PATH.read_text(encoding="utf-8"))
    payload["classes"][1]["target_index"] = 0
    payload["classes"][1]["target_class"] = CLASS_NAMES[0]
    path = tmp_path / "mapping.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(HistoricalAuditError, match="Duplicate deployed target index"):
        load_explicit_mapping(path)


def test_source_folder_order_cannot_change_deployed_indices():
    mapping = load_explicit_mapping(MAPPING_PATH)
    reversed_labels = list(reversed(list(mapping)))

    validate_full_mapping(reversed_labels, mapping)

    assert mapping["Potato___Late_blight"]["target_index"] == 22
    assert mapping["Potato___healthy"]["target_index"] == 23
    assert mapping["Tomato___healthy"]["target_index"] == 38


def test_exact_duplicate_uses_deterministic_canonical():
    first = record("root/a.jpg", "a" * 64)
    second = record("root/b.jpg", "a" * 64)

    groups, summary = audit_internal_exact_duplicates([second, first])

    expected = min(first["record_id"], second["record_id"])
    statuses = {row["record_id"]: row["candidate_status"] for row in (first, second)}
    assert summary["group_count"] == 1
    assert summary["copy_count_beyond_first"] == 1
    assert groups[0]["record_ids"][0] == expected
    assert statuses[expected] == "INCLUDE_CANDIDATE"
    assert list(statuses.values()).count("EXCLUDE_EXACT_DUPLICATE") == 1


def test_cross_class_exact_duplicate_is_a_label_conflict():
    first = record("root/a.jpg", "b" * 64, target_index=0)
    second = record("root/b.jpg", "b" * 64, target_index=1)

    _, summary = audit_internal_exact_duplicates([first, second])

    assert summary["label_conflict_group_count"] == 1
    assert {first["candidate_status"], second["candidate_status"]} == {
        "EXCLUDE_LABEL_CONFLICT"
    }


def test_locked_benchmark_exact_match_is_excluded():
    historical = record("root/a.jpg", "c" * 64)
    locked = {
        "dataset": "PlantDoc",
        "source_label": "Apple Scab Leaf",
        "target_class": CLASS_NAMES[0],
        "sha256": "c" * 64,
    }

    summary = audit_cross_exact([historical], [], [locked])

    assert summary["plantdoc_test_historical_image_count"] == 1
    assert historical["candidate_status"] == "EXCLUDE_BENCHMARK_LEAKAGE"
    assert historical["benchmark_leakage"] == "true"


def test_possible_cross_class_perceptual_match_requires_review():
    first = record("root/a.jpg", "1" * 64, target_index=0)
    second = record("root/b.jpg", "2" * 64, target_index=1)
    records = {first["record_id"]: first, second["record_id"]: second}
    signal = {
        "first_record_id": first["record_id"],
        "second_record_id": second["record_id"],
        "risk_type": "INTERNAL_HISTORICAL",
        "verification_status": "POSSIBLE_NEAR_DUPLICATE",
    }

    summary = apply_perceptual_policy([first, second], records, [signal])

    assert summary["internal_possible_conflict_image_count"] == 2
    assert {first["candidate_status"], second["candidate_status"]} == {
        "REVIEW_PERCEPTUAL_CONFLICT"
    }


def test_record_ids_are_stable_and_path_sensitive():
    assert stable_record_id("root/a.jpg") == stable_record_id("root/a.jpg")
    assert stable_record_id("root/a.jpg") != stable_record_id("root/b.jpg")


def test_clean_manifest_filter_keeps_only_include_candidates():
    included = record("root/a.jpg", "d" * 64)
    excluded = record(
        "root/b.jpg", "e" * 64, status="EXCLUDE_BENCHMARK_LEAKAGE"
    )

    assert clean_historical_records([excluded, included]) == [included]


def test_full_coverage_validation_reports_missing_classes():
    all_classes = [
        record(f"root/{index}.jpg", f"{index:064x}", target_index=index)
        for index in range(39)
    ]
    complete = coverage_counts(all_classes)
    missing = coverage_counts(all_classes[:-1])

    assert complete["coverage_count"] == 39
    assert complete["coverage_missing"] == []
    assert missing["coverage_count"] == 38
    assert missing["coverage_missing"] == [CLASS_NAMES[38]]
