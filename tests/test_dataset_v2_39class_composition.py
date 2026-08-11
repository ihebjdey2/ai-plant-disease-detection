from collections import Counter

import pytest

from training.taxonomy import CLASS_NAMES
from scripts.build_dataset_v2_39class_composition import (
    CompositionError,
    HISTORICAL_DOMAIN,
    REAL_WORLD_DOMAIN,
    build_summary,
    compose_records,
    finalize_historical_reviews,
    stable_composition_id,
)


def digest(number):
    return f"{number:064x}"


def historical(index, target, number=None, status="INCLUDE_CANDIDATE"):
    number = index + 1 if number is None else number
    return {
        "record_id": f"hist_{number}",
        "dataset": "Historical Mendeley 39-class source",
        "source_version": "1",
        "source_path": f"root/class-{index}/image-{number}.jpg",
        "source_label": f"historical-{target}",
        "target_index": str(index),
        "target_class": target,
        "sha256": digest(number),
        "dhash": f"{number:016x}",
        "phash": f"{number:016x}",
        "format": "JPEG",
        "mode": "RGB",
        "width": "256",
        "height": "256",
        "bytes": "100",
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


def real_world(index, target, number=1000, dataset="PlantDoc"):
    return {
        "record_id": f"real_{number}",
        "dataset": dataset,
        "source_version": "revision",
        "role": "training_candidate",
        "source_path": f"train/class-{index}/image-{number}.jpg",
        "local_file": f"class-{index}/image-{number}.jpg",
        "source_label": f"real-{target}",
        "mapping_status": "MATCHED",
        "target_class": target,
        "sha256": digest(number),
        "dhash": f"{number:016x}",
        "phash": f"{number:016x}",
        "format": "JPEG",
        "mode": "RGB",
        "width": "640",
        "height": "480",
        "bytes": "200",
        "original_or_augmented": "original",
        "candidate_status": "APPROVED_CANDIDATE",
        "cleaning_status": "INCLUDE",
        "exclusion_reason": "",
        "canonical_record_id": "",
        "exact_duplicate_group_id": "",
        "near_duplicate_group_id": "",
        "benchmark_leakage": "false",
        "label_conflict": "false",
        "manual_review_required": "false",
        "review_resolution": "NOT_REQUIRED",
        "refined_similarity_status": "NOT_SCREENED",
        "refined_group_id": "",
        "refined_group_representative": "",
    }


def test_historical_and_real_world_compose_with_full_39_class_coverage():
    historical_rows = [
        historical(index, target) for index, target in enumerate(CLASS_NAMES)
    ]
    real_rows = [real_world(0, CLASS_NAMES[0], number=2000)]

    combined = compose_records(historical_rows, real_rows)

    assert len(combined) == 40
    assert {int(row["target_index"]) for row in combined} == set(range(39))
    assert Counter(row["source_domain"] for row in combined) == {
        HISTORICAL_DOMAIN: 39,
        REAL_WORLD_DOMAIN: 1,
    }


def test_missing_class_fails_coverage_validation():
    rows = [historical(0, "Class A")]

    with pytest.raises(CompositionError, match="Incomplete class-index coverage"):
        compose_records(rows, [], class_names=["Class A", "Class B"])


def test_target_index_is_derived_from_taxonomy_not_input_order():
    class_names = ["Class B", "Class A"]
    rows = [historical(1, "Class A", number=10), historical(0, "Class B", number=11)]

    combined = compose_records(list(reversed(rows)), [], class_names=class_names)

    by_target = {row["target_class"]: row["target_index"] for row in combined}
    assert by_target == {"Class B": "0", "Class A": "1"}


def test_composition_ids_are_deterministic_and_identity_sensitive():
    first = stable_composition_id("REAL_WORLD", "PlantDoc", "rec_1", 0, "a" * 64)
    second = stable_composition_id("REAL_WORLD", "PlantDoc", "rec_1", 0, "a" * 64)
    changed = stable_composition_id("REAL_WORLD", "PlantDoc", "rec_2", 0, "a" * 64)

    assert first == second
    assert first != changed


def test_review_input_is_rejected_instead_of_silently_filtered():
    row = historical(0, "Class A", status="REVIEW_PERCEPTUAL_CONFLICT")

    with pytest.raises(CompositionError, match="non-INCLUDE"):
        compose_records([row], [], class_names=["Class A"])


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("role", "locked_test", "Locked or invalid role"),
        ("benchmark_leakage", "true", "benchmark leak"),
    ],
)
def test_locked_or_leaking_benchmark_rows_are_rejected(field, value, message):
    row = real_world(0, "Class A")
    row[field] = value

    with pytest.raises(CompositionError, match=message):
        compose_records([], [row], class_names=["Class A"])


def test_cross_domain_sha_collision_stops_composition():
    historical_row = historical(0, "Class A", number=99)
    real_row = real_world(0, "Class A", number=99)

    with pytest.raises(CompositionError, match="cross-domain SHA-256 collision"):
        compose_records([historical_row], [real_row], class_names=["Class A"])


def test_refined_group_members_share_split_group_id():
    first = real_world(0, "Class A", number=101)
    second = real_world(0, "Class A", number=102)
    first["refined_group_id"] = second["refined_group_id"] = "refined_shared"

    combined = compose_records([], [second, first], class_names=["Class A"])

    assert len({row["split_group_id"] for row in combined}) == 1
    assert all(row["refined_group_id"] == "refined_shared" for row in combined)


def test_singletons_receive_distinct_deterministic_groups():
    first = historical(0, "Class A", number=201)
    second = historical(0, "Class A", number=202)

    forward = compose_records([first, second], [], class_names=["Class A"])
    reverse = compose_records([second, first], [], class_names=["Class A"])

    assert forward == reverse
    assert len({row["split_group_id"] for row in forward}) == 2
    assert all(row["split_group_id"].startswith("split_single_") for row in forward)


def test_historical_reviews_are_finalized_without_changing_clean_count():
    late = historical(31, "Tomato Late blight", number=301, status="REVIEW_PERCEPTUAL_CONFLICT")
    healthy = historical(38, "Tomato healthy", number=302, status="REVIEW_PERCEPTUAL_CONFLICT")
    clean = [historical(0, CLASS_NAMES[0], number=303)]

    finalization = finalize_historical_reviews([late, healthy, *clean], clean)

    assert finalization["review_count_before"] == 2
    assert finalization["review_count_after"] == 0
    assert finalization["historical_clean_count_before"] == 1
    assert finalization["historical_clean_count_after"] == 1
    assert {row["cleaning_status"] for row in finalization["records"]} == {"EXCLUDE"}
    assert {row["review_resolution"] for row in finalization["records"]} == {
        "CONSERVATIVE_FINAL_EXCLUSION"
    }


def test_summary_calculates_domain_counts_and_group_statistics():
    historical_rows = [historical(0, "Class A", number=401)]
    real_rows = [real_world(1, "Class B", number=402)]
    combined = compose_records(historical_rows, real_rows, class_names=["Class A", "Class B"])
    finalization = {
        "review_count_after": 0,
        "records": [],
    }

    summary = build_summary(
        combined,
        finalization,
        {"perceptual_overlap_with_dataset_v2": 0},
        {},
        class_names=["Class A", "Class B"],
    )

    assert summary["counts_by_domain"] == {
        HISTORICAL_DOMAIN: 1,
        REAL_WORLD_DOMAIN: 1,
    }
    assert summary["historical_contribution_percentage"] == 50.0
    assert summary["real_world_contribution_percentage"] == 50.0
    assert summary["split_group_count"] == 2
    assert summary["singleton_group_count"] == 2
