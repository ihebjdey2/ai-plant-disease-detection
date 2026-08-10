from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter

import pytest

from app.taxonomy import CLASS_NAMES
from scripts.build_dataset_v2_group_aware_split import (
    HISTORICAL_DOMAIN,
    REAL_WORLD_DOMAIN,
    SPLIT_SEED,
    STRATEGY_VERSION,
    SplitError,
    allocate_groups,
    apply_assignments,
    file_sha256,
    run,
    validate_and_group,
    validate_assigned_rows,
)


FIELDS = [
    "record_id",
    "composition_record_id",
    "source_domain",
    "source_dataset",
    "source_path",
    "target_index",
    "target_class",
    "sha256",
    "split_group_id",
    "composition_status",
]


def digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def make_row(
    index: int,
    number: int,
    *,
    domain: str = HISTORICAL_DOMAIN,
    source: str = "Historical Mendeley 39-class source",
    group_id: str | None = None,
    sha256: str | None = None,
    source_path: str | None = None,
) -> dict[str, str]:
    identity = f"{index}-{domain}-{source}-{number}"
    return {
        "record_id": f"record-{identity}",
        "composition_record_id": f"comp-{identity}",
        "source_domain": domain,
        "source_dataset": source,
        "source_path": source_path or f"train/{CLASS_NAMES[index]}/{number}.jpg",
        "target_index": str(index),
        "target_class": CLASS_NAMES[index],
        "sha256": sha256 or digest(identity),
        "split_group_id": group_id or f"group-{identity}",
        "composition_status": "INCLUDE_CANDIDATE",
    }


def assigned(rows: list[dict[str, str]], seed: int = SPLIT_SEED):
    groups = validate_and_group(rows)
    assignments = allocate_groups(groups, seed)
    return apply_assignments(rows, assignments, seed), groups, assignments


def full_taxonomy_rows(groups_per_class: int = 30) -> list[dict[str, str]]:
    return [
        make_row(index, number)
        for index in range(len(CLASS_NAMES))
        for number in range(groups_per_class)
    ]


def test_deterministic_80_10_10_split_and_seed_metadata():
    rows = [make_row(0, number) for number in range(100)]
    first, _, first_assignments = assigned(rows)
    second, _, second_assignments = assigned(list(reversed(rows)))

    assert first_assignments == second_assignments
    assert Counter(row["split"] for row in first) == {
        "TRAIN": 80,
        "VALIDATION": 10,
        "TEST": 10,
    }
    assert {row["split_seed"] for row in first} == {str(SPLIT_SEED)}
    assert {row["split_strategy_version"] for row in first} == {
        STRATEGY_VERSION
    }


def test_different_seed_changes_deterministic_tie_breaking():
    rows = [make_row(0, number) for number in range(100)]
    _, _, first = assigned(rows, SPLIT_SEED)
    _, _, second = assigned(rows, SPLIT_SEED + 1)
    assert first != second


def test_split_group_is_indivisible():
    rows = []
    for number in range(30):
        group_id = f"family-{number}"
        rows.append(make_row(0, number * 2, group_id=group_id))
        rows.append(make_row(0, number * 2 + 1, group_id=group_id))
    output, _, _ = assigned(rows)
    group_splits: dict[str, set[str]] = {}
    for row in output:
        group_splits.setdefault(row["split_group_id"], set()).add(row["split"])
    assert all(len(splits) == 1 for splits in group_splits.values())


def test_all_39_classes_cover_train_validation_and_test():
    output, _, _ = assigned(full_taxonomy_rows())
    assert validate_assigned_rows(output) == {
        "group_leakage_count": 0,
        "sha_leakage_count": 0,
        "benchmark_contamination_count": 0,
    }
    for split in ("TRAIN", "VALIDATION", "TEST"):
        assert {int(row["target_index"]) for row in output if row["split"] == split} == set(
            range(39)
        )


def test_scarce_real_world_stratum_stays_in_training():
    rows = [
        make_row(
            34,
            number,
            domain=REAL_WORLD_DOMAIN,
            source="PlantDoc",
        )
        for number in range(2)
    ]
    output, _, _ = assigned(rows)
    assert {row["split"] for row in output} == {"TRAIN"}


def test_historical_only_class_has_all_three_partitions():
    rows = [make_row(5, number) for number in range(30)]
    output, _, _ = assigned(rows)
    counts = Counter(row["split"] for row in output)
    assert counts == {"TRAIN": 24, "VALIDATION": 3, "TEST": 3}


def test_domain_aware_allocation_preserves_each_large_domain():
    rows = [make_row(0, number) for number in range(100)]
    rows.extend(
        make_row(
            0,
            number,
            domain=REAL_WORLD_DOMAIN,
            source="PlantDoc",
        )
        for number in range(100)
    )
    output, _, _ = assigned(rows)
    for domain in (HISTORICAL_DOMAIN, REAL_WORLD_DOMAIN):
        counts = Counter(
            row["split"] for row in output if row["source_domain"] == domain
        )
        assert counts == {"TRAIN": 80, "VALIDATION": 10, "TEST": 10}


def test_source_aware_allocation_preserves_each_large_source():
    rows = []
    for source in ("PlantDoc", "PLDD-UP"):
        rows.extend(
            make_row(21, number, domain=REAL_WORLD_DOMAIN, source=source)
            for number in range(100)
        )
    output, _, _ = assigned(rows)
    for source in ("PlantDoc", "PLDD-UP"):
        counts = Counter(
            row["split"] for row in output if row["source_dataset"] == source
        )
        assert counts == {"TRAIN": 80, "VALIDATION": 10, "TEST": 10}


def test_sha_leakage_is_rejected():
    shared_sha = digest("same-content")
    rows = [
        {**make_row(0, 0, sha256=shared_sha), "split": "TRAIN"},
        {**make_row(0, 1, sha256=shared_sha), "split": "TEST"},
    ]
    with pytest.raises(SplitError, match="SHA leakage"):
        validate_assigned_rows(rows)


def test_group_leakage_is_rejected():
    rows = [
        {**make_row(0, 0, group_id="shared-family"), "split": "TRAIN"},
        {**make_row(0, 1, group_id="shared-family"), "split": "VALIDATION"},
    ]
    with pytest.raises(SplitError, match="Group leakage"):
        validate_assigned_rows(rows)


def test_plantdoc_test_benchmark_is_rejected():
    row = make_row(
        0,
        0,
        domain=REAL_WORLD_DOMAIN,
        source="PlantDoc",
        source_path="test/Apple Scab/image.jpg",
    )
    with pytest.raises(SplitError, match="PlantDoc TEST"):
        validate_and_group([row])


def write_input(path, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def builder_args(tmp_path):
    return argparse.Namespace(
        input_manifest=tmp_path / "input.csv",
        output_manifest=tmp_path / "split.csv",
        train_manifest=tmp_path / "train.csv",
        validation_manifest=tmp_path / "validation.csv",
        test_manifest=tmp_path / "test.csv",
        output_summary=tmp_path / "summary.json",
        quality_report=tmp_path / "quality.csv",
        test_lock=tmp_path / "test-lock.json",
        seed=SPLIT_SEED,
    )


def test_builder_creates_locked_test_metadata_and_is_byte_deterministic(tmp_path):
    args = builder_args(tmp_path)
    write_input(args.input_manifest, full_taxonomy_rows())
    summary = run(args)
    paths = (
        args.output_manifest,
        args.train_manifest,
        args.validation_manifest,
        args.test_manifest,
        args.output_summary,
        args.quality_report,
        args.test_lock,
    )
    first_hashes = {path.name: file_sha256(path) for path in paths}
    run(args)
    second_hashes = {path.name: file_sha256(path) for path in paths}

    lock = json.loads(args.test_lock.read_text(encoding="utf-8"))
    assert first_hashes == second_hashes
    assert lock["evaluation_role"] == "FINAL_INTERNAL_TEST"
    assert lock["test_manifest_sha256"] == file_sha256(args.test_manifest)
    assert lock["test_record_count"] == summary["test_images"]
    assert "may not be used for model-development decisions" in lock["policy"]
