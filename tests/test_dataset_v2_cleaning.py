import csv
import json
from pathlib import Path

import pytest

from scripts.build_dataset_v2_manifest import (
    MASTER_FIELDS,
    ManifestBuildError,
    build_manifests,
    validate_manifests,
)


INVENTORY_FIELDS = [
    "dataset",
    "source_version",
    "role",
    "source_label",
    "mapping_status",
    "target_class",
    "source_path",
    "local_file",
    "sha256",
    "dhash",
    "format",
    "mode",
    "width",
    "height",
    "bytes",
    "original_or_augmented",
    "candidate_status",
]

PAIR_FIELDS = [
    "first_dataset",
    "first_role",
    "first_path",
    "second_dataset",
    "second_role",
    "second_path",
    "hamming_distance",
]


def record(
    path,
    target="Potato Early blight",
    *,
    dataset="Source A",
    role="training_candidate",
    mapping="MATCHED",
    sha=None,
    dhash="0000000000000000",
):
    return {
        "dataset": dataset,
        "source_version": "1",
        "role": role,
        "source_label": path.split("/")[0],
        "mapping_status": mapping,
        "target_class": target,
        "source_path": path,
        "local_file": path,
        "sha256": sha or (path.encode().hex() + "0" * 64)[:64],
        "dhash": dhash,
        "format": "JPEG",
        "mode": "RGB",
        "width": "224",
        "height": "224",
        "bytes": "100",
        "original_or_augmented": "original",
        "candidate_status": "APPROVED_CANDIDATE",
    }


def pair(first, second, distance=1):
    return {
        "first_dataset": first["dataset"],
        "first_role": first["role"],
        "first_path": first["source_path"],
        "second_dataset": second["dataset"],
        "second_role": second["role"],
        "second_path": second["source_path"],
        "hamming_distance": str(distance),
    }


def write_csv(path, rows, fields):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def run_build(tmp_path, records, exact_groups=None, pairs=None, suffix=""):
    inventory = tmp_path / "inventory.csv"
    exact = tmp_path / "exact.json"
    full = tmp_path / "pairs.csv"
    members = tmp_path / f"members{suffix}.csv"
    groups = tmp_path / f"groups{suffix}.csv"
    queue = tmp_path / f"queue{suffix}.csv"
    master = tmp_path / f"master{suffix}.csv"
    clean = tmp_path / f"clean{suffix}.csv"
    summary = tmp_path / f"summary{suffix}.json"
    write_csv(inventory, records, INVENTORY_FIELDS)
    exact.write_text(json.dumps(exact_groups or []), encoding="utf-8")
    write_csv(full, pairs or [], PAIR_FIELDS)
    result = build_manifests(
        inventory,
        exact,
        members,
        groups,
        queue,
        master,
        clean,
        summary,
        full,
    )
    with master.open(encoding="utf-8", newline="") as handle:
        master_rows = list(csv.DictReader(handle))
    with clean.open(encoding="utf-8", newline="") as handle:
        clean_rows = list(csv.DictReader(handle))
    return result, master_rows, clean_rows, {
        "inventory": inventory,
        "exact": exact,
        "full": full,
        "members": members,
        "groups": groups,
        "queue": queue,
        "master": master,
        "clean": clean,
        "summary": summary,
    }


def by_path(rows):
    return {row["source_path"]: row for row in rows}


def test_same_target_exact_duplicate_keeps_deterministic_canonical(tmp_path):
    digest = "a" * 64
    rows = [record("z/image.jpg", sha=digest), record("a/image.jpg", sha=digest)]
    groups = [
        {
            "sha256": digest,
            "member_count": 2,
            "label_conflict": False,
            "touches_locked_test": False,
        }
    ]

    _, master, clean, _ = run_build(tmp_path, rows, groups)
    indexed = by_path(master)

    assert [row["source_path"] for row in clean] == ["a/image.jpg"]
    assert indexed["z/image.jpg"]["exclusion_reason"] == "EXACT_DUPLICATE_COPY"
    assert indexed["z/image.jpg"]["canonical_record_id"] == indexed["a/image.jpg"]["record_id"]


def test_conflicting_target_exact_duplicate_excludes_every_member(tmp_path):
    digest = "b" * 64
    rows = [
        record("early/a.jpg", sha=digest),
        record("healthy/b.jpg", target="Potato healthy", sha=digest),
    ]
    groups = [
        {
            "sha256": digest,
            "member_count": 2,
            "label_conflict": True,
            "touches_locked_test": False,
        }
    ]

    _, master, clean, _ = run_build(tmp_path, rows, groups)

    assert clean == []
    assert {row["exclusion_reason"] for row in master} == {"EXACT_LABEL_CONFLICT"}
    assert all(row["label_conflict"] == "true" for row in master)


def test_exact_benchmark_leakage_and_locked_test_are_excluded(tmp_path):
    digest = "c" * 64
    train = record("train/a.jpg", sha=digest)
    test = record(
        "test/a.jpg",
        dataset="PlantDoc",
        role="locked_test",
        sha=digest,
    )
    groups = [
        {
            "sha256": digest,
            "member_count": 2,
            "label_conflict": False,
            "touches_locked_test": True,
        }
    ]

    _, master, clean, _ = run_build(tmp_path, [train, test], groups)
    indexed = by_path(master)

    assert clean == []
    assert indexed["train/a.jpg"]["exclusion_reason"] == "EXACT_BENCHMARK_LEAKAGE"
    assert indexed["train/a.jpg"]["benchmark_leakage"] == "true"
    assert indexed["test/a.jpg"]["exclusion_reason"] == "LOCKED_BENCHMARK"


def test_ambiguous_and_unsupported_mappings_are_excluded(tmp_path):
    rows = [
        record("ambiguous/a.jpg", target="", mapping="AMBIGUOUS"),
        record("unsupported/b.jpg", target="", mapping="NOT_SUPPORTED"),
    ]

    _, master, clean, _ = run_build(tmp_path, rows)

    assert clean == []
    assert by_path(master)["ambiguous/a.jpg"]["exclusion_reason"] == "UNRESOLVED_MAPPING"
    assert by_path(master)["unsupported/b.jpg"]["exclusion_reason"] == "UNSUPPORTED_CLASS"


def test_same_target_perceptual_pair_is_grouped_but_included(tmp_path):
    first = record("early/a.jpg", dhash="0" * 16)
    second = record("early/b.jpg", dhash="0" * 15 + "1")

    _, master, clean, _ = run_build(tmp_path, [first, second], pairs=[pair(first, second)])

    assert len(clean) == 2
    assert len({row["near_duplicate_group_id"] for row in master}) == 1
    assert all(row["cleaning_status"] == "INCLUDE" for row in master)


def test_different_target_perceptual_pair_requires_review(tmp_path):
    first = record("early/a.jpg")
    second = record("healthy/b.jpg", target="Potato healthy")

    _, master, clean, paths = run_build(
        tmp_path, [first, second], pairs=[pair(first, second, 0)]
    )

    assert clean == []
    assert {row["cleaning_status"] for row in master} == {"REVIEW"}
    assert {row["exclusion_reason"] for row in master} == {"PERCEPTUAL_LABEL_CONFLICT"}
    with paths["queue"].open(encoding="utf-8", newline="") as handle:
        queue = list(csv.DictReader(handle))
    assert len(queue) == 1
    assert queue[0]["review_category"] == "DIFFERENT_TARGET"


def test_non_exact_train_to_test_pair_requires_training_review(tmp_path):
    train = record("train/a.jpg")
    test = record("test/b.jpg", dataset="PlantDoc", role="locked_test")

    _, master, clean, paths = run_build(
        tmp_path, [train, test], pairs=[pair(train, test, 1)]
    )
    indexed = by_path(master)

    assert clean == []
    assert indexed["train/a.jpg"]["exclusion_reason"] == "PERCEPTUAL_BENCHMARK_MATCH"
    assert indexed["train/a.jpg"]["manual_review_required"] == "true"
    assert indexed["test/b.jpg"]["exclusion_reason"] == "LOCKED_BENCHMARK"
    with paths["queue"].open(encoding="utf-8", newline="") as handle:
        queue = list(csv.DictReader(handle))
    assert queue[0]["review_category"] == "TRAIN_TO_LOCKED_BENCHMARK"


def test_rerun_without_full_pair_file_is_deterministic(tmp_path):
    first = record("early/a.jpg")
    second = record("early/b.jpg")
    _, _, _, paths = run_build(tmp_path, [first, second], pairs=[pair(first, second)])
    expected_master = paths["master"].read_bytes()
    expected_clean = paths["clean"].read_bytes()

    paths["full"].unlink()
    build_manifests(
        paths["inventory"],
        paths["exact"],
        paths["members"],
        paths["groups"],
        paths["queue"],
        paths["master"],
        paths["clean"],
        paths["summary"],
        paths["full"],
    )

    assert paths["master"].read_bytes() == expected_master
    assert paths["clean"].read_bytes() == expected_clean


def test_clean_manifest_validation_rejects_duplicate_include_hashes():
    first = {field: "" for field in MASTER_FIELDS}
    first.update(
        {
            "record_id": "rec_a",
            "role": "training_candidate",
            "mapping_status": "MATCHED",
            "target_class": "Potato healthy",
            "sha256": "d" * 64,
            "cleaning_status": "INCLUDE",
            "label_conflict": "false",
            "manual_review_required": "false",
        }
    )
    second = dict(first, record_id="rec_b")

    with pytest.raises(ManifestBuildError, match="duplicate SHA-256"):
        validate_manifests([first, second], [first, second])
