import hashlib
import json
from pathlib import Path

import pytest
from PIL import Image

from scripts.prepare_plantdoc_evaluation import (
    PreparationError,
    load_mapping,
    prepare_dataset,
)


def save_image(path, color):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (12, 8), color=color).save(path)


def write_mapping(path, classes):
    path.write_text(
        json.dumps({"benchmark_name": "test", "classes": classes}),
        encoding="utf-8",
    )


def matched(target_class, target_directory):
    return {
        "status": "MATCHED",
        "target_class": target_class,
        "target_directory": target_directory,
        "reason": "Verified semantic match.",
    }


def ambiguous():
    return {
        "status": "AMBIGUOUS",
        "target_class": None,
        "target_directory": None,
        "reason": "Insufficiently specific source label.",
    }


def test_repository_mapping_is_valid_and_has_all_review_statuses():
    mapping = load_mapping(Path("evaluation/mappings/plantdoc.json"))

    assert len(mapping["classes"]) == 27
    assert mapping["classes"]["Apple Scab Leaf"]["target_class"] == "Apple Apple scab"
    assert mapping["classes"]["Bell_pepper leaf spot"]["status"] == "AMBIGUOUS"


def test_preparation_maps_verified_class_and_leaves_source_unchanged(tmp_path):
    source = tmp_path / "source"
    matched_image = source / "test" / "Apple Scab Leaf" / "leaf.jpg"
    excluded_image = source / "test" / "Apple leaf" / "leaf.jpg"
    save_image(matched_image, (20, 80, 20))
    save_image(excluded_image, (30, 100, 30))
    before = {
        path.relative_to(source).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in source.rglob("*")
        if path.is_file()
    }
    mapping_path = tmp_path / "mapping.json"
    write_mapping(
        mapping_path,
        {
            "Apple Scab Leaf": matched("Apple Apple scab", "Apple___Apple_scab"),
            "Apple leaf": ambiguous(),
        },
    )
    output = tmp_path / "prepared"

    report = prepare_dataset(source, mapping_path, output)

    assert report["selection"]["prepared_image_count"] == 1
    assert report["selection"]["excluded_image_count"] == 1
    assert len(list((output / "Apple___Apple_scab").glob("*.jpg"))) == 1
    after = {
        path.relative_to(source).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in source.rglob("*")
        if path.is_file()
    }
    assert after == before


def test_unreviewed_source_label_is_rejected(tmp_path):
    source = tmp_path / "source"
    save_image(source / "test" / "Unknown class" / "leaf.jpg", (0, 100, 0))
    mapping_path = tmp_path / "mapping.json"
    write_mapping(mapping_path, {"Apple leaf": ambiguous()})

    with pytest.raises(PreparationError, match="coverage mismatch"):
        prepare_dataset(source, mapping_path, tmp_path / "prepared")


def test_exact_duplicate_detector_reports_and_removes_only_when_requested(tmp_path):
    source = tmp_path / "source"
    first = source / "test" / "Apple Scab Leaf" / "one.jpg"
    second = source / "test" / "Apple Scab Leaf" / "two.jpg"
    save_image(first, (10, 90, 10))
    second.write_bytes(first.read_bytes())
    mapping_path = tmp_path / "mapping.json"
    write_mapping(
        mapping_path,
        {"Apple Scab Leaf": matched("Apple Apple scab", "Apple___Apple_scab")},
    )

    report = prepare_dataset(
        source,
        mapping_path,
        tmp_path / "prepared",
        deduplicate_exact=True,
    )

    assert report["selection"]["exact_duplicate_group_count"] == 1
    assert report["selection"]["exact_duplicate_image_count"] == 1
    assert report["selection"]["exact_duplicates_removed"] == 1
    assert report["selection"]["prepared_image_count"] == 1
