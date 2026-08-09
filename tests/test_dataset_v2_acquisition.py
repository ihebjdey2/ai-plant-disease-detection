import hashlib
import io
import zipfile

import pytest
from PIL import Image

from scripts.audit_dataset_v2_sources import (
    DatasetAuditError,
    audit_index,
    exact_duplicate_groups,
    extract_plantdoc_train,
    extract_potato_originals,
    perceptual_duplicate_pairs,
    safe_component,
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
