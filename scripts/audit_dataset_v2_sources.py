"""Acquire and audit the approved Dataset V2 Step 5B sources.

This tool is deliberately limited to PlantDoc TRAIN and the original-image
subset published inside the Banu/Deb Potato archive. PlantDoc TEST is read only
for leakage checks and is never copied into a training directory.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import subprocess
import sys
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path, PurePosixPath
from typing import Iterable, Iterator, Sequence

from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training.taxonomy import CLASS_NAMES  # noqa: E402
from scripts.prepare_plantdoc_evaluation import (  # noqa: E402
    PlantDocSource,
    PreparationError,
    SourceImage,
    difference_hash,
    validate_image,
)


POTATO_ARCHIVE_SHA256 = (
    "549c7f3343422fa2b77b6fb2c5009a52215aa00626b2646435ba19f4826f8192"
)
POTATO_SOURCE_URL = "https://doi.org/10.17632/d5b3fzpw3g.1"
PLANTDOC_SOURCE_URL = "https://github.com/pratikkayal/PlantDoc-Dataset"
PLANTDOC_SOURCE_REVISION = "5467f6012d78d1c446145d5f582da6096f852ae8"
SEASONAL_CORN_SOURCE_URL = "https://doi.org/10.17632/vy629dngm8.1"
SEASONAL_CORN_ARCHIVE_SHA256 = (
    "575628df92e69c169fa82c8506253d7d5886a8931605bf765f0e2577022dc479"
)
PLDD_UP_SOURCE_URL = "https://doi.org/10.17632/3j4nfkvp2n.1"
PLDD_UP_ARCHIVES = {
    "EB": {
        "filename": "pldd-up-v1-eb.zip",
        "file_id": "5717ac85-cf61-461d-bf70-e1e5af2f2c53",
        "sha256": "cffd37bbb79e75c0e23c1486f88f0a7c873b3fe67f643c41db3abd794bdc01e5",
    },
    "Healthy": {
        "filename": "pldd-up-v1-healthy.zip",
        "file_id": "d4ce2acf-3af8-416e-90bc-bb834ba9da66",
        "sha256": "2b7e4107d7ba03c0ef9636831aa4de7d333435df238f667f562e27b1e46e59e2",
    },
    "LB": {
        "filename": "pldd-up-v1-lb.zip",
        "file_id": "35ff7712-865c-41bf-96b1-d25d84af7b95",
        "sha256": "f4d31182b5d2f147c256e1c73838eac9b17592b89ed2b1ae394f58863e74a447",
    },
}
SUPPORTED_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
VALID_MAPPING_STATUSES = {"MATCHED", "AMBIGUOUS", "NOT_SUPPORTED"}
FORMAT_SUFFIXES = {"JPEG": ".jpg", "MPO": ".jpg", "PNG": ".png", "WEBP": ".webp"}
WINDOWS_RESERVED_NAMES = {
    "CON",
    "PRN",
    "AUX",
    "NUL",
    *(f"COM{number}" for number in range(1, 10)),
    *(f"LPT{number}" for number in range(1, 10)),
}
MANIFEST_FIELDS = [
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
PERCEPTUAL_FIELDS = [
    "first_dataset",
    "first_role",
    "first_label",
    "first_target_class",
    "first_path",
    "first_width",
    "first_height",
    "second_dataset",
    "second_role",
    "second_label",
    "second_target_class",
    "second_path",
    "second_width",
    "second_height",
    "hamming_distance",
    "same_label",
    "same_target_class",
    "cross_role",
    "touches_locked_test",
]


@dataclass(frozen=True)
class ZipSource:
    path: Path
    expected_sha256: str
    file_id: str
    source_label: str | None = None


class DatasetAuditError(RuntimeError):
    """Raised when acquisition metadata or source integrity is invalid."""


def filesystem_path(path: Path) -> Path:
    """Use the Windows extended path namespace without changing manifest paths."""

    resolved = path.resolve()
    if os.name == "nt" and not str(resolved).startswith("\\\\?\\"):
        return Path("\\\\?\\" + str(resolved))
    return resolved


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_component(value: str) -> str:
    """Return a portable directory component while preserving readability."""

    cleaned = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", value).strip(" .")
    if not cleaned:
        cleaned = "unnamed"
    if cleaned.upper() in WINDOWS_RESERVED_NAMES:
        cleaned = f"_{cleaned}"
    return cleaned


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def write_csv(path: Path, rows: Sequence[dict], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def load_json(path: Path) -> dict:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DatasetAuditError(f"Cannot read JSON metadata {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise DatasetAuditError(f"JSON metadata must be an object: {path}")
    return payload


def normalize_mapping(classes: dict) -> dict[str, dict]:
    normalized: dict[str, dict] = {}
    for source_label, entry in classes.items():
        if not isinstance(entry, dict):
            raise DatasetAuditError(f"Invalid mapping entry for {source_label!r}.")
        status = entry.get("status")
        target = entry.get("target_class")
        if status not in VALID_MAPPING_STATUSES:
            raise DatasetAuditError(f"Invalid mapping status for {source_label!r}.")
        if not entry.get("reason"):
            raise DatasetAuditError(f"Mapping reason is required for {source_label!r}.")
        if status == "MATCHED":
            if target not in CLASS_NAMES:
                raise DatasetAuditError(
                    f"Unknown deployed class {target!r} for {source_label!r}."
                )
        elif target is not None:
            raise DatasetAuditError(
                f"Excluded mapping {source_label!r} must not specify a target."
            )
        normalized[source_label] = {
            "status": status,
            "target_class": target,
            "reason": entry["reason"],
        }
    return normalized


def load_plantdoc_train_mapping(base_path: Path, extra_path: Path) -> dict[str, dict]:
    base = load_json(base_path).get("classes")
    extra = load_json(extra_path).get("additional_classes")
    if not isinstance(base, dict) or not isinstance(extra, dict):
        raise DatasetAuditError("PlantDoc base and extra mappings must contain classes.")
    overlap = set(base).intersection(extra)
    if overlap:
        raise DatasetAuditError(f"PlantDoc extra mapping redefines labels: {sorted(overlap)}")
    return normalize_mapping({**base, **extra})


def load_mapping(path: Path) -> dict[str, dict]:
    classes = load_json(path).get("classes")
    if not isinstance(classes, dict) or not classes:
        raise DatasetAuditError(f"Mapping has no classes: {path}")
    return normalize_mapping(classes)


def validate_mapping_coverage(labels: Iterable[str], mapping: dict[str, dict]) -> None:
    observed = set(labels)
    declared = set(mapping)
    missing = sorted(observed - declared)
    stale = sorted(declared - observed)
    if missing or stale:
        raise DatasetAuditError(
            f"Mapping coverage mismatch; missing={missing or 'none'}, "
            f"stale={stale or 'none'}."
        )


def iter_plantdoc_bytes(
    source: PlantDocSource, images: Sequence[SourceImage]
) -> Iterator[tuple[SourceImage, bytes]]:
    """Read Git blobs in one process, or normal files for directory sources."""

    if not source.git_mode:
        for image in images:
            yield image, source.read_bytes(image)
        return

    process = subprocess.Popen(
        ["git", "-C", str(source.root), "cat-file", "--batch"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if process.stdin is None or process.stdout is None or process.stderr is None:
        process.kill()
        raise DatasetAuditError("Cannot open Git batch reader pipes.")
    try:
        for image in images:
            process.stdin.write(f"{image.identifier}\n".encode("ascii"))
            process.stdin.flush()
            header = process.stdout.readline().decode("ascii", errors="replace").strip()
            parts = header.split()
            if len(parts) != 3 or parts[1] != "blob":
                raise DatasetAuditError(
                    f"Unexpected Git blob header for {image.relative_path}: {header}"
                )
            size = int(parts[2])
            data = process.stdout.read(size)
            separator = process.stdout.read(1)
            if len(data) != size or separator != b"\n":
                raise DatasetAuditError(
                    f"Incomplete Git blob for {image.relative_path}."
                )
            yield image, data
        process.stdin.close()
        return_code = process.wait(timeout=30)
        if return_code != 0:
            message = process.stderr.read().decode("utf-8", errors="replace").strip()
            raise DatasetAuditError(f"Git batch reader failed: {message}")
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()


def _load_existing_index(destination: Path, dataset_id: str) -> dict | None:
    index_path = destination / ".source-index.json"
    if not index_path.is_file():
        if destination.exists() and any(destination.iterdir()):
            raise DatasetAuditError(
                f"Destination is not empty and has no source index: {destination}"
            )
        return None
    index = load_json(index_path)
    if index.get("dataset_id") != dataset_id:
        raise DatasetAuditError(f"Unexpected source index in {destination}.")
    for entry in index.get("files", []):
        local_path = filesystem_path(
            destination / PurePosixPath(entry["local_file"])
        )
        if not local_path.is_file():
            raise DatasetAuditError(f"Indexed local file is missing: {local_path}")
    return index


def extract_plantdoc_train(source_path: Path, destination: Path) -> dict:
    existing = _load_existing_index(destination, "plantdoc_train")
    if existing is not None:
        return existing

    source = PlantDocSource(
        source_path, revision=PLANTDOC_SOURCE_REVISION, split="train"
    )
    images = source.images()
    if not images:
        raise DatasetAuditError("PlantDoc TRAIN contains no source images.")
    files = []
    corrupted = []
    for index, (image, data) in enumerate(iter_plantdoc_bytes(source, images), start=1):
        try:
            image_format, image_mode = validate_image(data)
        except PreparationError as exc:
            corrupted.append({"source_path": image.relative_path, "error": str(exc)})
            continue
        digest = sha256_bytes(data)
        suffix = FORMAT_SUFFIXES.get(image_format, ".img")
        relative = Path(safe_component(image.label)) / (
            f"{index:05d}_{digest[:16]}{suffix}"
        )
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        files.append(
            {
                "source_label": image.label,
                "source_path": image.relative_path,
                "source_object_id": image.identifier,
                "local_file": relative.as_posix(),
                "source_sha256": digest,
                "source_format": image_format,
                "source_mode": image_mode,
            }
        )
    payload = {
        "dataset_id": "plantdoc_train",
        "source": PLANTDOC_SOURCE_URL,
        "source_revision": source.resolved_revision,
        "source_image_count": len(images),
        "materialized_image_count": len(files),
        "corrupted": corrupted,
        "files": files,
    }
    write_json(destination / ".source-index.json", payload)
    return payload


def _safe_zip_parts(name: str) -> tuple[str, ...]:
    normalized = name.replace("\\", "/")
    path = PurePosixPath(normalized)
    if path.is_absolute() or ".." in path.parts or re.match(r"^[A-Za-z]:", normalized):
        raise DatasetAuditError(f"Unsafe ZIP entry: {name}")
    return path.parts


def extract_potato_originals(
    archive_path: Path, destination: Path, expected_sha256: str
) -> dict:
    existing = _load_existing_index(destination, "potato_banu_deb_originals")
    if existing is not None:
        if existing.get("archive_sha256") != expected_sha256:
            raise DatasetAuditError("Existing Potato index uses a different archive hash.")
        return existing

    actual_sha256 = sha256_file(archive_path)
    if actual_sha256 != expected_sha256:
        raise DatasetAuditError(
            f"Potato archive SHA-256 mismatch: expected {expected_sha256}, "
            f"got {actual_sha256}."
        )
    files = []
    corrupted = []
    archive_class_counts: Counter[str] = Counter()
    original_class_counts: Counter[str] = Counter()
    augmented_class_counts: Counter[str] = Counter()
    other_class_counts: Counter[str] = Counter()

    with zipfile.ZipFile(archive_path) as archive:
        entries = [entry for entry in archive.infolist() if not entry.is_dir()]
        unsafe_entries = []
        for entry in entries:
            try:
                _safe_zip_parts(entry.filename)
            except DatasetAuditError:
                unsafe_entries.append(entry.filename)
        if unsafe_entries:
            raise DatasetAuditError(
                "Potato archive contains unsafe paths: "
                + ", ".join(unsafe_entries[:5])
            )

        destination.mkdir(parents=True, exist_ok=True)
        for index, entry in enumerate(entries, start=1):
            parts = _safe_zip_parts(entry.filename)
            if len(parts) != 3:
                other_class_counts["__unexpected_structure__"] += 1
                continue
            _, source_label, filename = parts
            archive_class_counts[source_label] += 1
            if filename.lower().startswith("aug_"):
                augmented_class_counts[source_label] += 1
                continue
            if not filename.lower().startswith("orig_"):
                other_class_counts[source_label] += 1
                continue
            original_class_counts[source_label] += 1
            data = archive.read(entry)
            try:
                image_format, image_mode = validate_image(data)
            except PreparationError as exc:
                corrupted.append({"source_path": entry.filename, "error": str(exc)})
                continue
            digest = sha256_bytes(data)
            suffix = FORMAT_SUFFIXES.get(image_format, ".img")
            relative = Path(safe_component(source_label)) / (
                f"{index:05d}_{digest[:16]}{suffix}"
            )
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
            files.append(
                {
                    "source_label": source_label,
                    "source_path": entry.filename.replace("\\", "/"),
                    "local_file": relative.as_posix(),
                    "source_sha256": digest,
                    "source_format": image_format,
                    "source_mode": image_mode,
                }
            )

    payload = {
        "dataset_id": "potato_banu_deb_originals",
        "source": POTATO_SOURCE_URL,
        "archive_sha256": actual_sha256,
        "archive_file_count": sum(archive_class_counts.values()),
        "archive_class_counts": dict(sorted(archive_class_counts.items())),
        "original_named_count": sum(original_class_counts.values()),
        "original_class_counts": dict(sorted(original_class_counts.items())),
        "augmented_named_count": sum(augmented_class_counts.values()),
        "augmented_class_counts": dict(sorted(augmented_class_counts.items())),
        "other_named_count": sum(other_class_counts.values()),
        "other_class_counts": dict(sorted(other_class_counts.items())),
        "materialized_image_count": len(files),
        "unsafe_entries": unsafe_entries,
        "corrupted": corrupted,
        "files": files,
    }
    write_json(destination / ".source-index.json", payload)
    return payload


def _portable_zip_parts(name: str) -> tuple[str, ...]:
    parts = _safe_zip_parts(name)
    if not parts or any(safe_component(part) != part for part in parts):
        raise DatasetAuditError(
            f"ZIP entry cannot be preserved exactly on Windows: {name}"
        )
    return parts


def _source_label_from_zip(
    parts: tuple[str, ...], archive: ZipSource, layout: str
) -> str:
    if layout == "seasonal_corn":
        if len(parts) != 3 or parts[0] != "Final corn dataset":
            raise DatasetAuditError(
                f"Unexpected Seasonal Corn archive path: {'/'.join(parts)}"
            )
        return parts[1]
    if layout == "pldd_up":
        if len(parts) != 2 or parts[0] != archive.source_label:
            raise DatasetAuditError(
                f"Unexpected PLDD-UP archive path: {'/'.join(parts)}"
            )
        return parts[0]
    raise DatasetAuditError(f"Unknown ZIP dataset layout: {layout}")


def extract_verified_original_archives(
    *,
    dataset_id: str,
    source_url: str,
    source_version: str,
    archives: Sequence[ZipSource],
    destination: Path,
    expected_class_counts: dict[str, int],
    layout: str,
) -> dict:
    """Materialize verified original files without renaming source paths."""

    expected_archive_hashes = {
        archive.path.name: archive.expected_sha256 for archive in archives
    }
    existing = _load_existing_index(destination, dataset_id)
    if existing is not None:
        if existing.get("archive_sha256") != expected_archive_hashes:
            raise DatasetAuditError(
                f"Existing {dataset_id} index uses different archive hashes."
            )
        return existing

    archive_metadata = []
    source_entries: list[tuple[ZipSource, str, str]] = []
    observed_counts: Counter[str] = Counter()
    seen_paths: set[str] = set()
    unsupported_files = []

    for archive in archives:
        actual_hash = sha256_file(archive.path)
        if actual_hash != archive.expected_sha256:
            raise DatasetAuditError(
                f"Archive SHA-256 mismatch for {archive.path.name}: "
                f"expected {archive.expected_sha256}, got {actual_hash}."
            )
        with zipfile.ZipFile(archive.path) as opened:
            entries = [entry for entry in opened.infolist() if not entry.is_dir()]
            uncompressed_bytes = 0
            for entry in entries:
                parts = _portable_zip_parts(entry.filename)
                label = _source_label_from_zip(parts, archive, layout)
                source_path = PurePosixPath(*parts).as_posix()
                if source_path in seen_paths:
                    raise DatasetAuditError(
                        f"Duplicate source path across archives: {source_path}"
                    )
                seen_paths.add(source_path)
                observed_counts[label] += 1
                uncompressed_bytes += entry.file_size
                source_entries.append((archive, entry.filename, label))
                if PurePosixPath(source_path).suffix.lower() not in SUPPORTED_IMAGE_SUFFIXES:
                    unsupported_files.append(source_path)
            archive_metadata.append(
                {
                    "filename": archive.path.name,
                    "file_id": archive.file_id,
                    "sha256": actual_hash,
                    "compressed_bytes": archive.path.stat().st_size,
                    "uncompressed_bytes": uncompressed_bytes,
                    "file_count": len(entries),
                }
            )

    if dict(sorted(observed_counts.items())) != dict(
        sorted(expected_class_counts.items())
    ):
        raise DatasetAuditError(
            f"{dataset_id} class counts differ from verified source claims: "
            f"expected={expected_class_counts}, observed={dict(observed_counts)}."
        )
    if unsupported_files:
        raise DatasetAuditError(
            f"{dataset_id} contains unsupported source files: "
            + ", ".join(unsupported_files[:5])
        )

    destination.mkdir(parents=True, exist_ok=True)
    files = []
    corrupted = []
    for archive in archives:
        with zipfile.ZipFile(archive.path) as opened:
            entries = [entry for entry in opened.infolist() if not entry.is_dir()]
            for entry in entries:
                parts = _portable_zip_parts(entry.filename)
                label = _source_label_from_zip(parts, archive, layout)
                source_path = PurePosixPath(*parts).as_posix()
                data = opened.read(entry)
                target = filesystem_path(destination.joinpath(*parts))
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(data)
                try:
                    image_format, image_mode = validate_image(data)
                except PreparationError as exc:
                    corrupted.append({"source_path": source_path, "error": str(exc)})
                    continue
                with Image.open(BytesIO(data)) as decoded:
                    width, height = decoded.size
                files.append(
                    {
                        "source_label": label,
                        "source_path": source_path,
                        "local_file": source_path,
                        "source_sha256": sha256_bytes(data),
                        "source_dhash": f"{difference_hash(data):016x}",
                        "source_format": image_format,
                        "source_mode": image_mode,
                        "width": width,
                        "height": height,
                        "bytes": len(data),
                        "original_or_augmented": "original",
                    }
                )

    payload = {
        "dataset_id": dataset_id,
        "source": source_url,
        "source_version": source_version,
        "archive_sha256": expected_archive_hashes,
        "archives": archive_metadata,
        "source_file_count": len(source_entries),
        "supported_image_candidate_count": len(source_entries),
        "valid_image_count": len(files),
        "materialized_image_count": len(source_entries),
        "original_image_count": len(source_entries),
        "augmented_image_count": 0,
        "class_counts": dict(sorted(observed_counts.items())),
        "unsupported_files": unsupported_files,
        "corrupted": corrupted,
        "files": files,
    }
    write_json(destination / ".source-index.json", payload)
    return payload


def audit_index(
    dataset: str,
    role: str,
    root: Path,
    index: dict,
    mapping: dict[str, dict],
    source_version: str = "",
    original_or_augmented: str = "original",
) -> list[dict]:
    validate_mapping_coverage(
        (entry["source_label"] for entry in index["files"]), mapping
    )
    records = []
    for entry in index["files"]:
        local_relative = PurePosixPath(entry["local_file"])
        path = filesystem_path(root.joinpath(*local_relative.parts))
        data = path.read_bytes()
        digest = sha256_bytes(data)
        if digest != entry["source_sha256"]:
            raise DatasetAuditError(f"Materialized image hash changed: {entry['local_file']}")
        image_format, image_mode = validate_image(data)
        with Image.open(BytesIO(data)) as image:
            width, height = image.size
        mapped = mapping[entry["source_label"]]
        mapping_status = mapped["status"]
        records.append(
            {
                "dataset": dataset,
                "source_version": index.get("source_version")
                or source_version
                or index.get("source_revision")
                or "",
                "role": role,
                "source_label": entry["source_label"],
                "mapping_status": mapping_status,
                "target_class": mapped["target_class"] or "",
                "source_path": entry["source_path"],
                "local_file": entry["local_file"],
                "sha256": digest,
                "dhash": entry.get("source_dhash")
                or f"{difference_hash(data):016x}",
                "format": image_format,
                "mode": image_mode,
                "width": width,
                "height": height,
                "bytes": len(data),
                "original_or_augmented": entry.get("original_or_augmented")
                or original_or_augmented,
                "candidate_status": (
                    "APPROVED_CANDIDATE"
                    if mapping_status == "MATCHED"
                    else mapping_status
                ),
            }
        )
    return records


def audit_locked_plantdoc_test(
    source_path: Path, mapping: dict[str, dict]
) -> list[dict]:
    source = PlantDocSource(
        source_path, revision=PLANTDOC_SOURCE_REVISION, split="test"
    )
    images = source.images()
    validate_mapping_coverage((image.label for image in images), mapping)
    records = []
    for image, data in iter_plantdoc_bytes(source, images):
        image_format, image_mode = validate_image(data)
        with Image.open(BytesIO(data)) as decoded:
            width, height = decoded.size
        mapped = mapping[image.label]
        records.append(
            {
                "dataset": "PlantDoc",
                "source_version": PLANTDOC_SOURCE_REVISION,
                "role": "locked_test",
                "source_label": image.label,
                "mapping_status": mapped["status"],
                "target_class": mapped["target_class"] or "",
                "source_path": image.relative_path,
                "local_file": "",
                "sha256": sha256_bytes(data),
                "dhash": f"{difference_hash(data):016x}",
                "format": image_format,
                "mode": image_mode,
                "width": width,
                "height": height,
                "bytes": len(data),
                "original_or_augmented": "original",
                "candidate_status": "LOCKED_BENCHMARK",
            }
        )
    return records


def source_summary(records: Sequence[dict]) -> dict:
    labels = Counter(record["source_label"] for record in records)
    statuses = Counter(record["mapping_status"] for record in records)
    formats = Counter(record["format"] for record in records)
    modes = Counter(record["mode"] for record in records)
    candidate_statuses = Counter(record["candidate_status"] for record in records)
    exact = sum(record["mapping_status"] == "MATCHED" for record in records)
    return {
        "valid_image_count": len(records),
        "exact_match_image_count": exact,
        "excluded_image_count": len(records) - exact,
        "class_counts": dict(sorted(labels.items())),
        "mapping_status_counts": dict(sorted(statuses.items())),
        "format_counts": dict(sorted(formats.items())),
        "mode_counts": dict(sorted(modes.items())),
        "minimum_dimensions": {
            "width": min((record["width"] for record in records), default=0),
            "height": min((record["height"] for record in records), default=0),
        },
        "maximum_dimensions": {
            "width": max((record["width"] for record in records), default=0),
            "height": max((record["height"] for record in records), default=0),
        },
        "candidate_status_counts": dict(sorted(candidate_statuses.items())),
    }


def exact_duplicate_groups(records: Sequence[dict]) -> list[dict]:
    by_hash: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        by_hash[record["sha256"]].append(record)
    groups = []
    for digest, members in sorted(by_hash.items()):
        if len(members) < 2:
            continue
        roles = sorted({member["role"] for member in members})
        source_labels = sorted({member["source_label"] for member in members})
        target_classes = sorted(
            {member["target_class"] for member in members if member["target_class"]}
        )
        label_conflict = len(source_labels) > 1 or len(target_classes) > 1
        groups.append(
            {
                "sha256": digest,
                "member_count": len(members),
                "roles": roles,
                "cross_role": len(roles) > 1,
                "touches_locked_test": "locked_test" in roles,
                "label_conflict": label_conflict,
                "source_labels": source_labels,
                "target_classes": target_classes,
                "members": [
                    {
                        "dataset": member["dataset"],
                        "role": member["role"],
                        "source_label": member["source_label"],
                        "mapping_status": member["mapping_status"],
                        "target_class": member["target_class"],
                        "source_path": member["source_path"],
                    }
                    for member in members
                ],
            }
        )
    return groups


def perceptual_duplicate_pairs(
    records: Sequence[dict], maximum_distance: int
) -> list[dict]:
    """Find every dHash pair within the threshold using lossless band indexing."""

    if not 0 <= maximum_distance <= 15:
        raise DatasetAuditError("Perceptual dHash distance must be between 0 and 15.")

    band_count = maximum_distance + 1
    base_width, wider_bands = divmod(64, band_count)

    def bands(value: int) -> list[tuple[int, int]]:
        result = []
        shift = 0
        for band_index in range(band_count):
            width = base_width + int(band_index < wider_bands)
            mask = (1 << width) - 1
            result.append((band_index, (value >> shift) & mask))
            shift += width
        return result

    pairs = []
    hashes = [int(record["dhash"], 16) for record in records]
    buckets: dict[tuple[int, int], list[int]] = defaultdict(list)
    for right_index, right in enumerate(records):
        right_bands = bands(hashes[right_index])
        candidate_indices: set[int] = set()
        for band in right_bands:
            candidate_indices.update(buckets[band])
        for left_index in sorted(candidate_indices):
            left = records[left_index]
            if left["sha256"] == right["sha256"]:
                continue
            distance = (hashes[left_index] ^ hashes[right_index]).bit_count()
            if distance > maximum_distance:
                continue
            pairs.append(
                {
                    "first_dataset": left["dataset"],
                    "first_role": left["role"],
                    "first_label": left["source_label"],
                    "first_target_class": left["target_class"],
                    "first_path": left["source_path"],
                    "first_width": left["width"],
                    "first_height": left["height"],
                    "second_dataset": right["dataset"],
                    "second_role": right["role"],
                    "second_label": right["source_label"],
                    "second_target_class": right["target_class"],
                    "second_path": right["source_path"],
                    "second_width": right["width"],
                    "second_height": right["height"],
                    "hamming_distance": distance,
                    "same_label": left["source_label"] == right["source_label"],
                    "same_target_class": left["target_class"]
                    == right["target_class"],
                    "cross_role": left["role"] != right["role"],
                    "touches_locked_test": "locked_test"
                    in {left["role"], right["role"]},
                }
            )
        for band in right_bands:
            buckets[band].append(right_index)
    return pairs


def apply_candidate_decisions(
    records: Sequence[dict], exact_groups: Sequence[dict], perceptual_pairs: Sequence[dict]
) -> None:
    by_identity = {
        (record["dataset"], record["role"], record["source_path"]): record
        for record in records
    }

    for group in exact_groups:
        training_members = [
            by_identity[(member["dataset"], member["role"], member["source_path"])]
            for member in group["members"]
            if member["role"] == "training_candidate"
        ]
        if group["touches_locked_test"] or group["label_conflict"]:
            status = "EXCLUDE_FROM_TRAINING"
        else:
            status = "NEEDS_MANUAL_REVIEW"
        for record in training_members:
            if record["mapping_status"] == "MATCHED":
                record["candidate_status"] = status

    for pair in perceptual_pairs:
        for prefix in ("first", "second"):
            role = pair[f"{prefix}_role"]
            if role != "training_candidate":
                continue
            identity = (
                pair[f"{prefix}_dataset"],
                role,
                pair[f"{prefix}_path"],
            )
            record = by_identity[identity]
            if record["candidate_status"] == "APPROVED_CANDIDATE":
                record["candidate_status"] = "NEEDS_MANUAL_REVIEW"


def candidate_counts(records: Sequence[dict]) -> dict:
    matched = [
        record
        for record in records
        if record["role"] == "training_candidate"
        and record["mapping_status"] == "MATCHED"
    ]
    by_dataset = Counter(record["dataset"] for record in matched)
    by_target = Counter(record["target_class"] for record in matched)
    by_target_and_dataset: dict[str, Counter[str]] = defaultdict(Counter)
    for record in matched:
        by_target_and_dataset[record["target_class"]][record["dataset"]] += 1
    return {
        "before_deduplication_total": len(matched),
        "by_dataset": dict(sorted(by_dataset.items())),
        "by_target_class": dict(sorted(by_target.items())),
        "by_target_class_and_dataset": {
            target: dict(sorted(datasets.items()))
            for target, datasets in sorted(by_target_and_dataset.items())
        },
    }


def compact_perceptual_report(
    full_path: Path,
    tracked_path: Path,
    aggregate_path: Path,
    fieldnames: Sequence[str],
    sample_limit_per_group: int = 25,
) -> dict:
    """Keep exhaustive dHash results locally and version a bounded review queue."""

    group_fields = [
        "first_dataset",
        "first_role",
        "first_label",
        "first_target_class",
        "second_dataset",
        "second_role",
        "second_label",
        "second_target_class",
        "hamming_distance",
        "same_target_class",
    ]
    counts: Counter[tuple[str, ...]] = Counter()
    samples: dict[tuple[str, ...], list[dict]] = defaultdict(list)
    critical: dict[tuple[str, str, str, str], dict] = {}

    with full_path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            key = tuple(row[field] for field in group_fields)
            counts[key] += 1
            if len(samples[key]) < sample_limit_per_group:
                sample = dict(row)
                sample["selection_reason"] = "DETERMINISTIC_GROUP_SAMPLE"
                samples[key].append(sample)
            if (
                row["touches_locked_test"].lower() == "true"
                or row["first_dataset"] != row["second_dataset"]
            ):
                identity = (
                    row["first_dataset"],
                    row["first_path"],
                    row["second_dataset"],
                    row["second_path"],
                )
                critical_row = dict(row)
                critical_row["selection_reason"] = (
                    "LOCKED_TEST"
                    if row["touches_locked_test"].lower() == "true"
                    else "CROSS_DATASET"
                )
                critical[identity] = critical_row

    review_rows = list(critical.values())
    critical_identities = set(critical)
    for group_samples in samples.values():
        for row in group_samples:
            identity = (
                row["first_dataset"],
                row["first_path"],
                row["second_dataset"],
                row["second_path"],
            )
            if identity not in critical_identities:
                review_rows.append(row)
    review_rows.sort(
        key=lambda row: (
            row["first_dataset"],
            row["first_path"],
            row["second_dataset"],
            row["second_path"],
        )
    )

    aggregate_rows = []
    for key, count in sorted(counts.items()):
        aggregate_rows.append(
            {**dict(zip(group_fields, key)), "candidate_pair_count": count}
        )
    write_csv(tracked_path, review_rows, [*fieldnames, "selection_reason"])
    write_csv(
        aggregate_path,
        aggregate_rows,
        [*group_fields, "candidate_pair_count"],
    )
    return {
        "full_local_report": "training/datasets/local-audits/perceptual-duplicate-candidates-full.csv",
        "full_local_report_sha256": sha256_file(full_path),
        "full_candidate_pair_count": sum(counts.values()),
        "aggregate_group_count": len(aggregate_rows),
        "tracked_review_row_count": len(review_rows),
        "tracked_review_policy": (
            "All locked-test and cross-dataset pairs, plus the first "
            f"{sample_limit_per_group} deterministic rows per aggregate group."
        ),
    }


def write_source_metadata_files(
    source_dir: Path,
    seasonal_index: dict,
    seasonal_summary: dict,
    pldd_index: dict,
    pldd_summary: dict,
) -> None:
    seasonal_payload = {
        "dataset": "Seasonal Corn Leaf Disease Dataset: A Multi-Year Collection for Robust Analysis",
        "canonical_source": "Mendeley Data",
        "doi": "10.17632/vy629dngm8.1",
        "url": SEASONAL_CORN_SOURCE_URL,
        "version": 1,
        "authors": ["MD Hasan Ahmad"],
        "publisher": "Mendeley Data",
        "license": "CC BY 4.0",
        "attribution_required": True,
        "retrieval_date": "2026-08-09",
        "reported": {
            "original_images": 2943,
            "augmented_images": 7500,
            "classes": {
                "Bacterial Leaf Streak": 190,
                "Common_rust": 129,
                "Gray_leaf_spot": 1497,
                "Healthy": 1038,
                "Maize Chlorotic Mottle Virus": 89,
            },
            "capture": "Several corn farms in Gurudaspur, Natore, Rajshahi, Bangladesh; high-resolution field images with environmental metadata.",
        },
        "verified": {
            key: value for key, value in seasonal_index.items() if key != "files"
        }
        | {"image_audit": seasonal_summary},
        "original_augmented_distinction": "The official V1 archive contains exactly the 2,943 reported originals and class counts; no augmented files or augmentation directory are present in the archive.",
        "training_eligibility": "Only MATCHED original images are candidates; unsupported classes are excluded and duplicate findings still apply.",
    }
    pldd_payload = {
        "dataset": "PLDD-UP: Potato Leaf Disease Dataset from Uttar Pradesh, India",
        "canonical_source": "Mendeley Data",
        "doi": "10.17632/3j4nfkvp2n.1",
        "url": PLDD_UP_SOURCE_URL,
        "version": 1,
        "authors": [
            "Prakash Kumar Singh",
            "Arun Yadav",
            "Divakar Yadav",
            "Sarthak Tiwari",
            "Aseem Chandel",
        ],
        "publisher": "Mendeley Data",
        "license": "CC BY 4.0",
        "attribution_required": True,
        "retrieval_date": "2026-08-09",
        "reported": {
            "images": 15519,
            "classes": {"EB": 4803, "Healthy": 4600, "LB": 6116},
            "capture": "Operational fields in Mainpuri, Etawah, and Jaswantnagar, Uttar Pradesh, during the October 2025-March 2026 Rabi season; Nikon D5300 and smartphones of at least 12 MP under natural daylight.",
        },
        "semantic_verification": {
            "EB": "Official EB archive filenames explicitly use early-blight; mapped to Potato Early blight.",
            "LB": "Official LB archive filenames explicitly use late-blight; mapped to Potato Late blight.",
            "Healthy": "The official class and archive are explicitly named Healthy; mapped to Potato healthy.",
        },
        "verified": {
            key: value for key, value in pldd_index.items() if key != "files"
        }
        | {"image_audit": pldd_summary},
        "original_augmented_distinction": "The official description states that original-resolution real-world captures are preserved; no augmented split or augmented filename convention is present in the three V1 archives.",
        "training_eligibility": "All three classes are semantic matches, subject to global duplicate and manual-review decisions.",
    }
    write_json(source_dir / "seasonal-corn.json", seasonal_payload)
    write_json(source_dir / "pldd-up.json", pldd_payload)


def run(args: argparse.Namespace) -> dict:
    plantdoc_source = args.plantdoc_source.expanduser().resolve()
    potato_archive = args.potato_archive.expanduser().resolve()
    seasonal_corn_archive = args.seasonal_corn_archive.expanduser().resolve()
    pldd_up_download_dir = args.pldd_up_download_dir.expanduser().resolve()
    raw_root = args.raw_root.expanduser().resolve()
    plantdoc_root = raw_root / "plantdoc-train"
    potato_root = raw_root / "potato-banu-deb-originals"
    seasonal_corn_root = raw_root / "seasonal_corn"
    pldd_up_root = raw_root / "pldd_up"

    plantdoc_index = extract_plantdoc_train(plantdoc_source, plantdoc_root)
    potato_index = extract_potato_originals(
        potato_archive, potato_root, args.potato_archive_sha256
    )
    seasonal_corn_index = extract_verified_original_archives(
        dataset_id="seasonal_corn_originals",
        source_url=SEASONAL_CORN_SOURCE_URL,
        source_version="1",
        archives=[
            ZipSource(
                seasonal_corn_archive,
                SEASONAL_CORN_ARCHIVE_SHA256,
                "e086f779-470a-4c7c-ba81-97734e9f8dd6",
            )
        ],
        destination=seasonal_corn_root,
        expected_class_counts={
            "Bacterial Leaf Streak": 190,
            "Common_rust": 129,
            "Gray_leaf_spot": 1497,
            "Healthy": 1038,
            "Maize Chlorotic Mottle Virus": 89,
        },
        layout="seasonal_corn",
    )
    pldd_up_index = extract_verified_original_archives(
        dataset_id="pldd_up",
        source_url=PLDD_UP_SOURCE_URL,
        source_version="1",
        archives=[
            ZipSource(
                pldd_up_download_dir / metadata["filename"],
                metadata["sha256"],
                metadata["file_id"],
                source_label,
            )
            for source_label, metadata in PLDD_UP_ARCHIVES.items()
        ],
        destination=pldd_up_root,
        expected_class_counts={"EB": 4803, "Healthy": 4600, "LB": 6116},
        layout="pldd_up",
    )

    plantdoc_train_mapping = load_plantdoc_train_mapping(
        args.plantdoc_test_mapping, args.plantdoc_train_extra_mapping
    )
    plantdoc_test_mapping = load_mapping(args.plantdoc_test_mapping)
    potato_mapping = load_mapping(args.potato_mapping)
    seasonal_corn_mapping = load_mapping(args.seasonal_corn_mapping)
    pldd_up_mapping = load_mapping(args.pldd_up_mapping)

    plantdoc_train = audit_index(
        "PlantDoc",
        "training_candidate",
        plantdoc_root,
        plantdoc_index,
        plantdoc_train_mapping,
        source_version=PLANTDOC_SOURCE_REVISION,
    )
    potato_originals = audit_index(
        "Potato Leaf Disease Dataset",
        "training_candidate",
        potato_root,
        potato_index,
        potato_mapping,
        source_version="1",
    )
    seasonal_corn = audit_index(
        "Seasonal Corn Leaf Disease Dataset",
        "training_candidate",
        seasonal_corn_root,
        seasonal_corn_index,
        seasonal_corn_mapping,
    )
    pldd_up = audit_index(
        "PLDD-UP",
        "training_candidate",
        pldd_up_root,
        pldd_up_index,
        pldd_up_mapping,
    )
    plantdoc_test = audit_locked_plantdoc_test(plantdoc_source, plantdoc_test_mapping)
    training_records = [
        *plantdoc_train,
        *potato_originals,
        *seasonal_corn,
        *pldd_up,
    ]
    all_records = [*training_records, *plantdoc_test]

    exact_groups = exact_duplicate_groups(all_records)
    perceptual_pairs = perceptual_duplicate_pairs(
        all_records, args.perceptual_maximum_distance
    )
    apply_candidate_decisions(all_records, exact_groups, perceptual_pairs)

    write_csv(args.manifest_dir / "plantdoc-train.csv", plantdoc_train, MANIFEST_FIELDS)
    write_csv(
        args.manifest_dir / "potato-banu-deb-originals.csv",
        potato_originals,
        MANIFEST_FIELDS,
    )
    write_csv(
        args.manifest_dir / "seasonal-corn-originals.csv",
        seasonal_corn,
        MANIFEST_FIELDS,
    )
    write_csv(args.manifest_dir / "pldd-up.csv", pldd_up, MANIFEST_FIELDS)
    write_csv(
        args.manifest_dir / "dataset-v2-global-audit.csv",
        all_records,
        MANIFEST_FIELDS,
    )
    full_perceptual_path = (
        args.local_audit_dir / "perceptual-duplicate-candidates-full.csv"
    )
    write_csv(full_perceptual_path, perceptual_pairs, PERCEPTUAL_FIELDS)
    perceptual_report = compact_perceptual_report(
        full_perceptual_path,
        args.report_dir / "perceptual-duplicate-candidates.csv",
        args.report_dir / "perceptual-duplicate-summary.csv",
        PERCEPTUAL_FIELDS,
    )
    write_json(args.report_dir / "exact-duplicate-groups.json", exact_groups)

    source_summaries = {
        "plantdoc_train": source_summary(plantdoc_train),
        "potato_originals": source_summary(potato_originals),
        "seasonal_corn_originals": source_summary(seasonal_corn),
        "pldd_up": source_summary(pldd_up),
        "plantdoc_test_locked": source_summary(plantdoc_test),
    }
    write_source_metadata_files(
        args.source_dir,
        seasonal_corn_index,
        source_summaries["seasonal_corn_originals"],
        pldd_up_index,
        source_summaries["pldd_up"],
    )

    summary = {
        "audit_name": "Dataset V2 Step 5B.1 global acquisition audit",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "scope": {
            "training_candidates": [
                "PlantDoc TRAIN",
                "Banu/Deb Potato originals",
                "Seasonal Corn originals",
                "PLDD-UP",
            ],
            "locked_benchmark": "PlantDoc TEST",
            "training_performed": False,
            "split_created": False,
            "images_removed": False,
        },
        "acquisition": {
            "plantdoc": {
                key: value
                for key, value in plantdoc_index.items()
                if key != "files"
            },
            "potato": {
                key: value for key, value in potato_index.items() if key != "files"
            },
            "seasonal_corn": {
                key: value
                for key, value in seasonal_corn_index.items()
                if key != "files"
            },
            "pldd_up": {
                key: value for key, value in pldd_up_index.items() if key != "files"
            },
        },
        "sources": source_summaries,
        "candidate_counts": candidate_counts(training_records),
        "duplicates": {
            "sha256_exact_group_count": len(exact_groups),
            "sha256_exact_duplicate_image_count": sum(
                group["member_count"] - 1 for group in exact_groups
            ),
            "sha256_cross_role_group_count": sum(
                group["cross_role"] for group in exact_groups
            ),
            "sha256_groups_touching_locked_test": sum(
                group["touches_locked_test"] for group in exact_groups
            ),
            "sha256_training_images_matching_locked_test": sum(
                member["role"] == "training_candidate"
                for group in exact_groups
                if group["touches_locked_test"]
                for member in group["members"]
            ),
            "sha256_label_conflict_group_count": sum(
                group["label_conflict"] for group in exact_groups
            ),
            "sha256_cross_training_source_group_count": sum(
                not group["touches_locked_test"]
                and len(
                    {
                        member["dataset"]
                        for member in group["members"]
                        if member["role"] == "training_candidate"
                    }
                )
                > 1
                for group in exact_groups
            ),
            "perceptual_maximum_hamming_distance": args.perceptual_maximum_distance,
            "perceptual_candidate_pair_count": len(perceptual_pairs),
            "perceptual_cross_role_pair_count": sum(
                pair["cross_role"] for pair in perceptual_pairs
            ),
            "perceptual_pairs_touching_locked_test": sum(
                pair["touches_locked_test"] for pair in perceptual_pairs
            ),
            "perceptual_reporting": perceptual_report,
        },
        "artifacts": {
            "plantdoc_manifest": "training/datasets/manifests/plantdoc-train.csv",
            "potato_manifest": "training/datasets/manifests/potato-banu-deb-originals.csv",
            "seasonal_corn_manifest": "training/datasets/manifests/seasonal-corn-originals.csv",
            "pldd_up_manifest": "training/datasets/manifests/pldd-up.csv",
            "global_manifest": "training/datasets/manifests/dataset-v2-global-audit.csv",
            "exact_duplicates": "training/datasets/reports/exact-duplicate-groups.json",
            "perceptual_candidates": "training/datasets/reports/perceptual-duplicate-candidates.csv",
            "perceptual_summary": "training/datasets/reports/perceptual-duplicate-summary.csv",
        },
    }
    write_json(args.report_dir / "step5b-audit-summary.json", summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--plantdoc-source",
        type=Path,
        default=PROJECT_ROOT / "evaluation" / "datasets" / "plantdoc",
    )
    parser.add_argument(
        "--potato-archive",
        type=Path,
        default=PROJECT_ROOT
        / "training"
        / "datasets"
        / "downloads"
        / "potato-leaf-disease-d5b3fzpw3g-v1.zip",
    )
    parser.add_argument(
        "--potato-archive-sha256", default=POTATO_ARCHIVE_SHA256
    )
    parser.add_argument(
        "--seasonal-corn-archive",
        type=Path,
        default=PROJECT_ROOT
        / "training"
        / "datasets"
        / "downloads"
        / "seasonal-corn-v1.zip",
    )
    parser.add_argument(
        "--pldd-up-download-dir",
        type=Path,
        default=PROJECT_ROOT / "training" / "datasets" / "downloads",
    )
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=PROJECT_ROOT / "training" / "datasets" / "raw",
    )
    parser.add_argument(
        "--manifest-dir",
        type=Path,
        default=PROJECT_ROOT / "training" / "datasets" / "manifests",
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=PROJECT_ROOT / "training" / "datasets" / "reports",
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=PROJECT_ROOT / "training" / "datasets" / "sources",
    )
    parser.add_argument(
        "--local-audit-dir",
        type=Path,
        default=PROJECT_ROOT / "training" / "datasets" / "local-audits",
    )
    parser.add_argument(
        "--plantdoc-test-mapping",
        type=Path,
        default=PROJECT_ROOT / "evaluation" / "mappings" / "plantdoc.json",
    )
    parser.add_argument(
        "--plantdoc-train-extra-mapping",
        type=Path,
        default=PROJECT_ROOT
        / "training"
        / "datasets"
        / "mappings"
        / "plantdoc-train-extra.json",
    )
    parser.add_argument(
        "--potato-mapping",
        type=Path,
        default=PROJECT_ROOT
        / "training"
        / "datasets"
        / "mappings"
        / "potato-banu-deb.json",
    )
    parser.add_argument(
        "--seasonal-corn-mapping",
        type=Path,
        default=PROJECT_ROOT
        / "training"
        / "datasets"
        / "mappings"
        / "seasonal-corn.json",
    )
    parser.add_argument(
        "--pldd-up-mapping",
        type=Path,
        default=PROJECT_ROOT
        / "training"
        / "datasets"
        / "mappings"
        / "pldd-up.json",
    )
    parser.add_argument(
        "--perceptual-maximum-distance", type=int, default=4, choices=range(0, 16)
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        summary = run(args)
    except (DatasetAuditError, PreparationError, OSError, zipfile.BadZipFile) as exc:
        parser.error(str(exc))
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
