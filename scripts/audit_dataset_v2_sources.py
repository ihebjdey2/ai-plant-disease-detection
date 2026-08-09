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
import re
import subprocess
import sys
import zipfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path, PurePosixPath
from typing import Iterable, Iterator, Sequence

from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.services.prediction_service import CLASS_NAMES  # noqa: E402
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
]


class DatasetAuditError(RuntimeError):
    """Raised when acquisition metadata or source integrity is invalid."""


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
        local_path = destination / PurePosixPath(entry["local_file"])
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


def audit_index(
    dataset: str,
    role: str,
    root: Path,
    index: dict,
    mapping: dict[str, dict],
) -> list[dict]:
    validate_mapping_coverage(
        (entry["source_label"] for entry in index["files"]), mapping
    )
    records = []
    for entry in index["files"]:
        local_relative = PurePosixPath(entry["local_file"])
        path = root.joinpath(*local_relative.parts)
        data = path.read_bytes()
        digest = sha256_bytes(data)
        if digest != entry["source_sha256"]:
            raise DatasetAuditError(f"Materialized image hash changed: {entry['local_file']}")
        image_format, image_mode = validate_image(data)
        with Image.open(path) as image:
            width, height = image.size
        mapped = mapping[entry["source_label"]]
        records.append(
            {
                "dataset": dataset,
                "role": role,
                "source_label": entry["source_label"],
                "mapping_status": mapped["status"],
                "target_class": mapped["target_class"] or "",
                "source_path": entry["source_path"],
                "local_file": entry["local_file"],
                "sha256": digest,
                "dhash": f"{difference_hash(data):016x}",
                "format": image_format,
                "mode": image_mode,
                "width": width,
                "height": height,
                "bytes": len(data),
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
            }
        )
    return records


def source_summary(records: Sequence[dict]) -> dict:
    labels = Counter(record["source_label"] for record in records)
    statuses = Counter(record["mapping_status"] for record in records)
    formats = Counter(record["format"] for record in records)
    exact = sum(record["mapping_status"] == "MATCHED" for record in records)
    return {
        "valid_image_count": len(records),
        "exact_match_image_count": exact,
        "excluded_image_count": len(records) - exact,
        "class_counts": dict(sorted(labels.items())),
        "mapping_status_counts": dict(sorted(statuses.items())),
        "format_counts": dict(sorted(formats.items())),
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
        groups.append(
            {
                "sha256": digest,
                "member_count": len(members),
                "roles": roles,
                "cross_role": len(roles) > 1,
                "touches_locked_test": "locked_test" in roles,
                "members": [
                    {
                        "dataset": member["dataset"],
                        "role": member["role"],
                        "source_label": member["source_label"],
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
    pairs = []
    hashes = [int(record["dhash"], 16) for record in records]
    for left_index, left in enumerate(records):
        for right_index in range(left_index + 1, len(records)):
            right = records[right_index]
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
                    "first_path": left["source_path"],
                    "second_dataset": right["dataset"],
                    "second_role": right["role"],
                    "second_label": right["source_label"],
                    "second_path": right["source_path"],
                    "hamming_distance": distance,
                    "same_label": left["source_label"] == right["source_label"],
                    "cross_role": left["role"] != right["role"],
                    "touches_locked_test": "locked_test"
                    in {left["role"], right["role"]},
                }
            )
    return pairs


def run(args: argparse.Namespace) -> dict:
    plantdoc_source = args.plantdoc_source.expanduser().resolve()
    potato_archive = args.potato_archive.expanduser().resolve()
    raw_root = args.raw_root.expanduser().resolve()
    plantdoc_root = raw_root / "plantdoc-train"
    potato_root = raw_root / "potato-banu-deb-originals"

    plantdoc_index = extract_plantdoc_train(plantdoc_source, plantdoc_root)
    potato_index = extract_potato_originals(
        potato_archive, potato_root, args.potato_archive_sha256
    )

    plantdoc_train_mapping = load_plantdoc_train_mapping(
        args.plantdoc_test_mapping, args.plantdoc_train_extra_mapping
    )
    plantdoc_test_mapping = load_mapping(args.plantdoc_test_mapping)
    potato_mapping = load_mapping(args.potato_mapping)

    plantdoc_train = audit_index(
        "PlantDoc", "training_candidate", plantdoc_root, plantdoc_index, plantdoc_train_mapping
    )
    potato_originals = audit_index(
        "Potato Leaf Disease Dataset",
        "training_candidate",
        potato_root,
        potato_index,
        potato_mapping,
    )
    plantdoc_test = audit_locked_plantdoc_test(
        plantdoc_source, plantdoc_test_mapping
    )
    all_records = [*plantdoc_train, *potato_originals, *plantdoc_test]

    exact_groups = exact_duplicate_groups(all_records)
    perceptual_pairs = perceptual_duplicate_pairs(
        all_records, args.perceptual_maximum_distance
    )

    write_csv(
        args.manifest_dir / "plantdoc-train.csv", plantdoc_train, MANIFEST_FIELDS
    )
    write_csv(
        args.manifest_dir / "potato-banu-deb-originals.csv",
        potato_originals,
        MANIFEST_FIELDS,
    )
    duplicate_fields = [
        "first_dataset",
        "first_role",
        "first_label",
        "first_path",
        "second_dataset",
        "second_role",
        "second_label",
        "second_path",
        "hamming_distance",
        "same_label",
        "cross_role",
        "touches_locked_test",
    ]
    write_csv(
        args.report_dir / "perceptual-duplicate-candidates.csv",
        perceptual_pairs,
        duplicate_fields,
    )
    write_json(args.report_dir / "exact-duplicate-groups.json", exact_groups)

    summary = {
        "audit_name": "Dataset V2 Step 5B acquisition audit",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "scope": {
            "training_candidates": ["PlantDoc TRAIN", "Potato originals"],
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
        },
        "sources": {
            "plantdoc_train": source_summary(plantdoc_train),
            "potato_originals": source_summary(potato_originals),
            "plantdoc_test_locked": source_summary(plantdoc_test),
        },
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
            "perceptual_maximum_hamming_distance": args.perceptual_maximum_distance,
            "perceptual_candidate_pair_count": len(perceptual_pairs),
            "perceptual_cross_role_pair_count": sum(
                pair["cross_role"] for pair in perceptual_pairs
            ),
            "perceptual_pairs_touching_locked_test": sum(
                pair["touches_locked_test"] for pair in perceptual_pairs
            ),
        },
        "artifacts": {
            "plantdoc_manifest": "training/datasets/manifests/plantdoc-train.csv",
            "potato_manifest": "training/datasets/manifests/potato-banu-deb-originals.csv",
            "exact_duplicates": "training/datasets/reports/exact-duplicate-groups.json",
            "perceptual_candidates": "training/datasets/reports/perceptual-duplicate-candidates.csv",
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
        "--perceptual-maximum-distance", type=int, default=4, choices=range(0, 65)
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
