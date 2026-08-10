from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import statistics
import subprocess
import sys
import time
import zipfile
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath
from typing import Iterable, Iterator, Sequence

import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.taxonomy import CLASS_NAMES  # noqa: E402
from scripts.refine_dataset_v2_perceptual_groups import (  # noqa: E402
    DATASET_ROOTS,
    HIGH_RISK_PHASH_MAX,
    MAX_ASPECT_LOG_DIFFERENCE,
    ORB_FEATURES,
    ORB_GOOD_MATCH_MINIMUM,
    ORB_IMAGE_MAX_DIMENSION,
    ORB_INLIER_RATIO_MINIMUM,
    ORB_MATCH_RATIO_MINIMUM,
    POSSIBLE_ORB_GOOD_MATCH_MINIMUM,
    POSSIBLE_ORB_INLIER_RATIO_MINIMUM,
    POSSIBLE_ORB_MATCH_RATIO_MINIMUM,
    SAME_TARGET_PHASH_MAX,
    aspect_log_difference,
    build_anchor_groups,
    compute_phash,
    filesystem_io_path,
    hash_distance,
)


DATASET_NAME = "Historical Mendeley 39-class source"
SOURCE_VERSION = "1"
DOI = "10.17632/tywbtsjrjv.1"
OFFICIAL_URL = "https://data.mendeley.com/datasets/tywbtsjrjv/1"
OFFICIAL_FILE_NAME = "Plant_leaf_diseases_dataset_without_augmentation.zip"
OFFICIAL_FILE_ID = "d5652a28-c1d8-4b76-97f3-72fb80f94efc"
OFFICIAL_ARCHIVE_BYTES = 868032562
OFFICIAL_ARCHIVE_SHA256 = (
    "ac3432453984d02a86197987e775a5429d0d59e7cc7c35bcf5a8f50349b90ff0"
)
PUBLISHED_TOTAL_COUNT = 61486
RETRIEVAL_DATE = "2026-08-10"
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
DHASH_MAXIMUM_DISTANCE = 4

MANIFEST_FIELDS = [
    "record_id",
    "dataset",
    "source_version",
    "source_path",
    "source_label",
    "target_index",
    "target_class",
    "sha256",
    "dhash",
    "phash",
    "format",
    "mode",
    "width",
    "height",
    "bytes",
    "integrity_status",
    "candidate_status",
    "candidate_reason",
    "canonical_record_id",
    "exact_duplicate_group_id",
    "perceptual_group_id",
    "exact_overlap_dataset_v2",
    "perceptual_overlap_dataset_v2",
    "benchmark_leakage",
]

LOCAL_PAIR_FIELDS = [
    "first_record_id",
    "second_record_id",
    "risk_type",
    "dhash_distance",
    "phash_distance",
    "aspect_log_difference",
    "first_keypoints",
    "second_keypoints",
    "orb_good_matches",
    "orb_match_ratio",
    "orb_inlier_ratio",
    "verification_status",
]

STATUS_PRIORITY = {
    "INCLUDE_CANDIDATE": 0,
    "REVIEW_PERCEPTUAL_CONFLICT": 1,
    "EXCLUDE_EXACT_DUPLICATE": 2,
    "EXCLUDE_LABEL_CONFLICT": 3,
    "EXCLUDE_BENCHMARK_LEAKAGE": 4,
    "INVALID_IMAGE": 5,
}


class HistoricalAuditError(RuntimeError):
    """Raised when historical-source evidence violates an audit invariant."""


def difference_hash_image(image: Image.Image) -> int:
    """Return the same deterministic 64-bit dHash used by Dataset V2 tooling."""

    grayscale = image.convert("L").resize((9, 8), Image.Resampling.LANCZOS)
    pixels = np.asarray(grayscale, dtype=np.int16)
    bits = (pixels[:, 1:] > pixels[:, :-1]).reshape(-1)
    return sum(int(bit) << (63 - index) for index, bit in enumerate(bits))


def difference_hash(data: bytes) -> int:
    with Image.open(io.BytesIO(data)) as image:
        image.load()
        return difference_hash_image(image)


def stable_record_id(source_path: str) -> str:
    identity = "\0".join((DATASET_NAME, SOURCE_VERSION, source_path))
    return f"hist_{hashlib.sha256(identity.encode('utf-8')).hexdigest()[:20]}"


def write_csv(path: Path, rows: Iterable[dict], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_archive(path: Path) -> dict:
    if not path.is_file():
        raise HistoricalAuditError(f"Official archive is unavailable: {path}")
    actual_bytes = path.stat().st_size
    actual_hash = file_sha256(path)
    if actual_bytes != OFFICIAL_ARCHIVE_BYTES:
        raise HistoricalAuditError(
            f"Archive size mismatch: expected {OFFICIAL_ARCHIVE_BYTES}, got {actual_bytes}."
        )
    if actual_hash != OFFICIAL_ARCHIVE_SHA256:
        raise HistoricalAuditError(
            f"Archive SHA-256 mismatch: expected {OFFICIAL_ARCHIVE_SHA256}, got {actual_hash}."
        )
    unsafe_paths: list[str] = []
    extraction_errors: list[str] = []
    with zipfile.ZipFile(path) as archive:
        bad_member = archive.testzip()
        if bad_member:
            extraction_errors.append(f"CRC failure: {bad_member}")
        entries = archive.infolist()
        for entry in entries:
            pure = PurePosixPath(entry.filename.replace("\\", "/"))
            if pure.is_absolute() or ".." in pure.parts:
                unsafe_paths.append(entry.filename)
        file_entries = [entry for entry in entries if not entry.is_dir()]
        uncompressed_bytes = sum(entry.file_size for entry in file_entries)
    if unsafe_paths or extraction_errors:
        raise HistoricalAuditError(
            f"Unsafe or corrupt archive: unsafe={unsafe_paths[:3]}, "
            f"errors={extraction_errors}."
        )
    return {
        "filename": path.name,
        "official_file_id": OFFICIAL_FILE_ID,
        "compressed_bytes": actual_bytes,
        "uncompressed_bytes": uncompressed_bytes,
        "sha256": actual_hash,
        "entry_count": len(entries),
        "file_count": len(file_entries),
        "unsafe_zip_paths": [],
        "extraction_errors": [],
    }


def load_explicit_mapping(
    path: Path, class_names: Sequence[str] = CLASS_NAMES
) -> dict[str, dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    entries = payload.get("classes")
    if not isinstance(entries, list):
        raise HistoricalAuditError("Historical mapping must contain a classes list.")
    mapping: dict[str, dict] = {}
    target_indices: set[int] = set()
    for entry in entries:
        source = entry.get("source_label")
        if not source or source in mapping:
            raise HistoricalAuditError(f"Duplicate or empty source mapping: {source!r}")
        status = entry.get("mapping_status")
        if status not in {"MATCHED", "AMBIGUOUS", "NOT_SUPPORTED"}:
            raise HistoricalAuditError(f"Invalid mapping status for {source}: {status}")
        if status == "MATCHED":
            index = entry.get("target_index")
            if not isinstance(index, int) or not 0 <= index < len(class_names):
                raise HistoricalAuditError(f"Invalid target index for {source}: {index}")
            if entry.get("target_class") != class_names[index]:
                raise HistoricalAuditError(
                    f"Target class/index mismatch for {source}: "
                    f"{entry.get('target_class')!r} != {class_names[index]!r}."
                )
            if index in target_indices:
                raise HistoricalAuditError(f"Duplicate deployed target index: {index}")
            target_indices.add(index)
        mapping[source] = dict(entry)
    return mapping


def validate_full_mapping(
    observed_labels: Iterable[str],
    mapping: dict[str, dict],
    class_names: Sequence[str] = CLASS_NAMES,
) -> None:
    observed = set(observed_labels)
    configured = set(mapping)
    if observed != configured:
        raise HistoricalAuditError(
            f"Source/mapping labels differ: missing={sorted(observed - configured)}, "
            f"not_observed={sorted(configured - observed)}."
        )
    matched = [entry for entry in mapping.values() if entry["mapping_status"] == "MATCHED"]
    indices = {entry["target_index"] for entry in matched}
    targets = {entry["target_class"] for entry in matched}
    if len(matched) != len(class_names) or indices != set(range(len(class_names))):
        raise HistoricalAuditError("The source does not map one-to-one to all deployed indices.")
    if targets != set(class_names):
        raise HistoricalAuditError("The source does not map one-to-one to all deployed classes.")


def _find_source_root(extracted_root: Path) -> Path:
    if not extracted_root.is_dir():
        raise HistoricalAuditError(f"Extracted source is unavailable: {extracted_root}")
    children = sorted(path for path in extracted_root.iterdir() if path.is_dir())
    if len(children) == 1 and not any(path.is_file() for path in extracted_root.iterdir()):
        return children[0]
    return extracted_root


def _initial_record(source_path: str, label: str, mapped: dict, data: bytes) -> dict:
    return {
        "record_id": stable_record_id(source_path),
        "dataset": DATASET_NAME,
        "source_version": SOURCE_VERSION,
        "source_path": source_path,
        "source_label": label,
        "target_index": mapped["target_index"],
        "target_class": mapped["target_class"],
        "sha256": hashlib.sha256(data).hexdigest(),
        "dhash": "",
        "phash": "",
        "format": "",
        "mode": "",
        "width": "",
        "height": "",
        "bytes": len(data),
        "integrity_status": "VALID",
        "candidate_status": "INCLUDE_CANDIDATE",
        "candidate_reason": "",
        "canonical_record_id": "",
        "exact_duplicate_group_id": "",
        "perceptual_group_id": "",
        "exact_overlap_dataset_v2": "false",
        "perceptual_overlap_dataset_v2": "false",
        "benchmark_leakage": "false",
    }


def audit_extracted_source(
    extracted_root: Path, mapping: dict[str, dict]
) -> tuple[list[dict], dict, dict[str, Path]]:
    source_root = _find_source_root(extracted_root)
    class_directories = sorted(path for path in source_root.iterdir() if path.is_dir())
    validate_full_mapping((path.name for path in class_directories), mapping)
    records: list[dict] = []
    record_paths: dict[str, Path] = {}
    unsupported: list[str] = []
    corrupt: list[dict] = []
    class_counts: Counter[str] = Counter()
    format_counts: Counter[str] = Counter()
    mode_counts: Counter[str] = Counter()
    started = time.monotonic()
    scanned = 0
    for class_directory in class_directories:
        label = class_directory.name
        mapped = mapping[label]
        for path in sorted(class_directory.iterdir(), key=lambda item: item.name):
            relative = PurePosixPath(source_root.name, label, path.name).as_posix()
            io_path = filesystem_io_path(path)
            if not io_path.is_file() or path.suffix.lower() not in ALLOWED_EXTENSIONS:
                unsupported.append(relative)
                continue
            scanned += 1
            data = io_path.read_bytes()
            record = _initial_record(relative, label, mapped, data)
            try:
                with Image.open(io.BytesIO(data)) as image:
                    image.load()
                    record["format"] = image.format or ""
                    record["mode"] = image.mode
                    record["width"], record["height"] = image.size
                    record["phash"] = compute_phash(image)
                    record["dhash"] = f"{difference_hash_image(image):016x}"
                class_counts[label] += 1
                format_counts[record["format"]] += 1
                mode_counts[record["mode"]] += 1
            except (OSError, ValueError, UnidentifiedImageError) as exc:
                record["integrity_status"] = "CORRUPT"
                record["candidate_status"] = "INVALID_IMAGE"
                record["candidate_reason"] = "IMAGE_DECODE_FAILED"
                corrupt.append({"source_path": relative, "error": type(exc).__name__})
            records.append(record)
            record_paths[record["record_id"]] = path
            if scanned % 2000 == 0:
                print(
                    f"Historical image audit {scanned} files in "
                    f"{time.monotonic() - started:.1f}s",
                    flush=True,
                )
    records.sort(key=lambda row: row["record_id"])
    valid = [row for row in records if row["integrity_status"] == "VALID"]
    inventory = {
        "source_root_name": source_root.name,
        "source_class_count": len(class_directories),
        "total_image_candidate_count": len(records),
        "valid_image_count": len(valid),
        "corrupt_image_count": len(corrupt),
        "unsupported_file_count": len(unsupported),
        "unsupported_files": unsupported,
        "corrupted": corrupt,
        "class_counts": dict(sorted(class_counts.items())),
        "format_counts": dict(sorted(format_counts.items())),
        "mode_counts": dict(sorted(mode_counts.items())),
        "minimum_dimensions": {
            "width": min((int(row["width"]) for row in valid), default=0),
            "height": min((int(row["height"]) for row in valid), default=0),
        },
        "maximum_dimensions": {
            "width": max((int(row["width"]) for row in valid), default=0),
            "height": max((int(row["height"]) for row in valid), default=0),
        },
    }
    return records, inventory, record_paths


def _set_status(record: dict, status: str, reason: str) -> None:
    if STATUS_PRIORITY[status] >= STATUS_PRIORITY[record["candidate_status"]]:
        record["candidate_status"] = status
        record["candidate_reason"] = reason


def audit_internal_exact_duplicates(records: Sequence[dict]) -> tuple[list[dict], dict]:
    groups: defaultdict[str, list[dict]] = defaultdict(list)
    for record in records:
        if record["integrity_status"] == "VALID":
            groups[record["sha256"]].append(record)
    report: list[dict] = []
    duplicate_copies = 0
    conflict_groups = 0
    for digest, members in sorted(groups.items()):
        if len(members) < 2:
            continue
        ordered = sorted(members, key=lambda row: row["record_id"])
        targets = sorted({row["target_class"] for row in ordered})
        group_id = f"hist_exact_{digest[:20]}"
        for member in ordered:
            member["exact_duplicate_group_id"] = group_id
        conflict = len(targets) > 1
        if conflict:
            conflict_groups += 1
            for member in ordered:
                _set_status(member, "EXCLUDE_LABEL_CONFLICT", "EXACT_CROSS_CLASS_LABEL_CONFLICT")
        else:
            canonical = ordered[0]
            canonical["canonical_record_id"] = canonical["record_id"]
            for member in ordered[1:]:
                member["canonical_record_id"] = canonical["record_id"]
                _set_status(member, "EXCLUDE_EXACT_DUPLICATE", "INTERNAL_EXACT_DUPLICATE_COPY")
        duplicate_copies += len(ordered) - 1
        report.append(
            {
                "group_id": group_id,
                "sha256": digest,
                "member_count": len(ordered),
                "copy_count_beyond_first": len(ordered) - 1,
                "label_conflict": conflict,
                "source_labels": sorted({row["source_label"] for row in ordered}),
                "target_classes": targets,
                "record_ids": [row["record_id"] for row in ordered],
            }
        )
    return report, {
        "group_count": len(report),
        "copy_count_beyond_first": duplicate_copies,
        "label_conflict_group_count": conflict_groups,
    }


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def audit_cross_exact(
    historical: Sequence[dict], dataset_v2: Sequence[dict], locked_test: Sequence[dict]
) -> dict:
    v2_by_hash: defaultdict[str, list[dict]] = defaultdict(list)
    test_by_hash: defaultdict[str, list[dict]] = defaultdict(list)
    for row in dataset_v2:
        v2_by_hash[row["sha256"]].append(row)
    for row in locked_test:
        test_by_hash[row["sha256"]].append(row)
    v2_pairs: list[dict] = []
    test_pairs: list[dict] = []
    v2_historical_ids: set[str] = set()
    test_historical_ids: set[str] = set()
    for record in historical:
        if record["integrity_status"] != "VALID":
            continue
        for other in v2_by_hash.get(record["sha256"], []):
            record["exact_overlap_dataset_v2"] = "true"
            v2_historical_ids.add(record["record_id"])
            v2_pairs.append(
                {
                    "historical_class": record["target_class"],
                    "dataset_v2_source": other["dataset"],
                    "dataset_v2_target": other["target_class"],
                }
            )
        for other in test_by_hash.get(record["sha256"], []):
            record["benchmark_leakage"] = "true"
            _set_status(
                record,
                "EXCLUDE_BENCHMARK_LEAKAGE",
                "HISTORICAL_BENCHMARK_LEAKAGE_EXACT",
            )
            test_historical_ids.add(record["record_id"])
            test_pairs.append(
                {
                    "historical_class": record["target_class"],
                    "plantdoc_test_label": other["source_label"],
                    "plantdoc_test_target": other["target_class"],
                }
            )
    return {
        "dataset_v2_pair_count": len(v2_pairs),
        "dataset_v2_historical_image_count": len(v2_historical_ids),
        "dataset_v2_by_relationship": _aggregate_relationships(v2_pairs),
        "plantdoc_test_pair_count": len(test_pairs),
        "plantdoc_test_historical_image_count": len(test_historical_ids),
        "plantdoc_test_by_relationship": _aggregate_relationships(test_pairs),
    }


def _aggregate_relationships(rows: Sequence[dict]) -> list[dict]:
    counts: Counter[tuple[tuple[str, str], ...]] = Counter()
    for row in rows:
        counts[tuple(sorted(row.items()))] += 1
    return [dict(key) | {"count": count} for key, count in sorted(counts.items())]


def _hash_bands(value: int, maximum_distance: int = DHASH_MAXIMUM_DISTANCE):
    band_count = maximum_distance + 1
    base_width, wider_bands = divmod(64, band_count)
    shift = 0
    for band_index in range(band_count):
        width = base_width + int(band_index < wider_bands)
        yield band_index, (value >> shift) & ((1 << width) - 1)
        shift += width


def internal_dhash_candidates(records: Sequence[dict]) -> Iterator[tuple[dict, dict, int]]:
    ordered = sorted(
        (row for row in records if row["integrity_status"] == "VALID"),
        key=lambda row: row["record_id"],
    )
    buckets: defaultdict[tuple[int, int], list[int]] = defaultdict(list)
    values = [int(row["dhash"], 16) for row in ordered]
    for right_index, right in enumerate(ordered):
        candidates: set[int] = set()
        bands = list(_hash_bands(values[right_index]))
        for band in bands:
            candidates.update(buckets[band])
        for left_index in sorted(candidates):
            left = ordered[left_index]
            if left["sha256"] == right["sha256"]:
                continue
            distance = (values[left_index] ^ values[right_index]).bit_count()
            if distance <= DHASH_MAXIMUM_DISTANCE:
                yield left, right, distance
        for band in bands:
            buckets[band].append(right_index)


def cross_dhash_candidates(
    historical: Sequence[dict], comparison: Sequence[dict]
) -> Iterator[tuple[dict, dict, int]]:
    comparison = sorted(comparison, key=lambda row: row["record_id"])
    buckets: defaultdict[tuple[int, int], list[int]] = defaultdict(list)
    comparison_hashes = [int(row["dhash"], 16) for row in comparison]
    for index, value in enumerate(comparison_hashes):
        for band in _hash_bands(value):
            buckets[band].append(index)
    for left in sorted(historical, key=lambda row: row["record_id"]):
        if left["integrity_status"] != "VALID":
            continue
        left_hash = int(left["dhash"], 16)
        candidates: set[int] = set()
        for band in _hash_bands(left_hash):
            candidates.update(buckets[band])
        for index in sorted(candidates):
            right = comparison[index]
            if left["sha256"] == right["sha256"]:
                continue
            distance = (left_hash ^ comparison_hashes[index]).bit_count()
            if distance <= DHASH_MAXIMUM_DISTANCE:
                yield left, right, distance


class AuditImageProvider:
    def __init__(
        self,
        project_root: Path,
        historical_paths: dict[str, Path],
        plantdoc_repo: Path,
    ):
        self.project_root = project_root
        self.historical_paths = historical_paths
        self.plantdoc_repo = plantdoc_repo

    def image_bytes(self, record: dict) -> bytes:
        if record.get("_origin") == "historical":
            return filesystem_io_path(self.historical_paths[record["record_id"]]).read_bytes()
        if record.get("role") == "locked_test":
            specification = f"{record['source_version']}:{record['source_path']}"
            try:
                return subprocess.run(
                    ["git", "-C", str(self.plantdoc_repo), "show", specification],
                    capture_output=True,
                    check=True,
                ).stdout
            except subprocess.CalledProcessError as exc:
                raise HistoricalAuditError(
                    f"Locked PlantDoc blob is unavailable: {specification}"
                ) from exc
        try:
            root = DATASET_ROOTS[record["dataset"]]
        except KeyError as exc:
            raise HistoricalAuditError(
                f"No local root configured for Dataset V2 source {record['dataset']}"
            ) from exc
        path = (self.project_root / root / Path(record["local_file"])).resolve()
        return filesystem_io_path(path).read_bytes()

    def phash(self, record: dict) -> str:
        if record.get("phash"):
            return record["phash"]
        with Image.open(io.BytesIO(self.image_bytes(record))) as image:
            return compute_phash(image)

    def orb_image(self, record: dict) -> np.ndarray:
        with Image.open(io.BytesIO(self.image_bytes(record))) as image:
            image.draft("L", (700, 700))
            grayscale = image.convert("L")
            scale = min(1.0, ORB_IMAGE_MAX_DIMENSION / max(grayscale.size))
            size = (
                max(1, round(grayscale.width * scale)),
                max(1, round(grayscale.height * scale)),
            )
            return np.asarray(
                grayscale.resize(size, Image.Resampling.LANCZOS), dtype=np.uint8
            )


def _external_record_id(record: dict) -> str:
    if record.get("record_id"):
        return record["record_id"]
    identity = "\0".join(
        (record["dataset"], record["source_version"], record["role"], record["source_path"])
    )
    return f"ext_{hashlib.sha256(identity.encode('utf-8')).hexdigest()[:20]}"


def prepare_comparison_records(rows: Sequence[dict], origin: str) -> list[dict]:
    prepared = []
    for source in rows:
        record = dict(source)
        record["record_id"] = _external_record_id(record)
        record["_origin"] = origin
        prepared.append(record)
    return prepared


def _load_phash_cache(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    return {
        row["record_id"]: row["phash"]
        for row in load_csv(path)
        if len(row.get("phash", "")) == 16
    }


def _save_phash_cache(path: Path, values: dict[str, str]) -> None:
    write_csv(
        path,
        ({"record_id": key, "phash": values[key]} for key in sorted(values)),
        ["record_id", "phash"],
    )


def select_refinement_candidates(
    pairs: Iterable[tuple[dict, dict, int]],
    risk_type: str,
    provider: AuditImageProvider,
    phash_cache: dict[str, str],
) -> tuple[list[dict], int]:
    selected: list[dict] = []
    screened = 0
    for first, second, dhash_distance in pairs:
        screened += 1
        first_phash = first.get("phash") or phash_cache.get(first["record_id"])
        if not first_phash:
            first_phash = provider.phash(first)
            phash_cache[first["record_id"]] = first_phash
        second_phash = second.get("phash") or phash_cache.get(second["record_id"])
        if not second_phash:
            second_phash = provider.phash(second)
            phash_cache[second["record_id"]] = second_phash
        phash_distance = hash_distance(first_phash, second_phash)
        aspect_difference = aspect_log_difference(first, second)
        same_target = first.get("target_class") == second.get("target_class")
        threshold = SAME_TARGET_PHASH_MAX if same_target else HIGH_RISK_PHASH_MAX
        keep = risk_type == "HISTORICAL_TO_PLANTDOC_TEST" or (
            aspect_difference <= MAX_ASPECT_LOG_DIFFERENCE
            and phash_distance <= threshold
        )
        if keep:
            selected.append(
                {
                    "first_record_id": first["record_id"],
                    "second_record_id": second["record_id"],
                    "risk_type": risk_type,
                    "dhash_distance": str(dhash_distance),
                    "phash_distance": str(phash_distance),
                    "aspect_log_difference": f"{aspect_difference:.8f}",
                }
            )
    return selected, screened


def calculate_orb_signals(
    records: dict[str, dict],
    candidates: Sequence[dict],
    provider: AuditImageProvider,
) -> list[dict]:
    involved = sorted(
        {
            record_id
            for candidate in candidates
            for record_id in (candidate["first_record_id"], candidate["second_record_id"])
        }
    )
    orb = cv2.ORB_create(nfeatures=ORB_FEATURES, fastThreshold=12, edgeThreshold=15)
    descriptors: dict[str, tuple[np.ndarray, np.ndarray | None]] = {}
    started = time.monotonic()
    for index, record_id in enumerate(involved, 1):
        keypoints, descriptor = orb.detectAndCompute(provider.orb_image(records[record_id]), None)
        points = (
            np.float32([keypoint.pt for keypoint in keypoints])
            if keypoints
            else np.empty((0, 2), dtype=np.float32)
        )
        descriptors[record_id] = points, descriptor
        if index % 500 == 0:
            print(
                f"Historical ORB descriptors {index}/{len(involved)} in "
                f"{time.monotonic() - started:.1f}s",
                flush=True,
            )
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    signals: list[dict] = []
    for index, candidate in enumerate(candidates, 1):
        first_points, first_descriptor = descriptors[candidate["first_record_id"]]
        second_points, second_descriptor = descriptors[candidate["second_record_id"]]
        matches = []
        if (
            first_descriptor is not None
            and second_descriptor is not None
            and len(first_descriptor) >= 2
            and len(second_descriptor) >= 2
        ):
            for pair in matcher.knnMatch(first_descriptor, second_descriptor, k=2):
                if len(pair) == 2 and pair[0].distance < 0.75 * pair[1].distance:
                    matches.append(pair[0])
        denominator = max(1, min(len(first_points), len(second_points)))
        match_ratio = len(matches) / denominator
        inlier_ratio = 0.0
        if len(matches) >= 8:
            source = np.float32([first_points[item.queryIdx] for item in matches]).reshape(-1, 1, 2)
            destination = np.float32([second_points[item.trainIdx] for item in matches]).reshape(-1, 1, 2)
            cv2.setRNGSeed(0)
            _, mask = cv2.findHomography(source, destination, cv2.RANSAC, 4.0)
            if mask is not None:
                inlier_ratio = float(mask.sum() / len(mask))
        signal = candidate | {
            "first_keypoints": str(len(first_points)),
            "second_keypoints": str(len(second_points)),
            "orb_good_matches": str(len(matches)),
            "orb_match_ratio": f"{match_ratio:.8f}",
            "orb_inlier_ratio": f"{inlier_ratio:.8f}",
        }
        signal["verification_status"] = signal_status(signal)
        signals.append(signal)
        if index % 3000 == 0:
            print(
                f"Historical ORB matching {index}/{len(candidates)} in "
                f"{time.monotonic() - started:.1f}s",
                flush=True,
            )
    return signals


def signal_status(signal: dict) -> str:
    verified = (
        int(signal["orb_good_matches"]) >= ORB_GOOD_MATCH_MINIMUM
        and float(signal["orb_match_ratio"]) >= ORB_MATCH_RATIO_MINIMUM
        and float(signal["orb_inlier_ratio"]) >= ORB_INLIER_RATIO_MINIMUM
    )
    if verified:
        return "VERIFIED_NEAR_DUPLICATE"
    possible = (
        int(signal["orb_good_matches"]) >= POSSIBLE_ORB_GOOD_MATCH_MINIMUM
        and float(signal["orb_match_ratio"]) >= POSSIBLE_ORB_MATCH_RATIO_MINIMUM
        and float(signal["orb_inlier_ratio"]) >= POSSIBLE_ORB_INLIER_RATIO_MINIMUM
    )
    return "POSSIBLE_NEAR_DUPLICATE" if possible else "NOT_VERIFIED"


def apply_perceptual_policy(
    historical: Sequence[dict],
    all_records: dict[str, dict],
    signals: Sequence[dict],
) -> dict:
    historical_ids = {row["record_id"] for row in historical}
    verified = [row for row in signals if row["verification_status"] == "VERIFIED_NEAR_DUPLICATE"]
    possible = [row for row in signals if row["verification_status"] == "POSSIBLE_NEAR_DUPLICATE"]
    internal_same_edges: list[tuple[str, str]] = []
    v2_rows: list[dict] = []
    benchmark_rows: list[dict] = []
    benchmark_verified_ids: set[str] = set()
    benchmark_possible_ids: set[str] = set()
    internal_conflict_ids: set[str] = set()
    internal_possible_conflict_ids: set[str] = set()
    for signal in verified:
        first = all_records[signal["first_record_id"]]
        second = all_records[signal["second_record_id"]]
        risk = signal["risk_type"]
        if risk == "INTERNAL_HISTORICAL":
            if first["target_class"] == second["target_class"]:
                internal_same_edges.append((first["record_id"], second["record_id"]))
            else:
                for record in (first, second):
                    if record["record_id"] in historical_ids:
                        _set_status(
                            record,
                            "REVIEW_PERCEPTUAL_CONFLICT",
                            "VERIFIED_CROSS_CLASS_PERCEPTUAL_CONFLICT",
                        )
                        internal_conflict_ids.add(record["record_id"])
        elif risk == "HISTORICAL_TO_DATASET_V2":
            historical_record = first if first["record_id"] in historical_ids else second
            other = second if historical_record is first else first
            historical_record["perceptual_overlap_dataset_v2"] = "true"
            v2_rows.append(
                {
                    "historical_class": historical_record["target_class"],
                    "dataset_v2_source": other["dataset"],
                    "dataset_v2_target": other["target_class"],
                }
            )
        elif risk == "HISTORICAL_TO_PLANTDOC_TEST":
            historical_record = first if first["record_id"] in historical_ids else second
            other = second if historical_record is first else first
            historical_record["benchmark_leakage"] = "true"
            _set_status(
                historical_record,
                "EXCLUDE_BENCHMARK_LEAKAGE",
                "HISTORICAL_BENCHMARK_LEAKAGE_PERCEPTUAL",
            )
            benchmark_verified_ids.add(historical_record["record_id"])
            benchmark_rows.append(
                {
                    "historical_class": historical_record["target_class"],
                    "plantdoc_test_label": other["source_label"],
                    "plantdoc_test_target": other["target_class"],
                }
            )
    for signal in possible:
        first = all_records[signal["first_record_id"]]
        second = all_records[signal["second_record_id"]]
        if (
            signal["risk_type"] == "INTERNAL_HISTORICAL"
            and first["target_class"] != second["target_class"]
        ):
            for record in (first, second):
                _set_status(
                    record,
                    "REVIEW_PERCEPTUAL_CONFLICT",
                    "POSSIBLE_CROSS_CLASS_PERCEPTUAL_CONFLICT",
                )
                internal_possible_conflict_ids.add(record["record_id"])
            continue
        if signal["risk_type"] != "HISTORICAL_TO_PLANTDOC_TEST":
            continue
        historical_record = first if first["record_id"] in historical_ids else second
        if historical_record["record_id"] not in benchmark_verified_ids:
            _set_status(
                historical_record,
                "REVIEW_PERCEPTUAL_CONFLICT",
                "POSSIBLE_HISTORICAL_BENCHMARK_LEAKAGE",
            )
            benchmark_possible_ids.add(historical_record["record_id"])

    groups = build_anchor_groups(historical_ids, internal_same_edges)
    for representative, members in sorted(groups.items()):
        group_identity = "\0".join(sorted(members))
        group_id = f"hist_near_{hashlib.sha256(group_identity.encode()).hexdigest()[:20]}"
        for record_id in members:
            all_records[record_id]["perceptual_group_id"] = group_id
    return {
        "screened_signal_count": len(signals),
        "verified_signal_count": len(verified),
        "possible_signal_count": len(possible),
        "internal_verified_same_target_pair_count": len(internal_same_edges),
        "internal_direct_group_count": len(groups),
        "internal_grouped_image_count": len({member for values in groups.values() for member in values}),
        "internal_verified_conflict_image_count": len(internal_conflict_ids),
        "internal_possible_conflict_image_count": len(internal_possible_conflict_ids),
        "dataset_v2_verified_pair_count": len(v2_rows),
        "dataset_v2_verified_historical_image_count": len(
            {
                record["record_id"]
                for record in historical
                if record["perceptual_overlap_dataset_v2"] == "true"
            }
        ),
        "dataset_v2_by_relationship": _aggregate_relationships(v2_rows),
        "plantdoc_test_verified_pair_count": len(benchmark_rows),
        "plantdoc_test_verified_historical_image_count": len(benchmark_verified_ids),
        "plantdoc_test_possible_historical_image_count": len(benchmark_possible_ids),
        "plantdoc_test_by_relationship": _aggregate_relationships(benchmark_rows),
    }


def clean_historical_records(records: Sequence[dict]) -> list[dict]:
    return [dict(row) for row in records if row["candidate_status"] == "INCLUDE_CANDIDATE"]


def coverage_counts(records: Sequence[dict], class_names: Sequence[str] = CLASS_NAMES) -> dict:
    counts = Counter(row["target_class"] for row in records)
    return {
        "coverage_count": sum(counts[name] > 0 for name in class_names),
        "coverage_missing": [name for name in class_names if counts[name] == 0],
        "by_target_class": {
            str(index): {"target_class": name, "count": counts[name]}
            for index, name in enumerate(class_names)
        },
    }


def combined_projection(historical: Sequence[dict], dataset_v2: Sequence[dict]) -> dict:
    historical_counts = Counter(row["target_class"] for row in historical)
    real_counts = Counter(row["target_class"] for row in dataset_v2)
    real_sources: defaultdict[str, set[str]] = defaultdict(set)
    for row in dataset_v2:
        real_sources[row["target_class"]].add(row["dataset"])
    rows = []
    for index, name in enumerate(CLASS_NAMES):
        rows.append(
            {
                "target_index": index,
                "target_class": name,
                "historical_count": historical_counts[name],
                "real_world_v2_count": real_counts[name],
                "real_world_source_count": len(real_sources[name]),
                "real_world_sources": sorted(real_sources[name]),
                "projected_total": historical_counts[name] + real_counts[name],
            }
        )
    totals = [row["projected_total"] for row in rows]
    largest = max(rows, key=lambda row: (row["projected_total"], -row["target_index"]))
    smallest = min(rows, key=lambda row: (row["projected_total"], row["target_index"]))
    return {
        "by_target_class": rows,
        "historical_only_classes": [row["target_class"] for row in rows if row["real_world_v2_count"] == 0],
        "classes_with_real_world_support": [row["target_class"] for row in rows if row["real_world_v2_count"] > 0],
        "classes_with_multiple_real_world_sources": [row["target_class"] for row in rows if row["real_world_source_count"] > 1],
        "largest_projected_class": largest,
        "smallest_projected_class": smallest,
        "median_projected_class_size": statistics.median(totals),
    }


def build_source_metadata(
    archive: dict, inventory: dict, audit_summary: dict
) -> dict:
    return {
        "canonical_name": "Data for: Identification of Plant Leaf Diseases Using a 9-layer Deep Convolutional Neural Network",
        "authors": ["Arun Pandian J", "Geetharamani Gopal"],
        "publisher": "Mendeley Data",
        "doi": DOI,
        "url": OFFICIAL_URL,
        "version": 1,
        "publication_date": "2019-04-18",
        "license": "CC0 1.0",
        "retrieval_date": RETRIEVAL_DATE,
        "official_file_name": OFFICIAL_FILE_NAME,
        "official_file_id": OFFICIAL_FILE_ID,
        "official_download_url": f"https://data.mendeley.com/public-files/datasets/tywbtsjrjv/files/{OFFICIAL_FILE_ID}/file_downloaded",
        "archive_sha256": archive["sha256"],
        "archive_compressed_bytes": archive["compressed_bytes"],
        "archive_uncompressed_bytes": archive["uncompressed_bytes"],
        "archive_unsafe_zip_paths": archive["unsafe_zip_paths"],
        "archive_extraction_errors": archive["extraction_errors"],
        "published_total_count": PUBLISHED_TOTAL_COUNT,
        "verified_original_count": inventory["valid_image_count"],
        "source_class_count": inventory["source_class_count"],
        "mapped_class_count": audit_summary["mapped_class_count"],
        "provenance_confidence": "STRONG_MATCH_BUT_NOT_PROVEN",
        "benchmark_leak_count": {
            "exact_historical_images": audit_summary["exact_leakage_to_plantdoc_test"],
            "verified_perceptual_historical_images": audit_summary["perceptual_leakage_to_plantdoc_test"],
        },
        "notes": [
            "Only the official non-augmented V1 archive was acquired.",
            "The verified 55,448-image, 39-class content exactly matches the notebook generator total and class count, but no notebook archive hash or persisted class_indices output proves byte-level identity.",
            "Raw files remain immutable and ignored by Git; all eligibility decisions are represented in manifests.",
            "No train/validation split, balancing, augmentation, or training was performed.",
        ],
    }


def output_hashes(paths: Sequence[Path]) -> dict[str, str]:
    return {path.as_posix(): file_sha256(path) for path in sorted(paths)}


def run(args: argparse.Namespace) -> dict:
    archive = inspect_archive(args.archive)
    mapping = load_explicit_mapping(args.mapping)
    historical, inventory, historical_paths = audit_extracted_source(args.raw_root, mapping)
    exact_groups, internal_exact = audit_internal_exact_duplicates(historical)
    dataset_v2 = prepare_comparison_records(load_csv(args.dataset_v2_manifest), "dataset_v2")
    locked_test = prepare_comparison_records(
        [row for row in load_csv(args.global_audit_manifest) if row["role"] == "locked_test"],
        "locked_test",
    )
    cross_exact = audit_cross_exact(historical, dataset_v2, locked_test)

    for row in historical:
        row["_origin"] = "historical"
    all_records = {
        row["record_id"]: row for row in [*historical, *dataset_v2, *locked_test]
    }
    if len(all_records) != len(historical) + len(dataset_v2) + len(locked_test):
        raise HistoricalAuditError("Record-ID collision across audit sources.")
    provider = AuditImageProvider(PROJECT_ROOT, historical_paths, args.plantdoc_repo)
    phash_cache = _load_phash_cache(args.phash_cache)
    internal_selected, internal_screened = select_refinement_candidates(
        internal_dhash_candidates(historical),
        "INTERNAL_HISTORICAL",
        provider,
        phash_cache,
    )
    v2_selected, v2_screened = select_refinement_candidates(
        cross_dhash_candidates(historical, dataset_v2),
        "HISTORICAL_TO_DATASET_V2",
        provider,
        phash_cache,
    )
    test_selected, test_screened = select_refinement_candidates(
        cross_dhash_candidates(historical, locked_test),
        "HISTORICAL_TO_PLANTDOC_TEST",
        provider,
        phash_cache,
    )
    _save_phash_cache(args.phash_cache, phash_cache)
    candidates = sorted(
        [*internal_selected, *v2_selected, *test_selected],
        key=lambda row: (
            row["risk_type"],
            row["first_record_id"],
            row["second_record_id"],
        ),
    )
    signals = calculate_orb_signals(all_records, candidates, provider)
    write_csv(args.local_pairs, signals, LOCAL_PAIR_FIELDS)
    perceptual = apply_perceptual_policy(historical, all_records, signals)
    perceptual["dhash_screened_pair_count"] = {
        "internal": internal_screened,
        "dataset_v2": v2_screened,
        "plantdoc_test": test_screened,
    }
    perceptual["orb_candidate_count"] = {
        "internal": len(internal_selected),
        "dataset_v2": len(v2_selected),
        "plantdoc_test": len(test_selected),
    }

    for row in historical:
        row.pop("_origin", None)
    clean = clean_historical_records(historical)
    coverage = coverage_counts(clean)
    projection = combined_projection(clean, dataset_v2)
    status_counts = Counter(row["candidate_status"] for row in historical)
    background = [row for row in historical if row["target_index"] == 4]
    background_valid = [row for row in background if row["integrity_status"] == "VALID"]
    background_exact_groups = {
        row["exact_duplicate_group_id"]
        for row in background_valid
        if row["exact_duplicate_group_id"]
    }
    audit_summary = {
        "schema_version": 1,
        "audit_date": RETRIEVAL_DATE,
        "verified_image_count": inventory["total_image_candidate_count"],
        "valid_image_count": inventory["valid_image_count"],
        "corrupt_image_count": inventory["corrupt_image_count"],
        "unsupported_file_count": inventory["unsupported_file_count"],
        "source_class_count": inventory["source_class_count"],
        "mapped_class_count": len([entry for entry in mapping.values() if entry["mapping_status"] == "MATCHED"]),
        "inventory": inventory,
        "exact_internal_duplicate_groups": internal_exact["group_count"],
        "exact_internal_duplicate_copies": internal_exact["copy_count_beyond_first"],
        "label_conflict_groups": internal_exact["label_conflict_group_count"],
        "exact_overlap_with_dataset_v2": cross_exact["dataset_v2_historical_image_count"],
        "exact_overlap_with_dataset_v2_pairs": cross_exact["dataset_v2_pair_count"],
        "exact_overlap_with_dataset_v2_by_relationship": cross_exact["dataset_v2_by_relationship"],
        "perceptual_overlap_with_dataset_v2": perceptual["dataset_v2_verified_historical_image_count"],
        "perceptual_overlap_with_dataset_v2_pairs": perceptual["dataset_v2_verified_pair_count"],
        "perceptual_overlap_with_dataset_v2_by_relationship": perceptual["dataset_v2_by_relationship"],
        "exact_leakage_to_plantdoc_test": cross_exact["plantdoc_test_historical_image_count"],
        "exact_leakage_to_plantdoc_test_pairs": cross_exact["plantdoc_test_pair_count"],
        "exact_leakage_to_plantdoc_test_by_relationship": cross_exact["plantdoc_test_by_relationship"],
        "perceptual_leakage_to_plantdoc_test": perceptual["plantdoc_test_verified_historical_image_count"],
        "perceptual_leakage_to_plantdoc_test_pairs": perceptual["plantdoc_test_verified_pair_count"],
        "possible_perceptual_leakage_to_plantdoc_test": perceptual["plantdoc_test_possible_historical_image_count"],
        "perceptual_leakage_to_plantdoc_test_by_relationship": perceptual["plantdoc_test_by_relationship"],
        "perceptual_audit": perceptual,
        "candidate_status_counts": dict(sorted(status_counts.items())),
        "clean_candidate_count": len(clean),
        "coverage_count": coverage["coverage_count"],
        "coverage_missing": coverage["coverage_missing"],
        "clean_candidates_by_target": coverage["by_target_class"],
        "background_audit": {
            "target_index": 4,
            "target_class": CLASS_NAMES[4],
            "source_label": "Background_without_leaves",
            "verified_image_count": len(background_valid),
            "formats": dict(sorted(Counter(row["format"] for row in background_valid).items())),
            "minimum_dimensions": {
                "width": min(int(row["width"]) for row in background_valid),
                "height": min(int(row["height"]) for row in background_valid),
            },
            "maximum_dimensions": {
                "width": max(int(row["width"]) for row in background_valid),
                "height": max(int(row["height"]) for row in background_valid),
            },
            "exact_duplicate_group_count": len(background_exact_groups),
            "semantic_mapping": "MATCHED_TO_DEPLOYED_INDEX_4",
            "provenance_scope": "Files are members of the same hash-verified official non-augmented archive; the source documentation does not establish universal OOD coverage.",
        },
        "combined_coverage_projection": projection,
        "provenance_confidence": "STRONG_MATCH_BUT_NOT_PROVEN",
        "provenance_evidence": {
            "notebook_total_images": 55448,
            "official_non_augmented_valid_images": inventory["valid_image_count"],
            "notebook_class_count": 39,
            "official_source_class_count": inventory["source_class_count"],
            "matching_preprocessing": "RGB, 224x224, float32 rescale 1./255",
            "limitation": "The notebook does not persist class_indices, a source archive hash, or immutable file identifiers.",
        },
        "invariants": {
            "raw_data_modified": False,
            "plantdoc_test_modified": False,
            "final_combined_manifest_created": False,
            "split_created": False,
            "augmentation_performed": False,
            "training_performed": False,
            "deployed_taxonomy_modified": False,
        },
    }
    metadata = build_source_metadata(archive, inventory, audit_summary)
    write_csv(args.manifest, historical, MANIFEST_FIELDS)
    write_csv(args.clean_manifest, clean, MANIFEST_FIELDS)
    write_json(args.exact_report, exact_groups)
    write_json(args.summary, audit_summary)
    write_json(args.source_metadata, metadata)
    return audit_summary


def parse_args() -> argparse.Namespace:
    datasets = PROJECT_ROOT / "training" / "datasets"
    parser = argparse.ArgumentParser(
        description="Audit the official historical Mendeley 39-class source without training."
    )
    parser.add_argument(
        "--archive",
        type=Path,
        default=datasets / "downloads" / OFFICIAL_FILE_NAME,
    )
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=datasets / "raw" / "historical-mendeley-39",
    )
    parser.add_argument(
        "--mapping",
        type=Path,
        default=datasets / "mappings" / "historical-mendeley-39.json",
    )
    parser.add_argument(
        "--dataset-v2-manifest",
        type=Path,
        default=datasets / "manifests" / "dataset-v2-clean-candidates.csv",
    )
    parser.add_argument(
        "--global-audit-manifest",
        type=Path,
        default=datasets / "manifests" / "dataset-v2-global-audit.csv",
    )
    parser.add_argument(
        "--plantdoc-repo",
        type=Path,
        default=PROJECT_ROOT / "evaluation" / "datasets" / "plantdoc",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=datasets / "manifests" / "historical-mendeley-39.csv",
    )
    parser.add_argument(
        "--clean-manifest",
        type=Path,
        default=datasets / "manifests" / "historical-mendeley-39-clean-candidates.csv",
    )
    parser.add_argument(
        "--exact-report",
        type=Path,
        default=datasets / "reports" / "historical-39class-exact-duplicate-groups.json",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=datasets / "reports" / "historical-39class-audit-summary.json",
    )
    parser.add_argument(
        "--source-metadata",
        type=Path,
        default=datasets / "sources" / "historical-mendeley-39.json",
    )
    parser.add_argument(
        "--local-pairs",
        type=Path,
        default=datasets / "local-audits" / "historical-39class-perceptual-signals.csv",
    )
    parser.add_argument(
        "--phash-cache",
        type=Path,
        default=datasets / "local-audits" / "historical-39class-external-phashes.csv",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = run(args)
    print(
        json.dumps(
            {
                "valid_images": summary["valid_image_count"],
                "clean_candidates": summary["clean_candidate_count"],
                "coverage": f"{summary['coverage_count']}/39",
                "provenance": summary["provenance_confidence"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
