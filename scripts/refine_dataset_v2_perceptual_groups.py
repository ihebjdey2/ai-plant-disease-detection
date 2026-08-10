from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
import subprocess
import sys
import time
from typing import Iterable, Sequence

import cv2
import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.build_dataset_v2_manifest import (  # noqa: E402
    ManifestBuildError,
    apply_exact_policy,
    initialize_cleaning,
    load_inventory,
    load_perceptual_members,
    write_csv,
)


class PerceptualResolutionError(RuntimeError):
    """Raised when a refinement input is incomplete or inconsistent."""


def filesystem_io_path(path: Path) -> Path:
    """Return a Windows extended-length path without changing its target."""
    if sys.platform == "win32" and not str(path).startswith("\\\\?\\"):
        return Path(f"\\\\?\\{path}")
    return path


PHASH_SIZE = 8
PHASH_IMAGE_SIZE = 32
SAME_TARGET_PHASH_MAX = 4
HIGH_RISK_PHASH_MAX = 16
MAX_ASPECT_LOG_DIFFERENCE = 0.03
ORB_FEATURES = 800
ORB_IMAGE_MAX_DIMENSION = 512
ORB_GOOD_MATCH_MINIMUM = 40
ORB_MATCH_RATIO_MINIMUM = 0.15
ORB_INLIER_RATIO_MINIMUM = 0.60
POSSIBLE_ORB_GOOD_MATCH_MINIMUM = 15
POSSIBLE_ORB_MATCH_RATIO_MINIMUM = 0.05
POSSIBLE_ORB_INLIER_RATIO_MINIMUM = 0.50

REFINED_MEMBER_FIELDS = [
    "record_id",
    "phash",
    "review_resolution",
    "refined_similarity_status",
    "refined_group_id",
    "refined_group_representative",
    "resolved_cleaning_status",
    "resolved_exclusion_reason",
]

REFINED_GROUP_FIELDS = [
    "refined_group_id",
    "representative_record_id",
    "member_count",
    "training_member_count",
    "locked_test_member_count",
    "target_classes",
    "datasets",
    "similarity_method",
    "minimum_dhash_distance",
    "maximum_dhash_distance",
    "minimum_phash_distance",
    "maximum_phash_distance",
    "minimum_orb_match_ratio",
    "maximum_orb_match_ratio",
    "minimum_orb_inlier_ratio",
    "maximum_orb_inlier_ratio",
    "risk_status",
]

HUMAN_REVIEW_FIELDS = [
    "review_id",
    "risk_type",
    "dataset_a",
    "record_a",
    "target_a",
    "dataset_b",
    "record_b",
    "target_b",
    "dhash_distance",
    "phash_distance",
    "geometry_similarity",
    "orb_good_matches",
    "orb_match_ratio",
    "orb_inlier_ratio",
    "recommended_action",
    "reason",
]

ORB_SIGNAL_FIELDS = [
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
]

DATASET_ROOTS = {
    "PlantDoc": Path("training/datasets/raw/plantdoc-train"),
    "Potato Leaf Disease Dataset": Path(
        "training/datasets/raw/potato-banu-deb-originals"
    ),
    "Seasonal Corn Leaf Disease Dataset": Path(
        "training/datasets/raw/seasonal_corn"
    ),
    "PLDD-UP": Path("training/datasets/raw/pldd_up"),
}


def compute_phash(image: Image.Image) -> str:
    """Return a deterministic 64-bit DCT perceptual hash."""

    grayscale = image.convert("L").resize(
        (PHASH_IMAGE_SIZE, PHASH_IMAGE_SIZE), Image.Resampling.LANCZOS
    )
    pixels = np.asarray(grayscale, dtype=np.float32)
    low_frequency = cv2.dct(pixels)[:PHASH_SIZE, :PHASH_SIZE]
    median = float(np.median(low_frequency))
    bits = (low_frequency > median).reshape(-1)
    value = sum(int(bit) << (63 - index) for index, bit in enumerate(bits))
    return f"{value:016x}"


def hash_distance(first: str, second: str) -> int:
    return (int(first, 16) ^ int(second, 16)).bit_count()


def aspect_log_difference(first: dict, second: dict) -> float:
    first_aspect = int(first["width"]) / int(first["height"])
    second_aspect = int(second["width"]) / int(second["height"])
    return abs(math.log(first_aspect / second_aspect))


def geometry_similarity(first: dict, second: dict) -> float:
    return math.exp(-aspect_log_difference(first, second))


def signal_is_verified(signal: dict[str, str]) -> bool:
    return (
        int(signal["orb_good_matches"]) >= ORB_GOOD_MATCH_MINIMUM
        and float(signal["orb_match_ratio"]) >= ORB_MATCH_RATIO_MINIMUM
        and float(signal["orb_inlier_ratio"]) >= ORB_INLIER_RATIO_MINIMUM
    )


def signal_is_possible(signal: dict[str, str]) -> bool:
    return (
        signal["risk_type"]
        in {"DIFFERENT_TARGET", "TRAIN_TO_LOCKED_BENCHMARK"}
        and not signal_is_verified(signal)
        and int(signal["orb_good_matches"]) >= POSSIBLE_ORB_GOOD_MATCH_MINIMUM
        and float(signal["orb_match_ratio"]) >= POSSIBLE_ORB_MATCH_RATIO_MINIMUM
        and float(signal["orb_inlier_ratio"])
        >= POSSIBLE_ORB_INLIER_RATIO_MINIMUM
    )


def pair_key(first: str, second: str) -> tuple[str, str]:
    return tuple(sorted((first, second)))


def pair_category(first: dict, second: dict) -> str:
    roles = {first["role"], second["role"]}
    if roles == {"training_candidate", "locked_test"}:
        return "TRAIN_TO_LOCKED_BENCHMARK"
    if (
        first["role"] == second["role"] == "training_candidate"
        and first["target_class"]
        and second["target_class"]
        and first["target_class"] != second["target_class"]
    ):
        return "DIFFERENT_TARGET"
    if first["target_class"] and first["target_class"] == second["target_class"]:
        return "SAME_TARGET"
    return "OTHER"


def build_anchor_groups(
    record_ids: Iterable[str], verified_pairs: Iterable[tuple[str, str]]
) -> dict[str, list[str]]:
    """Group only when a record directly matches the deterministic representative."""

    adjacency: dict[str, set[str]] = defaultdict(set)
    for first, second in verified_pairs:
        adjacency[first].add(second)
        adjacency[second].add(first)
    groups: dict[str, list[str]] = {}
    for record_id in sorted(set(record_ids) & set(adjacency)):
        matching_representatives = [
            representative
            for representative in groups
            if representative in adjacency[record_id]
        ]
        representative = (
            min(matching_representatives)
            if matching_representatives
            else record_id
        )
        groups.setdefault(representative, []).append(record_id)
    return {
        representative: sorted(set(members))
        for representative, members in sorted(groups.items())
        if len(set(members)) >= 2
    }


def refined_group_id(representative: str) -> str:
    digest = hashlib.sha256(representative.encode("utf-8")).hexdigest()
    return f"refined_{digest[:20]}"


class ImageProvider:
    def __init__(self, project_root: Path, plantdoc_repo: Path):
        self.project_root = project_root
        self.plantdoc_repo = plantdoc_repo

    def image_bytes(self, record: dict[str, str]) -> bytes:
        if record["role"] == "locked_test":
            specification = f"{record['source_version']}:{record['source_path']}"
            try:
                return subprocess.run(
                    [
                        "git",
                        "-C",
                        str(self.plantdoc_repo),
                        "show",
                        specification,
                    ],
                    capture_output=True,
                    check=True,
                ).stdout
            except subprocess.CalledProcessError as exc:
                raise PerceptualResolutionError(
                    f"Locked PlantDoc blob is unavailable: {specification}"
                ) from exc
        try:
            root = DATASET_ROOTS[record["dataset"]]
        except KeyError as exc:
            raise PerceptualResolutionError(
                f"No local root configured for dataset {record['dataset']}"
            ) from exc
        path = (self.project_root / root / Path(record["local_file"])).resolve()
        try:
            return filesystem_io_path(path).read_bytes()
        except OSError as exc:
            raise PerceptualResolutionError(
                f"Audited source image is unavailable: {record['local_file']}"
            ) from exc

    def phash(self, record: dict[str, str]) -> str:
        with Image.open(io.BytesIO(self.image_bytes(record))) as image:
            image.draft("L", (64, 64))
            return compute_phash(image)

    def orb_image(self, record: dict[str, str]) -> np.ndarray:
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


def load_step5c_records(
    inventory_path: Path, exact_path: Path, perceptual_members_path: Path
) -> list[dict[str, str]]:
    records = load_inventory(inventory_path)
    initialize_cleaning(records)
    apply_exact_policy(records, exact_path)
    load_perceptual_members(records, perceptual_members_path)
    return records


def load_old_groups(path: Path) -> dict[str, str]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return {
            row["record_id"]: row["near_duplicate_group_id"]
            for row in csv.DictReader(handle)
        }


def load_or_build_phashes(
    records: dict[str, dict[str, str]],
    screened_ids: Sequence[str],
    cache_path: Path,
    provider: ImageProvider,
    recompute: bool,
) -> dict[str, str]:
    expected = set(screened_ids)
    if cache_path.exists() and not recompute:
        with cache_path.open("r", encoding="utf-8", newline="") as handle:
            result = {row["record_id"]: row["phash"] for row in csv.DictReader(handle)}
        if set(result) == expected and all(len(value) == 16 for value in result.values()):
            return result
    started = time.monotonic()
    result = {}
    for index, record_id in enumerate(sorted(expected), 1):
        result[record_id] = provider.phash(records[record_id])
        if index % 1000 == 0:
            print(
                f"pHash {index}/{len(expected)} in {time.monotonic() - started:.1f}s",
                flush=True,
            )
    write_csv(
        cache_path,
        ({"record_id": record_id, "phash": result[record_id]} for record_id in sorted(result)),
        ["record_id", "phash"],
    )
    return result


def pair_records(
    row: dict[str, str],
    identity_index: dict[tuple[str, str, str], str],
    records: dict[str, dict[str, str]],
) -> tuple[str, str, dict[str, str], dict[str, str]]:
    try:
        first_id = identity_index[
            (row["first_dataset"], row["first_role"], row["first_path"])
        ]
        second_id = identity_index[
            (row["second_dataset"], row["second_role"], row["second_path"])
        ]
    except KeyError as exc:
        raise PerceptualResolutionError(
            f"dHash pair references an unknown record: {exc}"
        ) from exc
    return first_id, second_id, records[first_id], records[second_id]


def select_orb_candidates(
    full_pairs_path: Path,
    records: dict[str, dict[str, str]],
    phashes: dict[str, str],
) -> list[dict[str, str]]:
    identity_index = {
        (record["dataset"], record["role"], record["source_path"]): record_id
        for record_id, record in records.items()
    }
    candidates = []
    with full_pairs_path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            first_id, second_id, first, second = pair_records(
                row, identity_index, records
            )
            category = pair_category(first, second)
            phash_distance = hash_distance(phashes[first_id], phashes[second_id])
            aspect_difference = aspect_log_difference(first, second)
            selected = category == "TRAIN_TO_LOCKED_BENCHMARK" or (
                aspect_difference <= MAX_ASPECT_LOG_DIFFERENCE
                and (
                    (category == "DIFFERENT_TARGET" and phash_distance <= HIGH_RISK_PHASH_MAX)
                    or (category == "SAME_TARGET" and phash_distance <= SAME_TARGET_PHASH_MAX)
                )
            )
            if not selected:
                continue
            candidates.append(
                {
                    "first_record_id": first_id,
                    "second_record_id": second_id,
                    "risk_type": category,
                    "dhash_distance": row["hamming_distance"],
                    "phash_distance": str(phash_distance),
                    "aspect_log_difference": f"{aspect_difference:.8f}",
                }
            )
    return candidates


def load_or_build_orb_signals(
    records: dict[str, dict[str, str]],
    candidates: Sequence[dict[str, str]],
    cache_path: Path,
    provider: ImageProvider,
    recompute: bool,
) -> list[dict[str, str]]:
    expected_keys = {
        pair_key(row["first_record_id"], row["second_record_id"])
        for row in candidates
    }
    if cache_path.exists() and not recompute:
        with cache_path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        actual_keys = {
            pair_key(row["first_record_id"], row["second_record_id"])
            for row in rows
        }
        if actual_keys == expected_keys and len(rows) == len(candidates):
            return rows

    record_ids = sorted(
        {
            record_id
            for row in candidates
            for record_id in (row["first_record_id"], row["second_record_id"])
        }
    )
    orb = cv2.ORB_create(
        nfeatures=ORB_FEATURES, fastThreshold=12, edgeThreshold=15
    )
    descriptors = {}
    started = time.monotonic()
    for index, record_id in enumerate(record_ids, 1):
        keypoints, descriptor = orb.detectAndCompute(
            provider.orb_image(records[record_id]), None
        )
        points = (
            np.float32([keypoint.pt for keypoint in keypoints])
            if keypoints
            else np.empty((0, 2), dtype=np.float32)
        )
        descriptors[record_id] = (points, descriptor)
        if index % 500 == 0:
            print(
                f"ORB descriptors {index}/{len(record_ids)} in "
                f"{time.monotonic() - started:.1f}s",
                flush=True,
            )

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    rows = []
    for index, candidate in enumerate(candidates, 1):
        first_id = candidate["first_record_id"]
        second_id = candidate["second_record_id"]
        first_points, first_descriptor = descriptors[first_id]
        second_points, second_descriptor = descriptors[second_id]
        good_matches = []
        if (
            first_descriptor is not None
            and second_descriptor is not None
            and len(first_descriptor) >= 2
            and len(second_descriptor) >= 2
        ):
            for matches in matcher.knnMatch(
                first_descriptor, second_descriptor, k=2
            ):
                if len(matches) == 2 and matches[0].distance < 0.75 * matches[1].distance:
                    good_matches.append(matches[0])
        denominator = max(1, min(len(first_points), len(second_points)))
        match_ratio = len(good_matches) / denominator
        inlier_ratio = 0.0
        if len(good_matches) >= 8:
            source = np.float32(
                [first_points[match.queryIdx] for match in good_matches]
            ).reshape(-1, 1, 2)
            destination = np.float32(
                [second_points[match.trainIdx] for match in good_matches]
            ).reshape(-1, 1, 2)
            cv2.setRNGSeed(0)
            _, mask = cv2.findHomography(
                source, destination, cv2.RANSAC, 4.0
            )
            if mask is not None:
                inlier_ratio = float(mask.sum() / len(mask))
        rows.append(
            candidate
            | {
                "first_keypoints": str(len(first_points)),
                "second_keypoints": str(len(second_points)),
                "orb_good_matches": str(len(good_matches)),
                "orb_match_ratio": f"{match_ratio:.8f}",
                "orb_inlier_ratio": f"{inlier_ratio:.8f}",
            }
        )
        if index % 3000 == 0:
            print(
                f"ORB matching {index}/{len(candidates)} in "
                f"{time.monotonic() - started:.1f}s",
                flush=True,
            )
    write_csv(cache_path, rows, ORB_SIGNAL_FIELDS)
    return rows


def _entity_members(
    entity: str, groups: dict[str, list[str]], records: dict[str, dict]
) -> list[str]:
    if entity in groups:
        return [
            record_id
            for record_id in groups[entity]
            if records[record_id]["role"] == "training_candidate"
        ]
    return [entity] if records[entity]["role"] == "training_candidate" else []


def resolve_records(
    records: dict[str, dict[str, str]],
    phashes: dict[str, str],
    signals: Sequence[dict[str, str]],
) -> tuple[list[dict], list[dict], list[dict], dict]:
    verified = [signal for signal in signals if signal_is_verified(signal)]
    possible = [signal for signal in signals if signal_is_possible(signal)]
    verified_pairs = [
        (signal["first_record_id"], signal["second_record_id"])
        for signal in verified
    ]
    groups = build_anchor_groups(phashes, verified_pairs)
    record_representative = {
        record_id: representative
        for representative, members in groups.items()
        for record_id in members
    }
    signal_by_pair = {
        pair_key(signal["first_record_id"], signal["second_record_id"]): signal
        for signal in signals
    }

    benchmark_groups = set()
    conflict_groups = set()
    group_risk = {}
    for representative, members in groups.items():
        roles = {records[record_id]["role"] for record_id in members}
        targets = {
            records[record_id]["target_class"]
            for record_id in members
            if records[record_id]["role"] == "training_candidate"
            and records[record_id]["target_class"]
        }
        if "locked_test" in roles:
            benchmark_groups.add(representative)
            group_risk[representative] = "VERIFIED_BENCHMARK_LEAKAGE"
        elif len(targets) > 1:
            conflict_groups.add(representative)
            group_risk[representative] = "VERIFIED_LABEL_CONFLICT"
        else:
            group_risk[representative] = "VERIFIED_NEAR_DUPLICATE"

    possible_cases: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for signal in possible:
        first = signal["first_record_id"]
        second = signal["second_record_id"]
        first_entity = record_representative.get(first, first)
        second_entity = record_representative.get(second, second)
        if first_entity == second_entity:
            continue
        key = (*pair_key(first_entity, second_entity), signal["risk_type"])
        possible_cases[key].append(signal)

    benchmark_exclusions = {
        record_id
        for representative in benchmark_groups
        for record_id in _entity_members(representative, groups, records)
    }
    conflict_exclusions = {
        record_id
        for representative in conflict_groups
        for record_id in _entity_members(representative, groups, records)
    }
    uncertain_records: set[str] = set()
    uncertain_risk: dict[str, str] = {}
    for (first_entity, second_entity, risk_type), _ in possible_cases.items():
        for entity in (first_entity, second_entity):
            for record_id in _entity_members(entity, groups, records):
                uncertain_records.add(record_id)
                if risk_type == "TRAIN_TO_LOCKED_BENCHMARK":
                    uncertain_risk[record_id] = "PERCEPTUAL_BENCHMARK_MATCH"
                else:
                    uncertain_risk.setdefault(
                        record_id, "PERCEPTUAL_LABEL_CONFLICT"
                    )
    uncertain_records -= benchmark_exclusions | conflict_exclusions

    resolution_rows = []
    for record_id in sorted(phashes):
        record = records[record_id]
        representative = record_representative.get(record_id, "")
        group_id = refined_group_id(representative) if representative else ""
        base_status = record["cleaning_status"]
        base_reason = record["exclusion_reason"]
        if base_status == "EXCLUDE":
            resolved_status = base_status
            resolved_reason = base_reason
            resolution = "NOT_REQUIRED"
        elif record_id in benchmark_exclusions:
            resolved_status = "EXCLUDE"
            resolved_reason = "VERIFIED_PERCEPTUAL_BENCHMARK_LEAKAGE"
            resolution = "VERIFIED_BENCHMARK_LEAKAGE"
        elif record_id in conflict_exclusions:
            resolved_status = "EXCLUDE"
            resolved_reason = "VERIFIED_PERCEPTUAL_LABEL_CONFLICT"
            resolution = "VERIFIED_LABEL_CONFLICT"
        elif record_id in uncertain_records:
            resolved_status = "REVIEW"
            resolved_reason = uncertain_risk[record_id]
            resolution = "STILL_UNCERTAIN"
        elif base_status == "REVIEW":
            resolved_status = "INCLUDE"
            resolved_reason = ""
            resolution = "FALSE_POSITIVE_PERCEPTUAL_SCREEN"
        else:
            resolved_status = base_status
            resolved_reason = base_reason
            resolution = "NOT_REQUIRED"

        if representative:
            similarity = group_risk[representative]
        elif record_id in uncertain_records:
            similarity = "POSSIBLE_NEAR_DUPLICATE"
        else:
            similarity = "NOT_SIMILAR"
        resolution_rows.append(
            {
                "record_id": record_id,
                "phash": phashes[record_id],
                "review_resolution": resolution,
                "refined_similarity_status": similarity,
                "refined_group_id": group_id,
                "refined_group_representative": representative,
                "resolved_cleaning_status": resolved_status,
                "resolved_exclusion_reason": resolved_reason,
            }
        )

    group_rows = []
    for representative, members in sorted(groups.items()):
        evidence = [
            signal_by_pair[pair_key(representative, member)]
            for member in members
            if member != representative
        ]
        group_rows.append(
            {
                "refined_group_id": refined_group_id(representative),
                "representative_record_id": representative,
                "member_count": len(members),
                "training_member_count": sum(
                    records[record_id]["role"] == "training_candidate"
                    for record_id in members
                ),
                "locked_test_member_count": sum(
                    records[record_id]["role"] == "locked_test"
                    for record_id in members
                ),
                "target_classes": " | ".join(
                    sorted(
                        {
                            records[record_id]["target_class"]
                            for record_id in members
                            if records[record_id]["target_class"]
                        }
                    )
                ),
                "datasets": " | ".join(
                    sorted({records[record_id]["dataset"] for record_id in members})
                ),
                "similarity_method": (
                    "direct representative match: dHash<=4 + pHash + geometry + ORB/RANSAC"
                ),
                "minimum_dhash_distance": min(
                    int(signal["dhash_distance"]) for signal in evidence
                ),
                "maximum_dhash_distance": max(
                    int(signal["dhash_distance"]) for signal in evidence
                ),
                "minimum_phash_distance": min(
                    int(signal["phash_distance"]) for signal in evidence
                ),
                "maximum_phash_distance": max(
                    int(signal["phash_distance"]) for signal in evidence
                ),
                "minimum_orb_match_ratio": f"{min(float(signal['orb_match_ratio']) for signal in evidence):.8f}",
                "maximum_orb_match_ratio": f"{max(float(signal['orb_match_ratio']) for signal in evidence):.8f}",
                "minimum_orb_inlier_ratio": f"{min(float(signal['orb_inlier_ratio']) for signal in evidence):.8f}",
                "maximum_orb_inlier_ratio": f"{max(float(signal['orb_inlier_ratio']) for signal in evidence):.8f}",
                "risk_status": group_risk[representative],
            }
        )

    human_rows = []
    for key, case_signals in sorted(possible_cases.items()):
        selected = min(
            case_signals,
            key=lambda signal: (
                int(signal["phash_distance"]),
                -float(signal["orb_match_ratio"]),
                -int(signal["orb_good_matches"]),
                signal["first_record_id"],
                signal["second_record_id"],
            ),
        )
        first = records[selected["first_record_id"]]
        second = records[selected["second_record_id"]]
        review_digest = hashlib.sha256("\0".join(key).encode("utf-8")).hexdigest()
        human_rows.append(
            {
                "review_id": f"review_{review_digest[:20]}",
                "risk_type": (
                    "PERCEPTUAL_BENCHMARK_MATCH"
                    if key[2] == "TRAIN_TO_LOCKED_BENCHMARK"
                    else "PERCEPTUAL_LABEL_CONFLICT"
                ),
                "dataset_a": first["dataset"],
                "record_a": first["record_id"],
                "target_a": first["target_class"],
                "dataset_b": second["dataset"],
                "record_b": second["record_id"],
                "target_b": second["target_class"],
                "dhash_distance": selected["dhash_distance"],
                "phash_distance": selected["phash_distance"],
                "geometry_similarity": f"{geometry_similarity(first, second):.8f}",
                "orb_good_matches": selected["orb_good_matches"],
                "orb_match_ratio": selected["orb_match_ratio"],
                "orb_inlier_ratio": selected["orb_inlier_ratio"],
                "recommended_action": "MANUAL_IDENTITY_REVIEW",
                "reason": (
                    "Signals fall between the verified-match and false-positive bands."
                ),
            }
        )

    diagnostics = {
        "verified_relations": len(verified),
        "verified_relations_by_type": dict(
            sorted(Counter(signal["risk_type"] for signal in verified).items())
        ),
        "possible_relations": len(possible),
        "possible_review_cases": len(human_rows),
        "refined_group_count": len(groups),
        "largest_refined_group": max(map(len, groups.values()), default=1),
        "benchmark_group_count": len(benchmark_groups),
        "label_conflict_group_count": len(conflict_groups),
        "verified_benchmark_training_records": len(benchmark_exclusions),
        "verified_label_conflict_training_records": len(conflict_exclusions),
    }
    return resolution_rows, group_rows, human_rows, diagnostics


def histogram_quantiles(histogram: Counter[int]) -> dict[str, int]:
    total = sum(histogram.values())
    result = {}
    for quantile in (0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1):
        position = quantile * (total - 1)
        cumulative = 0
        for value in sorted(histogram):
            cumulative += histogram[value]
            if cumulative > position:
                result[str(quantile)] = value
                break
    return result


def analyze_full_pairs(
    full_pairs_path: Path,
    records: dict[str, dict[str, str]],
    phashes: dict[str, str],
    old_groups: dict[str, str],
) -> dict:
    identity_index = {
        (record["dataset"], record["role"], record["source_path"]): record_id
        for record_id, record in records.items()
    }
    largest_old_group = Counter(old_groups.values()).most_common(1)[0][0]
    giant_members = {
        record_id
        for record_id, group_id in old_groups.items()
        if group_id == largest_old_group
    }
    category_counts = Counter()
    category_phash: dict[str, Counter[int]] = defaultdict(Counter)
    giant_dhash = Counter()
    giant_phash = Counter()
    with full_pairs_path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            first_id, second_id, first, second = pair_records(
                row, identity_index, records
            )
            category = pair_category(first, second)
            phash_distance = hash_distance(phashes[first_id], phashes[second_id])
            category_counts[category] += 1
            category_phash[category][phash_distance] += 1
            if first_id in giant_members and second_id in giant_members:
                giant_dhash[int(row["hamming_distance"])] += 1
                giant_phash[phash_distance] += 1
    giant_records = [records[record_id] for record_id in giant_members]
    dimensions = Counter(
        (int(record["width"]), int(record["height"])) for record in giant_records
    )
    aspects = sorted(
        int(record["width"]) / int(record["height"]) for record in giant_records
    )
    return {
        "candidate_pair_counts": dict(sorted(category_counts.items())),
        "phash_quantiles_by_category": {
            category: histogram_quantiles(histogram)
            for category, histogram in sorted(category_phash.items())
        },
        "giant_component": {
            "step5c_group_id": largest_old_group,
            "record_count": len(giant_records),
            "class_counts": dict(
                sorted(Counter(record["target_class"] for record in giant_records).items())
            ),
            "dimension_counts": [
                {"width": width, "height": height, "count": count}
                for (width, height), count in dimensions.most_common()
            ],
            "aspect_ratio": {
                "minimum": min(aspects),
                "median": aspects[len(aspects) // 2],
                "maximum": max(aspects),
            },
            "dhash_distance_histogram": dict(sorted(giant_dhash.items())),
            "phash_distance_quantiles": histogram_quantiles(giant_phash),
            "dominant_cause": (
                "Coarse dHash collisions on nearly identical panoramic geometry, "
                "amplified by unrestricted connected-component chaining; verified "
                "near duplicates and burst/recompressed observations are also present."
            ),
        },
    }


def decision_summary(
    records: dict[str, dict[str, str]],
    resolution_rows: Sequence[dict],
    group_rows: Sequence[dict],
    old_groups: dict[str, str],
) -> dict:
    resolution = {row["record_id"]: row for row in resolution_rows}
    semantic_ids = [
        record_id
        for record_id, record in records.items()
        if record["role"] == "training_candidate"
        and record["mapping_status"] == "MATCHED"
    ]

    def final_status(record_id: str) -> str:
        if record_id in resolution:
            return resolution[record_id]["resolved_cleaning_status"]
        return records[record_id]["cleaning_status"]

    old_review = {
        record_id
        for record_id in semantic_ids
        if records[record_id]["cleaning_status"] == "REVIEW"
    }
    transitions = Counter(final_status(record_id) for record_id in old_review)
    by_source = {}
    for dataset in sorted({records[record_id]["dataset"] for record_id in semantic_ids}):
        ids = [
            record_id
            for record_id in semantic_ids
            if records[record_id]["dataset"] == dataset
        ]
        counts = Counter(final_status(record_id) for record_id in ids)
        by_source[dataset] = {
            "raw": len(ids),
            "include": counts["INCLUDE"],
            "review": counts["REVIEW"],
            "exclude": counts["EXCLUDE"],
        }
    group_by_record = {
        row["record_id"]: row["refined_group_id"]
        for row in resolution_rows
        if row["refined_group_id"]
    }
    by_target = {}
    for target in sorted({records[record_id]["target_class"] for record_id in semantic_ids}):
        ids = [
            record_id
            for record_id in semantic_ids
            if records[record_id]["target_class"] == target
        ]
        counts = Counter(final_status(record_id) for record_id in ids)
        by_target[target] = {
            "raw": len(ids),
            "include": counts["INCLUDE"],
            "review": counts["REVIEW"],
            "exclude": counts["EXCLUDE"],
            "refined_group_count": len(
                {group_by_record[record_id] for record_id in ids if record_id in group_by_record}
            ),
            "source_count": len({records[record_id]["dataset"] for record_id in ids}),
        }
    giant_group = Counter(old_groups.values()).most_common(1)[0][0]
    giant_ids = {
        record_id
        for record_id, group_id in old_groups.items()
        if group_id == giant_group
    }
    refined_groups = defaultdict(list)
    for record_id in giant_ids:
        group_id = group_by_record.get(record_id)
        if group_id:
            refined_groups[group_id].append(record_id)
    refined_groups = {
        group_id: members
        for group_id, members in refined_groups.items()
        if len(members) >= 2
    }
    giant_breakdown = {
        "refined_group_count": len(refined_groups),
        "grouped_record_count": sum(len(members) for members in refined_groups.values()),
        "ungrouped_record_count": len(giant_ids)
        - sum(len(members) for members in refined_groups.values()),
        "largest_refined_group": max(map(len, refined_groups.values()), default=1),
        "group_size_distribution": dict(
            sorted(Counter(len(members) for members in refined_groups.values()).items())
        ),
        "groups_by_target_signature": dict(
            sorted(
                Counter(
                    " | ".join(
                        sorted({records[record_id]["target_class"] for record_id in members})
                    )
                    for members in refined_groups.values()
                ).items()
            )
        ),
    }
    pldd_by_target = {}
    for target in (
        "Potato Early blight",
        "Potato Late blight",
        "Potato healthy",
    ):
        ids = [
            record_id
            for record_id in semantic_ids
            if records[record_id]["dataset"] == "PLDD-UP"
            and records[record_id]["target_class"] == target
        ]
        counts = Counter(final_status(record_id) for record_id in ids)
        pldd_by_target[target] = {
            "raw": len(ids),
            "include": counts["INCLUDE"],
            "review": counts["REVIEW"],
            "exclude": counts["EXCLUDE"],
            "refined_group_count": len(
                {
                    group_by_record[record_id]
                    for record_id in ids
                    if record_id in group_by_record
                }
            ),
        }
    final_counts = Counter(final_status(record_id) for record_id in semantic_ids)
    return {
        "initial_semantic_decisions": {
            "include": sum(records[record_id]["cleaning_status"] == "INCLUDE" for record_id in semantic_ids),
            "review": len(old_review),
            "exclude": sum(records[record_id]["cleaning_status"] == "EXCLUDE" for record_id in semantic_ids),
        },
        "review_transitions": {
            "review_to_include": transitions["INCLUDE"],
            "review_to_exclude": transitions["EXCLUDE"],
            "review_to_review": transitions["REVIEW"],
        },
        "final_semantic_decisions": {
            "include": final_counts["INCLUDE"],
            "review": final_counts["REVIEW"],
            "exclude": final_counts["EXCLUDE"],
        },
        "by_source": by_source,
        "by_target_class": by_target,
        "pldd_up": {
            "before": {
                "include": 12342,
                "review": 2918,
                "exclude": 259,
            },
            "after": by_source["PLDD-UP"],
            "by_target_class_after": pldd_by_target,
        },
        "giant_component_after_refinement": giant_breakdown,
        "refined_group_report_count": len(group_rows),
    }


def benchmark_decisions(
    signals: Sequence[dict[str, str]], records: dict[str, dict[str, str]]
) -> list[dict]:
    decisions = []
    for signal in signals:
        if signal["risk_type"] != "TRAIN_TO_LOCKED_BENCHMARK":
            continue
        first = records[signal["first_record_id"]]
        second = records[signal["second_record_id"]]
        train = first if first["role"] == "training_candidate" else second
        test = second if first["role"] == "training_candidate" else first
        if signal_is_verified(signal):
            decision = "LIKELY_SAME_OBSERVATION"
            action = "EXCLUDE_TRAINING_SIDE"
        elif signal_is_possible(signal):
            decision = "UNCERTAIN"
            action = "REVIEW_TRAINING_SIDE"
        else:
            decision = "CLEAR_FALSE_POSITIVE"
            action = "INCLUDE_TRAINING_SIDE"
        decisions.append(
            {
                "training_record_id": train["record_id"],
                "training_path": train["source_path"],
                "training_target": train["target_class"],
                "locked_test_record_id": test["record_id"],
                "locked_test_path": test["source_path"],
                "locked_test_target": test["target_class"],
                "dhash_distance": int(signal["dhash_distance"]),
                "phash_distance": int(signal["phash_distance"]),
                "orb_good_matches": int(signal["orb_good_matches"]),
                "orb_match_ratio": float(signal["orb_match_ratio"]),
                "orb_inlier_ratio": float(signal["orb_inlier_ratio"]),
                "decision": decision,
                "action": action,
            }
        )
    return sorted(decisions, key=lambda row: row["training_record_id"])


def parse_args() -> argparse.Namespace:
    datasets = PROJECT_ROOT / "training" / "datasets"
    parser = argparse.ArgumentParser(
        description="Refine Dataset V2 dHash candidates with pHash, geometry, and ORB."
    )
    parser.add_argument(
        "--inventory",
        type=Path,
        default=datasets / "manifests" / "dataset-v2-global-audit.csv",
    )
    parser.add_argument(
        "--exact-report",
        type=Path,
        default=datasets / "reports" / "exact-duplicate-groups.json",
    )
    parser.add_argument(
        "--step5c-members",
        type=Path,
        default=datasets / "reports" / "perceptual-group-members.csv",
    )
    parser.add_argument(
        "--full-pairs",
        type=Path,
        default=datasets / "local-audits" / "perceptual-duplicate-candidates-full.csv",
    )
    parser.add_argument(
        "--phash-cache",
        type=Path,
        default=datasets / "local-audits" / "phash-records-step5c1.csv",
    )
    parser.add_argument(
        "--orb-cache",
        type=Path,
        default=datasets / "local-audits" / "orb-candidate-signals-step5c1.csv",
    )
    parser.add_argument(
        "--plantdoc-repo",
        type=Path,
        default=PROJECT_ROOT / "evaluation" / "datasets" / "plantdoc",
    )
    parser.add_argument(
        "--refined-groups-output",
        type=Path,
        default=datasets / "reports" / "refined-near-duplicate-groups.csv",
    )
    parser.add_argument(
        "--refined-members-output",
        type=Path,
        default=datasets / "reports" / "refined-group-members.csv",
    )
    parser.add_argument(
        "--human-review-output",
        type=Path,
        default=datasets / "reports" / "perceptual-human-review.csv",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=datasets / "reports" / "perceptual-resolution-summary.json",
    )
    parser.add_argument("--recompute-signals", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        record_list = load_step5c_records(
            args.inventory, args.exact_report, args.step5c_members
        )
        records = {record["record_id"]: record for record in record_list}
        old_groups = load_old_groups(args.step5c_members)
        screened_ids = sorted(old_groups)
        provider = ImageProvider(PROJECT_ROOT, args.plantdoc_repo.resolve())
        phashes = load_or_build_phashes(
            records,
            screened_ids,
            args.phash_cache,
            provider,
            args.recompute_signals,
        )
        candidates = select_orb_candidates(args.full_pairs, records, phashes)
        signals = load_or_build_orb_signals(
            records,
            candidates,
            args.orb_cache,
            provider,
            args.recompute_signals,
        )
        resolution_rows, group_rows, human_rows, diagnostics = resolve_records(
            records, phashes, signals
        )
        pair_analysis = analyze_full_pairs(
            args.full_pairs, records, phashes, old_groups
        )
        decisions = decision_summary(
            records, resolution_rows, group_rows, old_groups
        )
        summary = {
            "schema_version": 1,
            "screening": {
                "dhash_role": "candidate_generation_only",
                "dhash_maximum_distance": 4,
                "screened_record_count": len(screened_ids),
                "raw_candidate_pair_count": sum(
                    pair_analysis["candidate_pair_counts"].values()
                ),
            },
            "verification_rule": {
                "phash": "64-bit DCT pHash",
                "same_target_phash_maximum": SAME_TARGET_PHASH_MAX,
                "high_risk_phash_maximum": HIGH_RISK_PHASH_MAX,
                "maximum_aspect_log_difference": MAX_ASPECT_LOG_DIFFERENCE,
                "orb_features": ORB_FEATURES,
                "orb_image_maximum_dimension": ORB_IMAGE_MAX_DIMENSION,
                "verified_minimum_good_matches": ORB_GOOD_MATCH_MINIMUM,
                "verified_minimum_match_ratio": ORB_MATCH_RATIO_MINIMUM,
                "verified_minimum_inlier_ratio": ORB_INLIER_RATIO_MINIMUM,
                "possible_minimum_good_matches": POSSIBLE_ORB_GOOD_MATCH_MINIMUM,
                "possible_minimum_match_ratio": POSSIBLE_ORB_MATCH_RATIO_MINIMUM,
                "possible_minimum_inlier_ratio": POSSIBLE_ORB_INLIER_RATIO_MINIMUM,
                "grouping": "deterministic direct-to-representative anchor grouping",
                "representative_rule": "lowest stable record_id that directly verifies against each member",
            },
            "diagnostics": diagnostics,
            "pair_analysis": pair_analysis,
            "decisions": decisions,
            "benchmark_cases": benchmark_decisions(signals, records),
            "safety": {
                "raw_data_modified": False,
                "split_created": False,
                "training_performed": False,
                "tensorflow_required": False,
            },
        }
        write_csv(
            args.refined_members_output,
            resolution_rows,
            REFINED_MEMBER_FIELDS,
        )
        write_csv(
            args.refined_groups_output,
            group_rows,
            REFINED_GROUP_FIELDS,
        )
        write_csv(
            args.human_review_output,
            human_rows,
            HUMAN_REVIEW_FIELDS,
        )
        args.summary_output.parent.mkdir(parents=True, exist_ok=True)
        args.summary_output.write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    except (
        OSError,
        csv.Error,
        json.JSONDecodeError,
        ManifestBuildError,
        PerceptualResolutionError,
    ) as exc:
        print(f"Perceptual resolution failed: {exc}", file=sys.stderr)
        return 1
    final = summary["decisions"]["final_semantic_decisions"]
    print(f"Strongly verified relations: {diagnostics['verified_relations']}")
    print(f"Refined groups: {diagnostics['refined_group_count']}")
    print(f"Human review cases: {diagnostics['possible_review_cases']}")
    print(
        f"Final semantic decisions: INCLUDE={final['include']} "
        f"REVIEW={final['review']} EXCLUDE={final['exclude']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
