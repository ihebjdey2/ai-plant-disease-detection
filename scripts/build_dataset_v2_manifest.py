from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
import sys
from typing import Iterable, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.taxonomy import CLASS_NAMES  # noqa: E402


class ManifestBuildError(RuntimeError):
    """Raised when an audited input or cleaning invariant is invalid."""


MASTER_FIELDS = [
    "record_id",
    "dataset",
    "source_version",
    "role",
    "source_path",
    "local_file",
    "source_label",
    "mapping_status",
    "target_class",
    "sha256",
    "dhash",
    "phash",
    "format",
    "mode",
    "width",
    "height",
    "bytes",
    "original_or_augmented",
    "candidate_status",
    "cleaning_status",
    "exclusion_reason",
    "canonical_record_id",
    "exact_duplicate_group_id",
    "near_duplicate_group_id",
    "benchmark_leakage",
    "label_conflict",
    "manual_review_required",
    "review_resolution",
    "refined_similarity_status",
    "refined_group_id",
    "refined_group_representative",
]

PERCEPTUAL_MEMBER_FIELDS = [
    "record_id",
    "near_duplicate_group_id",
    "direct_perceptual_label_conflict",
    "direct_benchmark_match",
    "minimum_direct_distance",
]

PERCEPTUAL_SUMMARY_FIELDS = [
    "near_duplicate_group_id",
    "risk_category",
    "member_count",
    "training_member_count",
    "locked_test_member_count",
    "dataset_count",
    "datasets",
    "target_class_count",
    "target_classes",
    "edge_count",
    "minimum_hamming_distance",
    "same_target_same_source_pairs",
    "same_target_cross_source_pairs",
    "different_target_pairs",
    "train_to_locked_benchmark_pairs",
    "other_pairs",
]

REVIEW_FIELDS = [
    "near_duplicate_group_id",
    "review_category",
    "pair_count_in_group",
    "group_member_count",
    "first_record_id",
    "first_dataset",
    "first_role",
    "first_path",
    "first_label",
    "first_target_class",
    "first_width",
    "first_height",
    "second_record_id",
    "second_dataset",
    "second_role",
    "second_path",
    "second_label",
    "second_target_class",
    "second_width",
    "second_height",
    "hamming_distance",
]

EXACT_CONFLICT_FIELDS = [
    "exact_duplicate_group_id",
    "sha256",
    "record_id",
    "dataset",
    "role",
    "source_path",
    "source_label",
    "mapping_status",
    "target_class",
    "benchmark_leakage",
    "cleaning_status",
    "exclusion_reason",
]


def stable_record_id(record: dict[str, str]) -> str:
    identity = "\0".join(
        (
            record["dataset"],
            record["source_version"],
            record["role"],
            record["source_path"],
        )
    )
    return f"rec_{hashlib.sha256(identity.encode('utf-8')).hexdigest()[:20]}"


def bool_text(value: bool) -> str:
    return "true" if value else "false"


def write_csv(path: Path, rows: Iterable[dict], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def load_inventory(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        records = [dict(row) for row in csv.DictReader(handle)]
    if not records:
        raise ManifestBuildError(f"Audited inventory is empty: {path}")
    identities: set[tuple[str, str, str]] = set()
    record_ids: set[str] = set()
    for record in records:
        identity = (record["dataset"], record["role"], record["source_path"])
        if identity in identities:
            raise ManifestBuildError(f"Duplicate audited identity: {identity}")
        identities.add(identity)
        record["record_id"] = stable_record_id(record)
        if record["record_id"] in record_ids:
            raise ManifestBuildError(f"Stable record ID collision: {record['record_id']}")
        record_ids.add(record["record_id"])
    return sorted(records, key=lambda row: row["record_id"])


def initialize_cleaning(records: Sequence[dict[str, str]]) -> None:
    for record in records:
        record.update(
            {
                "cleaning_status": "INCLUDE",
                "exclusion_reason": "",
                "canonical_record_id": "",
                "exact_duplicate_group_id": "",
                "near_duplicate_group_id": "",
                "benchmark_leakage": "false",
                "label_conflict": "false",
                "manual_review_required": "false",
                "phash": "",
                "review_resolution": "NOT_REQUIRED",
                "refined_similarity_status": "NOT_SCREENED",
                "refined_group_id": "",
                "refined_group_representative": "",
            }
        )
        if record["role"] == "locked_test":
            exclude(record, "LOCKED_BENCHMARK")
        elif record["mapping_status"] == "AMBIGUOUS":
            exclude(record, "UNRESOLVED_MAPPING")
        elif record["mapping_status"] == "NOT_SUPPORTED":
            exclude(record, "UNSUPPORTED_CLASS")
        elif record["mapping_status"] != "MATCHED":
            exclude(record, "NOT_SEMANTIC_MATCH")


def exclude(record: dict[str, str], reason: str) -> None:
    record["cleaning_status"] = "EXCLUDE"
    record["exclusion_reason"] = reason
    record["manual_review_required"] = "false"


def review(record: dict[str, str], reason: str) -> None:
    if record["cleaning_status"] != "INCLUDE":
        return
    record["cleaning_status"] = "REVIEW"
    record["exclusion_reason"] = reason
    record["manual_review_required"] = "true"


def include(record: dict[str, str]) -> None:
    if record["cleaning_status"] == "EXCLUDE":
        return
    record["cleaning_status"] = "INCLUDE"
    record["exclusion_reason"] = ""
    record["manual_review_required"] = "false"


def apply_perceptual_resolution(
    records: Sequence[dict[str, str]], resolution_path: Path
) -> dict[str, int]:
    allowed_resolutions = {
        "NOT_REQUIRED",
        "FALSE_POSITIVE_PERCEPTUAL_SCREEN",
        "VERIFIED_NEAR_DUPLICATE",
        "VERIFIED_LABEL_CONFLICT",
        "VERIFIED_BENCHMARK_LEAKAGE",
        "STILL_UNCERTAIN",
        "CONSERVATIVE_FINAL_EXCLUSION",
    }
    allowed_similarity = {
        "NOT_SCREENED",
        "NOT_SIMILAR",
        "VERIFIED_NEAR_DUPLICATE",
        "POSSIBLE_NEAR_DUPLICATE",
        "VERIFIED_LABEL_CONFLICT",
        "VERIFIED_BENCHMARK_LEAKAGE",
    }
    by_id = {record["record_id"]: record for record in records}
    seen: set[str] = set()
    counts = Counter()
    with resolution_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        record_id = row["record_id"]
        if record_id not in by_id:
            raise ManifestBuildError(
                f"Perceptual resolution references unknown record: {record_id}"
            )
        if record_id in seen:
            raise ManifestBuildError(f"Duplicate refined record ID: {record_id}")
        seen.add(record_id)
        resolution = row["review_resolution"]
        similarity = row["refined_similarity_status"]
        if resolution not in allowed_resolutions:
            raise ManifestBuildError(f"Invalid review resolution: {resolution}")
        if similarity not in allowed_similarity:
            raise ManifestBuildError(f"Invalid refined similarity status: {similarity}")
        record = by_id[record_id]
        record["phash"] = row["phash"]
        record["review_resolution"] = resolution
        record["refined_similarity_status"] = similarity
        record["refined_group_id"] = row["refined_group_id"]
        record["refined_group_representative"] = row[
            "refined_group_representative"
        ]
        if record["cleaning_status"] == "EXCLUDE":
            counts["protected_existing_exclusion"] += 1
            continue
        resolved_status = row["resolved_cleaning_status"]
        resolved_reason = row["resolved_exclusion_reason"]
        if resolved_status == "INCLUDE":
            include(record)
        elif resolved_status == "REVIEW":
            review(record, resolved_reason)
        elif resolved_status == "EXCLUDE":
            exclude(record, resolved_reason)
        else:
            raise ManifestBuildError(
                f"Invalid resolved cleaning status: {resolved_status}"
            )
        counts[f"resolved_{resolved_status.lower()}"] += 1
    counts["resolution_rows"] = len(rows)
    return dict(counts)


def finalize_unresolved_perceptual_reviews(
    records: Sequence[dict[str, str]],
) -> dict[str, int]:
    """Exclude only Step 5C.1 reviews that remained perceptually unresolved."""
    finalized = 0
    for record in records:
        if record["cleaning_status"] != "REVIEW":
            continue
        if record["review_resolution"] != "STILL_UNCERTAIN":
            continue
        exclude(record, "UNRESOLVED_PERCEPTUAL_IDENTITY")
        record["review_resolution"] = "CONSERVATIVE_FINAL_EXCLUSION"
        finalized += 1
    return {"conservative_final_exclusions": finalized}


def apply_exact_policy(
    records: Sequence[dict[str, str]], exact_report_path: Path
) -> dict[str, int]:
    groups = json.loads(exact_report_path.read_text(encoding="utf-8"))
    by_sha: dict[str, list[dict[str, str]]] = defaultdict(list)
    for record in records:
        by_sha[record["sha256"]].append(record)
    duplicate_hashes = {digest for digest, members in by_sha.items() if len(members) > 1}
    reported_hashes = {group["sha256"] for group in groups}
    if duplicate_hashes != reported_hashes:
        raise ManifestBuildError("Exact duplicate report does not match the audited inventory.")

    stats = Counter()
    for group in sorted(groups, key=lambda item: item["sha256"]):
        members = sorted(
            by_sha[group["sha256"]],
            key=lambda row: (
                row["dataset"],
                row["source_version"],
                row["source_path"],
                row["record_id"],
            ),
        )
        if len(members) != group["member_count"]:
            raise ManifestBuildError(
                f"Exact group count mismatch for SHA-256 {group['sha256']}"
            )
        group_id = f"exact_{group['sha256'][:20]}"
        for member in members:
            member["exact_duplicate_group_id"] = group_id
            member["label_conflict"] = bool_text(bool(group["label_conflict"]))

        stats["groups"] += 1
        stats["copies_beyond_canonical"] += len(members) - 1
        if group["label_conflict"]:
            stats["label_conflict_groups"] += 1
        if group["touches_locked_test"]:
            stats["benchmark_groups"] += 1
            for member in members:
                if member["role"] == "training_candidate":
                    member["benchmark_leakage"] = "true"
                    exclude(member, "EXACT_BENCHMARK_LEAKAGE")
                    stats["benchmark_training_exclusions"] += 1
            continue
        if group["label_conflict"]:
            for member in members:
                if member["role"] == "training_candidate":
                    exclude(member, "EXACT_LABEL_CONFLICT")
                    stats["label_conflict_record_exclusions"] += 1
            continue

        eligible = [
            member
            for member in members
            if member["role"] == "training_candidate"
            and member["mapping_status"] == "MATCHED"
        ]
        if not eligible:
            continue
        canonical = eligible[0]
        for member in eligible:
            member["canonical_record_id"] = canonical["record_id"]
        for duplicate in eligible[1:]:
            exclude(duplicate, "EXACT_DUPLICATE_COPY")
            stats["exact_copy_exclusions"] += 1
    return dict(stats)


class UnionFind:
    def __init__(self, size: int):
        self.parent = list(range(size))
        self.size = [1] * size

    def find(self, item: int) -> int:
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, first: int, second: int) -> None:
        first_root = self.find(first)
        second_root = self.find(second)
        if first_root == second_root:
            return
        if self.size[first_root] < self.size[second_root]:
            first_root, second_root = second_root, first_root
        self.parent[second_root] = first_root
        self.size[first_root] += self.size[second_root]


def pair_identity(row: dict[str, str], prefix: str) -> tuple[str, str, str]:
    return (
        row[f"{prefix}_dataset"],
        row[f"{prefix}_role"],
        row[f"{prefix}_path"],
    )


def pair_category(first: dict[str, str], second: dict[str, str]) -> str:
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
        if first["dataset"] == second["dataset"]:
            return "SAME_TARGET_SAME_SOURCE"
        return "SAME_TARGET_CROSS_SOURCE"
    return "OTHER"


def _pair_records(
    row: dict[str, str], identity_index: dict[tuple[str, str, str], int], records: Sequence[dict]
) -> tuple[int, int, dict, dict]:
    try:
        first_index = identity_index[pair_identity(row, "first")]
        second_index = identity_index[pair_identity(row, "second")]
    except KeyError as exc:
        raise ManifestBuildError(f"Perceptual pair references an unknown record: {exc}") from exc
    return first_index, second_index, records[first_index], records[second_index]


def build_perceptual_reports(
    records: Sequence[dict[str, str]],
    full_report_path: Path,
    members_path: Path,
    summary_path: Path,
    review_path: Path,
) -> dict[str, int]:
    identity_index = {
        (record["dataset"], record["role"], record["source_path"]): index
        for index, record in enumerate(records)
    }
    union_find = UnionFind(len(records))
    touched: set[int] = set()
    with full_report_path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            first_index, second_index, _, _ = _pair_records(
                row, identity_index, records
            )
            union_find.union(first_index, second_index)
            touched.update((first_index, second_index))

    components: dict[int, list[int]] = defaultdict(list)
    for index in sorted(touched):
        components[union_find.find(index)].append(index)

    group_ids: dict[int, str] = {}
    record_groups: dict[int, str] = {}
    for root, member_indices in components.items():
        member_ids = sorted(records[index]["record_id"] for index in member_indices)
        digest = hashlib.sha256("\n".join(member_ids).encode("utf-8")).hexdigest()
        group_ids[root] = f"near_{digest[:20]}"
        for index in member_indices:
            record_groups[index] = group_ids[root]

    aggregate: dict[str, dict] = {}
    for root, indices in components.items():
        group_id = group_ids[root]
        members = [records[index] for index in indices]
        aggregate[group_id] = {
            "member_count": len(members),
            "training_member_count": sum(
                member["role"] == "training_candidate" for member in members
            ),
            "locked_test_member_count": sum(
                member["role"] == "locked_test" for member in members
            ),
            "datasets": sorted({member["dataset"] for member in members}),
            "target_classes": sorted(
                {member["target_class"] for member in members if member["target_class"]}
            ),
            "counts": Counter(),
            "minimum_hamming_distance": 64,
        }

    direct_label_conflict: set[int] = set()
    direct_benchmark: set[int] = set()
    direct_minimum: dict[int, int] = {}
    representatives: dict[tuple[str, str], tuple[tuple, dict]] = {}

    with full_report_path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            first_index, second_index, first, second = _pair_records(
                row, identity_index, records
            )
            group_id = record_groups[first_index]
            if group_id != record_groups[second_index]:
                raise ManifestBuildError("Perceptual component assignment is inconsistent.")
            category = pair_category(first, second)
            distance = int(row["hamming_distance"])
            info = aggregate[group_id]
            info["counts"][category] += 1
            info["minimum_hamming_distance"] = min(
                info["minimum_hamming_distance"], distance
            )
            for index in (first_index, second_index):
                direct_minimum[index] = min(direct_minimum.get(index, 64), distance)

            eligible_review = False
            if category == "TRAIN_TO_LOCKED_BENCHMARK":
                for index, record in ((first_index, first), (second_index, second)):
                    if record["role"] == "training_candidate":
                        direct_benchmark.add(index)
                        eligible_review |= record["cleaning_status"] == "INCLUDE"
            elif category == "DIFFERENT_TARGET":
                direct_label_conflict.update((first_index, second_index))
                eligible_review = any(
                    record["cleaning_status"] == "INCLUDE" for record in (first, second)
                )

            if category not in {"TRAIN_TO_LOCKED_BENCHMARK", "DIFFERENT_TARGET"}:
                continue
            if not eligible_review:
                continue
            key = (group_id, category)
            rank = (
                distance,
                0 if first["dataset"] != second["dataset"] else 1,
                first["record_id"],
                second["record_id"],
            )
            candidate = {
                "near_duplicate_group_id": group_id,
                "review_category": category,
                "first_record_id": first["record_id"],
                "first_dataset": first["dataset"],
                "first_role": first["role"],
                "first_path": first["source_path"],
                "first_label": first["source_label"],
                "first_target_class": first["target_class"],
                "first_width": first["width"],
                "first_height": first["height"],
                "second_record_id": second["record_id"],
                "second_dataset": second["dataset"],
                "second_role": second["role"],
                "second_path": second["source_path"],
                "second_label": second["source_label"],
                "second_target_class": second["target_class"],
                "second_width": second["width"],
                "second_height": second["height"],
                "hamming_distance": distance,
            }
            if key not in representatives or rank < representatives[key][0]:
                representatives[key] = (rank, candidate)

    member_rows = []
    for index in sorted(touched, key=lambda item: records[item]["record_id"]):
        member_rows.append(
            {
                "record_id": records[index]["record_id"],
                "near_duplicate_group_id": record_groups[index],
                "direct_perceptual_label_conflict": bool_text(
                    index in direct_label_conflict
                ),
                "direct_benchmark_match": bool_text(index in direct_benchmark),
                "minimum_direct_distance": direct_minimum[index],
            }
        )
    write_csv(members_path, member_rows, PERCEPTUAL_MEMBER_FIELDS)

    summary_rows = []
    for group_id, info in sorted(aggregate.items()):
        counts = info["counts"]
        if counts["TRAIN_TO_LOCKED_BENCHMARK"]:
            risk = "TRAIN_TO_LOCKED_BENCHMARK"
        elif counts["DIFFERENT_TARGET"]:
            risk = "DIFFERENT_TARGET"
        elif counts["SAME_TARGET_CROSS_SOURCE"]:
            risk = "LOW_SAME_TARGET_CROSS_SOURCE"
        else:
            risk = "LOW_SAME_TARGET_SAME_SOURCE"
        summary_rows.append(
            {
                "near_duplicate_group_id": group_id,
                "risk_category": risk,
                "member_count": info["member_count"],
                "training_member_count": info["training_member_count"],
                "locked_test_member_count": info["locked_test_member_count"],
                "dataset_count": len(info["datasets"]),
                "datasets": " | ".join(info["datasets"]),
                "target_class_count": len(info["target_classes"]),
                "target_classes": " | ".join(info["target_classes"]),
                "edge_count": sum(counts.values()),
                "minimum_hamming_distance": info["minimum_hamming_distance"],
                "same_target_same_source_pairs": counts["SAME_TARGET_SAME_SOURCE"],
                "same_target_cross_source_pairs": counts["SAME_TARGET_CROSS_SOURCE"],
                "different_target_pairs": counts["DIFFERENT_TARGET"],
                "train_to_locked_benchmark_pairs": counts[
                    "TRAIN_TO_LOCKED_BENCHMARK"
                ],
                "other_pairs": counts["OTHER"],
            }
        )
    write_csv(summary_path, summary_rows, PERCEPTUAL_SUMMARY_FIELDS)

    review_rows = []
    for (group_id, category), (_, row) in sorted(
        representatives.items(),
        key=lambda item: (
            0 if item[0][1] == "TRAIN_TO_LOCKED_BENCHMARK" else 1,
            item[0][0],
        ),
    ):
        info = aggregate[group_id]
        row["pair_count_in_group"] = info["counts"][category]
        row["group_member_count"] = info["member_count"]
        review_rows.append(row)
    write_csv(review_path, review_rows, REVIEW_FIELDS)

    return {
        "pair_count": sum(sum(info["counts"].values()) for info in aggregate.values()),
        "group_count": len(aggregate),
        "grouped_record_count": len(touched),
        "different_target_pair_count": sum(
            info["counts"]["DIFFERENT_TARGET"] for info in aggregate.values()
        ),
        "different_target_group_count": sum(
            bool(info["counts"]["DIFFERENT_TARGET"]) for info in aggregate.values()
        ),
        "benchmark_pair_count": sum(
            info["counts"]["TRAIN_TO_LOCKED_BENCHMARK"]
            for info in aggregate.values()
        ),
        "benchmark_group_count": sum(
            bool(info["counts"]["TRAIN_TO_LOCKED_BENCHMARK"])
            for info in aggregate.values()
        ),
        "review_queue_rows": len(review_rows),
    }


def load_perceptual_members(
    records: Sequence[dict[str, str]], members_path: Path
) -> dict[str, int]:
    by_id = {record["record_id"]: record for record in records}
    seen: set[str] = set()
    with members_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        record_id = row["record_id"]
        if record_id not in by_id:
            raise ManifestBuildError(f"Perceptual report references unknown ID: {record_id}")
        if record_id in seen:
            raise ManifestBuildError(f"Duplicate perceptual member ID: {record_id}")
        seen.add(record_id)
        record = by_id[record_id]
        record["near_duplicate_group_id"] = row["near_duplicate_group_id"]
        if row["direct_benchmark_match"] == "true":
            review(record, "PERCEPTUAL_BENCHMARK_MATCH")
        elif row["direct_perceptual_label_conflict"] == "true":
            review(record, "PERCEPTUAL_LABEL_CONFLICT")
    return {
        "grouped_record_count": len(rows),
        "group_count": len({row["near_duplicate_group_id"] for row in rows}),
    }


def validate_manifests(
    master: Sequence[dict[str, str]],
    clean: Sequence[dict[str, str]],
    require_final: bool = False,
) -> None:
    allowed_statuses = {"INCLUDE", "EXCLUDE", "REVIEW"}
    allowed_reasons = {
        "",
        "NOT_SEMANTIC_MATCH",
        "LOCKED_BENCHMARK",
        "EXACT_BENCHMARK_LEAKAGE",
        "EXACT_DUPLICATE_COPY",
        "EXACT_LABEL_CONFLICT",
        "PERCEPTUAL_LABEL_CONFLICT",
        "PERCEPTUAL_BENCHMARK_MATCH",
        "UNRESOLVED_MAPPING",
        "UNSUPPORTED_CLASS",
        "VERIFIED_PERCEPTUAL_LABEL_CONFLICT",
        "VERIFIED_PERCEPTUAL_BENCHMARK_LEAKAGE",
        "UNRESOLVED_PERCEPTUAL_IDENTITY",
    }
    for record in master:
        if record["cleaning_status"] not in allowed_statuses:
            raise ManifestBuildError(f"Invalid cleaning status: {record['cleaning_status']}")
        if record["exclusion_reason"] not in allowed_reasons:
            raise ManifestBuildError(f"Invalid cleaning reason: {record['exclusion_reason']}")
        if record["target_class"] and record["target_class"] not in CLASS_NAMES:
            raise ManifestBuildError(f"Unknown deployed target: {record['target_class']}")
        if record["cleaning_status"] == "INCLUDE":
            if record["role"] == "locked_test":
                raise ManifestBuildError("A locked benchmark record was included.")
            if record["benchmark_leakage"] == "true":
                raise ManifestBuildError("An exact benchmark leak was included.")
            if record["mapping_status"] != "MATCHED":
                raise ManifestBuildError("An unresolved semantic mapping was included.")
            if record["label_conflict"] == "true":
                raise ManifestBuildError("An exact label conflict was included.")
            if record["refined_similarity_status"] in {
                "VERIFIED_LABEL_CONFLICT",
                "VERIFIED_BENCHMARK_LEAKAGE",
            }:
                raise ManifestBuildError(
                    "A verified perceptual conflict or benchmark leak was included."
                )
            if record["review_resolution"] in {
                "STILL_UNCERTAIN",
                "CONSERVATIVE_FINAL_EXCLUSION",
            }:
                raise ManifestBuildError(
                    "An unresolved perceptual identity record was included."
                )
        if record["review_resolution"] == "CONSERVATIVE_FINAL_EXCLUSION" and (
            record["cleaning_status"] != "EXCLUDE"
            or record["exclusion_reason"] != "UNRESOLVED_PERCEPTUAL_IDENTITY"
        ):
            raise ManifestBuildError(
                "A conservative final exclusion has inconsistent decision metadata."
            )
    expected_clean = [row for row in master if row["cleaning_status"] == "INCLUDE"]
    if [row["record_id"] for row in clean] != [row["record_id"] for row in expected_clean]:
        raise ManifestBuildError("Clean manifest is not the exact INCLUDE subset.")
    hashes = [row["sha256"] for row in clean]
    if len(hashes) != len(set(hashes)):
        raise ManifestBuildError("Clean manifest contains duplicate SHA-256 contents.")
    if any(row["manual_review_required"] == "true" for row in clean):
        raise ManifestBuildError("Clean manifest contains unresolved review records.")
    if require_final and any(row["cleaning_status"] == "REVIEW" for row in master):
        raise ManifestBuildError("Final Dataset V2 manifest contains REVIEW records.")
    if require_final and any(
        row["review_resolution"] == "STILL_UNCERTAIN" for row in master
    ):
        raise ManifestBuildError(
            "Final Dataset V2 manifest contains unresolved perceptual identities."
        )


def write_exact_conflict_report(
    records: Sequence[dict[str, str]], path: Path
) -> None:
    rows = [
        row
        for row in records
        if row["label_conflict"] == "true" and row["exact_duplicate_group_id"]
    ]
    write_csv(
        path,
        sorted(
            rows,
            key=lambda row: (row["exact_duplicate_group_id"], row["record_id"]),
        ),
        EXACT_CONFLICT_FIELDS,
    )


def summarize(
    master: Sequence[dict[str, str]], exact_stats: dict[str, int], perceptual: dict[str, int]
) -> dict:
    training_matched = [
        row
        for row in master
        if row["role"] == "training_candidate" and row["mapping_status"] == "MATCHED"
    ]

    def decision_counts(rows: Sequence[dict[str, str]]) -> dict[str, int]:
        return dict(sorted(Counter(row["cleaning_status"] for row in rows).items()))

    by_source = {}
    for dataset in sorted({row["dataset"] for row in training_matched}):
        rows = [row for row in training_matched if row["dataset"] == dataset]
        counts = decision_counts(rows)
        by_source[dataset] = {
            "semantic_candidates": len(rows),
            "included": counts.get("INCLUDE", 0),
            "excluded": counts.get("EXCLUDE", 0),
            "review": counts.get("REVIEW", 0),
            "unique_contents": len({row["sha256"] for row in rows}),
        }

    by_target = {}
    for target in sorted({row["target_class"] for row in training_matched}):
        rows = [row for row in training_matched if row["target_class"] == target]
        counts = decision_counts(rows)
        by_target[target] = {
            "raw_candidates": len(rows),
            "included": counts.get("INCLUDE", 0),
            "excluded": counts.get("EXCLUDE", 0),
            "review": counts.get("REVIEW", 0),
            "source_count": len({row["dataset"] for row in rows}),
            "near_duplicate_group_count": len(
                {row["near_duplicate_group_id"] for row in rows if row["near_duplicate_group_id"]}
            ),
            "refined_group_count": len(
                {row["refined_group_id"] for row in rows if row["refined_group_id"]}
            ),
        }

    semantic_decisions = decision_counts(training_matched)
    full_decisions = decision_counts(master)
    represented_classes = sorted(
        target for target, counts in by_target.items() if counts["included"] > 0
    )
    missing_classes = sorted(set(CLASS_NAMES) - set(represented_classes))
    counts_by_source = {
        source: {
            "raw": counts["semantic_candidates"],
            "include": counts["included"],
            "exclude": counts["excluded"],
            "review": counts["review"],
        }
        for source, counts in by_source.items()
    }
    counts_by_target = {
        target: {
            "raw": counts["raw_candidates"],
            "include": counts["included"],
            "exclude": counts["excluded"],
            "review": counts["review"],
            "source_count": counts["source_count"],
            "refined_group_count": counts["refined_group_count"],
        }
        for target, counts in by_target.items()
    }
    include_count = semantic_decisions.get("INCLUDE", 0)
    review_count = semantic_decisions.get("REVIEW", 0)
    exclude_count = semantic_decisions.get("EXCLUDE", 0)
    conservative_final_exclusions = sum(
        row["exclusion_reason"] == "UNRESOLVED_PERCEPTUAL_IDENTITY"
        for row in training_matched
    )
    include_rows = [row for row in master if row["cleaning_status"] == "INCLUDE"]
    supports_full_taxonomy = len(represented_classes) == len(CLASS_NAMES)
    return {
        "schema_version": 2,
        "semantic_raw": len(training_matched),
        "include": include_count,
        "review": review_count,
        "exclude": exclude_count,
        "represented_class_count": len(represented_classes),
        "represented_classes": represented_classes,
        "missing_class_count": len(missing_classes),
        "missing_classes": missing_classes,
        "conservative_final_exclusion_count": conservative_final_exclusions,
        "conservative_final_exclusion_percentage_of_semantic_pool": (
            conservative_final_exclusions / len(training_matched) * 100
            if training_matched
            else 0.0
        ),
        "full_39_class_retraining_supported_by_candidate_pool": supports_full_taxonomy,
        "missing_class_data_requirement": (
            "Recover or reproducibly reconstruct the historical PlantVillage-style "
            "training source before designing a full 39-output retraining plan."
            if not supports_full_taxonomy
            else "No missing deployed classes."
        ),
        "counts_by_source": counts_by_source,
        "counts_by_target": counts_by_target,
        "inventory_record_count": len(master),
        "raw_semantic_candidate_count": len(training_matched),
        "semantic_candidate_decisions": {
            "include": include_count,
            "review": review_count,
            "exclude": exclude_count,
        },
        "full_master_decisions": {
            "include": full_decisions.get("INCLUDE", 0),
            "review": full_decisions.get("REVIEW", 0),
            "exclude": full_decisions.get("EXCLUDE", 0),
        },
        "exclusions_by_reason": dict(
            sorted(
                Counter(
                    row["exclusion_reason"]
                    for row in master
                    if row["cleaning_status"] == "EXCLUDE"
                ).items()
            )
        ),
        "reviews_by_reason": dict(
            sorted(
                Counter(
                    row["exclusion_reason"]
                    for row in master
                    if row["cleaning_status"] == "REVIEW"
                ).items()
            )
        ),
        "exact_duplicate_audit": exact_stats,
        "perceptual_audit": perceptual,
        "by_source": by_source,
        "by_target_class": by_target,
        "invariants": {
            "raw_data_modified": False,
            "split_created": False,
            "training_performed": False,
            "clean_manifest_contains_only_include": True,
            "clean_manifest_has_unique_sha256": True,
            "review_count_is_zero": review_count == 0,
            "locked_benchmark_is_excluded": not any(
                row["role"] == "locked_test" and row["cleaning_status"] == "INCLUDE"
                for row in master
            ),
            "unresolved_perceptual_identity_is_excluded": not any(
                row["review_resolution"] == "STILL_UNCERTAIN"
                or (
                    row["review_resolution"] == "CONSERVATIVE_FINAL_EXCLUSION"
                    and row["cleaning_status"] != "EXCLUDE"
                )
                for row in master
            ),
            "exact_benchmark_leakage_is_excluded": not any(
                row["benchmark_leakage"] == "true" for row in include_rows
            ),
            "exact_label_conflicts_are_excluded": not any(
                row["label_conflict"] == "true" for row in include_rows
            ),
            "verified_perceptual_risks_are_excluded": not any(
                row["refined_similarity_status"]
                in {"VERIFIED_LABEL_CONFLICT", "VERIFIED_BENCHMARK_LEAKAGE"}
                for row in include_rows
            ),
            "ambiguous_and_unsupported_mappings_are_excluded": all(
                row["mapping_status"] == "MATCHED" for row in include_rows
            ),
            "all_targets_use_deployed_taxonomy": all(
                not row["target_class"] or row["target_class"] in CLASS_NAMES
                for row in master
            ),
        },
    }


def build_manifests(
    inventory_path: Path,
    exact_report_path: Path,
    perceptual_members_path: Path,
    perceptual_summary_path: Path,
    perceptual_review_path: Path,
    master_path: Path,
    clean_path: Path,
    cleaning_summary_path: Path,
    full_perceptual_path: Path | None = None,
    exact_conflicts_path: Path | None = None,
    perceptual_resolution_path: Path | None = None,
    finalize_candidates: bool = False,
) -> dict:
    records = load_inventory(inventory_path)
    initialize_cleaning(records)
    exact_stats = apply_exact_policy(records, exact_report_path)

    if full_perceptual_path is not None and full_perceptual_path.exists():
        perceptual_stats = build_perceptual_reports(
            records,
            full_perceptual_path,
            perceptual_members_path,
            perceptual_summary_path,
            perceptual_review_path,
        )
    else:
        required = (
            perceptual_members_path,
            perceptual_summary_path,
            perceptual_review_path,
        )
        missing = [str(path) for path in required if not path.exists()]
        if missing:
            raise ManifestBuildError(
                "The full local dHash audit is unavailable and compact reports are missing: "
                + ", ".join(missing)
            )
        perceptual_stats = load_perceptual_members(records, perceptual_members_path)
        with perceptual_summary_path.open("r", encoding="utf-8", newline="") as handle:
            summary_rows = list(csv.DictReader(handle))
        with perceptual_review_path.open("r", encoding="utf-8", newline="") as handle:
            review_rows = list(csv.DictReader(handle))
        perceptual_stats.update(
            {
                "different_target_group_count": sum(
                    int(row["different_target_pairs"]) > 0 for row in summary_rows
                ),
                "benchmark_group_count": sum(
                    int(row["train_to_locked_benchmark_pairs"]) > 0
                    for row in summary_rows
                ),
                "different_target_pair_count": sum(
                    int(row["different_target_pairs"]) for row in summary_rows
                ),
                "benchmark_pair_count": sum(
                    int(row["train_to_locked_benchmark_pairs"]) for row in summary_rows
                ),
                "pair_count": sum(int(row["edge_count"]) for row in summary_rows),
                "review_queue_rows": len(review_rows),
            }
        )

    member_stats = load_perceptual_members(records, perceptual_members_path)
    perceptual_stats.update(member_stats)
    if perceptual_resolution_path is not None and perceptual_resolution_path.exists():
        perceptual_stats["resolution"] = apply_perceptual_resolution(
            records, perceptual_resolution_path
        )
    elif finalize_candidates:
        raise ManifestBuildError(
            "Final candidate generation requires the Step 5C.1 perceptual resolution report."
        )
    if finalize_candidates:
        perceptual_stats["finalization"] = finalize_unresolved_perceptual_reviews(
            records
        )
    master = sorted(records, key=lambda row: row["record_id"])
    clean = [row for row in master if row["cleaning_status"] == "INCLUDE"]
    validate_manifests(master, clean, require_final=finalize_candidates)
    write_csv(master_path, master, MASTER_FIELDS)
    write_csv(clean_path, clean, MASTER_FIELDS)
    if exact_conflicts_path is not None:
        write_exact_conflict_report(master, exact_conflicts_path)
    summary = summarize(master, exact_stats, perceptual_stats)
    cleaning_summary_path.parent.mkdir(parents=True, exist_ok=True)
    cleaning_summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


def parse_args() -> argparse.Namespace:
    datasets = PROJECT_ROOT / "training" / "datasets"
    parser = argparse.ArgumentParser(
        description="Build deterministic Dataset V2 master and clean-candidate manifests."
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
        "--full-perceptual-report",
        type=Path,
        default=datasets / "local-audits" / "perceptual-duplicate-candidates-full.csv",
    )
    parser.add_argument(
        "--perceptual-members-report",
        type=Path,
        default=datasets / "reports" / "perceptual-group-members.csv",
    )
    parser.add_argument(
        "--perceptual-summary-report",
        type=Path,
        default=datasets / "reports" / "perceptual-groups-summary.csv",
    )
    parser.add_argument(
        "--perceptual-review-queue",
        type=Path,
        default=datasets / "reports" / "perceptual-review-queue.csv",
    )
    parser.add_argument(
        "--master-output",
        type=Path,
        default=datasets / "manifests" / "dataset-v2-master.csv",
    )
    parser.add_argument(
        "--clean-output",
        type=Path,
        default=datasets / "manifests" / "dataset-v2-clean-candidates.csv",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=datasets / "reports" / "dataset-v2-final-candidate-summary.json",
    )
    parser.add_argument(
        "--exact-conflicts-report",
        type=Path,
        default=datasets / "reports" / "exact-label-conflicts.csv",
    )
    parser.add_argument(
        "--perceptual-resolution",
        type=Path,
        default=datasets / "reports" / "refined-group-members.csv",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        summary = build_manifests(
            args.inventory,
            args.exact_report,
            args.perceptual_members_report,
            args.perceptual_summary_report,
            args.perceptual_review_queue,
            args.master_output,
            args.clean_output,
            args.summary_output,
            args.full_perceptual_report,
            args.exact_conflicts_report,
            args.perceptual_resolution,
            True,
        )
    except (ManifestBuildError, OSError, csv.Error, json.JSONDecodeError) as exc:
        print(f"Dataset V2 manifest build failed: {exc}", file=sys.stderr)
        return 1
    decisions = summary["semantic_candidate_decisions"]
    print(f"Master records: {summary['inventory_record_count']}")
    print(f"Raw semantic candidates: {summary['raw_semantic_candidate_count']}")
    print(f"INCLUDE: {decisions['include']}")
    print(f"REVIEW: {decisions['review']}")
    print(f"EXCLUDE: {decisions['exclude']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
