from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.taxonomy import CLASS_NAMES  # noqa: E402


HISTORICAL_DOMAIN = "HISTORICAL_CONTROLLED"
REAL_WORLD_DOMAIN = "REAL_WORLD"
COMPOSITION_STATUS = "INCLUDE_CANDIDATE"
HISTORICAL_DATASET = "Historical Mendeley 39-class source"
ALLOWED_REAL_WORLD_DATASETS = {
    "PlantDoc",
    "PLDD-UP",
    "Potato Leaf Disease Dataset",
    "Seasonal Corn Leaf Disease Dataset",
}
FINAL_REVIEW_REASON = "UNRESOLVED_HISTORICAL_PERCEPTUAL_IDENTITY"
FINAL_REVIEW_RESOLUTION = "CONSERVATIVE_FINAL_EXCLUSION"

COMBINED_FIELDS = [
    "record_id",
    "composition_record_id",
    "source_domain",
    "source_dataset",
    "source_version",
    "source_record_id",
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
    "exact_group_id",
    "refined_group_id",
    "split_group_id",
    "provenance_status",
    "composition_status",
]


class CompositionError(RuntimeError):
    """Raised when an audited input or final composition invariant fails."""


def load_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise CompositionError(f"Required manifest is unavailable: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = [dict(row) for row in csv.DictReader(handle)]
    if not rows:
        raise CompositionError(f"Required manifest is empty: {path}")
    return rows


def load_json(path: Path) -> dict:
    if not path.is_file():
        raise CompositionError(f"Required report is unavailable: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise CompositionError(f"Expected a JSON object: {path}")
    return payload


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


def stable_composition_id(
    source_domain: str,
    source_dataset: str,
    source_record_id: str,
    target_index: int,
    sha256: str,
) -> str:
    identity = "\0".join(
        (
            source_domain,
            source_dataset,
            source_record_id,
            str(target_index),
            sha256,
        )
    )
    return f"comp_{hashlib.sha256(identity.encode('utf-8')).hexdigest()[:24]}"


def stable_split_group_id(
    source_domain: str,
    composition_record_id: str,
    exact_group_id: str,
    refined_group_id: str,
) -> str:
    if refined_group_id:
        kind = "refined"
        source_group = refined_group_id
    elif exact_group_id:
        kind = "exact"
        source_group = exact_group_id
    else:
        kind = "singleton"
        source_group = composition_record_id
    identity = "\0".join((source_domain, kind, source_group))
    prefix = "split_single" if kind == "singleton" else "split_group"
    return f"{prefix}_{hashlib.sha256(identity.encode('utf-8')).hexdigest()[:24]}"


def _require_fields(row: dict, fields: Sequence[str], context: str) -> None:
    missing = [field for field in fields if field not in row]
    if missing:
        raise CompositionError(f"{context} is missing fields: {missing}")


def _taxonomy_index(target_class: str, class_names: Sequence[str]) -> int:
    try:
        return list(class_names).index(target_class)
    except ValueError as exc:
        raise CompositionError(f"Unknown deployed taxonomy class: {target_class!r}") from exc


def _base_combined_record(
    source: dict,
    source_domain: str,
    source_dataset: str,
    source_record_id: str,
    target_index: int,
    exact_group_id: str,
    refined_group_id: str,
    provenance_status: str,
) -> dict[str, str]:
    composition_id = stable_composition_id(
        source_domain,
        source_dataset,
        source_record_id,
        target_index,
        source["sha256"],
    )
    split_group_id = stable_split_group_id(
        source_domain,
        composition_id,
        exact_group_id,
        refined_group_id,
    )
    return {
        "record_id": composition_id,
        "composition_record_id": composition_id,
        "source_domain": source_domain,
        "source_dataset": source_dataset,
        "source_version": source["source_version"],
        "source_record_id": source_record_id,
        "source_path": source["source_path"],
        "source_label": source["source_label"],
        "target_index": str(target_index),
        "target_class": source["target_class"],
        "sha256": source["sha256"],
        "dhash": source.get("dhash", ""),
        "phash": source.get("phash", ""),
        "format": source["format"],
        "mode": source.get("mode", ""),
        "width": source["width"],
        "height": source["height"],
        "bytes": source["bytes"],
        "exact_group_id": exact_group_id,
        "refined_group_id": refined_group_id,
        "split_group_id": split_group_id,
        "provenance_status": provenance_status,
        "composition_status": COMPOSITION_STATUS,
    }


def normalize_historical_record(
    source: dict, class_names: Sequence[str] = CLASS_NAMES
) -> dict[str, str]:
    _require_fields(
        source,
        (
            "record_id",
            "dataset",
            "source_version",
            "source_path",
            "source_label",
            "target_index",
            "target_class",
            "sha256",
            "format",
            "width",
            "height",
            "bytes",
            "integrity_status",
            "candidate_status",
            "exact_duplicate_group_id",
            "perceptual_group_id",
            "benchmark_leakage",
        ),
        "Historical row",
    )
    if source["dataset"] != HISTORICAL_DATASET:
        raise CompositionError(f"Unexpected historical dataset: {source['dataset']!r}")
    if source["integrity_status"] != "VALID":
        raise CompositionError(f"Invalid historical image entered composition: {source['record_id']}")
    if source["candidate_status"] != "INCLUDE_CANDIDATE":
        raise CompositionError(
            f"Historical non-INCLUDE row entered composition: {source['record_id']}"
        )
    if source["benchmark_leakage"].lower() != "false":
        raise CompositionError(f"Historical benchmark leak entered composition: {source['record_id']}")
    target_index = _taxonomy_index(source["target_class"], class_names)
    if source["target_index"] != str(target_index):
        raise CompositionError(
            f"Historical target index mismatch for {source['record_id']}: "
            f"{source['target_index']} != {target_index}"
        )
    return _base_combined_record(
        source,
        HISTORICAL_DOMAIN,
        source["dataset"],
        source["record_id"],
        target_index,
        source.get("exact_duplicate_group_id", ""),
        source.get("perceptual_group_id", ""),
        "AUDITED_HISTORICAL_CLEAN",
    )


def normalize_real_world_record(
    source: dict, class_names: Sequence[str] = CLASS_NAMES
) -> dict[str, str]:
    _require_fields(
        source,
        (
            "record_id",
            "dataset",
            "source_version",
            "role",
            "source_path",
            "source_label",
            "mapping_status",
            "target_class",
            "sha256",
            "format",
            "width",
            "height",
            "bytes",
            "cleaning_status",
            "exact_duplicate_group_id",
            "refined_group_id",
            "benchmark_leakage",
            "manual_review_required",
        ),
        "Real-world row",
    )
    if source["dataset"] not in ALLOWED_REAL_WORLD_DATASETS:
        raise CompositionError(f"Unexpected real-world dataset: {source['dataset']!r}")
    if source["role"] != "training_candidate":
        raise CompositionError(f"Locked or invalid role entered composition: {source['role']!r}")
    if source["mapping_status"] != "MATCHED":
        raise CompositionError(f"Unresolved real-world mapping: {source['record_id']}")
    if source["cleaning_status"] != "INCLUDE":
        raise CompositionError(f"Real-world non-INCLUDE row entered composition: {source['record_id']}")
    if source["benchmark_leakage"].lower() != "false":
        raise CompositionError(f"Real-world benchmark leak entered composition: {source['record_id']}")
    if source["manual_review_required"].lower() != "false":
        raise CompositionError(f"Real-world review row entered composition: {source['record_id']}")
    target_index = _taxonomy_index(source["target_class"], class_names)
    return _base_combined_record(
        source,
        REAL_WORLD_DOMAIN,
        source["dataset"],
        source["record_id"],
        target_index,
        source.get("exact_duplicate_group_id", ""),
        source.get("refined_group_id", ""),
        "AUDITED_REAL_WORLD_CLEAN",
    )


def finalize_historical_reviews(
    historical_full: Sequence[dict], historical_clean: Sequence[dict]
) -> dict:
    clean_ids = {row["record_id"] for row in historical_clean}
    if len(clean_ids) != len(historical_clean):
        raise CompositionError("Historical clean manifest contains duplicate record IDs.")
    reviews = sorted(
        (
            row
            for row in historical_full
            if row.get("candidate_status") == "REVIEW_PERCEPTUAL_CONFLICT"
        ),
        key=lambda row: row["record_id"],
    )
    if len(reviews) != 2:
        raise CompositionError(f"Expected exactly two historical review records, found {len(reviews)}.")
    if any(row["record_id"] in clean_ids for row in reviews):
        raise CompositionError("A historical review record is present in the clean manifest.")
    targets = {row["target_class"] for row in reviews}
    if targets != {"Tomato Late blight", "Tomato healthy"}:
        raise CompositionError(f"Unexpected historical review targets: {sorted(targets)}")
    records = [
        {
            "source_record_id": row["record_id"],
            "source_path": row["source_path"],
            "source_label": row["source_label"],
            "target_index": int(row["target_index"]),
            "target_class": row["target_class"],
            "sha256": row["sha256"],
            "cleaning_status": "EXCLUDE",
            "exclusion_reason": FINAL_REVIEW_REASON,
            "review_resolution": FINAL_REVIEW_RESOLUTION,
        }
        for row in reviews
    ]
    return {
        "schema_version": 1,
        "decision": "Conservative metadata-only exclusion; no relabeling or raw-file deletion.",
        "review_count_before": len(reviews),
        "review_count_after": 0,
        "historical_clean_count_before": len(historical_clean),
        "historical_clean_count_after": len(historical_clean),
        "records": records,
    }


def validate_input_reports(
    historical_summary: dict,
    real_world_summary: dict,
    historical_full: Sequence[dict],
    historical_clean: Sequence[dict],
    real_world_clean: Sequence[dict],
) -> None:
    expected_historical_clean = int(historical_summary["clean_candidate_count"])
    expected_real_clean = int(real_world_summary["include"])
    if len(historical_clean) != expected_historical_clean:
        raise CompositionError(
            f"Historical clean count mismatch: {len(historical_clean)} != {expected_historical_clean}"
        )
    if len(real_world_clean) != expected_real_clean:
        raise CompositionError(
            f"Real-world clean count mismatch: {len(real_world_clean)} != {expected_real_clean}"
        )
    if len(historical_full) != int(historical_summary["verified_image_count"]):
        raise CompositionError("Historical full manifest count differs from its audit summary.")
    cross_fields = (
        "exact_overlap_with_dataset_v2",
        "perceptual_overlap_with_dataset_v2",
        "exact_leakage_to_plantdoc_test",
        "perceptual_leakage_to_plantdoc_test",
    )
    unexpected = {field: historical_summary[field] for field in cross_fields if historical_summary[field] != 0}
    if unexpected:
        raise CompositionError(f"Known cross-source or benchmark collision is non-zero: {unexpected}")
    if int(real_world_summary["review"]) != 0:
        raise CompositionError("Real-world final manifest still reports REVIEW records.")


def compose_records(
    historical_clean: Sequence[dict],
    real_world_clean: Sequence[dict],
    class_names: Sequence[str] = CLASS_NAMES,
) -> list[dict[str, str]]:
    historical = [normalize_historical_record(row, class_names) for row in historical_clean]
    real_world = [normalize_real_world_record(row, class_names) for row in real_world_clean]

    historical_hashes: defaultdict[str, list[str]] = defaultdict(list)
    real_world_hashes: defaultdict[str, list[str]] = defaultdict(list)
    for row in historical:
        historical_hashes[row["sha256"]].append(row["source_record_id"])
    for row in real_world:
        real_world_hashes[row["sha256"]].append(row["source_record_id"])
    collisions = sorted(set(historical_hashes) & set(real_world_hashes))
    if collisions:
        details = [
            {
                "sha256": digest,
                "historical": historical_hashes[digest],
                "real_world": real_world_hashes[digest],
            }
            for digest in collisions[:10]
        ]
        raise CompositionError(f"Unexpected cross-domain SHA-256 collision: {details}")

    combined = sorted(
        [*historical, *real_world], key=lambda row: row["composition_record_id"]
    )
    composition_ids = [row["composition_record_id"] for row in combined]
    if len(composition_ids) != len(set(composition_ids)):
        raise CompositionError("Duplicate deterministic composition_record_id detected.")
    if len(combined) != len(historical_clean) + len(real_world_clean):
        raise CompositionError("Final candidate count does not reconcile with input manifests.")
    indices = {int(row["target_index"]) for row in combined}
    expected_indices = set(range(len(class_names)))
    if indices != expected_indices:
        raise CompositionError(
            f"Incomplete class-index coverage: missing={sorted(expected_indices - indices)}, "
            f"unexpected={sorted(indices - expected_indices)}"
        )
    if {row["target_class"] for row in combined} != set(class_names):
        raise CompositionError("Final class names do not match the deployed taxonomy.")
    if any(row["composition_status"] != COMPOSITION_STATUS for row in combined):
        raise CompositionError("Final manifest contains a non-eligible composition status.")
    if any("test" in row["source_path"].lower().split("/")[:1] for row in combined):
        raise CompositionError("A test path unexpectedly entered the final composition.")
    return combined


def build_summary(
    combined: Sequence[dict],
    finalization: dict,
    historical_summary: dict,
    input_hashes: dict[str, str],
    class_names: Sequence[str] = CLASS_NAMES,
) -> dict:
    domain_counts = Counter(row["source_domain"] for row in combined)
    source_counts = Counter(row["source_dataset"] for row in combined)
    class_domain_counts: defaultdict[str, Counter[str]] = defaultdict(Counter)
    class_sources: defaultdict[str, set[str]] = defaultdict(set)
    for row in combined:
        class_domain_counts[row["target_class"]][row["source_domain"]] += 1
        if row["source_domain"] == REAL_WORLD_DOMAIN:
            class_sources[row["target_class"]].add(row["source_dataset"])

    counts_by_class = []
    for index, target_class in enumerate(class_names):
        historical = class_domain_counts[target_class][HISTORICAL_DOMAIN]
        real_world = class_domain_counts[target_class][REAL_WORLD_DOMAIN]
        combined_count = historical + real_world
        counts_by_class.append(
            {
                "target_index": index,
                "target_class": target_class,
                "historical_count": historical,
                "real_world_count": real_world,
                "combined_count": combined_count,
                "real_world_source_count": len(class_sources[target_class]),
                "real_world_sources": sorted(class_sources[target_class]),
                "historical_fraction": round(historical / combined_count, 8),
                "real_world_fraction": round(real_world / combined_count, 8),
            }
        )

    class_sizes = [row["combined_count"] for row in counts_by_class]
    smallest = sorted(counts_by_class, key=lambda row: (row["combined_count"], row["target_index"]))
    largest = sorted(
        counts_by_class,
        key=lambda row: (-row["combined_count"], row["target_index"]),
    )
    split_sizes = Counter(row["split_group_id"] for row in combined)
    split_domains: defaultdict[str, set[str]] = defaultdict(set)
    split_targets: defaultdict[str, set[str]] = defaultdict(set)
    for row in combined:
        split_domains[row["split_group_id"]].add(row["source_domain"])
        split_targets[row["split_group_id"]].add(row["target_class"])
    cross_domain_groups = [group for group, domains in split_domains.items() if len(domains) > 1]
    if cross_domain_groups:
        raise CompositionError(f"Unexpected cross-domain split groups: {cross_domain_groups[:10]}")
    cross_target_groups = [group for group, targets in split_targets.items() if len(targets) > 1]
    if cross_target_groups:
        raise CompositionError(f"Unexpected cross-target split groups: {cross_target_groups[:10]}")
    largest_group_size = max(split_sizes.values())
    largest_group_id = min(
        group for group, size in split_sizes.items() if size == largest_group_size
    )

    historical_only = [row["target_class"] for row in counts_by_class if row["real_world_count"] == 0]
    real_supported = [row["target_class"] for row in counts_by_class if row["real_world_count"] > 0]
    multiple_sources = [
        row["target_class"] for row in counts_by_class if row["real_world_source_count"] > 1
    ]
    total = len(combined)
    benchmark_leakage_count = sum(
        row["source_dataset"] == "PlantDoc" and row["source_path"].startswith("test/")
        for row in combined
    )
    if benchmark_leakage_count:
        raise CompositionError("PlantDoc TEST contamination detected in final composition.")

    return {
        "schema_version": 1,
        "composition_name": "Dataset V2 final 39-class candidate pool",
        "input_artifact_sha256": dict(sorted(input_hashes.items())),
        "historical_review_finalization": finalization,
        "total_candidates": total,
        "historical_candidates": domain_counts[HISTORICAL_DOMAIN],
        "real_world_candidates": domain_counts[REAL_WORLD_DOMAIN],
        "historical_contribution_percentage": round(
            100 * domain_counts[HISTORICAL_DOMAIN] / total, 6
        ),
        "real_world_contribution_percentage": round(
            100 * domain_counts[REAL_WORLD_DOMAIN] / total, 6
        ),
        "class_count": len(counts_by_class),
        "class_indices_present": [row["target_index"] for row in counts_by_class],
        "historical_only_class_count": len(historical_only),
        "historical_only_classes": historical_only,
        "real_world_supported_class_count": len(real_supported),
        "real_world_supported_classes": real_supported,
        "classes_with_multiple_real_world_sources": multiple_sources,
        "counts_by_class": counts_by_class,
        "counts_by_domain": dict(sorted(domain_counts.items())),
        "counts_by_source": dict(sorted(source_counts.items())),
        "class_imbalance": {
            "minimum_combined_class_size": min(class_sizes),
            "maximum_combined_class_size": max(class_sizes),
            "median_combined_class_size": statistics.median(class_sizes),
            "mean_combined_class_size": round(statistics.mean(class_sizes), 6),
            "maximum_to_minimum_ratio": round(max(class_sizes) / min(class_sizes), 6),
            "five_smallest_classes": smallest[:5],
            "five_largest_classes": largest[:5],
        },
        "exact_cross_domain_collision_count": 0,
        "verified_cross_domain_perceptual_collision_count": int(
            historical_summary["perceptual_overlap_with_dataset_v2"]
        ),
        "cross_domain_shared_split_group_count": len(cross_domain_groups),
        "benchmark_leakage_count": benchmark_leakage_count,
        "split_group_count": len(split_sizes),
        "singleton_group_count": sum(size == 1 for size in split_sizes.values()),
        "multi_record_group_count": sum(size > 1 for size in split_sizes.values()),
        "largest_group_size": largest_group_size,
        "largest_group": {
            "split_group_id": largest_group_id,
            "size": largest_group_size,
            "source_domain": min(split_domains[largest_group_id]),
            "target_class": min(split_targets[largest_group_id]),
        },
        "invariants": {
            "source_manifest_counts_reconciled": True,
            "class_coverage_is_39": len(counts_by_class) == 39,
            "target_indices_are_0_through_38": [row["target_index"] for row in counts_by_class]
            == list(range(39)),
            "only_clean_input_rows_included": True,
            "historical_review_count_is_zero_after_finalization": finalization[
                "review_count_after"
            ]
            == 0,
            "locked_benchmark_rows_absent": benchmark_leakage_count == 0,
            "known_benchmark_leaks_absent": True,
            "cross_domain_exact_collisions_absent": True,
            "known_cross_domain_perceptual_collisions_absent": historical_summary[
                "perceptual_overlap_with_dataset_v2"
            ]
            == 0,
            "cross_domain_split_groups_absent": not cross_domain_groups,
            "cross_target_split_groups_absent": not cross_target_groups,
            "composition_record_ids_unique": len(combined)
            == len({row["composition_record_id"] for row in combined}),
            "source_provenance_complete": all(
                row["source_domain"]
                and row["source_dataset"]
                and row["source_version"]
                and row["source_record_id"]
                and row["source_path"]
                for row in combined
            ),
            "split_assignment_created": False,
            "balancing_performed": False,
            "augmentation_performed": False,
            "training_performed": False,
        },
    }


def run(args: argparse.Namespace) -> dict:
    historical_full = load_csv(args.historical_full)
    historical_clean = load_csv(args.historical_clean)
    real_world_clean = load_csv(args.real_world_clean)
    historical_summary = load_json(args.historical_summary)
    real_world_summary = load_json(args.real_world_summary)
    validate_input_reports(
        historical_summary,
        real_world_summary,
        historical_full,
        historical_clean,
        real_world_clean,
    )
    finalization = finalize_historical_reviews(historical_full, historical_clean)
    combined = compose_records(historical_clean, real_world_clean)
    inputs = {
        "historical_full_manifest": file_sha256(args.historical_full),
        "historical_clean_manifest": file_sha256(args.historical_clean),
        "real_world_clean_manifest": file_sha256(args.real_world_clean),
        "historical_audit_summary": file_sha256(args.historical_summary),
        "real_world_final_summary": file_sha256(args.real_world_summary),
    }
    summary = build_summary(combined, finalization, historical_summary, inputs)
    write_csv(args.output_manifest, combined, COMBINED_FIELDS)
    write_json(args.output_summary, summary)
    write_json(args.review_finalization, finalization)
    return summary


def parse_args() -> argparse.Namespace:
    datasets = PROJECT_ROOT / "training" / "datasets"
    parser = argparse.ArgumentParser(
        description="Compose audited historical and real-world manifests without splitting or training."
    )
    parser.add_argument(
        "--historical-full",
        type=Path,
        default=datasets / "manifests" / "historical-mendeley-39.csv",
    )
    parser.add_argument(
        "--historical-clean",
        type=Path,
        default=datasets / "manifests" / "historical-mendeley-39-clean-candidates.csv",
    )
    parser.add_argument(
        "--real-world-clean",
        type=Path,
        default=datasets / "manifests" / "dataset-v2-clean-candidates.csv",
    )
    parser.add_argument(
        "--historical-summary",
        type=Path,
        default=datasets / "reports" / "historical-39class-audit-summary.json",
    )
    parser.add_argument(
        "--real-world-summary",
        type=Path,
        default=datasets / "reports" / "dataset-v2-final-candidate-summary.json",
    )
    parser.add_argument(
        "--output-manifest",
        type=Path,
        default=datasets / "manifests" / "dataset-v2-39class-combined.csv",
    )
    parser.add_argument(
        "--output-summary",
        type=Path,
        default=datasets / "reports" / "dataset-v2-39class-composition-summary.json",
    )
    parser.add_argument(
        "--review-finalization",
        type=Path,
        default=datasets / "reports" / "historical-review-finalization.json",
    )
    return parser.parse_args()


def main() -> int:
    summary = run(parse_args())
    print(
        json.dumps(
            {
                "total_candidates": summary["total_candidates"],
                "coverage": f"{summary['class_count']}/39",
                "historical_candidates": summary["historical_candidates"],
                "real_world_candidates": summary["real_world_candidates"],
                "split_groups": summary["split_group_count"],
                "benchmark_leakage": summary["benchmark_leakage_count"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
