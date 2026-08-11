from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training.taxonomy import CLASS_NAMES  # noqa: E402


SPLITS = ("TRAIN", "VALIDATION", "TEST")
TARGET_RATIOS = {"TRAIN": 0.80, "VALIDATION": 0.10, "TEST": 0.10}
SPLIT_SEED = 20260810
STRATEGY_VERSION = "dataset-v2-group-aware-80-10-10-v1"
HISTORICAL_DOMAIN = "HISTORICAL_CONTROLLED"
REAL_WORLD_DOMAIN = "REAL_WORLD"
EVALUATION_ROLES = {
    "TRAIN": "MODEL_TRAINING",
    "VALIDATION": "MODEL_DEVELOPMENT_VALIDATION",
    "TEST": "FINAL_INTERNAL_TEST",
}
REQUIRED_FIELDS = {
    "composition_record_id",
    "source_domain",
    "source_dataset",
    "source_path",
    "target_index",
    "target_class",
    "sha256",
    "split_group_id",
    "composition_status",
}
REAL_WORLD_SOURCE_LABELS = {
    "PLDD-UP": "PLDD-UP",
    "Seasonal Corn Leaf Disease Dataset": "Seasonal Corn",
    "PlantDoc": "PlantDoc TRAIN-source",
    "Potato Leaf Disease Dataset": "Banu/Deb Potato",
}


class SplitError(RuntimeError):
    """Raised when split inputs or leakage invariants are invalid."""


@dataclass(frozen=True)
class SplitGroup:
    group_id: str
    target_index: int
    target_class: str
    source_domain: str
    source_datasets: tuple[str, ...]
    image_count: int
    record_ids: tuple[str, ...]

    @property
    def stratum_key(self) -> tuple[int, str, tuple[str, ...]]:
        return (self.target_index, self.source_domain, self.source_datasets)


def load_csv(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    if not path.is_file():
        raise SplitError(f"Required composition manifest is unavailable: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = list(reader.fieldnames or [])
        rows = [dict(row) for row in reader]
    if not rows:
        raise SplitError(f"Required composition manifest is empty: {path}")
    missing = REQUIRED_FIELDS - set(fields)
    if missing:
        raise SplitError(f"Composition manifest is missing fields: {sorted(missing)}")
    return rows, fields


def write_csv(path: Path, rows: Iterable[Mapping[str, object]], fields: Sequence[str]) -> None:
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


def stable_hash(*parts: object) -> str:
    encoded = "\0".join(str(part) for part in parts).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def percentage(part: int, total: int) -> float:
    return round((100.0 * part / total), 8) if total else 0.0


def is_plantdoc_test(row: Mapping[str, str]) -> bool:
    if row.get("source_dataset", "").casefold() != "plantdoc":
        return False
    parts = [
        part.casefold()
        for part in row.get("source_path", "").replace("\\", "/").split("/")
        if part
    ]
    return bool(parts and parts[0] in {"test", "testing", "test_split"})


def validate_and_group(rows: Sequence[dict[str, str]]) -> list[SplitGroup]:
    record_ids = [row["composition_record_id"] for row in rows]
    if len(record_ids) != len(set(record_ids)):
        raise SplitError("Duplicate composition_record_id detected.")
    if any(is_plantdoc_test(row) for row in rows):
        raise SplitError("PlantDoc TEST benchmark record entered Dataset V2 splitting.")

    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        group_id = row["split_group_id"].strip()
        if not group_id:
            raise SplitError(f"Missing split_group_id: {row['composition_record_id']}")
        if row["composition_status"] != "INCLUDE_CANDIDATE":
            raise SplitError(
                f"Non-candidate record entered splitting: {row['composition_record_id']}"
            )
        try:
            target_index = int(row["target_index"])
        except ValueError as exc:
            raise SplitError(f"Invalid target_index: {row['target_index']}") from exc
        if not 0 <= target_index < len(CLASS_NAMES):
            raise SplitError(f"Out-of-range target_index: {target_index}")
        if row["target_class"] != CLASS_NAMES[target_index]:
            raise SplitError(
                f"Taxonomy mismatch for {row['composition_record_id']}: "
                f"{target_index} != {row['target_class']}"
            )
        if row["source_domain"] not in {HISTORICAL_DOMAIN, REAL_WORLD_DOMAIN}:
            raise SplitError(f"Unsupported source_domain: {row['source_domain']}")
        if not row["sha256"].strip():
            raise SplitError(f"Missing SHA-256: {row['composition_record_id']}")
        grouped[group_id].append(row)

    groups: list[SplitGroup] = []
    for group_id in sorted(grouped):
        members = grouped[group_id]
        target_indices = {int(row["target_index"]) for row in members}
        target_classes = {row["target_class"] for row in members}
        domains = {row["source_domain"] for row in members}
        if len(target_indices) != 1 or len(target_classes) != 1 or len(domains) != 1:
            raise SplitError(
                f"Incompatible target class/index/domain inside split group: {group_id}"
            )
        groups.append(
            SplitGroup(
                group_id=group_id,
                target_index=min(target_indices),
                target_class=min(target_classes),
                source_domain=min(domains),
                source_datasets=tuple(sorted({row["source_dataset"] for row in members})),
                image_count=len(members),
                record_ids=tuple(sorted(row["composition_record_id"] for row in members)),
            )
        )
    return groups


def integer_targets(total: int) -> dict[str, int]:
    raw = {split: total * TARGET_RATIOS[split] for split in SPLITS}
    result = {split: int(raw[split]) for split in SPLITS}
    remainder = total - sum(result.values())
    priority = {split: index for index, split in enumerate(SPLITS)}
    order = sorted(
        SPLITS,
        key=lambda split: (-(raw[split] - result[split]), priority[split]),
    )
    for split in order[:remainder]:
        result[split] += 1
    return result


def minimum_group_counts(group_count: int) -> dict[str, int]:
    if group_count >= 20:
        return {"TRAIN": 1, "VALIDATION": 2, "TEST": 2}
    if group_count >= 10:
        return {"TRAIN": 1, "VALIDATION": 1, "TEST": 1}
    return {"TRAIN": group_count, "VALIDATION": 0, "TEST": 0}


def allocate_stratum(groups: Sequence[SplitGroup], seed: int) -> dict[str, str]:
    if not groups:
        return {}
    minimums = minimum_group_counts(len(groups))
    if len(groups) < 10:
        return {group.group_id: "TRAIN" for group in groups}

    total_images = sum(group.image_count for group in groups)
    image_targets = integer_targets(total_images)
    group_targets = integer_targets(len(groups))
    for split in SPLITS:
        group_targets[split] = max(group_targets[split], minimums[split])

    # Randomize deterministically at the group level. Sorting by group size would
    # systematically concentrate the largest duplicate families in holdouts.
    ordered = sorted(
        groups,
        key=lambda group: (stable_hash(seed, group.group_id), group.group_id),
    )
    image_counts = Counter({split: 0 for split in SPLITS})
    group_counts = Counter({split: 0 for split in SPLITS})
    assignments: dict[str, str] = {}

    for position, group in enumerate(ordered):
        remaining_after = len(ordered) - position - 1
        eligible = list(SPLITS)
        unmet = {
            split: max(0, minimums[split] - group_counts[split]) for split in SPLITS
        }
        if sum(unmet.values()) > remaining_after:
            eligible = [split for split in SPLITS if unmet[split] > 0]

        def objective(split: str) -> tuple[float, str]:
            projected_images = dict(image_counts)
            projected_groups = dict(group_counts)
            projected_images[split] += group.image_count
            projected_groups[split] += 1
            image_error = sum(
                (
                    (projected_images[name] - image_targets[name])
                    / max(1, image_targets[name])
                )
                ** 2
                for name in SPLITS
            )
            group_error = sum(
                (
                    (projected_groups[name] - group_targets[name])
                    / max(1, group_targets[name])
                )
                ** 2
                for name in SPLITS
            )
            return (
                image_error + (0.05 * group_error),
                stable_hash(seed, group.group_id, split),
            )

        chosen = min(eligible, key=objective)
        assignments[group.group_id] = chosen
        image_counts[chosen] += group.image_count
        group_counts[chosen] += 1

    if any(group_counts[split] < minimums[split] for split in SPLITS):
        raise SplitError("Allocator failed its minimum group-support policy.")
    return assignments


def allocate_groups(groups: Sequence[SplitGroup], seed: int) -> dict[str, str]:
    strata: dict[tuple[int, str, tuple[str, ...]], list[SplitGroup]] = defaultdict(list)
    for group in groups:
        strata[group.stratum_key].append(group)
    assignments: dict[str, str] = {}
    for key in sorted(strata):
        stratum_assignments = allocate_stratum(strata[key], seed)
        overlap = set(assignments).intersection(stratum_assignments)
        if overlap:
            raise SplitError(f"Group assigned twice: {min(overlap)}")
        assignments.update(stratum_assignments)
    if len(assignments) != len(groups):
        raise SplitError("Not every split group received an assignment.")
    return assignments


def apply_assignments(
    rows: Sequence[dict[str, str]], assignments: Mapping[str, str], seed: int
) -> list[dict[str, str]]:
    assigned: list[dict[str, str]] = []
    for row in sorted(rows, key=lambda item: item["composition_record_id"]):
        split = assignments.get(row["split_group_id"])
        if split not in SPLITS:
            raise SplitError(
                f"Missing or invalid assignment for group: {row['split_group_id']}"
            )
        output = dict(row)
        output.update(
            {
                "split": split,
                "split_seed": str(seed),
                "split_strategy_version": STRATEGY_VERSION,
                "evaluation_role": EVALUATION_ROLES[split],
            }
        )
        assigned.append(output)
    return assigned


def validate_assigned_rows(rows: Sequence[dict[str, str]]) -> dict[str, int]:
    if not rows:
        raise SplitError("Assigned split is empty.")
    record_ids = [row["composition_record_id"] for row in rows]
    if len(record_ids) != len(set(record_ids)):
        raise SplitError("A composition record appears more than once in split outputs.")
    if any(row.get("split") not in SPLITS for row in rows):
        raise SplitError("An invalid split value was generated.")

    group_splits: dict[str, set[str]] = defaultdict(set)
    sha_splits: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        group_splits[row["split_group_id"]].add(row["split"])
        sha_splits[row["sha256"]].add(row["split"])
    leaking_groups = [group_id for group_id, splits in group_splits.items() if len(splits) > 1]
    if leaking_groups:
        raise SplitError(f"Group leakage detected: {len(leaking_groups)} groups")
    leaking_hashes = [digest for digest, splits in sha_splits.items() if len(splits) > 1]
    if leaking_hashes:
        raise SplitError(f"SHA leakage detected: {len(leaking_hashes)} hashes")
    benchmark_count = sum(is_plantdoc_test(row) for row in rows)
    if benchmark_count:
        raise SplitError(f"PlantDoc TEST contamination detected: {benchmark_count}")

    expected_indices = set(range(len(CLASS_NAMES)))
    for split in SPLITS:
        actual = {int(row["target_index"]) for row in rows if row["split"] == split}
        if actual != expected_indices:
            missing = sorted(expected_indices - actual)
            raise SplitError(f"{split} does not cover all classes; missing indices: {missing}")
    return {
        "group_leakage_count": 0,
        "sha_leakage_count": 0,
        "benchmark_contamination_count": 0,
    }


def split_counts(rows: Sequence[dict[str, str]]) -> dict[str, int]:
    counts = Counter(row["split"] for row in rows)
    return {split: counts[split] for split in SPLITS}


def class_quality_rows(
    rows: Sequence[dict[str, str]], groups: Sequence[SplitGroup], assignments: Mapping[str, str]
) -> list[dict[str, object]]:
    counts: dict[int, Counter] = defaultdict(Counter)
    real_groups: dict[int, Counter] = defaultdict(Counter)
    for row in rows:
        index = int(row["target_index"])
        counts[index]["total"] += 1
        counts[index][row["split"].lower()] += 1
        domain_key = "historical" if row["source_domain"] == HISTORICAL_DOMAIN else "real_world"
        counts[index][f"{domain_key}_total"] += 1
        counts[index][f"{domain_key}_{row['split'].lower()}"] += 1
    for group in groups:
        if group.source_domain == REAL_WORLD_DOMAIN:
            real_groups[group.target_index]["total"] += 1
            real_groups[group.target_index][assignments[group.group_id].lower()] += 1

    result: list[dict[str, object]] = []
    for index, target_class in enumerate(CLASS_NAMES):
        item = counts[index]
        real = real_groups[index]
        validation_real = item["real_world_validation"]
        test_real = item["real_world_test"]
        validation_groups = real["validation"]
        test_groups = real["test"]
        if item["real_world_total"] == 0:
            status = "NOT_APPLICABLE_NO_REAL_WORLD_DATA"
        elif validation_real == 0 and test_real == 0:
            status = "NO_REAL_WORLD_HOLDOUT"
        elif (
            validation_real >= 20
            and test_real >= 20
            and validation_groups >= 2
            and test_groups >= 2
        ):
            status = "ROBUST_REAL_WORLD_HOLDOUT"
        else:
            status = "LIMITED_REAL_WORLD_HOLDOUT"
        total = item["total"]
        result.append(
            {
                "target_index": index,
                "target_class": target_class,
                "total": total,
                "train": item["train"],
                "validation": item["validation"],
                "test": item["test"],
                "train_percentage": percentage(item["train"], total),
                "validation_percentage": percentage(item["validation"], total),
                "test_percentage": percentage(item["test"], total),
                "train_deviation_percentage_points": round(
                    percentage(item["train"], total) - 80.0, 8
                ),
                "validation_deviation_percentage_points": round(
                    percentage(item["validation"], total) - 10.0, 8
                ),
                "test_deviation_percentage_points": round(
                    percentage(item["test"], total) - 10.0, 8
                ),
                "historical_total": item["historical_total"],
                "historical_train": item["historical_train"],
                "historical_validation": item["historical_validation"],
                "historical_test": item["historical_test"],
                "real_world_total": item["real_world_total"],
                "real_world_train": item["real_world_train"],
                "real_world_validation": validation_real,
                "real_world_test": test_real,
                "real_world_group_count": real["total"],
                "real_world_validation_group_count": validation_groups,
                "real_world_test_group_count": test_groups,
                "real_world_holdout_status": status,
            }
        )
    return result


def distribution_rows(
    rows: Sequence[dict[str, str]], field: str
) -> list[dict[str, object]]:
    values = sorted({row[field] for row in rows})
    result: list[dict[str, object]] = []
    for value in values:
        selected = [row for row in rows if row[field] == value]
        counts = split_counts(selected)
        total = len(selected)
        item: dict[str, object] = {
            field: value,
            "display_name": REAL_WORLD_SOURCE_LABELS.get(value, value),
            "total": total,
        }
        for split in SPLITS:
            lower = split.lower()
            item[lower] = counts[split]
            item[f"{lower}_percentage"] = percentage(counts[split], total)
            item[f"classes_in_{lower}"] = len(
                {row["target_class"] for row in selected if row["split"] == split}
            )
        result.append(item)
    return result


def group_report(
    groups: Sequence[SplitGroup], assignments: Mapping[str, str]
) -> dict[str, object]:
    by_split = Counter(assignments.values())
    singleton = Counter()
    multi = Counter()
    largest: dict[str, dict[str, object]] = {}
    for split in SPLITS:
        split_groups = [group for group in groups if assignments[group.group_id] == split]
        singleton[split] = sum(group.image_count == 1 for group in split_groups)
        multi[split] = sum(group.image_count > 1 for group in split_groups)
        group = min(
            split_groups,
            key=lambda item: (-item.image_count, item.group_id),
        )
        largest[split.lower()] = {
            "split_group_id": group.group_id,
            "image_count": group.image_count,
            "target_index": group.target_index,
            "target_class": group.target_class,
            "source_domain": group.source_domain,
            "source_datasets": list(group.source_datasets),
        }
    return {
        "total_groups": len(groups),
        "groups_by_split": {split.lower(): by_split[split] for split in SPLITS},
        "singleton_groups_by_split": {
            split.lower(): singleton[split] for split in SPLITS
        },
        "multi_record_groups_by_split": {
            split.lower(): multi[split] for split in SPLITS
        },
        "largest_group_by_split": largest,
    }


def build_warnings(
    quality: Sequence[dict[str, object]], groups: Sequence[SplitGroup]
) -> list[dict[str, object]]:
    warnings: list[dict[str, object]] = []
    for row in quality:
        if row["validation"] < 20 or row["test"] < 20:
            warnings.append(
                {
                    "code": "CLASS_BELOW_PREFERRED_HOLDOUT_IMAGES",
                    "target_index": row["target_index"],
                    "target_class": row["target_class"],
                    "validation": row["validation"],
                    "test": row["test"],
                }
            )
        if row["real_world_holdout_status"] in {
            "NO_REAL_WORLD_HOLDOUT",
            "LIMITED_REAL_WORLD_HOLDOUT",
        }:
            warnings.append(
                {
                    "code": "INSUFFICIENT_DOMAIN_HOLDOUT_SUPPORT",
                    "target_index": row["target_index"],
                    "target_class": row["target_class"],
                    "real_world_group_count": row["real_world_group_count"],
                    "status": row["real_world_holdout_status"],
                }
            )

    strata: dict[tuple[int, str, tuple[str, ...]], int] = Counter(
        group.stratum_key for group in groups
    )
    for (target_index, domain, sources), count in sorted(strata.items()):
        if domain == REAL_WORLD_DOMAIN and count < 10:
            warnings.append(
                {
                    "code": "INSUFFICIENT_SOURCE_HOLDOUT_SUPPORT",
                    "target_index": target_index,
                    "target_class": CLASS_NAMES[target_index],
                    "source_domain": domain,
                    "source_datasets": list(sources),
                    "independent_group_count": count,
                    "allocation_policy": "TRAIN_ONLY",
                }
            )
    return warnings


def build_summary(
    rows: Sequence[dict[str, str]],
    groups: Sequence[SplitGroup],
    assignments: Mapping[str, str],
    quality: Sequence[dict[str, object]],
    seed: int,
    input_hash: str,
    output_hashes: Mapping[str, str],
    invariants: Mapping[str, int],
) -> dict[str, object]:
    counts = split_counts(rows)
    total = len(rows)
    domains = distribution_rows(rows, "source_domain")
    real_rows = [row for row in rows if row["source_domain"] == REAL_WORLD_DOMAIN]
    sources = distribution_rows(real_rows, "source_dataset")
    coverage = {
        split.lower(): len(
            {row["target_index"] for row in rows if row["split"] == split}
        )
        for split in SPLITS
    }
    holdout = [
        {
            key: row[key]
            for key in (
                "target_index",
                "target_class",
                "real_world_total",
                "real_world_train",
                "real_world_validation",
                "real_world_test",
                "real_world_group_count",
                "real_world_validation_group_count",
                "real_world_test_group_count",
                "real_world_holdout_status",
            )
        }
        for row in quality
        if row["real_world_total"] > 0
    ]
    groups_payload = group_report(groups, assignments)
    return {
        "seed": seed,
        "strategy_version": STRATEGY_VERSION,
        "target_ratios": {split.lower(): TARGET_RATIOS[split] for split in SPLITS},
        "total_images": total,
        "train_images": counts["TRAIN"],
        "validation_images": counts["VALIDATION"],
        "test_images": counts["TEST"],
        "train_percentage": percentage(counts["TRAIN"], total),
        "validation_percentage": percentage(counts["VALIDATION"], total),
        "test_percentage": percentage(counts["TEST"], total),
        "class_coverage": coverage,
        "total_groups": groups_payload["total_groups"],
        "train_groups": groups_payload["groups_by_split"]["train"],
        "validation_groups": groups_payload["groups_by_split"]["validation"],
        "test_groups": groups_payload["groups_by_split"]["test"],
        **dict(invariants),
        "counts_by_class": list(quality),
        "counts_by_domain": domains,
        "counts_by_source": sources,
        "real_world_holdout_quality": holdout,
        "group_distribution": groups_payload,
        "warnings": build_warnings(quality, groups),
        "input_composition_manifest_sha256": input_hash,
        "output_artifact_sha256": dict(sorted(output_hashes.items())),
        "policies": {
            "allocation_unit": "split_group_id",
            "validation_role": "MODEL_DEVELOPMENT_VALIDATION",
            "internal_test_role": "FINAL_INTERNAL_TEST",
            "internal_test_locked": True,
            "plantdoc_test_external": True,
            "balancing_performed": False,
            "augmentation_performed": False,
            "training_performed": False,
        },
    }


def run(args: argparse.Namespace) -> dict[str, object]:
    rows, input_fields = load_csv(args.input_manifest)
    groups = validate_and_group(rows)
    assignments = allocate_groups(groups, args.seed)
    assigned = apply_assignments(rows, assignments, args.seed)
    invariants = validate_assigned_rows(assigned)
    quality = class_quality_rows(assigned, groups, assignments)
    output_fields = [
        *input_fields,
        "split",
        "split_seed",
        "split_strategy_version",
        "evaluation_role",
    ]

    write_csv(args.output_manifest, assigned, output_fields)
    write_csv(
        args.train_manifest,
        (row for row in assigned if row["split"] == "TRAIN"),
        output_fields,
    )
    write_csv(
        args.validation_manifest,
        (row for row in assigned if row["split"] == "VALIDATION"),
        output_fields,
    )
    write_csv(
        args.test_manifest,
        (row for row in assigned if row["split"] == "TEST"),
        output_fields,
    )
    quality_fields = list(quality[0])
    write_csv(args.quality_report, quality, quality_fields)

    output_hashes = {
        "combined_split_manifest": file_sha256(args.output_manifest),
        "train_manifest": file_sha256(args.train_manifest),
        "validation_manifest": file_sha256(args.validation_manifest),
        "test_manifest": file_sha256(args.test_manifest),
        "split_quality_report": file_sha256(args.quality_report),
    }
    input_hash = file_sha256(args.input_manifest)
    summary = build_summary(
        assigned,
        groups,
        assignments,
        quality,
        args.seed,
        input_hash,
        output_hashes,
        invariants,
    )
    write_json(args.output_summary, summary)
    write_json(
        args.test_lock,
        {
            "created_from_composition_sha256": input_hash,
            "evaluation_role": "FINAL_INTERNAL_TEST",
            "policy": (
                "Internal TEST is frozen and may not be used for model-development decisions."
            ),
            "split_seed": args.seed,
            "strategy_version": STRATEGY_VERSION,
            "test_manifest_sha256": output_hashes["test_manifest"],
            "test_record_count": summary["test_images"],
        },
    )
    return summary


def parse_args() -> argparse.Namespace:
    datasets = PROJECT_ROOT / "training" / "datasets"
    manifests = datasets / "manifests"
    reports = datasets / "reports"
    parser = argparse.ArgumentParser(
        description=(
            "Create the deterministic group-aware 80/10/10 Dataset V2 split "
            "without decoding images or training a model."
        )
    )
    parser.add_argument(
        "--input-manifest",
        type=Path,
        default=manifests / "dataset-v2-39class-combined.csv",
    )
    parser.add_argument(
        "--output-manifest",
        type=Path,
        default=manifests / "dataset-v2-39class-split.csv",
    )
    parser.add_argument(
        "--train-manifest", type=Path, default=manifests / "dataset-v2-train.csv"
    )
    parser.add_argument(
        "--validation-manifest",
        type=Path,
        default=manifests / "dataset-v2-validation.csv",
    )
    parser.add_argument(
        "--test-manifest", type=Path, default=manifests / "dataset-v2-test.csv"
    )
    parser.add_argument(
        "--output-summary",
        type=Path,
        default=reports / "dataset-v2-39class-split-summary.json",
    )
    parser.add_argument(
        "--quality-report",
        type=Path,
        default=reports / "dataset-v2-39class-split-quality.csv",
    )
    parser.add_argument(
        "--test-lock", type=Path, default=reports / "dataset-v2-test-lock.json"
    )
    parser.add_argument("--seed", type=int, default=SPLIT_SEED)
    return parser.parse_args()


def main() -> int:
    summary = run(parse_args())
    print(
        json.dumps(
            {
                "seed": summary["seed"],
                "strategy_version": summary["strategy_version"],
                "total_images": summary["total_images"],
                "train_images": summary["train_images"],
                "validation_images": summary["validation_images"],
                "test_images": summary["test_images"],
                "class_coverage": summary["class_coverage"],
                "group_leakage_count": summary["group_leakage_count"],
                "sha_leakage_count": summary["sha_leakage_count"],
                "benchmark_contamination_count": summary[
                    "benchmark_contamination_count"
                ],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
