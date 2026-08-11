from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training.taxonomy import CLASS_NAMES  # noqa: E402
from training.data_pipeline import (  # noqa: E402
    DEFAULT_POLICY_PATH,
    SOURCE_ROOT_ENV,
    ManifestRecord,
    TrainingPolicyError,
    configured_source_roots,
    count_records,
    load_development_manifest,
    load_policy,
    preprocess_image,
    resolve_record_path,
)


DEFAULT_TRAIN_MANIFEST = (
    PROJECT_ROOT / "training" / "datasets" / "manifests" / "dataset-v2-train.csv"
)
DEFAULT_VALIDATION_MANIFEST = (
    PROJECT_ROOT
    / "training"
    / "datasets"
    / "manifests"
    / "dataset-v2-validation.csv"
)
DEFAULT_WEIGHT_REPORT = (
    PROJECT_ROOT
    / "training"
    / "datasets"
    / "reports"
    / "model-v2-class-weight-analysis.csv"
)
DEFAULT_POLICY_SUMMARY = (
    PROJECT_ROOT
    / "training"
    / "datasets"
    / "reports"
    / "model-v2-training-policy-summary.json"
)
WEIGHT_FIELDS = [
    "target_index",
    "target_class",
    "train_count",
    "relative_frequency",
    "unweighted_weight",
    "inverse_sqrt_candidate",
    "median_sqrt_candidate",
    "normalized_weight",
    "clipped_recommended_weight",
]


def stable_hash(seed: int, value: str) -> str:
    return hashlib.sha256(f"{seed}\0{value}".encode("utf-8")).hexdigest()


def write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=WEIGHT_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def analyze_class_weights(
    train_records: Sequence[ManifestRecord],
    *,
    clip_minimum: float = 0.5,
    clip_maximum: float = 3.0,
) -> tuple[list[dict[str, object]], dict[str, float | int]]:
    if not train_records or {record.split for record in train_records} != {"TRAIN"}:
        raise TrainingPolicyError("Class weights must be calculated from TRAIN records only.")
    counts = Counter(record.target_index for record in train_records)
    if set(counts) != set(range(len(CLASS_NAMES))):
        missing = sorted(set(range(len(CLASS_NAMES))) - set(counts))
        raise TrainingPolicyError(f"TRAIN class-weight analysis is missing indices: {missing}")
    if not 0 < clip_minimum <= clip_maximum:
        raise TrainingPolicyError("Invalid class-weight clipping range.")

    total = len(train_records)
    class_counts = [counts[index] for index in range(len(CLASS_NAMES))]
    median_count = float(statistics.median(class_counts))
    mean_count = float(statistics.mean(class_counts))
    median_raw = {
        index: math.sqrt(median_count / counts[index]) for index in range(len(CLASS_NAMES))
    }
    normalization_factor = total / sum(
        counts[index] * median_raw[index] for index in range(len(CLASS_NAMES))
    )

    rows: list[dict[str, object]] = []
    for index, target_class in enumerate(CLASS_NAMES):
        count = counts[index]
        normalized = median_raw[index] * normalization_factor
        clipped = min(clip_maximum, max(clip_minimum, normalized))
        rows.append(
            {
                "target_index": index,
                "target_class": target_class,
                "train_count": count,
                "relative_frequency": round(count / total, 10),
                "unweighted_weight": 1.0,
                "inverse_sqrt_candidate": round(math.sqrt(total / count), 8),
                "median_sqrt_candidate": round(median_raw[index], 8),
                "normalized_weight": round(normalized, 8),
                "clipped_recommended_weight": round(clipped, 8),
            }
        )

    weighted_mean = sum(
        row["train_count"] * row["clipped_recommended_weight"] for row in rows
    ) / total
    stats: dict[str, float | int] = {
        "train_count": total,
        "minimum_class_count": min(class_counts),
        "maximum_class_count": max(class_counts),
        "median_class_count": median_count,
        "mean_class_count": round(mean_count, 8),
        "maximum_to_minimum_ratio": round(max(class_counts) / min(class_counts), 8),
        "normalization_factor": round(normalization_factor, 8),
        "recommended_sample_weighted_mean": round(weighted_mean, 8),
        "clip_minimum": clip_minimum,
        "clip_maximum": clip_maximum,
    }
    return rows, stats


def inspect_available_samples(
    records: Sequence[ManifestRecord],
    policy: Mapping[str, object],
    source_roots: Mapping[str, Path],
    sample_size: int,
) -> dict[str, object]:
    seed = int(policy["experiment_seed"])
    eligible: list[tuple[ManifestRecord, Path]] = []
    for record in records:
        if record.source_dataset not in source_roots:
            continue
        path = resolve_record_path(record, source_roots)
        if path.is_file():
            eligible.append((record, path))
    selected = sorted(
        eligible,
        key=lambda item: (
            stable_hash(seed, item[0].composition_record_id),
            item[0].composition_record_id,
        ),
    )[:sample_size]
    inspected: list[dict[str, object]] = []
    for record, path in selected:
        image = preprocess_image(path, policy)
        if image.shape != (224, 224, 3):
            raise TrainingPolicyError(f"Unexpected preprocessed shape: {image.shape}")
        if str(image.dtype) != "float32":
            raise TrainingPolicyError(f"Unexpected preprocessed dtype: {image.dtype}")
        minimum = float(image.min())
        maximum = float(image.max())
        if minimum < 0.0 or maximum > 1.0:
            raise TrainingPolicyError(f"Preprocessed range is outside [0,1]: {path.name}")
        inspected.append(
            {
                "composition_record_id": record.composition_record_id,
                "source_dataset": record.source_dataset,
                "split": record.split,
                "target_index": record.target_index,
                "shape": list(image.shape),
                "dtype": str(image.dtype),
                "minimum": round(minimum, 8),
                "maximum": round(maximum, 8),
            }
        )
    return {
        "eligible_local_images": len(eligible),
        "inspected_image_count": len(inspected),
        "sample_preprocessing_status": (
            "PASSED" if inspected else "SKIPPED_NO_CONFIGURED_LOCAL_IMAGES"
        ),
        "samples": inspected,
    }


def build_policy_summary(
    train_records: Sequence[ManifestRecord],
    validation_records: Sequence[ManifestRecord],
    policy: Mapping[str, object],
    weight_stats: Mapping[str, float | int],
) -> dict[str, object]:
    train_counts = count_records(train_records)
    validation_counts = count_records(validation_records)
    return {
        "policy_version": policy["policy_version"],
        "experiment_seed": policy["experiment_seed"],
        "train_count": train_counts["total"],
        "validation_count": validation_counts["total"],
        "historical_train_count": train_counts["domain_counts"].get(
            "HISTORICAL_CONTROLLED", 0
        ),
        "real_world_train_count": train_counts["domain_counts"].get("REAL_WORLD", 0),
        "class_count": len(CLASS_NAMES),
        "train_min_class_count": weight_stats["minimum_class_count"],
        "train_max_class_count": weight_stats["maximum_class_count"],
        "train_median_class_count": weight_stats["median_class_count"],
        "train_mean_class_count": weight_stats["mean_class_count"],
        "train_maximum_to_minimum_ratio": weight_stats[
            "maximum_to_minimum_ratio"
        ],
        "train_counts_by_class": train_counts["class_counts"],
        "train_counts_by_domain": train_counts["domain_counts"],
        "train_counts_by_source": train_counts["source_counts"],
        "recommended_weight_method": policy["class_weight_policy"][
            "recommended_method"
        ],
        "recommended_weight_sample_weighted_mean": weight_stats[
            "recommended_sample_weighted_mean"
        ],
        "augmentation_policy": policy["augmentation"],
        "architecture": policy["architecture"],
        "transfer_learning_phases": {
            "phase1": policy["phase1"],
            "phase2": policy["phase2"],
        },
        "model_selection_metric": policy["selection_metrics"]["primary"],
        "locked_test_confirmed": policy["locked_test_policy"][
            "development_loading_forbidden"
        ],
        "plantdoc_locked_confirmed": policy["locked_test_policy"][
            "plantdoc_test_locked"
        ],
        "confidence_threshold_changed": False,
        "training_performed": False,
        "test_evaluation_performed": False,
    }


def run_dry_run(
    args: argparse.Namespace,
    *,
    source_root_overrides: Mapping[str, Path | str] | None = None,
) -> dict[str, object]:
    policy = load_policy(args.policy)
    train_records = load_development_manifest(args.train_manifest, training=True)
    validation_records = load_development_manifest(
        args.validation_manifest, training=False
    )
    if {record.target_index for record in train_records} != set(range(39)):
        raise TrainingPolicyError("TRAIN does not cover target indices 0..38.")
    if {record.target_index for record in validation_records} != set(range(39)):
        raise TrainingPolicyError("VALIDATION does not cover target indices 0..38.")

    clipping = policy["class_weight_policy"]["clipping"]
    weight_rows, weight_stats = analyze_class_weights(
        train_records,
        clip_minimum=float(clipping["minimum"]),
        clip_maximum=float(clipping["maximum"]),
    )
    summary = build_policy_summary(
        train_records, validation_records, policy, weight_stats
    )
    if not args.no_write_reports:
        write_csv(args.weight_report, weight_rows)
        write_json(args.policy_summary, summary)

    roots = configured_source_roots(source_root_overrides)
    sample_audit = inspect_available_samples(
        [*train_records, *validation_records],
        policy,
        roots,
        args.sample_size,
    )
    configured = sorted(source for source, root in roots.items() if root.is_dir())
    missing = sorted(
        {
            record.source_dataset
            for record in [*train_records, *validation_records]
            if record.source_dataset not in roots or not roots[record.source_dataset].is_dir()
        }
    )
    return {
        "dry_run": True,
        "policy_version": policy["policy_version"],
        "train_count": len(train_records),
        "validation_count": len(validation_records),
        "train_class_coverage": len({record.target_index for record in train_records}),
        "validation_class_coverage": len(
            {record.target_index for record in validation_records}
        ),
        "class_weights_validated_from": "TRAIN_ONLY",
        "recommended_weight_sample_weighted_mean": weight_stats[
            "recommended_sample_weighted_mean"
        ],
        "configured_local_sources": configured,
        "unavailable_local_sources": [
            {"source_dataset": source, "environment_variable": SOURCE_ROOT_ENV[source]}
            for source in missing
        ],
        "sample_audit": sample_audit,
        "internal_test_loaded": False,
        "plantdoc_test_loaded": False,
        "training_performed": False,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate Model V2 policy and development manifests without training."
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY_PATH)
    parser.add_argument("--train-manifest", type=Path, default=DEFAULT_TRAIN_MANIFEST)
    parser.add_argument(
        "--validation-manifest", type=Path, default=DEFAULT_VALIDATION_MANIFEST
    )
    parser.add_argument("--weight-report", type=Path, default=DEFAULT_WEIGHT_REPORT)
    parser.add_argument("--policy-summary", type=Path, default=DEFAULT_POLICY_SUMMARY)
    parser.add_argument("--sample-size", type=int, default=8)
    parser.add_argument("--no-write-reports", action="store_true")
    args = parser.parse_args()
    if not args.dry_run:
        parser.error("Only --dry-run is available in Step 5G; training is forbidden.")
    if args.sample_size < 0:
        parser.error("--sample-size must be non-negative.")
    return args


def main() -> int:
    result = run_dry_run(parse_args())
    print(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
