"""Fail-closed, VALIDATION-only comparison for Model V2 Experiments A and B.

This module deliberately has no TensorFlow, model-loading, or image-loading
dependency.  It compares persisted prediction indices against the immutable
VALIDATION manifest and never accepts a TEST manifest.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

from training.taxonomy import CLASS_NAMES


EXPERIMENT_A = "agri-diagnose-v2-exp-a"
EXPERIMENT_B = "agri-diagnose-v2-exp-b"
EXPECTED_VALIDATION_COUNT = 7_362
EXPECTED_REAL_WORLD_COUNT = 1_816
BOOTSTRAP_SEED = 20_260_810
BOOTSTRAP_REPETITIONS = 10_000
REAL_WORLD_DOMAIN = "REAL_WORLD"
HISTORICAL_DOMAIN = "HISTORICAL_CONTROLLED"
EXPECTED_REAL_WORLD_CLASS_INDICES = (
    0,
    8,
    9,
    11,
    12,
    21,
    22,
    23,
    26,
    29,
    30,
    31,
    32,
    33,
    37,
)
TOMATO_INDICES = tuple(
    index for index, class_name in enumerate(CLASS_NAMES) if class_name.startswith("Tomato ")
)
POTATO_EARLY_INDEX = CLASS_NAMES.index("Potato Early blight")
POTATO_LATE_INDEX = CLASS_NAMES.index("Potato Late blight")
TEST_PATH_MARKERS = (
    "dataset-v2-test",
    "internal-test",
    "internal_test",
    "plantdoc-test",
    "plantdoc_test",
)
COMPARISON_JSON = "experiment-a-vs-b-validation-comparison.json"
COMPARISON_MARKDOWN = "experiment-a-vs-b-validation-comparison.md"
PER_CLASS_CSV = "experiment-a-vs-b-validation-per-class.csv"
CONFUSION_CSV = "experiment-a-vs-b-validation-confusion-changes.csv"


class ValidationComparisonError(RuntimeError):
    """Raised when comparison inputs violate the VALIDATION-only contract."""


@dataclass(frozen=True)
class ValidationRecord:
    record_id: str
    source_domain: str
    target_index: int


@dataclass(frozen=True)
class ValidationBundle:
    name: str
    root: Path
    loss: float
    accuracy: float
    true_indices: np.ndarray
    predicted_indices: np.ndarray
    metrics: Mapping[str, object]
    manifest_sha256: str
    source_hashes: Mapping[str, str]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_with_canonical_lf(path: Path) -> str:
    """Hash text bytes after canonicalizing line endings only to LF."""
    content = Path(path).read_bytes()
    canonical = content.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return hashlib.sha256(canonical).hexdigest()


def _read_json(path: Path, label: str) -> dict[str, object]:
    if not path.is_file() or path.stat().st_size == 0:
        raise ValidationComparisonError(f"Missing {label}: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValidationComparisonError(f"Invalid {label}: {path}") from exc
    if not isinstance(payload, dict):
        raise ValidationComparisonError(f"{label} must contain a JSON object.")
    return payload


def _require_false(payload: Mapping[str, object], key: str, label: str) -> None:
    if payload.get(key) is not False:
        raise ValidationComparisonError(f"{label} must explicitly set {key}=false.")


def _forbidden_manifest_path(path: Path) -> bool:
    normalized = path.as_posix().casefold()
    return any(marker in normalized for marker in TEST_PATH_MARKERS)


def load_validation_manifest(path: Path) -> tuple[list[ValidationRecord], str]:
    """Load only the authoritative development VALIDATION manifest."""
    path = Path(path)
    if _forbidden_manifest_path(path):
        raise ValidationComparisonError("TEST-like manifest paths are forbidden.")
    if not path.is_file():
        raise ValidationComparisonError(f"VALIDATION manifest is unavailable: {path}")
    required = {
        "composition_record_id",
        "source_domain",
        "target_index",
        "target_class",
        "split",
        "evaluation_role",
    }
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ValidationComparisonError(
                f"VALIDATION manifest is missing fields: {sorted(missing)}"
            )
        rows = list(reader)
    if len(rows) != EXPECTED_VALIDATION_COUNT:
        raise ValidationComparisonError(
            f"VALIDATION manifest must contain {EXPECTED_VALIDATION_COUNT} records."
        )
    records: list[ValidationRecord] = []
    seen: set[str] = set()
    for row in rows:
        record_id = row["composition_record_id"]
        if not record_id or record_id in seen:
            raise ValidationComparisonError("VALIDATION record identifiers must be unique.")
        seen.add(record_id)
        if row["split"] != "VALIDATION":
            raise ValidationComparisonError("Comparison accepts VALIDATION records only.")
        if row["evaluation_role"] != "MODEL_DEVELOPMENT_VALIDATION":
            raise ValidationComparisonError("Invalid VALIDATION evaluation role.")
        try:
            target_index = int(row["target_index"])
        except ValueError as exc:
            raise ValidationComparisonError("Invalid VALIDATION target index.") from exc
        if not 0 <= target_index < len(CLASS_NAMES):
            raise ValidationComparisonError("VALIDATION target index is out of range.")
        if row["target_class"] != CLASS_NAMES[target_index]:
            raise ValidationComparisonError("VALIDATION taxonomy does not match 39 classes.")
        records.append(
            ValidationRecord(record_id, row["source_domain"], target_index)
        )
    if {record.target_index for record in records} != set(range(len(CLASS_NAMES))):
        raise ValidationComparisonError("VALIDATION coverage must remain 39/39.")
    domains = {record.source_domain for record in records}
    if domains != {HISTORICAL_DOMAIN, REAL_WORLD_DOMAIN}:
        raise ValidationComparisonError("VALIDATION domain taxonomy is inconsistent.")
    real_world = [record for record in records if record.source_domain == REAL_WORLD_DOMAIN]
    if len(real_world) != EXPECTED_REAL_WORLD_COUNT or {
        record.target_index for record in real_world
    } != set(EXPECTED_REAL_WORLD_CLASS_INDICES):
        raise ValidationComparisonError(
            "REAL_WORLD VALIDATION composition must remain 1,816 images / 15 classes."
        )
    return records, sha256_with_canonical_lf(path)


def _index_array(value: object, label: str) -> np.ndarray:
    if not isinstance(value, list) or len(value) != EXPECTED_VALIDATION_COUNT:
        raise ValidationComparisonError(
            f"{label} must contain {EXPECTED_VALIDATION_COUNT} indices."
        )
    if any(
        isinstance(item, bool) or not isinstance(item, int) or not 0 <= item < 39
        for item in value
    ):
        raise ValidationComparisonError(f"{label} contains invalid class indices.")
    return np.asarray(value, dtype=np.int32)


def _finite_float(value: object, label: str, *, nonnegative: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValidationComparisonError(f"{label} must be numeric.")
    result = float(value)
    if not math.isfinite(result) or (nonnegative and result < 0):
        raise ValidationComparisonError(f"{label} must be finite and valid.")
    return result


def _metric_block(
    true: np.ndarray,
    predicted: np.ndarray,
    indices: np.ndarray,
    labels: Sequence[int],
) -> dict[str, object]:
    if len(indices) == 0 or not labels:
        raise ValidationComparisonError("Cannot calculate metrics for an empty slice.")
    y_true = true[indices]
    y_pred = predicted[indices]
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=list(labels), zero_division=0
    )
    macro = precision_recall_fscore_support(
        y_true, y_pred, labels=list(labels), average="macro", zero_division=0
    )
    weighted = precision_recall_fscore_support(
        y_true, y_pred, labels=list(labels), average="weighted", zero_division=0
    )
    return {
        "image_count": int(len(indices)),
        "supported_class_count": len(labels),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(macro[0]),
        "macro_recall": float(macro[1]),
        "macro_f1": float(macro[2]),
        "weighted_precision": float(weighted[0]),
        "weighted_recall": float(weighted[1]),
        "weighted_f1": float(weighted[2]),
        "per_class": [
            {
                "target_index": int(target),
                "target_class": CLASS_NAMES[target],
                "precision": float(precision[offset]),
                "recall": float(recall[offset]),
                "f1": float(f1[offset]),
                "support": int(support[offset]),
            }
            for offset, target in enumerate(labels)
        ],
    }


def _assert_close(actual: object, expected: float, label: str) -> None:
    number = _finite_float(actual, label)
    if not math.isclose(number, expected, rel_tol=1e-7, abs_tol=1e-7):
        raise ValidationComparisonError(f"Stored {label} is inconsistent with predictions.")


def _validate_stored_block(
    stored: object, calculated: Mapping[str, object], label: str
) -> None:
    if not isinstance(stored, dict):
        raise ValidationComparisonError(f"Missing stored {label} metrics.")
    for key in ("image_count", "supported_class_count"):
        if stored.get(key) != calculated[key]:
            raise ValidationComparisonError(f"Stored {label} {key} is inconsistent.")
    for key in (
        "accuracy",
        "macro_precision",
        "macro_recall",
        "macro_f1",
        "weighted_precision",
        "weighted_recall",
        "weighted_f1",
    ):
        _assert_close(stored.get(key), float(calculated[key]), f"{label}.{key}")
    stored_rows = stored.get("per_class")
    expected_rows = calculated["per_class"]
    if not isinstance(stored_rows, list) or len(stored_rows) != len(expected_rows):
        raise ValidationComparisonError(f"Stored {label} per-class metrics are incomplete.")
    for stored_row, expected_row in zip(stored_rows, expected_rows):
        if not isinstance(stored_row, dict):
            raise ValidationComparisonError(f"Stored {label} per-class row is invalid.")
        for key in ("target_index", "target_class", "support"):
            if stored_row.get(key) != expected_row[key]:
                raise ValidationComparisonError(
                    f"Stored {label} per-class identity/support is inconsistent."
                )
        for key in ("precision", "recall", "f1"):
            _assert_close(
                stored_row.get(key),
                float(expected_row[key]),
                f"{label}.per_class.{key}",
            )


def load_validation_bundle(
    root: Path,
    expected_experiment: str,
    records: Sequence[ValidationRecord],
    manifest_sha256: str,
) -> ValidationBundle:
    root = Path(root).resolve()
    suffix = "a" if expected_experiment == EXPERIMENT_A else "b"
    files = {
        "metrics": root / "validation-metrics.json",
        "experiment": root / "experiment.json",
        "preflight": root / "preflight.json",
        "summary": root / f"model-v2-exp-{suffix}-summary.json",
    }
    payloads = {
        name: _read_json(path, f"Experiment {suffix.upper()} {name}")
        for name, path in files.items()
    }
    experiment = payloads["experiment"]
    preflight = payloads["preflight"]
    summary = payloads["summary"]
    if experiment.get("experiment") != expected_experiment:
        raise ValidationComparisonError("Experiment artifact identity mismatch.")
    if experiment.get("validation_manifest_sha256") != manifest_sha256:
        raise ValidationComparisonError("VALIDATION manifest hash mismatch.")
    for payload, label in ((experiment, "experiment"), (preflight, "preflight")):
        _require_false(payload, "internal_test_loaded", label)
        _require_false(payload, "plantdoc_test_loaded", label)
    _require_false(summary, "test_sets_evaluated", "summary")

    metrics = payloads["metrics"]
    true = _index_array(metrics.get("true_indices"), "true_indices")
    predicted = _index_array(metrics.get("predicted_indices"), "predicted_indices")
    manifest_true = np.asarray([record.target_index for record in records], dtype=np.int32)
    if not np.array_equal(true, manifest_true):
        raise ValidationComparisonError(
            "Prediction order/targets do not match the immutable VALIDATION manifest."
        )
    overall_indices = np.arange(len(records), dtype=np.int32)
    real_indices = np.asarray(
        [index for index, record in enumerate(records) if record.source_domain == REAL_WORLD_DOMAIN],
        dtype=np.int32,
    )
    if len(real_indices) == 0:
        raise ValidationComparisonError("REAL_WORLD VALIDATION slice is unavailable.")
    real_labels = sorted(set(int(true[index]) for index in real_indices))
    calculated_overall = _metric_block(true, predicted, overall_indices, list(range(39)))
    calculated_real = _metric_block(true, predicted, real_indices, real_labels)
    _validate_stored_block(metrics.get("overall_validation"), calculated_overall, "overall")
    _validate_stored_block(metrics.get("real_world_validation"), calculated_real, "real_world")
    accuracy = _finite_float(metrics.get("accuracy"), "accuracy")
    if not math.isclose(
        accuracy, float(calculated_overall["accuracy"]), rel_tol=1e-6, abs_tol=1e-6
    ):
        raise ValidationComparisonError("Top-level VALIDATION accuracy is inconsistent.")
    loss = _finite_float(metrics.get("loss"), "loss", nonnegative=True)
    return ValidationBundle(
        expected_experiment,
        root,
        loss,
        accuracy,
        true,
        predicted,
        metrics,
        manifest_sha256,
        {name: sha256_file(path) for name, path in files.items()},
    )


def _delta_metrics(a: Mapping[str, object], b: Mapping[str, object]) -> dict[str, float]:
    return {
        key: float(b[key]) - float(a[key])
        for key in (
            "accuracy",
            "macro_precision",
            "macro_recall",
            "macro_f1",
            "weighted_precision",
            "weighted_recall",
            "weighted_f1",
        )
    }


def _per_class_deltas(
    a: Mapping[str, object], b: Mapping[str, object], partition: str
) -> list[dict[str, object]]:
    a_rows = {int(row["target_index"]): row for row in a["per_class"]}
    b_rows = {int(row["target_index"]): row for row in b["per_class"]}
    if set(a_rows) != set(b_rows):
        raise ValidationComparisonError("A/B per-class label support differs.")
    rows: list[dict[str, object]] = []
    for index in sorted(a_rows):
        left, right = a_rows[index], b_rows[index]
        if left["support"] != right["support"]:
            raise ValidationComparisonError("A/B per-class support differs.")
        rows.append(
            {
                "partition": partition,
                "target_index": index,
                "target_class": CLASS_NAMES[index],
                "support": int(left["support"]),
                **{
                    f"a_{metric}": float(left[metric])
                    for metric in ("precision", "recall", "f1")
                },
                **{
                    f"b_{metric}": float(right[metric])
                    for metric in ("precision", "recall", "f1")
                },
                **{
                    f"delta_{metric}": float(right[metric]) - float(left[metric])
                    for metric in ("precision", "recall", "f1")
                },
            }
        )
    return rows


def _tomato_metrics(
    true: np.ndarray, predicted: np.ndarray, partition_indices: np.ndarray
) -> dict[str, object]:
    tomato = np.asarray(
        [index for index in partition_indices if int(true[index]) in TOMATO_INDICES],
        dtype=np.int32,
    )
    labels = sorted(set(int(true[index]) for index in tomato))
    return _metric_block(true, predicted, tomato, labels)


def _potato_confusion(
    true: np.ndarray, predicted: np.ndarray, indices: np.ndarray
) -> dict[str, object]:
    y_true, y_pred = true[indices], predicted[indices]
    early_support = int(np.sum(y_true == POTATO_EARLY_INDEX))
    late_support = int(np.sum(y_true == POTATO_LATE_INDEX))
    if not early_support or not late_support:
        raise ValidationComparisonError("Potato Early/Late support is required.")
    early_late = int(
        np.sum((y_true == POTATO_EARLY_INDEX) & (y_pred == POTATO_LATE_INDEX))
    )
    late_early = int(
        np.sum((y_true == POTATO_LATE_INDEX) & (y_pred == POTATO_EARLY_INDEX))
    )
    return {
        "early_support": early_support,
        "late_support": late_support,
        "early_to_late_count": early_late,
        "early_to_late_rate": early_late / early_support,
        "late_to_early_count": late_early,
        "late_to_early_rate": late_early / late_support,
        "bidirectional_count": early_late + late_early,
        "bidirectional_rate": (early_late + late_early) / (early_support + late_support),
    }


def _numeric_delta(a: Mapping[str, object], b: Mapping[str, object]) -> dict[str, float]:
    return {
        key: float(b[key]) - float(value)
        for key, value in a.items()
        if key in b and isinstance(value, (int, float)) and not key.endswith("support")
    }


def _confusion_changes(
    true: np.ndarray,
    predicted_a: np.ndarray,
    predicted_b: np.ndarray,
    indices: np.ndarray,
    partition: str,
) -> list[dict[str, object]]:
    labels = list(range(39))
    matrix_a = confusion_matrix(true[indices], predicted_a[indices], labels=labels)
    matrix_b = confusion_matrix(true[indices], predicted_b[indices], labels=labels)
    support = np.bincount(true[indices], minlength=39)
    rows: list[dict[str, object]] = []
    for source in labels:
        if not support[source]:
            continue
        for target in labels:
            if source == target:
                continue
            count_a, count_b = int(matrix_a[source, target]), int(matrix_b[source, target])
            if count_a == count_b:
                continue
            rate_a = count_a / int(support[source])
            rate_b = count_b / int(support[source])
            rows.append(
                {
                    "partition": partition,
                    "true_index": source,
                    "true_class": CLASS_NAMES[source],
                    "predicted_index": target,
                    "predicted_class": CLASS_NAMES[target],
                    "true_support": int(support[source]),
                    "a_count": count_a,
                    "b_count": count_b,
                    "delta_count": count_b - count_a,
                    "a_rate": rate_a,
                    "b_rate": rate_b,
                    "delta_rate": rate_b - rate_a,
                    "direction": "increased" if count_b > count_a else "decreased",
                }
            )
    return sorted(
        rows,
        key=lambda row: (
            -abs(float(row["delta_rate"])),
            -abs(int(row["delta_count"])),
            int(row["true_index"]),
            int(row["predicted_index"]),
        ),
    )


def _fast_macro_f1(true: np.ndarray, predicted: np.ndarray, labels: Sequence[int]) -> float:
    matrix = confusion_matrix(true, predicted, labels=list(range(39)))
    selected = np.asarray(labels, dtype=np.int32)
    true_positive = np.diag(matrix)[selected].astype(np.float64)
    precision_denominator = matrix[:, selected].sum(axis=0)
    recall_denominator = matrix[selected, :].sum(axis=1)
    precision = np.divide(
        true_positive,
        precision_denominator,
        out=np.zeros_like(true_positive),
        where=precision_denominator != 0,
    )
    recall = np.divide(
        true_positive,
        recall_denominator,
        out=np.zeros_like(true_positive),
        where=recall_denominator != 0,
    )
    return float(
        np.mean(
            np.divide(
                2 * precision * recall,
                precision + recall,
                out=np.zeros_like(precision),
                where=(precision + recall) != 0,
            )
        )
    )


def paired_class_aware_bootstrap(
    true: np.ndarray,
    predicted_a: np.ndarray,
    predicted_b: np.ndarray,
    indices: np.ndarray,
    *,
    repetitions: int = BOOTSTRAP_REPETITIONS,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, object]:
    """Paired bootstrap that preserves the exact true-class support vector."""
    if isinstance(repetitions, bool) or not isinstance(repetitions, int) or repetitions <= 0:
        raise ValidationComparisonError("Bootstrap repetitions must be a positive integer.")
    if seed != BOOTSTRAP_SEED:
        raise ValidationComparisonError(
            f"Bootstrap seed is locked to {BOOTSTRAP_SEED}."
        )
    labels = sorted(set(int(true[index]) for index in indices))
    strata = [indices[true[indices] == label] for label in labels]
    if any(len(stratum) == 0 for stratum in strata):
        raise ValidationComparisonError("Bootstrap class strata must not be empty.")
    rng = np.random.default_rng(seed)
    macro_deltas = np.empty(repetitions, dtype=np.float64)
    accuracy_deltas = np.empty(repetitions, dtype=np.float64)
    for iteration in range(repetitions):
        sampled = np.concatenate(
            [rng.choice(stratum, size=len(stratum), replace=True) for stratum in strata]
        )
        sampled_true = true[sampled]
        sampled_a = predicted_a[sampled]
        sampled_b = predicted_b[sampled]
        macro_deltas[iteration] = _fast_macro_f1(
            sampled_true, sampled_b, labels
        ) - _fast_macro_f1(sampled_true, sampled_a, labels)
        accuracy_deltas[iteration] = float(np.mean(sampled_b == sampled_true)) - float(
            np.mean(sampled_a == sampled_true)
        )

    def summarize(values: np.ndarray, point: float) -> dict[str, float]:
        lower, upper = np.quantile(values, (0.025, 0.975))
        return {
            "point_delta_b_minus_a": point,
            "bootstrap_mean_delta": float(np.mean(values)),
            "ci95_lower": float(lower),
            "ci95_upper": float(upper),
        }

    point_macro = _fast_macro_f1(true[indices], predicted_b[indices], labels) - _fast_macro_f1(
        true[indices], predicted_a[indices], labels
    )
    point_accuracy = float(np.mean(predicted_b[indices] == true[indices])) - float(
        np.mean(predicted_a[indices] == true[indices])
    )
    return {
        "method": "paired true-class-stratified bootstrap with replacement",
        "seed": seed,
        "repetitions": repetitions,
        "image_count": int(len(indices)),
        "class_support": {
            CLASS_NAMES[label]: int(len(stratum))
            for label, stratum in zip(labels, strata)
        },
        "macro_f1": summarize(macro_deltas, point_macro),
        "accuracy": summarize(accuracy_deltas, point_accuracy),
    }


def build_validation_comparison(
    experiment_a_dir: Path,
    experiment_b_dir: Path,
    validation_manifest: Path,
    *,
    bootstrap_repetitions: int = BOOTSTRAP_REPETITIONS,
) -> tuple[dict[str, object], list[dict[str, object]], list[dict[str, object]]]:
    records, manifest_hash = load_validation_manifest(validation_manifest)
    a = load_validation_bundle(experiment_a_dir, EXPERIMENT_A, records, manifest_hash)
    b = load_validation_bundle(experiment_b_dir, EXPERIMENT_B, records, manifest_hash)
    if a.root == b.root:
        raise ValidationComparisonError("Experiment A and B artifact roots must differ.")
    if not np.array_equal(a.true_indices, b.true_indices):
        raise ValidationComparisonError("A/B VALIDATION targets are not paired.")
    true = a.true_indices
    overall_indices = np.arange(len(records), dtype=np.int32)
    real_indices = np.asarray(
        [index for index, record in enumerate(records) if record.source_domain == REAL_WORLD_DOMAIN],
        dtype=np.int32,
    )
    labels_real = sorted(set(int(true[index]) for index in real_indices))
    overall_a = _metric_block(true, a.predicted_indices, overall_indices, list(range(39)))
    overall_b = _metric_block(true, b.predicted_indices, overall_indices, list(range(39)))
    real_a = _metric_block(true, a.predicted_indices, real_indices, labels_real)
    real_b = _metric_block(true, b.predicted_indices, real_indices, labels_real)
    overall_delta = _delta_metrics(overall_a, overall_b)
    overall_delta["loss"] = b.loss - a.loss
    real_delta = _delta_metrics(real_a, real_b)
    gap_a = float(overall_a["macro_f1"]) - float(real_a["macro_f1"])
    gap_b = float(overall_b["macro_f1"]) - float(real_b["macro_f1"])

    per_class = [
        *_per_class_deltas(overall_a, overall_b, "OVERALL_VALIDATION"),
        *_per_class_deltas(real_a, real_b, "REAL_WORLD_VALIDATION"),
    ]
    tomato_overall_a = _tomato_metrics(true, a.predicted_indices, overall_indices)
    tomato_overall_b = _tomato_metrics(true, b.predicted_indices, overall_indices)
    tomato_real_a = _tomato_metrics(true, a.predicted_indices, real_indices)
    tomato_real_b = _tomato_metrics(true, b.predicted_indices, real_indices)

    potato: dict[str, object] = {}
    for name, indices in (
        ("overall_validation", overall_indices),
        ("real_world_validation", real_indices),
    ):
        left = _potato_confusion(true, a.predicted_indices, indices)
        right = _potato_confusion(true, b.predicted_indices, indices)
        potato[name] = {"experiment_a": left, "experiment_b": right, "delta": _numeric_delta(left, right)}

    confusion_rows = [
        *_confusion_changes(
            true, a.predicted_indices, b.predicted_indices, overall_indices, "OVERALL_VALIDATION"
        ),
        *_confusion_changes(
            true, a.predicted_indices, b.predicted_indices, real_indices, "REAL_WORLD_VALIDATION"
        ),
    ]
    change_key = lambda row: (
        -abs(float(row["delta_rate"])),
        -abs(int(row["delta_count"])),
        str(row["partition"]),
        int(row["true_index"]),
        int(row["predicted_index"]),
    )
    largest_increases = sorted(
        (row for row in confusion_rows if row["delta_count"] > 0), key=change_key
    )[:15]
    largest_decreases = sorted(
        (row for row in confusion_rows if row["delta_count"] < 0), key=change_key
    )[:15]

    comparison = {
        "schema_version": 1,
        "comparison": "agri-diagnose-v2-exp-a-vs-b-validation-only",
        "baseline_experiment": EXPERIMENT_A,
        "candidate_experiment": EXPERIMENT_B,
        "validation_manifest_sha256": manifest_hash,
        "validation_image_count": len(records),
        "real_world_validation_image_count": int(len(real_indices)),
        "overall_validation": {
            "experiment_a": {**overall_a, "loss": a.loss},
            "experiment_b": {**overall_b, "loss": b.loss},
            "delta_b_minus_a": overall_delta,
        },
        "real_world_validation": {
            "experiment_a": real_a,
            "experiment_b": real_b,
            "delta_b_minus_a": real_delta,
        },
        "generalization_gap_macro_f1": {
            "experiment_a": gap_a,
            "experiment_b": gap_b,
            "delta_b_minus_a": gap_b - gap_a,
            "gap_reduction_a_minus_b": gap_a - gap_b,
        },
        "tomato_aggregate": {
            "definition": "true-Tomato disease-classification slice; unsupported classes are excluded from each macro average",
            "overall_validation": {
                "experiment_a": tomato_overall_a,
                "experiment_b": tomato_overall_b,
                "delta_b_minus_a": _delta_metrics(tomato_overall_a, tomato_overall_b),
            },
            "real_world_validation": {
                "experiment_a": tomato_real_a,
                "experiment_b": tomato_real_b,
                "delta_b_minus_a": _delta_metrics(tomato_real_a, tomato_real_b),
            },
        },
        "potato_early_late_bidirectional_confusion": potato,
        "major_confusion_pair_changes": {
            "ranking": "absolute true-class error-rate delta, then absolute count delta",
            "largest_increases": largest_increases,
            "largest_decreases": largest_decreases,
        },
        "paired_class_aware_bootstrap": {
            "overall_validation": paired_class_aware_bootstrap(
                true,
                a.predicted_indices,
                b.predicted_indices,
                overall_indices,
                repetitions=bootstrap_repetitions,
                seed=BOOTSTRAP_SEED,
            ),
            "real_world_validation": paired_class_aware_bootstrap(
                true,
                a.predicted_indices,
                b.predicted_indices,
                real_indices,
                repetitions=bootstrap_repetitions,
                seed=BOOTSTRAP_SEED,
            ),
        },
        "safety": {
            "partition": "VALIDATION_ONLY",
            "internal_test_loaded": False,
            "plantdoc_test_loaded": False,
            "test_sets_evaluated": False,
            "models_loaded": False,
            "images_loaded": False,
            "inference_performed": False,
        },
    }
    return comparison, per_class, confusion_rows


def _write_json(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_csv(
    path: Path,
    rows: Sequence[Mapping[str, object]],
    *,
    fieldnames: Sequence[str] | None = None,
) -> None:
    columns = list(fieldnames or (list(rows[0]) if rows else ()))
    if not columns:
        raise ValidationComparisonError(f"CSV schema is unavailable: {path.name}")
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _markdown_report(comparison: Mapping[str, object]) -> str:
    overall = comparison["overall_validation"]
    real = comparison["real_world_validation"]
    bootstrap = comparison["paired_class_aware_bootstrap"]["real_world_validation"]["macro_f1"]
    return (
        "# Model V2 Experiment A vs B — VALIDATION only\n\n"
        "No INTERNAL TEST or PlantDoc TEST data was loaded or evaluated.\n\n"
        "| Metric | Experiment A | Experiment B | Delta B - A |\n"
        "|---|---:|---:|---:|\n"
        f"| Overall Macro-F1 | {overall['experiment_a']['macro_f1']:.6f} | "
        f"{overall['experiment_b']['macro_f1']:.6f} | {overall['delta_b_minus_a']['macro_f1']:+.6f} |\n"
        f"| Overall accuracy | {overall['experiment_a']['accuracy']:.6f} | "
        f"{overall['experiment_b']['accuracy']:.6f} | {overall['delta_b_minus_a']['accuracy']:+.6f} |\n"
        f"| VALIDATION loss | {overall['experiment_a']['loss']:.6f} | "
        f"{overall['experiment_b']['loss']:.6f} | {overall['delta_b_minus_a']['loss']:+.6f} |\n"
        f"| Real-world Macro-F1 | {real['experiment_a']['macro_f1']:.6f} | "
        f"{real['experiment_b']['macro_f1']:.6f} | {real['delta_b_minus_a']['macro_f1']:+.6f} |\n"
        f"| Real-world accuracy | {real['experiment_a']['accuracy']:.6f} | "
        f"{real['experiment_b']['accuracy']:.6f} | {real['delta_b_minus_a']['accuracy']:+.6f} |\n\n"
        "## Paired class-aware bootstrap\n\n"
        f"Real-world Macro-F1 delta 95% CI: [{bootstrap['ci95_lower']:.6f}, "
        f"{bootstrap['ci95_upper']:.6f}] using "
        f"{comparison['paired_class_aware_bootstrap']['real_world_validation']['repetitions']} "
        "fixed-seed paired resamples. This is VALIDATION uncertainty analysis, not TEST evaluation.\n"
    )


def write_validation_comparison(
    experiment_a_dir: Path,
    experiment_b_dir: Path,
    validation_manifest: Path,
    output_dir: Path,
    *,
    bootstrap_repetitions: int = BOOTSTRAP_REPETITIONS,
) -> dict[str, Path]:
    """Validate, compare, and persist JSON/Markdown/CSV artifacts atomically."""
    a_root = Path(experiment_a_dir).resolve()
    output = Path(output_dir).resolve()
    if output == a_root or a_root in output.parents:
        raise ValidationComparisonError("Comparison outputs must not modify Experiment A.")
    if not a_root.is_dir():
        raise ValidationComparisonError(f"Experiment A artifact root is unavailable: {a_root}")
    before = {
        path: sha256_file(path)
        for path in a_root.iterdir()
        if path.is_file()
    }
    comparison, per_class, confusion = build_validation_comparison(
        experiment_a_dir,
        experiment_b_dir,
        validation_manifest,
        bootstrap_repetitions=bootstrap_repetitions,
    )
    output.mkdir(parents=True, exist_ok=True)
    paths = {
        "json": output / COMPARISON_JSON,
        "markdown": output / COMPARISON_MARKDOWN,
        "per_class_csv": output / PER_CLASS_CSV,
        "confusion_csv": output / CONFUSION_CSV,
    }
    _write_json(paths["json"], comparison)
    temporary = paths["markdown"].with_suffix(".md.tmp")
    temporary.write_text(_markdown_report(comparison), encoding="utf-8")
    temporary.replace(paths["markdown"])
    _write_csv(paths["per_class_csv"], per_class)
    _write_csv(
        paths["confusion_csv"],
        confusion,
        fieldnames=(
            "partition",
            "true_index",
            "true_class",
            "predicted_index",
            "predicted_class",
            "true_support",
            "a_count",
            "b_count",
            "delta_count",
            "a_rate",
            "b_rate",
            "delta_rate",
            "direction",
        ),
    )
    after = {
        path: sha256_file(path)
        for path in a_root.iterdir()
        if path.is_file()
    }
    if before != after:
        raise ValidationComparisonError("Experiment A artifacts changed during comparison.")
    return paths
