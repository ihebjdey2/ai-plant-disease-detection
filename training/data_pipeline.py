from __future__ import annotations

import csv
import json
import os
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
from PIL import Image, UnidentifiedImageError

from app.taxonomy import CLASS_NAMES


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_POLICY_PATH = PROJECT_ROOT / "training" / "config" / "model-v2-training-policy.json"
ALLOWED_DEVELOPMENT_SPLITS = {"TRAIN", "VALIDATION"}
SOURCE_ROOT_ENV = {
    "Historical Mendeley 39-class source": "AGRIDIAGNOSE_HISTORICAL_ROOT",
    "PLDD-UP": "AGRIDIAGNOSE_PLDD_UP_ROOT",
    "Seasonal Corn Leaf Disease Dataset": "AGRIDIAGNOSE_SEASONAL_CORN_ROOT",
    "PlantDoc": "AGRIDIAGNOSE_PLANTDOC_TRAIN_ROOT",
    "Potato Leaf Disease Dataset": "AGRIDIAGNOSE_BANU_DEB_ROOT",
}
REQUIRED_MANIFEST_FIELDS = {
    "composition_record_id",
    "source_domain",
    "source_dataset",
    "source_path",
    "target_index",
    "target_class",
    "split",
    "evaluation_role",
}


class TrainingPolicyError(RuntimeError):
    """Raised when a development input violates the approved Model V2 policy."""


@dataclass(frozen=True)
class ManifestRecord:
    composition_record_id: str
    source_domain: str
    source_dataset: str
    source_path: str
    target_index: int
    target_class: str
    split: str
    evaluation_role: str


def load_policy(path: Path = DEFAULT_POLICY_PATH) -> dict:
    if not path.is_file():
        raise TrainingPolicyError(f"Training-policy config is unavailable: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    validate_policy(payload)
    return payload


def validate_policy(policy: Mapping[str, object]) -> None:
    required = {
        "policy_version",
        "experiment_seed",
        "input",
        "architecture",
        "augmentation",
        "class_weight_policy",
        "phase1",
        "phase2",
        "optimizer",
        "loss",
        "batch_size",
        "callbacks",
        "selection_metrics",
        "locked_test_policy",
    }
    missing = required - set(policy)
    if missing:
        raise TrainingPolicyError(f"Policy config is missing sections: {sorted(missing)}")
    input_policy = policy["input"]
    architecture = policy["architecture"]
    augmentation = policy["augmentation"]
    locked = policy["locked_test_policy"]
    if not isinstance(input_policy, Mapping) or (
        input_policy.get("width"), input_policy.get("height"), input_policy.get("channels")
    ) != (224, 224, 3):
        raise TrainingPolicyError("Model V2 input must remain RGB 224x224x3.")
    if input_policy.get("scaling") != "pixel / 255.0":
        raise TrainingPolicyError("Model V2 preprocessing must remain pixel / 255.0.")
    if not isinstance(architecture, Mapping) or architecture.get("output_classes") != 39:
        raise TrainingPolicyError("Model V2 must preserve all 39 output classes.")
    if not isinstance(augmentation, Mapping) or augmentation.get("enabled_for") != ["TRAIN"]:
        raise TrainingPolicyError("Random augmentation must be enabled for TRAIN only.")
    if not isinstance(locked, Mapping) or not locked.get("development_loading_forbidden"):
        raise TrainingPolicyError("Internal TEST must remain development-locked.")
    if not locked.get("plantdoc_test_locked"):
        raise TrainingPolicyError("PlantDoc TEST must remain development-locked.")


def load_development_manifest(
    path: Path,
    *,
    training: bool,
    development_selection: bool = True,
) -> list[ManifestRecord]:
    if not path.is_file():
        raise TrainingPolicyError(f"Development manifest is unavailable: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or [])
        missing = REQUIRED_MANIFEST_FIELDS - fields
        if missing:
            raise TrainingPolicyError(f"Manifest is missing fields: {sorted(missing)}")
        source_rows = [dict(row) for row in reader]
    if not source_rows:
        raise TrainingPolicyError(f"Development manifest is empty: {path}")

    splits = {row["split"] for row in source_rows}
    if "TEST" in splits and (training or development_selection):
        raise TrainingPolicyError(
            "Internal TEST is frozen and cannot be loaded for training or model selection."
        )
    unexpected = splits - ALLOWED_DEVELOPMENT_SPLITS
    if unexpected:
        raise TrainingPolicyError(f"Unsupported development splits: {sorted(unexpected)}")
    expected = "TRAIN" if training else "VALIDATION"
    if splits != {expected}:
        raise TrainingPolicyError(
            f"Expected a {expected}-only manifest; found {sorted(splits)}."
        )

    records: list[ManifestRecord] = []
    seen: set[str] = set()
    for row in source_rows:
        record_id = row["composition_record_id"]
        normalized_parts = [
            part.casefold()
            for part in row["source_path"].replace("\\", "/").split("/")
            if part
        ]
        if (
            row["source_dataset"].casefold() == "plantdoc"
            and normalized_parts
            and normalized_parts[0] in {"test", "testing", "test_split"}
        ):
            raise TrainingPolicyError(
                "PlantDoc TEST is frozen and cannot be loaded for Model V2 development."
            )
        if record_id in seen:
            raise TrainingPolicyError(f"Duplicate composition_record_id: {record_id}")
        seen.add(record_id)
        try:
            target_index = int(row["target_index"])
        except ValueError as exc:
            raise TrainingPolicyError(f"Invalid target_index: {row['target_index']}") from exc
        if not 0 <= target_index < len(CLASS_NAMES):
            raise TrainingPolicyError(f"Out-of-range target_index: {target_index}")
        if row["target_class"] != CLASS_NAMES[target_index]:
            raise TrainingPolicyError(f"Taxonomy mismatch for record: {record_id}")
        if expected == "TRAIN" and row["evaluation_role"] != "MODEL_TRAINING":
            raise TrainingPolicyError(f"Invalid TRAIN evaluation role: {record_id}")
        if expected == "VALIDATION" and row["evaluation_role"] != "MODEL_DEVELOPMENT_VALIDATION":
            raise TrainingPolicyError(f"Invalid VALIDATION evaluation role: {record_id}")
        records.append(
            ManifestRecord(
                composition_record_id=record_id,
                source_domain=row["source_domain"],
                source_dataset=row["source_dataset"],
                source_path=row["source_path"],
                target_index=target_index,
                target_class=row["target_class"],
                split=row["split"],
                evaluation_role=row["evaluation_role"],
            )
        )
    return records


def configured_source_roots(
    overrides: Mapping[str, Path | str] | None = None,
) -> dict[str, Path]:
    result: dict[str, Path] = {}
    overrides = overrides or {}
    for source, env_name in SOURCE_ROOT_ENV.items():
        raw_value = overrides.get(source) or os.getenv(env_name)
        if raw_value:
            result[source] = Path(raw_value).expanduser().resolve()
    return result


def resolve_record_path(record: ManifestRecord, source_roots: Mapping[str, Path]) -> Path:
    root = source_roots.get(record.source_dataset)
    if root is None:
        env_name = SOURCE_ROOT_ENV.get(record.source_dataset, "an approved source-root variable")
        raise TrainingPolicyError(
            f"No local root configured for {record.source_dataset}; configure {env_name}."
        )
    root = Path(root).expanduser().resolve()
    relative = Path(record.source_path)
    if relative.is_absolute():
        raise TrainingPolicyError("Manifest source paths must remain relative.")
    resolved = (root / relative).resolve()
    if not resolved.is_relative_to(root):
        raise TrainingPolicyError("Manifest source path escapes its configured root.")
    return resolved


def preprocess_image(path: Path, policy: Mapping[str, object]) -> np.ndarray:
    input_policy = policy["input"]
    width = int(input_policy["width"])
    height = int(input_policy["height"])
    try:
        with Image.open(path) as image:
            array = np.asarray(
                image.convert("RGB").resize((width, height), Image.Resampling.BILINEAR),
                dtype=np.float32,
            )
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise TrainingPolicyError(f"Invalid training image: {path.name}") from exc
    return array / np.float32(255.0)


def augmentation_enabled(*, training: bool, split: str) -> bool:
    if split == "TEST":
        raise TrainingPolicyError("Random augmentation is forbidden for internal TEST.")
    return training and split == "TRAIN"


def build_augmentation(policy: Mapping[str, object], seed: int | None = None):
    """Build dynamic TRAIN-only augmentation; TensorFlow is imported lazily."""
    import tensorflow as tf

    config = policy["augmentation"]
    experiment_seed = int(seed if seed is not None else policy["experiment_seed"])
    flip_mode = "horizontal_and_vertical"
    zoom = config["zoom_range"]
    zoom_factor = (float(zoom[0]) - 1.0, float(zoom[1]) - 1.0)
    return tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip(flip_mode, seed=experiment_seed),
            tf.keras.layers.RandomRotation(
                float(config["rotation_degrees"]) / 360.0,
                fill_mode="reflect",
                seed=experiment_seed + 1,
            ),
            tf.keras.layers.RandomTranslation(
                float(config["translation_fraction"]),
                float(config["translation_fraction"]),
                fill_mode="reflect",
                seed=experiment_seed + 2,
            ),
            tf.keras.layers.RandomZoom(
                height_factor=zoom_factor,
                width_factor=zoom_factor,
                fill_mode="reflect",
                seed=experiment_seed + 3,
            ),
            tf.keras.layers.RandomBrightness(
                float(config["brightness_factor"]),
                value_range=(0.0, 1.0),
                seed=experiment_seed + 4,
            ),
            tf.keras.layers.RandomContrast(
                float(config["contrast_factor"]), seed=experiment_seed + 5
            ),
        ],
        name="model_v2_train_augmentation",
    )


def set_experiment_seeds(policy: Mapping[str, object]) -> int:
    """Set the documented Python, NumPy, and TensorFlow experiment seed."""
    import tensorflow as tf

    seed = int(policy["experiment_seed"])
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)
    return seed


def build_tf_dataset(
    records: Sequence[ManifestRecord],
    policy: Mapping[str, object],
    source_roots: Mapping[str, Path],
    *,
    training: bool,
    batch_size: int | None = None,
):
    """Create a deterministic manifest-driven tf.data pipeline without training."""
    import tensorflow as tf

    if not records:
        raise TrainingPolicyError("Cannot build a dataset from zero records.")
    splits = {record.split for record in records}
    expected = "TRAIN" if training else "VALIDATION"
    if splits != {expected}:
        raise TrainingPolicyError(f"Expected {expected}-only records; found {sorted(splits)}")
    if not augmentation_enabled(training=training, split=expected) and training:
        raise TrainingPolicyError("TRAIN augmentation policy is inconsistent.")

    paths = [str(resolve_record_path(record, source_roots)) for record in records]
    labels = [record.target_index for record in records]
    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))
    options = tf.data.Options()
    options.experimental_deterministic = True
    dataset = dataset.with_options(options)
    seed = int(policy["experiment_seed"])
    if training:
        dataset = dataset.shuffle(
            buffer_size=len(records), seed=seed, reshuffle_each_iteration=True
        )

    height = int(policy["input"]["height"])
    width = int(policy["input"]["width"])

    def decode(path, label):
        content = tf.io.read_file(path)
        image = tf.io.decode_image(content, channels=3, expand_animations=False)
        image.set_shape((None, None, 3))
        image = tf.image.resize(image, (height, width), method="bilinear")
        image = tf.cast(image, tf.float32) / 255.0
        return image, tf.cast(label, tf.int32)

    dataset = dataset.map(decode, num_parallel_calls=tf.data.AUTOTUNE)
    actual_batch_size = int(batch_size or policy["batch_size"]["default"])
    dataset = dataset.batch(actual_batch_size)
    if training:
        augmenter = build_augmentation(policy, seed)
        dataset = dataset.map(
            lambda images, labels: (augmenter(images, training=True), labels),
            num_parallel_calls=tf.data.AUTOTUNE,
        )
    return dataset.prefetch(tf.data.AUTOTUNE)


def count_records(records: Sequence[ManifestRecord]) -> dict[str, object]:
    class_counts = Counter(record.target_index for record in records)
    domain_counts = Counter(record.source_domain for record in records)
    source_counts = Counter(record.source_dataset for record in records)
    return {
        "total": len(records),
        "class_counts": {str(index): class_counts[index] for index in range(39)},
        "domain_counts": dict(sorted(domain_counts.items())),
        "source_counts": dict(sorted(source_counts.items())),
    }


def compute_validation_metrics(
    records: Sequence[ManifestRecord], score_matrix: np.ndarray
) -> dict[str, object]:
    """Compute overall and REAL_WORLD validation metrics, never TEST metrics."""
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support

    if not records or {record.split for record in records} != {"VALIDATION"}:
        raise TrainingPolicyError("Metrics tooling accepts VALIDATION records only.")
    scores = np.asarray(score_matrix)
    if scores.shape != (len(records), len(CLASS_NAMES)):
        raise TrainingPolicyError(
            f"Expected score shape {(len(records), len(CLASS_NAMES))}; got {scores.shape}."
        )
    true = np.asarray([record.target_index for record in records], dtype=np.int32)
    predicted = np.argmax(scores, axis=1)

    def metrics(indices: np.ndarray) -> dict[str, object]:
        y_true = true[indices]
        y_pred = predicted[indices]
        supported = sorted(set(int(value) for value in y_true))
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, labels=supported, zero_division=0
        )
        weighted = precision_recall_fscore_support(
            y_true, y_pred, average="weighted", zero_division=0
        )
        return {
            "image_count": int(len(indices)),
            "supported_class_indices": supported,
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "macro_precision": float(np.mean(precision)),
            "macro_recall": float(np.mean(recall)),
            "macro_f1": float(np.mean(f1)),
            "weighted_f1": float(weighted[2]),
            "per_class": [
                {
                    "target_index": index,
                    "target_class": CLASS_NAMES[index],
                    "precision": float(precision[offset]),
                    "recall": float(recall[offset]),
                    "f1": float(f1[offset]),
                    "support": int(support[offset]),
                }
                for offset, index in enumerate(supported)
            ],
        }

    overall_indices = np.arange(len(records))
    real_indices = np.asarray(
        [index for index, record in enumerate(records) if record.source_domain == "REAL_WORLD"],
        dtype=np.int32,
    )
    return {
        "overall_validation": metrics(overall_indices),
        "real_world_validation": metrics(real_indices) if len(real_indices) else None,
    }
