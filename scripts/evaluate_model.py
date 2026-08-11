"""Offline evaluation for the deployed 39-class AgriDiagnose model."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image, UnidentifiedImageError


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training.taxonomy import CLASS_NAMES  # noqa: E402
from config import Config  # noqa: E402


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
BACKGROUND_CLASS_INDEX = 4
DEFAULT_MODEL_PATH = PROJECT_ROOT / "plant_disease_model.h5"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "evaluation" / "results"

# This is the only accepted directory vocabulary. It deliberately avoids
# guessing aliases used by other PlantVillage distributions.
DATASET_DIRECTORY_TO_CLASS = {
    "Apple___Apple_scab": "Apple Apple scab",
    "Apple___Black_rot": "Apple Black rot",
    "Apple___Cedar_apple_rust": "Apple Cedar apple rust",
    "Apple___healthy": "Apple healthy",
    "Background_without_leaves": "Background without leaves",
    "Blueberry___healthy": "Blueberry healthy",
    "Cherry___Powdery_mildew": "Cherry Powdery mildew",
    "Cherry___healthy": "Cherry healthy",
    "Corn___Cercospora_leaf_spot": "Corn Cercospora leaf spot",
    "Corn___Common_rust": "Corn Common rust",
    "Corn___Northern_Leaf_Blight": "Corn Northern Leaf Blight",
    "Corn___healthy": "Corn healthy",
    "Grape___Black_rot": "Grape Black rot",
    "Grape___Esca": "Grape Esca",
    "Grape___Leaf_blight": "Grape Leaf blight",
    "Grape___healthy": "Grape healthy",
    "Orange___Huanglongbing": "Orange Huanglongbing",
    "Peach___Bacterial_spot": "Peach Bacterial spot",
    "Peach___healthy": "Peach healthy",
    "Bell_pepper___Bacterial_spot": "Bell pepper Bacterial spot",
    "Bell_pepper___healthy": "Bell pepper healthy",
    "Potato___Early_blight": "Potato Early blight",
    "Potato___Late_blight": "Potato Late blight",
    "Potato___healthy": "Potato healthy",
    "Raspberry___healthy": "Raspberry healthy",
    "Soybean___healthy": "Soybean healthy",
    "Squash___Powdery_mildew": "Squash Powdery mildew",
    "Strawberry___Leaf_scorch": "Strawberry Leaf scorch",
    "Strawberry___healthy": "Strawberry healthy",
    "Tomato___Bacterial_spot": "Tomato Bacterial spot",
    "Tomato___Early_blight": "Tomato Early blight",
    "Tomato___Late_blight": "Tomato Late blight",
    "Tomato___Leaf_Mold": "Tomato Leaf Mold",
    "Tomato___Septoria_leaf_spot": "Tomato Septoria leaf spot",
    "Tomato___Spider_mites": "Tomato Spider mites",
    "Tomato___Target_Spot": "Tomato Target Spot",
    "Tomato___Yellow_Leaf_Curl_Virus": "Tomato Yellow Leaf Curl Virus",
    "Tomato___mosaic_virus": "Tomato mosaic virus",
    "Tomato___healthy": "Tomato healthy",
}


class EvaluationError(RuntimeError):
    """Raised for controlled dataset, mapping, model, or evaluation errors."""


@dataclass(frozen=True)
class EvaluationSample:
    path: Path
    class_index: int
    class_name: str


@dataclass
class DatasetAudit:
    dataset_name: str
    samples: list[EvaluationSample]
    images_per_class: dict[str, int]
    mapped_directories: list[str]
    missing_classes: list[str]
    unexpected_classes: list[str]
    corrupted_images: list[str]
    candidate_image_count: int
    ignored_file_count: int

    @property
    def valid_image_count(self) -> int:
        return len(self.samples)

    @property
    def classes_found(self) -> int:
        return sum(count > 0 for count in self.images_per_class.values())

    @property
    def minimum_samples_per_class(self) -> int:
        counts = [count for count in self.images_per_class.values() if count > 0]
        return min(counts, default=0)

    @property
    def maximum_samples_per_class(self) -> int:
        return max(self.images_per_class.values(), default=0)

    def as_dict(self) -> dict:
        return {
            "directory_name": self.dataset_name,
            "candidate_image_count": self.candidate_image_count,
            "valid_image_count": self.valid_image_count,
            "corrupted_image_count": len(self.corrupted_images),
            "ignored_file_count": self.ignored_file_count,
            "classes_found": self.classes_found,
            "minimum_samples_per_found_class": self.minimum_samples_per_class,
            "maximum_samples_per_found_class": self.maximum_samples_per_class,
            "images_per_class": self.images_per_class,
            "missing_classes": self.missing_classes,
            "unexpected_classes": self.unexpected_classes,
            "corrupted_images": self.corrupted_images,
        }


def validate_mapping() -> None:
    mapped_classes = list(DATASET_DIRECTORY_TO_CLASS.values())
    if len(DATASET_DIRECTORY_TO_CLASS) != 39:
        raise EvaluationError(
            f"Evaluation mapping has {len(DATASET_DIRECTORY_TO_CLASS)} entries; 39 are required."
        )
    if len(CLASS_NAMES) != 39:
        raise EvaluationError(f"Deployed class mapping has {len(CLASS_NAMES)} entries; 39 are required.")
    if mapped_classes != CLASS_NAMES:
        raise EvaluationError(
            "Evaluation directory mapping does not exactly match the deployed CLASS_NAMES order."
        )


def validate_image_file(path: Path) -> None:
    try:
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            image.convert("RGB").load()
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise EvaluationError("invalid or corrupted image") from exc


def load_image_for_evaluation(path: Path) -> np.ndarray:
    """Apply the exact deployed preprocessing without augmentation."""
    with Image.open(path) as image:
        return (
            np.asarray(image.convert("RGB").resize((224, 224)), dtype=np.float32)
            / 255.0
        )


def audit_dataset(dataset_path: Path) -> DatasetAudit:
    validate_mapping()
    dataset_path = dataset_path.expanduser().resolve()
    if not dataset_path.is_dir():
        raise EvaluationError(f"Dataset directory does not exist: {dataset_path}")

    class_to_index = {class_name: index for index, class_name in enumerate(CLASS_NAMES)}
    images_per_class = {class_name: 0 for class_name in CLASS_NAMES}
    samples: list[EvaluationSample] = []
    corrupted_images: list[str] = []
    mapped_directories: list[str] = []
    unexpected_classes: list[str] = []
    candidate_image_count = 0
    ignored_file_count = 0

    for class_directory in sorted(path for path in dataset_path.iterdir() if path.is_dir()):
        class_name = DATASET_DIRECTORY_TO_CLASS.get(class_directory.name)
        if class_name is None:
            unexpected_classes.append(class_directory.name)
            continue
        mapped_directories.append(class_directory.name)
        class_index = class_to_index[class_name]
        for image_path in sorted(path for path in class_directory.rglob("*") if path.is_file()):
            if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                ignored_file_count += 1
                continue
            candidate_image_count += 1
            try:
                validate_image_file(image_path)
            except EvaluationError:
                corrupted_images.append(image_path.relative_to(dataset_path).as_posix())
                continue
            images_per_class[class_name] += 1
            samples.append(EvaluationSample(image_path, class_index, class_name))

    missing_classes = [
        class_name for class_name, count in images_per_class.items() if count == 0
    ]
    return DatasetAudit(
        dataset_name=dataset_path.name,
        samples=samples,
        images_per_class=images_per_class,
        mapped_directories=mapped_directories,
        missing_classes=missing_classes,
        unexpected_classes=unexpected_classes,
        corrupted_images=corrupted_images,
        candidate_image_count=candidate_image_count,
        ignored_file_count=ignored_file_count,
    )


def validate_audit(audit: DatasetAudit, allow_subset: bool = False) -> None:
    if audit.unexpected_classes:
        names = ", ".join(audit.unexpected_classes)
        raise EvaluationError(f"Unknown class directories: {names}")
    empty_directories = [
        directory
        for directory in audit.mapped_directories
        if audit.images_per_class[DATASET_DIRECTORY_TO_CLASS[directory]] == 0
    ]
    if empty_directories:
        names = ", ".join(empty_directories)
        raise EvaluationError(f"Mapped class directories contain no valid images: {names}")
    if audit.valid_image_count == 0:
        raise EvaluationError("The dataset contains no valid supported images.")
    if audit.missing_classes and not allow_subset:
        raise EvaluationError(
            f"Full evaluation requires all 39 classes; {len(audit.missing_classes)} are missing. "
            "Use --allow-subset only for an explicitly reported partial evaluation."
        )


def validate_model_output_count(model) -> int:
    output_shape = model.output_shape
    if isinstance(output_shape, list):
        raise EvaluationError("Multi-output models are not supported by this evaluator.")
    output_count = int(output_shape[-1])
    if output_count != len(CLASS_NAMES) or output_count != 39:
        raise EvaluationError(
            f"Model has {output_count} outputs; the deployed mapping requires exactly 39."
        )
    return output_count


def run_batched_inference(model, samples: Sequence[EvaluationSample], batch_size: int):
    if batch_size <= 0:
        raise EvaluationError("Batch size must be greater than zero.")
    score_batches: list[np.ndarray] = []
    inference_seconds = 0.0
    for offset in range(0, len(samples), batch_size):
        batch_samples = samples[offset : offset + batch_size]
        batch = np.stack(
            [load_image_for_evaluation(sample.path) for sample in batch_samples]
        )
        started = time.perf_counter()
        scores = np.asarray(model.predict(batch, verbose=0), dtype=np.float32)
        inference_seconds += time.perf_counter() - started
        if scores.shape != (len(batch_samples), len(CLASS_NAMES)):
            raise EvaluationError(
                f"Unexpected prediction shape {scores.shape}; expected "
                f"({len(batch_samples)}, {len(CLASS_NAMES)})."
            )
        if not np.isfinite(scores).all():
            raise EvaluationError("Model predictions contain non-finite values.")
        score_batches.append(scores)
    return np.concatenate(score_batches, axis=0), inference_seconds


def calculate_metrics(
    y_true: np.ndarray,
    scores: np.ndarray,
    confidence_threshold: float,
):
    from sklearn.metrics import (
        accuracy_score,
        classification_report,
        confusion_matrix,
        precision_recall_fscore_support,
    )

    labels = np.arange(len(CLASS_NAMES))
    evaluated_labels = np.unique(y_true)
    y_pred = np.argmax(scores, axis=1)
    confidences = np.max(scores, axis=1) * 100.0
    accuracy = float(accuracy_score(y_true, y_pred))
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    macro = precision_recall_fscore_support(
        y_true, y_pred, labels=evaluated_labels, average="macro", zero_division=0
    )
    weighted = precision_recall_fscore_support(
        y_true, y_pred, labels=evaluated_labels, average="weighted", zero_division=0
    )
    report = classification_report(
        y_true,
        y_pred,
        labels=evaluated_labels,
        target_names=[CLASS_NAMES[index] for index in evaluated_labels],
        output_dict=True,
        zero_division=0,
    )
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    per_class = {
        class_name: {
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "f1": float(f1[index]),
            "support": int(support[index]),
        }
        for index, class_name in enumerate(CLASS_NAMES)
    }
    confident_mask = confidences >= confidence_threshold
    background_prediction_mask = y_pred == BACKGROUND_CLASS_INDEX
    below_threshold_mask = ~confident_mask
    application_uncertain_mask = below_threshold_mask & ~background_prediction_mask
    confident_accuracy = (
        float(accuracy_score(y_true[confident_mask], y_pred[confident_mask]))
        if confident_mask.any()
        else None
    )
    confidence = {
        "threshold_percent": float(confidence_threshold),
        "average_top1_percent": float(np.mean(confidences)),
        "median_top1_percent": float(np.median(confidences)),
        "samples_at_or_above_threshold": int(confident_mask.sum()),
        "accuracy_at_or_above_threshold": confident_accuracy,
        "below_threshold_count": int(below_threshold_mask.sum()),
        "below_threshold_percent": float(np.mean(below_threshold_mask) * 100.0),
        "uncertain_prediction_count": int(application_uncertain_mask.sum()),
        "uncertain_prediction_percent": float(
            np.mean(application_uncertain_mask) * 100.0
        ),
        "no_leaf_prediction_count": int(background_prediction_mask.sum()),
        "no_leaf_below_threshold_count": int(
            (background_prediction_mask & below_threshold_mask).sum()
        ),
    }
    return {
        "accuracy": accuracy,
        "metric_scope": {
            "type": "evaluated_ground_truth_classes",
            "class_count": int(len(evaluated_labels)),
            "classes": [CLASS_NAMES[index] for index in evaluated_labels],
        },
        "macro": {
            "precision": float(macro[0]),
            "recall": float(macro[1]),
            "f1": float(macro[2]),
        },
        "weighted": {
            "precision": float(weighted[0]),
            "recall": float(weighted[1]),
            "f1": float(weighted[2]),
        },
        "per_class": per_class,
        "classification_report": report,
        "confidence_analysis": confidence,
        "background_class_evaluated": bool(BACKGROUND_CLASS_INDEX in evaluated_labels),
        "background_class": (
            per_class[CLASS_NAMES[BACKGROUND_CLASS_INDEX]]
            if BACKGROUND_CLASS_INDEX in evaluated_labels
            else None
        ),
    }, matrix


def save_confusion_matrix_csv(matrix: np.ndarray, output_path: Path) -> None:
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["actual/predicted", *CLASS_NAMES])
        for class_name, row in zip(CLASS_NAMES, matrix):
            writer.writerow([class_name, *(int(value) for value in row)])


def save_confusion_matrix_plot(matrix: np.ndarray, output_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(26, 24))
    image = axis.imshow(matrix, interpolation="nearest", cmap="Blues")
    figure.colorbar(image, ax=axis, fraction=0.03, pad=0.02)
    ticks = np.arange(len(CLASS_NAMES))
    axis.set_xticks(ticks, labels=CLASS_NAMES, rotation=90, fontsize=6)
    axis.set_yticks(ticks, labels=CLASS_NAMES, fontsize=6)
    axis.set_xlabel("Predicted class")
    axis.set_ylabel("Actual class")
    axis.set_title("AgriDiagnose 39-class confusion matrix")
    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def print_audit(audit: DatasetAudit) -> None:
    print("Dataset audit")
    print(f"  Candidate images: {audit.candidate_image_count}")
    print(f"  Valid images: {audit.valid_image_count}")
    print(f"  Corrupted images: {len(audit.corrupted_images)}")
    print(f"  Classes found: {audit.classes_found}/39")
    print(f"  Minimum samples in a found class: {audit.minimum_samples_per_class}")
    print(f"  Maximum samples in a found class: {audit.maximum_samples_per_class}")
    print(f"  Missing classes: {len(audit.missing_classes)}")
    print(f"  Unexpected classes: {len(audit.unexpected_classes)}")
    for class_name, count in audit.images_per_class.items():
        print(f"    {class_name}: {count}")
    for path in audit.corrupted_images:
        print(f"    CORRUPTED: {path}")


def evaluate(
    dataset_path: Path,
    model_path: Path,
    output_dir: Path,
    batch_size: int,
    allow_subset: bool,
    confidence_threshold: float,
) -> dict:
    audit = audit_dataset(dataset_path)
    print_audit(audit)
    validate_audit(audit, allow_subset=allow_subset)

    model_path = model_path.expanduser().resolve()
    if not model_path.is_file():
        raise EvaluationError(f"Model file does not exist: {model_path}")
    from tensorflow.keras.models import load_model

    model = load_model(model_path, compile=False)
    output_count = validate_model_output_count(model)
    scores, inference_seconds = run_batched_inference(model, audit.samples, batch_size)
    y_true = np.asarray([sample.class_index for sample in audit.samples], dtype=np.int64)
    metric_values, matrix = calculate_metrics(y_true, scores, confidence_threshold)

    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    save_confusion_matrix_csv(matrix, output_dir / "confusion_matrix.csv")
    save_confusion_matrix_plot(matrix, output_dir / "confusion_matrix.png")
    result = {
        "evaluation_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": {**audit.as_dict(), "allow_subset": allow_subset},
        "model": {
            "file_name": model_path.name,
            "sha256": file_sha256(model_path),
            "output_count": output_count,
        },
        "preprocessing": {
            "color_mode": "RGB",
            "resize": [224, 224],
            "dtype": "float32",
            "scaling": "image / 255.0",
            "augmentation": False,
        },
        **metric_values,
        "performance": {
            "batch_size": batch_size,
            "total_model_predict_seconds": inference_seconds,
            "average_model_predict_seconds_per_image": inference_seconds
            / audit.valid_image_count,
        },
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print("Evaluation complete")
    print(f"  Accuracy: {result['accuracy']:.6f}")
    print(f"  Macro F1: {result['macro']['f1']:.6f}")
    print(f"  Weighted F1: {result['weighted']['f1']:.6f}")
    print(f"  Results directory: {output_dir}")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate the existing AgriDiagnose model on a labeled test dataset."
    )
    parser.add_argument("--dataset", required=True, type=Path, help="Labeled test dataset root")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--allow-subset",
        action="store_true",
        help="Explicitly allow a partial-class evaluation; missing classes remain reported.",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=Config.PREDICTION_CONFIDENCE_THRESHOLD,
        help="Application-level uncertainty threshold in percent (default: 60).",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not 0 <= args.confidence_threshold <= 100:
        parser.error("--confidence-threshold must be between 0 and 100")
    try:
        evaluate(
            dataset_path=args.dataset,
            model_path=args.model,
            output_dir=args.output_dir,
            batch_size=args.batch_size,
            allow_subset=args.allow_subset,
            confidence_threshold=args.confidence_threshold,
        )
    except EvaluationError as exc:
        print(f"Evaluation failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
