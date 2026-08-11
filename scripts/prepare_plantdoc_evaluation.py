"""Prepare a conservative PlantDoc TEST subset for the existing evaluator.

The official repository contains Windows-invalid characters in a few file
names. This adapter can therefore read images directly from the immutable Git
tree, without checking out, renaming, or moving PlantDoc source files.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path, PurePosixPath
from typing import Iterable, Sequence

from PIL import Image, UnidentifiedImageError


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training.taxonomy import CLASS_NAMES  # noqa: E402
from scripts.evaluate_model import DATASET_DIRECTORY_TO_CLASS  # noqa: E402


ALLOWED_STATUSES = {"MATCHED", "AMBIGUOUS", "NOT_SUPPORTED"}
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
DEFAULT_SOURCE = PROJECT_ROOT / "evaluation" / "datasets" / "plantdoc"
DEFAULT_MAPPING = PROJECT_ROOT / "evaluation" / "mappings" / "plantdoc.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "evaluation" / "datasets" / "plantdoc_prepared"


class PreparationError(RuntimeError):
    """Raised when source integrity, mapping, or preparation is invalid."""


@dataclass(frozen=True)
class SourceImage:
    label: str
    relative_path: str
    identifier: str


class PlantDocSource:
    """Read the official test split from a Git tree or normal directory."""

    def __init__(self, root: Path, revision: str = "HEAD", split: str = "test"):
        self.root = root.expanduser().resolve()
        self.revision = revision
        self.split = split
        self.git_mode = (self.root / ".git").is_dir()
        self._split_root = self.root / split
        if not self.git_mode and not self._split_root.is_dir():
            raise PreparationError(
                f"PlantDoc source must be a Git repository or contain {split}/."
            )
        self.resolved_revision = (
            self._git_text("rev-parse", revision).strip() if self.git_mode else None
        )

    def _git(self, *arguments: str, binary: bool = True) -> bytes | str:
        completed = subprocess.run(
            ["git", "-C", str(self.root), *arguments],
            capture_output=True,
            check=False,
        )
        if completed.returncode != 0:
            message = completed.stderr.decode("utf-8", errors="replace").strip()
            raise PreparationError(f"Git source read failed: {message}")
        return completed.stdout if binary else completed.stdout.decode("utf-8")

    def _git_text(self, *arguments: str) -> str:
        return self._git(*arguments, binary=False)  # type: ignore[return-value]

    def images(self) -> list[SourceImage]:
        if self.git_mode:
            raw = self._git(
                "ls-tree",
                "-r",
                "-z",
                "--format=%(objectname)%x09%(path)",
                f"{self.revision}:{self.split}",
            )
            records: list[SourceImage] = []
            for record in raw.split(b"\0"):  # type: ignore[union-attr]
                if not record:
                    continue
                object_id, path_bytes = record.split(b"\t", 1)
                relative_path = path_bytes.decode("utf-8")
                parts = PurePosixPath(relative_path).parts
                if len(parts) < 2:
                    continue
                records.append(
                    SourceImage(
                        parts[0],
                        f"{self.split}/{relative_path}",
                        object_id.decode(),
                    )
                )
            return sorted(records, key=lambda item: item.relative_path)

        records = []
        paths = (candidate for candidate in self._split_root.rglob("*") if candidate.is_file())
        for path in sorted(paths):
            relative = path.relative_to(self._split_root)
            if len(relative.parts) < 2:
                continue
            records.append(
                SourceImage(
                    relative.parts[0],
                    f"{self.split}/{relative.as_posix()}",
                    relative.as_posix(),
                )
            )
        return records

    def read_bytes(self, image: SourceImage) -> bytes:
        if self.git_mode:
            return self._git("cat-file", "blob", image.identifier)  # type: ignore[return-value]
        relative = PurePosixPath(image.relative_path)
        return (self.root.joinpath(*relative.parts)).read_bytes()


def load_mapping(path: Path) -> dict:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PreparationError(f"Cannot read PlantDoc mapping: {exc}") from exc
    classes = payload.get("classes")
    if not isinstance(classes, dict) or not classes:
        raise PreparationError("PlantDoc mapping must contain a non-empty classes object.")
    for source_label, entry in classes.items():
        if not isinstance(entry, dict) or entry.get("status") not in ALLOWED_STATUSES:
            raise PreparationError(f"Invalid mapping status for {source_label!r}.")
        status = entry["status"]
        target_class = entry.get("target_class")
        target_directory = entry.get("target_directory")
        if not entry.get("reason"):
            raise PreparationError(f"Mapping reason is required for {source_label!r}.")
        if status == "MATCHED":
            if target_class not in CLASS_NAMES:
                raise PreparationError(f"Unknown deployed class for {source_label!r}.")
            if DATASET_DIRECTORY_TO_CLASS.get(target_directory) != target_class:
                raise PreparationError(f"Invalid evaluator directory for {source_label!r}.")
        elif target_class is not None or target_directory is not None:
            raise PreparationError(
                f"Excluded mapping {source_label!r} must not specify a target."
            )
    return payload


def validate_mapping_coverage(source_labels: Iterable[str], mapping: dict) -> None:
    actual = set(source_labels)
    declared = set(mapping["classes"])
    missing = sorted(actual - declared)
    stale = sorted(declared - actual)
    if missing or stale:
        raise PreparationError(
            f"Mapping coverage mismatch; missing={missing or 'none'}, stale={stale or 'none'}."
        )


def validate_image(data: bytes) -> tuple[str, str]:
    try:
        with Image.open(BytesIO(data)) as image:
            image_format = image.format or "UNKNOWN"
            image_mode = image.mode
            image.verify()
        with Image.open(BytesIO(data)) as image:
            image.convert("RGB").load()
        return image_format, image_mode
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise PreparationError(f"invalid or corrupted image: {exc}") from exc


def difference_hash(data: bytes) -> int:
    with Image.open(BytesIO(data)) as image:
        grayscale = image.convert("L").resize((9, 8), Image.Resampling.LANCZOS)
        pixels = list(grayscale.getdata())
    result = 0
    for row in range(8):
        offset = row * 9
        for column in range(8):
            result = (result << 1) | int(
                pixels[offset + column] > pixels[offset + column + 1]
            )
    return result


def perceptual_duplicate_pairs(
    records: Sequence[dict], maximum_distance: int = 4
) -> list[dict]:
    pairs = []
    for left_index, left in enumerate(records):
        for right in records[left_index + 1 :]:
            distance = (left["dhash"] ^ right["dhash"]).bit_count()
            if distance <= maximum_distance and left["sha256"] != right["sha256"]:
                pairs.append(
                    {
                        "first": left["source_path"],
                        "second": right["source_path"],
                        "hamming_distance": distance,
                        "same_label": left["source_label"] == right["source_label"],
                    }
                )
    return pairs


def prepare_dataset(
    source_path: Path,
    mapping_path: Path,
    output_path: Path,
    report_path: Path | None = None,
    deduplicate_exact: bool = False,
) -> dict:
    source = PlantDocSource(source_path)
    mapping = load_mapping(mapping_path)
    images = source.images()
    if not images:
        raise PreparationError("The official PlantDoc test split contains no files.")
    validate_mapping_coverage((image.label for image in images), mapping)

    source_counts = Counter(image.label for image in images)
    extension_counts = Counter(
        PurePosixPath(image.relative_path).suffix.lower() for image in images
    )
    format_counts: Counter[str] = Counter()
    mode_counts: Counter[str] = Counter()
    corrupted = []
    selected = []

    for image in images:
        extension = PurePosixPath(image.relative_path).suffix.lower()
        try:
            data = source.read_bytes(image)
            image_format, image_mode = validate_image(data)
        except PreparationError as exc:
            corrupted.append({"path": image.relative_path, "error": str(exc)})
            continue
        format_counts[image_format] += 1
        mode_counts[image_mode] += 1
        entry = mapping["classes"][image.label]
        if entry["status"] != "MATCHED" or extension not in SUPPORTED_EXTENSIONS:
            continue
        selected.append(
            {
                "source_label": image.label,
                "source_path": image.relative_path,
                "target_class": entry["target_class"],
                "target_directory": entry["target_directory"],
                "extension": extension,
                "data": data,
                "sha256": hashlib.sha256(data).hexdigest(),
                "dhash": difference_hash(data),
            }
        )

    selected_before_deduplication = len(selected)
    exact_groups = defaultdict(list)
    for record in selected:
        exact_groups[record["sha256"]].append(record)
    duplicate_groups = [group for group in exact_groups.values() if len(group) > 1]
    for group in duplicate_groups:
        labels = {record["target_class"] for record in group}
        if len(labels) > 1:
            paths = [record["source_path"] for record in group]
            raise PreparationError(f"Exact duplicate has conflicting labels: {paths}")

    removed = []
    if deduplicate_exact:
        retained = []
        seen_hashes = set()
        for record in selected:
            if record["sha256"] in seen_hashes:
                removed.append(record["source_path"])
                continue
            seen_hashes.add(record["sha256"])
            retained.append(record)
        selected = retained

    visual_pairs = perceptual_duplicate_pairs(selected)
    output_path = output_path.expanduser().resolve()
    if output_path.exists():
        raise PreparationError(f"Prepared output already exists: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="plantdoc-prepared-", dir=output_path.parent
    ) as temp:
        staging = Path(temp)
        class_counters: Counter[str] = Counter()
        for record in selected:
            target_directory = record["target_directory"]
            class_counters[target_directory] += 1
            destination_directory = staging / target_directory
            destination_directory.mkdir(parents=True, exist_ok=True)
            destination = destination_directory / (
                f"{class_counters[target_directory]:04d}_"
                f"{record['sha256'][:12]}{record['extension']}"
            )
            destination.write_bytes(record["data"])
        Path(temp).replace(output_path)

    mapping_status_counts = Counter(
        entry["status"] for entry in mapping["classes"].values()
    )
    report = {
        "benchmark_name": mapping.get("benchmark_name"),
        "source": {
            "repository": mapping.get("source", {}).get("repository"),
            "split": source.split,
            "revision": source.resolved_revision,
            "git_tree_mode": source.git_mode,
        },
        "dataset_integrity": {
            "total_test_files": len(images),
            "class_count": len(source_counts),
            "images_per_class": dict(sorted(source_counts.items())),
            "file_extensions": dict(sorted(extension_counts.items())),
            "decoded_formats": dict(sorted(format_counts.items())),
            "decoded_modes": dict(sorted(mode_counts.items())),
            "corrupted_images": corrupted,
            "empty_classes": [],
        },
        "mapping": {
            "status_counts": dict(sorted(mapping_status_counts.items())),
            "matched_classes": sorted(
                label
                for label, entry in mapping["classes"].items()
                if entry["status"] == "MATCHED"
            ),
            "excluded_classes": {
                label: {"status": entry["status"], "reason": entry["reason"]}
                for label, entry in mapping["classes"].items()
                if entry["status"] != "MATCHED"
            },
        },
        "selection": {
            "selected_before_exact_deduplication": selected_before_deduplication,
            "excluded_image_count": len(images) - selected_before_deduplication,
            "exact_duplicate_group_count": len(duplicate_groups),
            "exact_duplicate_image_count": sum(
                len(group) - 1 for group in duplicate_groups
            ),
            "exact_duplicate_groups": [
                [record["source_path"] for record in group]
                for group in duplicate_groups
            ],
            "exact_duplicates_removed": len(removed),
            "removed_source_paths": removed,
            "likely_visual_duplicate_pair_count": len(visual_pairs),
            "likely_visual_duplicate_pairs": visual_pairs,
            "prepared_image_count": len(selected),
            "prepared_images_per_class": {
                DATASET_DIRECTORY_TO_CLASS[directory]: count
                for directory, count in sorted(class_counters.items())
            },
        },
    }
    if report_path is not None:
        report_path = report_path.expanduser().resolve()
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare the verified PlantDoc TEST overlap for AgriDiagnose evaluation."
    )
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--mapping", type=Path, default=DEFAULT_MAPPING)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path)
    parser.add_argument(
        "--deduplicate-exact",
        action="store_true",
        help="Remove exact repeats only after recording them in the report.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = prepare_dataset(
            args.source,
            args.mapping,
            args.output,
            args.report,
            args.deduplicate_exact,
        )
    except PreparationError as exc:
        print(f"PlantDoc preparation failed: {exc}", file=sys.stderr)
        return 1
    integrity = report["dataset_integrity"]
    selection = report["selection"]
    print("PlantDoc TEST preparation complete")
    print(f"  Official test images: {integrity['total_test_files']}")
    print(f"  Official test classes: {integrity['class_count']}")
    print(f"  Corrupted images: {len(integrity['corrupted_images'])}")
    print(f"  Selected images: {selection['selected_before_exact_deduplication']}")
    print(f"  Exact duplicates removed: {selection['exact_duplicates_removed']}")
    print(f"  Prepared images: {selection['prepared_image_count']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
