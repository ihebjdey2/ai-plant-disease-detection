"""Optional, fail-closed Kaggle Dataset persistence for Experiment B.

Only recovery artifacts explicitly listed here can enter the staging directory
or be restored. The Kaggle API import is lazy so local tests and disabled runs
do not require Kaggle credentials or network access.
"""

from __future__ import annotations

import hashlib
import re
import shutil
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Callable, Mapping, Sequence

import tensorflow as tf


EXPERIMENT_NAME = "agri-diagnose-v2-exp-b"
STAGING_DIRECTORY_NAME = "agridiagnose-exp-b-persistent-staging"
RESULTS_DIRECTORY_NAME = "agridiagnose-exp-b-results"
CANDIDATE_RELATIVE_ROOT = PurePosixPath(
    "models/candidates/agri-diagnose-v2-exp-b"
)
RESULTS_RELATIVE_ROOT = PurePosixPath(RESULTS_DIRECTORY_NAME)
COMMON_FILENAMES = ("environment-runtime.json", "preflight.json")
PHASE_FILENAMES = {
    "phase1": ("phase1-best.keras", "phase1-history.csv", "phase1-complete.json"),
    "phase2": ("phase2-best.keras", "phase2-history.csv", "phase2-complete.json"),
}
FORBIDDEN_TEST_MARKERS = (
    "internal-test",
    "internal_test",
    "plantdoc-test",
    "plantdoc_test",
    "dataset-v2-test",
)
DATASET_HANDLE_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_-]*/[a-z0-9][a-z0-9_-]*$")


class PersistentBackupError(RuntimeError):
    """Raised before another epoch can start when persistence is not guaranteed."""


@dataclass(frozen=True)
class RecoveryLayout:
    candidate_dir: Path
    results_dir: Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_dataset_handle(handle: str) -> str:
    normalized = str(handle).strip()
    if not DATASET_HANDLE_PATTERN.fullmatch(normalized):
        raise PersistentBackupError(
            "Persistent backup dataset handle must use the owner/dataset format."
        )
    return normalized


def _contains_test_marker(path: Path | PurePosixPath | str) -> bool:
    value = str(path).replace("\\", "/").casefold()
    return any(marker in value for marker in FORBIDDEN_TEST_MARKERS)


def approved_relative_paths() -> dict[str, PurePosixPath]:
    paths = {
        name: RESULTS_RELATIVE_ROOT / name for name in COMMON_FILENAMES
    }
    for phase, filenames in PHASE_FILENAMES.items():
        paths[filenames[0]] = CANDIDATE_RELATIVE_ROOT / filenames[0]
        paths[filenames[1]] = RESULTS_RELATIVE_ROOT / filenames[1]
        paths[filenames[2]] = RESULTS_RELATIVE_ROOT / filenames[2]
    return paths


def select_recovery_files(layout: RecoveryLayout, *, phase: str) -> dict[PurePosixPath, Path]:
    if phase not in PHASE_FILENAMES:
        raise PersistentBackupError(f"Unknown Experiment B phase: {phase!r}")
    candidate_dir = Path(layout.candidate_dir)
    results_dir = Path(layout.results_dir)
    selected: dict[PurePosixPath, Path] = {}

    for name in COMMON_FILENAMES:
        source = results_dir / name
        if not source.is_file() or source.stat().st_size <= 0:
            raise PersistentBackupError(
                f"Required persistent recovery artifact is missing or empty: {name}"
            )
        selected[RESULTS_RELATIVE_ROOT / name] = source

    phases = ("phase1",) if phase == "phase1" else ("phase1", "phase2")
    for selected_phase in phases:
        checkpoint_name, history_name, marker_name = PHASE_FILENAMES[selected_phase]
        checkpoint = candidate_dir / checkpoint_name
        history = results_dir / history_name
        for relative, source in (
            (CANDIDATE_RELATIVE_ROOT / checkpoint_name, checkpoint),
            (RESULTS_RELATIVE_ROOT / history_name, history),
        ):
            if not source.is_file() or source.stat().st_size <= 0:
                raise PersistentBackupError(
                    "Required persistent recovery artifact is missing or empty: "
                    f"{source.name}"
                )
            selected[relative] = source
        marker = results_dir / marker_name
        if marker.is_file():
            if marker.stat().st_size <= 0:
                raise PersistentBackupError(
                    f"Persistent recovery artifact is empty: {marker.name}"
                )
            selected[RESULTS_RELATIVE_ROOT / marker_name] = marker

    if any(_contains_test_marker(path) for path in selected):
        raise PersistentBackupError("A TEST-related artifact entered the backup allowlist.")
    return selected


def _safe_remove_staging(path: Path) -> None:
    candidate = Path(path).resolve()
    if candidate.name not in {
        STAGING_DIRECTORY_NAME,
        f"{STAGING_DIRECTORY_NAME}.tmp",
    }:
        raise PersistentBackupError("Refusing to remove an unexpected staging path.")
    if candidate.exists():
        shutil.rmtree(candidate)


def stage_recovery_files(
    selected: Mapping[PurePosixPath, Path], staging_dir: Path
) -> dict[str, dict[str, int | str]]:
    staging = Path(staging_dir)
    if staging.name != STAGING_DIRECTORY_NAME:
        raise PersistentBackupError(
            f"Persistent staging directory must be named {STAGING_DIRECTORY_NAME}."
        )
    temporary = staging.with_name(f"{STAGING_DIRECTORY_NAME}.tmp")
    _safe_remove_staging(temporary)
    temporary.mkdir(parents=True)
    manifest: dict[str, dict[str, int | str]] = {}
    try:
        for relative, source in selected.items():
            if relative.is_absolute() or ".." in relative.parts or _contains_test_marker(relative):
                raise PersistentBackupError("Unsafe persistent staging destination.")
            destination = temporary.joinpath(*relative.parts)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            source_sha = sha256_file(source)
            if sha256_file(destination) != source_sha:
                raise PersistentBackupError(
                    f"Persistent staging byte verification failed: {source.name}"
                )
            manifest[relative.as_posix()] = {
                "sha256": source_sha,
                "size_bytes": destination.stat().st_size,
            }
        _safe_remove_staging(staging)
        temporary.replace(staging)
    except Exception:
        _safe_remove_staging(temporary)
        raise
    return manifest


def kagglehub_dataset_uploader(
    dataset_handle: str, staging_dir: Path, version_notes: str
) -> None:
    try:
        import kagglehub
    except ImportError as exc:
        raise PersistentBackupError(
            "kagglehub is required when persistent backup is enabled."
        ) from exc
    kagglehub.dataset_upload(
        dataset_handle,
        str(staging_dir),
        version_notes=version_notes,
    )


class KaggleDatasetBackupService:
    def __init__(
        self,
        *,
        enabled: bool,
        dataset_handle: str,
        layout: RecoveryLayout,
        staging_dir: Path,
        uploader: Callable[[str, Path, str], None] | None = None,
    ):
        self.enabled = enabled
        self.dataset_handle = (
            validate_dataset_handle(dataset_handle) if enabled else ""
        )
        self.layout = layout
        self.staging_dir = Path(staging_dir)
        self.uploader = uploader or kagglehub_dataset_uploader

    def persist(self, *, phase: str, completed_epoch: int) -> dict[str, object] | None:
        if not self.enabled:
            return None
        if type(completed_epoch) is not int or completed_epoch <= 0:
            raise PersistentBackupError("Completed epoch must be a positive integer.")
        selected = select_recovery_files(self.layout, phase=phase)
        manifest = stage_recovery_files(selected, self.staging_dir)
        note = f"Experiment B {phase} epoch {completed_epoch}"
        try:
            self.uploader(self.dataset_handle, self.staging_dir, note)
        except Exception as exc:
            print(
                "PERSISTENT_BACKUP_FAILED: local artifacts remain untouched; "
                f"{phase} epoch {completed_epoch} was not persisted."
            )
            raise PersistentBackupError(
                f"Experiment B {phase} epoch {completed_epoch} persistent upload failed."
            ) from exc
        print(f"Persistent backup uploaded: {note}")
        return {
            "experiment": EXPERIMENT_NAME,
            "phase": phase,
            "completed_epoch": completed_epoch,
            "dataset_handle": self.dataset_handle,
            "version_notes": note,
            "files": manifest,
        }


class KaggleDatasetBackupCallback(tf.keras.callbacks.Callback):
    """Upload only after earlier history/checkpoint callbacks finish epoch end."""

    def __init__(self, service: KaggleDatasetBackupService, *, phase: str):
        super().__init__()
        if phase not in PHASE_FILENAMES:
            raise PersistentBackupError(f"Unknown Experiment B phase: {phase!r}")
        self.service = service
        self.phase = phase

    def on_epoch_end(self, epoch, logs=None):
        del logs
        self.service.persist(phase=self.phase, completed_epoch=int(epoch) + 1)


def append_backup_callback(
    callbacks: Sequence[tf.keras.callbacks.Callback],
    service: KaggleDatasetBackupService,
    *,
    phase: str,
) -> list[tf.keras.callbacks.Callback]:
    values = list(callbacks)
    if service.enabled:
        values.append(KaggleDatasetBackupCallback(service, phase=phase))
    return values


def _scan_restore_source(source_dir: Path) -> dict[str, Path]:
    source = Path(source_dir).resolve()
    if not source.is_dir():
        raise PersistentBackupError("Persistent restore input directory is missing.")
    files = [path for path in source.rglob("*") if path.is_file()]
    forbidden = [path for path in files if _contains_test_marker(path.relative_to(source))]
    if forbidden:
        raise PersistentBackupError(
            "Persistent restore source contains a forbidden TEST-related file."
        )
    allowlisted_names = set(approved_relative_paths())
    matches: dict[str, Path] = {}
    for name in allowlisted_names:
        candidates = [path for path in files if path.name == name]
        if len(candidates) > 1:
            raise PersistentBackupError(
                f"Persistent restore source is ambiguous for {name}."
            )
        if candidates:
            matches[name] = candidates[0]
    if not matches:
        raise PersistentBackupError("No approved Experiment B recovery files were found.")
    required_common = set(COMMON_FILENAMES)
    missing_common = sorted(required_common - set(matches))
    if missing_common:
        raise PersistentBackupError(
            f"Persistent restore source is missing provenance files: {missing_common}"
        )
    for phase, filenames in PHASE_FILENAMES.items():
        checkpoint, history, _ = filenames
        checkpoint_present = checkpoint in matches
        history_present = history in matches
        if checkpoint_present != history_present:
            raise PersistentBackupError(
                f"Persistent restore source has incomplete {phase} checkpoint/history."
            )
    if "phase2-best.keras" in matches and "phase1-best.keras" not in matches:
        raise PersistentBackupError(
            "Persistent Phase 2 restore requires the complete Phase 1 lineage."
        )
    if "phase1-best.keras" not in matches:
        raise PersistentBackupError(
            "Persistent restore source is missing the Phase 1 checkpoint/history."
        )
    return matches


def validate_kaggle_restore_input(
    source_dir: Path, kaggle_input_root: Path = Path("/kaggle/input")
) -> Path:
    source = Path(source_dir).resolve()
    root = Path(kaggle_input_root).resolve()
    if source == root or not source.is_relative_to(root):
        raise PersistentBackupError(
            "Persistent restore source must be an attached directory under /kaggle/input."
        )
    if _contains_test_marker(source.relative_to(root)):
        raise PersistentBackupError("Persistent restore path is TEST-related.")
    return source


def restore_recovery_files(
    source_dir: Path,
    layout: RecoveryLayout,
) -> dict[str, dict[str, int | str]]:
    matches = _scan_restore_source(source_dir)
    relative_paths = approved_relative_paths()
    restored: dict[str, dict[str, int | str]] = {}
    operations: list[tuple[str, Path, PurePosixPath, Path, str]] = []
    for name, source in matches.items():
        relative = relative_paths[name]
        destination_root = (
            Path(layout.candidate_dir)
            if relative.parts[0] == "models"
            else Path(layout.results_dir)
        )
        destination = destination_root / name
        source_sha = sha256_file(source)
        if destination.exists():
            if not destination.is_file() or sha256_file(destination) != source_sha:
                raise PersistentBackupError(
                    f"Refusing to overwrite an existing different artifact: {name}"
                )
        operations.append((name, source, relative, destination, source_sha))

    for name, source, relative, destination, source_sha in operations:
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            restored[name] = {
                "sha256": source_sha,
                "size_bytes": destination.stat().st_size,
                "destination": relative.as_posix(),
            }
            if name.endswith((".keras", "-history.csv")):
                print(
                    f"Restored {name}: sha256={source_sha} "
                    f"size={destination.stat().st_size} (already identical)"
                )
            continue
        temporary = destination.with_suffix(destination.suffix + ".restore.tmp")
        shutil.copy2(source, temporary)
        if sha256_file(temporary) != source_sha:
            temporary.unlink(missing_ok=True)
            raise PersistentBackupError(f"Restore byte verification failed: {name}")
        temporary.replace(destination)
        restored[name] = {
            "sha256": source_sha,
            "size_bytes": destination.stat().st_size,
            "destination": relative.as_posix(),
        }
        if name.endswith((".keras", "-history.csv")):
            print(
                f"Restored {name}: sha256={source_sha} "
                f"size={destination.stat().st_size}"
            )
    return restored


__all__ = [
    "KaggleDatasetBackupCallback",
    "KaggleDatasetBackupService",
    "PersistentBackupError",
    "RecoveryLayout",
    "STAGING_DIRECTORY_NAME",
    "append_backup_callback",
    "approved_relative_paths",
    "restore_recovery_files",
    "select_recovery_files",
    "stage_recovery_files",
    "validate_kaggle_restore_input",
    "validate_dataset_handle",
]
