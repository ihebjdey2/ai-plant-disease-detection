from __future__ import annotations

from pathlib import Path

import pytest
import tensorflow as tf

from training.kaggle_persistence import (
    KaggleDatasetBackupService,
    PersistentBackupError,
    RecoveryLayout,
    STAGING_DIRECTORY_NAME,
    append_backup_callback,
    restore_recovery_files,
    select_recovery_files,
    validate_kaggle_restore_input,
)


def make_layout(tmp_path: Path) -> RecoveryLayout:
    candidate = tmp_path / "working/models/candidates/agri-diagnose-v2-exp-b"
    results = tmp_path / "working/agridiagnose-exp-b-results"
    candidate.mkdir(parents=True)
    results.mkdir(parents=True)
    (results / "environment-runtime.json").write_bytes(b"runtime")
    (results / "preflight.json").write_bytes(b"preflight")
    return RecoveryLayout(candidate, results)


def write_phase(layout: RecoveryLayout, phase: str, *, marker: bool = False) -> None:
    (layout.candidate_dir / f"{phase}-best.keras").write_bytes(
        f"{phase}-checkpoint".encode()
    )
    (layout.results_dir / f"{phase}-history.csv").write_bytes(
        f"{phase}-history".encode()
    )
    if marker:
        (layout.results_dir / f"{phase}-complete.json").write_bytes(
            f"{phase}-complete".encode()
        )


def test_persistence_disabled_does_not_change_callbacks_or_call_upload(tmp_path):
    layout = make_layout(tmp_path)
    uploads = []
    service = KaggleDatasetBackupService(
        enabled=False,
        dataset_handle="",
        layout=layout,
        staging_dir=tmp_path / STAGING_DIRECTORY_NAME,
        uploader=lambda *args: uploads.append(args),
    )
    callback = tf.keras.callbacks.TerminateOnNaN()

    assert append_backup_callback([callback], service, phase="phase1") == [callback]
    assert service.persist(phase="phase1", completed_epoch=1) is None
    assert uploads == []


def test_phase1_selection_contains_only_approved_available_files(tmp_path):
    layout = make_layout(tmp_path)
    write_phase(layout, "phase1", marker=True)
    (layout.results_dir / "unrelated.txt").write_bytes(b"never upload")
    selected = select_recovery_files(layout, phase="phase1")

    assert {path.name for path in selected} == {
        "environment-runtime.json",
        "preflight.json",
        "phase1-best.keras",
        "phase1-history.csv",
        "phase1-complete.json",
    }


def test_phase2_selection_preserves_both_phase_lineages_only(tmp_path):
    layout = make_layout(tmp_path)
    write_phase(layout, "phase1", marker=True)
    write_phase(layout, "phase2", marker=True)
    selected = select_recovery_files(layout, phase="phase2")

    assert {path.name for path in selected} == {
        "environment-runtime.json",
        "preflight.json",
        "phase1-best.keras",
        "phase1-history.csv",
        "phase1-complete.json",
        "phase2-best.keras",
        "phase2-history.csv",
        "phase2-complete.json",
    }


def test_callback_uploads_after_completed_epoch_with_allowlisted_staging(tmp_path):
    layout = make_layout(tmp_path)
    write_phase(layout, "phase1")
    staging = tmp_path / STAGING_DIRECTORY_NAME
    uploads = []

    def upload(handle, staged, note):
        uploads.append((handle, Path(staged), note))
        names = {path.name for path in Path(staged).rglob("*") if path.is_file()}
        assert names == {
            "environment-runtime.json",
            "preflight.json",
            "phase1-best.keras",
            "phase1-history.csv",
        }

    service = KaggleDatasetBackupService(
        enabled=True,
        dataset_handle="owner/private-checkpoints",
        layout=layout,
        staging_dir=staging,
        uploader=upload,
    )
    callbacks = append_backup_callback(
        [tf.keras.callbacks.TerminateOnNaN()], service, phase="phase1"
    )
    assert callbacks[-1].__class__.__name__ == "KaggleDatasetBackupCallback"

    callbacks[-1].on_epoch_end(3)
    assert uploads == [
        ("owner/private-checkpoints", staging, "Experiment B phase1 epoch 4")
    ]


def test_upload_failure_preserves_every_local_artifact_and_raises(tmp_path):
    layout = make_layout(tmp_path)
    write_phase(layout, "phase1")
    sources = [
        layout.candidate_dir / "phase1-best.keras",
        layout.results_dir / "phase1-history.csv",
        layout.results_dir / "environment-runtime.json",
        layout.results_dir / "preflight.json",
    ]
    before = {path: path.read_bytes() for path in sources}

    def fail_upload(*args):
        raise OSError("network unavailable")

    service = KaggleDatasetBackupService(
        enabled=True,
        dataset_handle="owner/private-checkpoints",
        layout=layout,
        staging_dir=tmp_path / STAGING_DIRECTORY_NAME,
        uploader=fail_upload,
    )
    with pytest.raises(PersistentBackupError, match="persistent upload failed"):
        service.persist(phase="phase1", completed_epoch=2)
    assert {path: path.read_bytes() for path in sources} == before


def build_restore_source(tmp_path: Path) -> tuple[Path, dict[str, bytes]]:
    source_layout = make_layout(tmp_path / "source")
    write_phase(source_layout, "phase1", marker=True)
    selected = select_recovery_files(source_layout, phase="phase1")
    staging = tmp_path / "source" / STAGING_DIRECTORY_NAME
    service = KaggleDatasetBackupService(
        enabled=True,
        dataset_handle="owner/private-checkpoints",
        layout=source_layout,
        staging_dir=staging,
        uploader=lambda *args: None,
    )
    service.persist(phase="phase1", completed_epoch=10)
    expected = {path.name: source.read_bytes() for path, source in selected.items()}
    return staging, expected


def test_restore_copies_only_approved_files_with_exact_bytes(tmp_path):
    source, expected = build_restore_source(tmp_path)
    target = make_layout(tmp_path / "target")
    for path in target.results_dir.iterdir():
        path.unlink()

    restored = restore_recovery_files(source, target)

    assert set(restored) == set(expected)
    for name, content in expected.items():
        destination = (
            target.candidate_dir / name
            if name.endswith(".keras")
            else target.results_dir / name
        )
        assert destination.read_bytes() == content


def test_restore_rejects_ambiguous_checkpoint_sources(tmp_path):
    source, _ = build_restore_source(tmp_path)
    duplicate = source / "duplicate/phase1-best.keras"
    duplicate.parent.mkdir()
    duplicate.write_bytes(b"other")
    with pytest.raises(PersistentBackupError, match="ambiguous"):
        restore_recovery_files(source, make_layout(tmp_path / "target"))


def test_restore_rejects_any_test_related_content(tmp_path):
    source, _ = build_restore_source(tmp_path)
    forbidden = source / "internal_test/notice.txt"
    forbidden.parent.mkdir()
    forbidden.write_bytes(b"forbidden")
    with pytest.raises(PersistentBackupError, match="TEST-related"):
        restore_recovery_files(source, make_layout(tmp_path / "target"))


def test_restore_input_must_be_one_non_test_directory_under_kaggle_input(tmp_path):
    root = tmp_path / "kaggle/input"
    approved = root / "private-checkpoints"
    approved.mkdir(parents=True)
    assert validate_kaggle_restore_input(approved, root) == approved.resolve()
    with pytest.raises(PersistentBackupError, match="under /kaggle/input"):
        validate_kaggle_restore_input(tmp_path / "elsewhere", root)
    forbidden = root / "plantdoc-test-checkpoints"
    forbidden.mkdir()
    with pytest.raises(PersistentBackupError, match="TEST-related"):
        validate_kaggle_restore_input(forbidden, root)


def test_restore_refuses_to_overwrite_different_existing_artifacts(tmp_path):
    source, _ = build_restore_source(tmp_path)
    target = make_layout(tmp_path / "target")
    (target.candidate_dir / "phase1-best.keras").write_bytes(b"existing")
    with pytest.raises(PersistentBackupError, match="Refusing to overwrite"):
        restore_recovery_files(source, target)
