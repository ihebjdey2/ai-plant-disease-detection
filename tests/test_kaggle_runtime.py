from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from scripts.bootstrap_kaggle_tf215_runtime import has_approved_isolated_python
from training.kaggle_runtime import (
    KaggleRuntimeLayout,
    NVIDIA_PYPI_INDEX_URL,
    UV_INDEX_STRATEGY,
    UV_VERSION,
    bootstrap_commands,
    bootstrap_environment,
    build_execution_config,
    discover_kaggle_source_roots,
    isolated_entrypoint_command,
    load_execution_config,
    require_kaggle_working_root,
    validate_runtime_payload,
    validate_uv_version,
    write_execution_config,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def valid_runtime_payload() -> dict[str, object]:
    return {
        "python_version": "3.11.15",
        "tensorflow_version": "2.15.0",
        "keras_version": "2.15.0",
        "numpy_version": "1.26.4",
        "tensorflow_built_with_cuda": True,
        "tensorflow_gpu_devices": ["/physical_device:GPU:0"],
        "gpu_smoke_test_passed": True,
        "gpu_smoke_device": "/job:localhost/device:GPU:0",
    }


def source_roots() -> dict[str, str]:
    return {
        "historical": "/kaggle/input/agridiagnose-historical",
        "pldd_up": "/kaggle/input/agridiagnose-pldd-up",
        "seasonal_corn": "/kaggle/input/agridiagnose-seasonal-corn",
        "plantdoc_train": "/kaggle/input/agridiagnose-plantdoc-train",
        "banu_deb": "/kaggle/input/agridiagnose-banu-deb",
    }


def test_runtime_layout_keeps_all_mutations_below_working_root():
    root = Path("/kaggle/working/agridiagnose-tf215-runtime")
    layout = KaggleRuntimeLayout(root, Path("/kaggle/working/project"))
    mutable_paths = [
        layout.uv_bin_dir,
        layout.uv_installer,
        layout.managed_python_dir,
        layout.uv_cache_dir,
        layout.experiment_venv,
        layout.runtime_report,
        layout.installed_packages,
        layout.execution_config,
    ]
    assert all(root == path or root in path.parents for path in mutable_paths)
    assert layout.experiment_python.as_posix().endswith(
        "venvs/agridiagnose-tf215/bin/python"
    )


def test_kaggle_working_root_guard_rejects_system_locations():
    assert "kaggle" in str(
        require_kaggle_working_root(
            Path("/kaggle/working/agridiagnose-tf215-runtime")
        )
    ).casefold()
    with pytest.raises(ValueError, match="/kaggle/working"):
        require_kaggle_working_root(Path("/usr/local/agridiagnose"))


def test_bootstrap_commands_use_isolated_uv_and_python():
    layout = KaggleRuntimeLayout(
        Path("/kaggle/working/runtime"), Path("/kaggle/working/project")
    )
    commands = bootstrap_commands(layout)
    flattened = "\n".join(" ".join(command) for command in commands)
    assert commands[0][0] == "curl"
    assert any(f"/uv/{UV_VERSION}/install.sh" in part for part in commands[0])
    assert commands[1] == ["sh", str(layout.uv_installer)]
    assert commands[2] == [str(layout.uv_executable), "--version"]
    assert commands[3][1:4] == ["python", "install", "3.11"]
    assert "--managed-python" in commands[4]
    assert commands[5][1:3] == ["pip", "install"]
    assert str(layout.requirements_path) in commands[5]
    assert commands[5][-4:] == [
        "--extra-index-url",
        NVIDIA_PYPI_INDEX_URL,
        "--index-strategy",
        UV_INDEX_STRATEGY,
    ]
    assert "python -m venv" not in flattened
    assert "python -m pip" not in flattened
    assert "uv-bootstrap/bin/python" not in flattened
    environment = bootstrap_environment(layout)
    assert environment["UV_PYTHON_INSTALL_DIR"] == str(
        layout.managed_python_dir
    )
    assert environment["UV_UNMANAGED_INSTALL"] == str(layout.uv_bin_dir)
    assert environment["UV_NO_MODIFY_PATH"] == "1"


def test_bootstrap_sanitizes_targeted_environment_contamination_only():
    layout = KaggleRuntimeLayout(
        Path("/kaggle/working/runtime"), Path("/kaggle/working/project")
    )
    inherited = {
        "PYTHONPATH": "/kaggle/lib/kagglesitepackages",
        "PYTHONHOME": "/usr/local",
        "VIRTUAL_ENV": "/kaggle/host-venv",
        "UV_INSTALL_DIR": "/usr/local/bin",
        "PATH": "/usr/local/cuda/bin:/usr/bin",
        "LD_LIBRARY_PATH": "/usr/local/cuda/lib64",
        "NVIDIA_VISIBLE_DEVICES": "all",
    }
    environment = bootstrap_environment(layout, inherited)
    assert "PYTHONPATH" not in environment
    assert "PYTHONHOME" not in environment
    assert "VIRTUAL_ENV" not in environment
    assert "UV_INSTALL_DIR" not in environment
    assert environment["PYTHONNOUSERSITE"] == "1"
    assert environment["PATH"] == inherited["PATH"]
    assert environment["LD_LIBRARY_PATH"] == inherited["LD_LIBRARY_PATH"]
    assert environment["NVIDIA_VISIBLE_DEVICES"] == "all"
    assert environment["UV_UNMANAGED_INSTALL"] == str(layout.uv_bin_dir)


def create_source_data_roots(base: Path, parents: dict[str, str]) -> dict[str, Path]:
    directory_names = {
        "historical": "historical-mendeley-39",
        "pldd_up": "pldd_up",
        "seasonal_corn": "seasonal_corn",
        "plantdoc_train": "plantdoc-train",
        "banu_deb": "potato-banu-deb-originals",
    }
    created = {}
    for key, directory_name in directory_names.items():
        path = base / parents[key] / directory_name
        path.mkdir(parents=True)
        created[key] = path.resolve()
    return created


def test_source_discovery_supports_standard_kaggle_slug_layout(tmp_path):
    input_root = tmp_path / "input"
    parents = {
        "historical": "agridiagnose-historical",
        "pldd_up": "agridiagnose-pldd-up",
        "seasonal_corn": "agridiagnose-seasonal-corn",
        "plantdoc_train": "agridiagnose-plantdoc-train",
        "banu_deb": "agridiagnose-banu-deb",
    }
    expected = create_source_data_roots(input_root, parents)
    assert discover_kaggle_source_roots(input_root) == expected


def test_source_discovery_supports_nested_owner_and_misleading_parents(tmp_path):
    input_root = tmp_path / "input"
    owner_root = input_root / "datasets" / "portable-owner"
    parents = {
        "historical": "agridiagnose-historical",
        "pldd_up": "agridiagnose-plddd-up",
        "seasonal_corn": "agridiagnose-seasonal-corn",
        "plantdoc_train": "agridiagnose-pldd-up",
        "banu_deb": "agridiagnose-banu-deb",
    }
    expected = create_source_data_roots(owner_root, parents)
    assert discover_kaggle_source_roots(input_root) == expected


def test_source_discovery_fails_when_a_required_source_is_missing(tmp_path):
    input_root = tmp_path / "input"
    parents = {
        key: f"{key}-source"
        for key in source_roots()
        if key != "plantdoc_train"
    }
    directory_names = {
        "historical": "historical-mendeley-39",
        "pldd_up": "pldd_up",
        "seasonal_corn": "seasonal_corn",
        "banu_deb": "potato-banu-deb-originals",
    }
    for key, parent in parents.items():
        (input_root / parent / directory_names[key]).mkdir(parents=True)
    with pytest.raises(ValueError, match="Missing Kaggle source root for plantdoc_train"):
        discover_kaggle_source_roots(input_root)


def test_source_discovery_rejects_duplicate_ambiguous_roots(tmp_path):
    input_root = tmp_path / "input"
    parents = {key: f"{key}-source" for key in source_roots()}
    create_source_data_roots(input_root, parents)
    (input_root / "duplicate" / "pldd_up").mkdir(parents=True)
    with pytest.raises(ValueError, match="Ambiguous Kaggle source root for pldd_up"):
        discover_kaggle_source_roots(input_root)


def test_source_discovery_rejects_forbidden_test_like_path(tmp_path):
    input_root = tmp_path / "input"
    parents = {key: f"{key}-source" for key in source_roots()}
    create_source_data_roots(input_root, parents)
    safe = input_root / "plantdoc_train-source" / "plantdoc-train"
    safe.rmdir()
    (input_root / "plantdoc-test" / "plantdoc-train").mkdir(parents=True)
    with pytest.raises(ValueError, match="TEST-like"):
        discover_kaggle_source_roots(input_root)


def test_kaggle_uv_install_dir_cannot_override_unmanaged_runtime_destination():
    layout = KaggleRuntimeLayout(
        Path("/kaggle/working/agridiagnose-tf215-runtime"),
        Path("/kaggle/working/ai-plant-disease-detection"),
    )
    environment = bootstrap_environment(
        layout,
        {
            "UV_INSTALL_DIR": "/usr/local/bin",
            "UV_UNMANAGED_INSTALL": "/tmp/wrong-uv-bin",
            "UV_NO_MODIFY_PATH": "0",
        },
    )
    assert "UV_INSTALL_DIR" not in environment
    assert environment["UV_UNMANAGED_INSTALL"] == str(layout.uv_bin_dir)
    assert environment["UV_NO_MODIFY_PATH"] == "1"


def test_partial_legacy_bootstrap_without_pip_cannot_block_standalone_uv(tmp_path):
    layout = KaggleRuntimeLayout(tmp_path, tmp_path / "project")
    legacy_python = layout.obsolete_uv_bootstrap_venv / "bin/python"
    legacy_python.parent.mkdir(parents=True)
    legacy_python.write_text("partial Kaggle bootstrap", encoding="utf-8")

    commands = bootstrap_commands(layout)
    flattened = "\n".join(" ".join(command) for command in commands)
    assert str(legacy_python) not in flattened
    assert " -m pip " not in f" {flattened} "
    assert commands[0][0] == "curl"
    assert commands[1] == ["sh", str(layout.uv_installer)]
    assert commands[2][0] == str(layout.uv_executable)


def test_standalone_uv_version_is_strictly_pinned():
    validate_uv_version(f"uv {UV_VERSION}\n")
    validate_uv_version(f"uv {UV_VERSION} (x86_64-unknown-linux-gnu)\n")
    with pytest.raises(RuntimeError, match="version mismatch"):
        validate_uv_version("uv 0.12.4")
    with pytest.raises(RuntimeError, match="version mismatch"):
        validate_uv_version(f"uv {UV_VERSION} unexpected text")


def test_partial_or_wrong_experiment_python_requires_recreation(tmp_path, monkeypatch):
    python_path = tmp_path / "venvs/agridiagnose-tf215/bin/python"
    environment = {"PYTHONNOUSERSITE": "1"}
    assert has_approved_isolated_python(python_path, environment) is False

    python_path.parent.mkdir(parents=True)
    python_path.write_text("partial", encoding="utf-8")

    def broken_run(*args, **kwargs):
        raise OSError("incomplete interpreter")

    monkeypatch.setattr(subprocess, "run", broken_run)
    assert has_approved_isolated_python(python_path, environment) is False


def test_existing_experiment_python_is_reused_only_when_it_is_311(
    tmp_path, monkeypatch
):
    python_path = tmp_path / "venvs/agridiagnose-tf215/bin/python"
    python_path.parent.mkdir(parents=True)
    python_path.write_text("placeholder", encoding="utf-8")
    captured = {}

    def approved_run(command, **kwargs):
        captured["command"] = command
        captured["environment"] = kwargs["env"]
        return subprocess.CompletedProcess(command, 0, stdout="3.11\n", stderr="")

    monkeypatch.setattr(subprocess, "run", approved_run)
    environment = {"PYTHONNOUSERSITE": "1"}
    assert has_approved_isolated_python(python_path, environment) is True
    assert "-I" in captured["command"]
    assert captured["environment"] == environment


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("python_version", "3.12.13"),
        ("tensorflow_version", "2.20.0"),
        ("keras_version", "3.13.2"),
        ("numpy_version", "2.0.2"),
        ("tensorflow_built_with_cuda", False),
        ("tensorflow_gpu_devices", []),
        ("gpu_smoke_test_passed", False),
        ("gpu_smoke_device", "/job:localhost/device:CPU:0"),
    ],
)
def test_runtime_gate_rejects_every_incompatible_dimension(field, value):
    payload = valid_runtime_payload()
    payload[field] = value
    with pytest.raises(RuntimeError, match="KAGGLE_TF215_GPU_RUNTIME_FAILED"):
        validate_runtime_payload(payload)


def test_runtime_gate_accepts_only_approved_tf215_gpu_payload():
    validate_runtime_payload(valid_runtime_payload())


def test_execution_config_defaults_to_training_disabled(tmp_path):
    payload = build_execution_config(source_roots())
    assert payload["start_training"] is False
    assert payload["internal_test_loaded"] is False
    assert payload["plantdoc_test_loaded"] is False
    path = tmp_path / "config.json"
    write_execution_config(path, payload)
    assert load_execution_config(path)["start_training"] is False


def test_execution_config_requires_double_training_authorization(tmp_path):
    payload = build_execution_config(source_roots(), start_training=True)
    path = tmp_path / "config.json"
    write_execution_config(path, payload)
    with pytest.raises(RuntimeError, match="TRAINING_DISABLED_BY_USER"):
        load_execution_config(path)
    assert load_execution_config(path, allow_training=True)["start_training"] is True


def test_execution_config_rejects_test_and_non_kaggle_roots():
    roots = source_roots()
    roots["plantdoc_train"] = "/kaggle/input/plantdoc-test"
    with pytest.raises(ValueError, match="TEST-like"):
        build_execution_config(roots)
    roots = source_roots()
    roots["historical"] = "C:/private/data"
    with pytest.raises(ValueError, match="/kaggle/input"):
        build_execution_config(roots)


def test_isolated_entrypoint_never_uses_kernel_python():
    layout = KaggleRuntimeLayout(
        Path("/kaggle/working/runtime"), Path("/kaggle/working/project")
    )
    command = isolated_entrypoint_command(
        layout,
        "preflight",
        config_path=layout.execution_config,
        output_path=layout.working_root / "preflight.json",
    )
    assert command[0] == str(layout.experiment_python)
    assert "run_kaggle_model_v2_experiment_a.py" in command[1]
    assert "--authorize-training" not in command


def test_isolated_requirements_keep_approved_scientific_stack():
    requirements = (PROJECT_ROOT / "requirements-kaggle-tf215.txt").read_text(
        encoding="utf-8"
    )
    assert "tensorflow[and-cuda]==2.15.0" in requirements
    assert "keras==2.15.0" in requirements
    assert "numpy==1.26.4" in requirements
    assert "tensorflow==2.20" not in requirements
    assert "keras==3" not in requirements


def test_tf215_cuda_resolution_uses_official_nvidia_package_index_only():
    layout = KaggleRuntimeLayout(
        Path("/kaggle/working/runtime"), Path("/kaggle/working/project")
    )
    install_command = bootstrap_commands(layout)[5]
    assert NVIDIA_PYPI_INDEX_URL == "https://pypi.nvidia.com"
    assert install_command[-4:] == [
        "--extra-index-url",
        NVIDIA_PYPI_INDEX_URL,
        "--index-strategy",
        "unsafe-first-match",
    ]
    assert "unsafe-best-match" not in install_command
    requirements = (PROJECT_ROOT / "requirements-kaggle-tf215.txt").read_text(
        encoding="utf-8"
    )
    assert "tensorflow[and-cuda]==2.15.0" in requirements


def test_bootstrap_and_entrypoint_are_source_only_and_safe():
    bootstrap = (PROJECT_ROOT / "scripts/bootstrap_kaggle_tf215_runtime.py").read_text(
        encoding="utf-8"
    )
    entrypoint = (
        PROJECT_ROOT / "scripts/run_kaggle_model_v2_experiment_a.py"
    ).read_text(encoding="utf-8")
    assert "model.fit(" not in bootstrap
    assert "python -m pip" not in bootstrap
    assert "uv-bootstrap/bin/python" not in bootstrap
    assert "UV_UNMANAGED_INSTALL" in (
        PROJECT_ROOT / "training/kaggle_runtime.py"
    ).read_text(encoding="utf-8")
    assert "plant_disease_model.h5" not in entrypoint
    assert "--authorize-training" in entrypoint
    assert "TRAINING_DISABLED_BY_USER" in entrypoint
    assert "KAGGLE_TF215_GPU_RUNTIME_FAILED" in entrypoint
