from __future__ import annotations

import json
from pathlib import Path

import pytest

from training.kaggle_runtime import (
    KaggleRuntimeLayout,
    UV_VERSION,
    bootstrap_commands,
    bootstrap_environment,
    build_execution_config,
    isolated_entrypoint_command,
    load_execution_config,
    require_kaggle_working_root,
    validate_runtime_payload,
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
        layout.uv_bootstrap_venv,
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
    commands = bootstrap_commands(layout, "/usr/bin/python3.12")
    assert commands[0][:3] == ["/usr/bin/python3.12", "-m", "venv"]
    assert commands[1][-1] == f"uv=={UV_VERSION}"
    assert commands[2][1:4] == ["python", "install", "3.11"]
    assert "--managed-python" in commands[3]
    assert commands[4][1:3] == ["pip", "install"]
    assert str(layout.requirements_path) in commands[4]
    assert bootstrap_environment(layout)["UV_PYTHON_INSTALL_DIR"] == str(
        layout.managed_python_dir
    )


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


def test_bootstrap_and_entrypoint_are_source_only_and_safe():
    bootstrap = (PROJECT_ROOT / "scripts/bootstrap_kaggle_tf215_runtime.py").read_text(
        encoding="utf-8"
    )
    entrypoint = (
        PROJECT_ROOT / "scripts/run_kaggle_model_v2_experiment_a.py"
    ).read_text(encoding="utf-8")
    assert "model.fit(" not in bootstrap
    assert "plant_disease_model.h5" not in entrypoint
    assert "--authorize-training" in entrypoint
    assert "TRAINING_DISABLED_BY_USER" in entrypoint
    assert "KAGGLE_TF215_GPU_RUNTIME_FAILED" in entrypoint
