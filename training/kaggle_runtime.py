from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence


UV_VERSION = "0.12.3"
UV_INSTALLER_URL = f"https://astral.sh/uv/{UV_VERSION}/install.sh"
NVIDIA_PYPI_INDEX_URL = "https://pypi.nvidia.com"
UV_INDEX_STRATEGY = "unsafe-first-match"
PYTHON_REQUEST = "3.11"
APPROVED_TENSORFLOW_PREFIX = "2.15."
APPROVED_KERAS_PREFIX = "2.15."
APPROVED_NUMPY_VERSION = "1.26.4"
SOURCE_ROOT_KEYS = (
    "historical",
    "pldd_up",
    "seasonal_corn",
    "plantdoc_train",
    "banu_deb",
)
SOURCE_DATA_ROOT_NAMES = {
    "historical": "historical-mendeley-39",
    "pldd_up": "pldd_up",
    "seasonal_corn": "seasonal_corn",
    "plantdoc_train": "plantdoc-train",
    "banu_deb": "potato-banu-deb-originals",
}
ALLOWED_BATCH_SIZES = (32, 16, 8)
FORBIDDEN_SOURCE_MARKERS = (
    "internal-test",
    "internal_test",
    "plantdoc-test",
    "plantdoc_test",
)


@dataclass(frozen=True)
class KaggleRuntimeLayout:
    working_root: Path
    project_root: Path

    @property
    def uv_bin_dir(self) -> Path:
        return self.working_root / "uv-bin"

    @property
    def uv_installer(self) -> Path:
        return self.working_root / "uv-installer.sh"

    @property
    def uv_executable(self) -> Path:
        return self.uv_bin_dir / "uv"

    @property
    def obsolete_uv_bootstrap_venv(self) -> Path:
        """Location created by the superseded host-pip bootstrap."""
        return self.working_root / "uv-bootstrap"

    @property
    def managed_python_dir(self) -> Path:
        return self.working_root / "uv-python"

    @property
    def uv_cache_dir(self) -> Path:
        return self.working_root / "uv-cache"

    @property
    def experiment_venv(self) -> Path:
        return self.working_root / "venvs/agridiagnose-tf215"

    @property
    def experiment_python(self) -> Path:
        return self.experiment_venv / "bin/python"

    @property
    def requirements_path(self) -> Path:
        return self.project_root / "requirements-kaggle-tf215.txt"

    @property
    def runtime_report(self) -> Path:
        return self.working_root / "tf215-gpu-runtime.json"

    @property
    def installed_packages(self) -> Path:
        return self.working_root / "installed-packages.txt"

    @property
    def execution_config(self) -> Path:
        return self.working_root / "experiment-a-config.json"


def require_kaggle_working_root(path: Path) -> Path:
    resolved = Path(path).expanduser().resolve()
    allowed = Path("/kaggle/working").resolve()
    if resolved != allowed and allowed not in resolved.parents:
        raise ValueError("Kaggle runtime files must stay under /kaggle/working.")
    return resolved


def bootstrap_environment(
    layout: KaggleRuntimeLayout,
    inherited: Mapping[str, str] | None = None,
) -> dict[str, str]:
    environment = dict(inherited or {})
    contaminated = {
        "PYTHONPATH",
        "PYTHONHOME",
        "VIRTUAL_ENV",
        # Kaggle may set this to /usr/local/bin. The standalone installer gives
        # it precedence over the runtime-controlled unmanaged destination.
        "UV_INSTALL_DIR",
    }
    for key in tuple(environment):
        if key.upper() in contaminated:
            environment.pop(key)
    environment.update({
        "PYTHONNOUSERSITE": "1",
        "UV_UNMANAGED_INSTALL": str(layout.uv_bin_dir),
        "UV_NO_MODIFY_PATH": "1",
        "UV_PYTHON_INSTALL_DIR": str(layout.managed_python_dir),
        "UV_CACHE_DIR": str(layout.uv_cache_dir),
        "UV_NO_CONFIG": "1",
    })
    return environment


def bootstrap_commands(layout: KaggleRuntimeLayout) -> list[list[str]]:
    return [
        [
            "curl",
            "--proto",
            "=https",
            "--tlsv1.2",
            "-LsSf",
            UV_INSTALLER_URL,
            "-o",
            str(layout.uv_installer),
        ],
        ["sh", str(layout.uv_installer)],
        [str(layout.uv_executable), "--version"],
        [
            str(layout.uv_executable),
            "python",
            "install",
            PYTHON_REQUEST,
            "--install-dir",
            str(layout.managed_python_dir),
        ],
        [
            str(layout.uv_executable),
            "venv",
            "--python",
            PYTHON_REQUEST,
            "--managed-python",
            str(layout.experiment_venv),
        ],
        [
            str(layout.uv_executable),
            "pip",
            "install",
            "--python",
            str(layout.experiment_python),
            "--requirements",
            str(layout.requirements_path),
            "--extra-index-url",
            NVIDIA_PYPI_INDEX_URL,
            "--index-strategy",
            UV_INDEX_STRATEGY,
        ],
        [
            str(layout.uv_executable),
            "pip",
            "check",
            "--python",
            str(layout.experiment_python),
        ],
    ]


def validate_uv_version(output: str) -> None:
    normalized = output.strip()
    match = re.fullmatch(
        r"uv\s+(\d+\.\d+\.\d+)(?:\s+\([^()\r\n]+\))?",
        normalized,
    )
    if match is None or match.group(1) != UV_VERSION:
        raise RuntimeError(
            f"Standalone uv version mismatch: expected {UV_VERSION}, got {normalized!r}."
        )


def validate_runtime_payload(payload: Mapping[str, object]) -> None:
    checks = {
        "python": str(payload.get("python_version", "")).startswith("3.11."),
        "tensorflow": str(payload.get("tensorflow_version", "")).startswith(
            APPROVED_TENSORFLOW_PREFIX
        ),
        "keras": str(payload.get("keras_version", "")).startswith(
            APPROVED_KERAS_PREFIX
        ),
        "numpy": payload.get("numpy_version") == APPROVED_NUMPY_VERSION,
        "cuda_build": payload.get("tensorflow_built_with_cuda") is True,
        "gpu": bool(payload.get("tensorflow_gpu_devices")),
        "gpu_smoke": payload.get("gpu_smoke_test_passed") is True,
        "gpu_device": "GPU" in str(payload.get("gpu_smoke_device", "")).upper(),
    }
    failed = sorted(name for name, passed in checks.items() if not passed)
    if failed:
        raise RuntimeError(
            "KAGGLE_TF215_GPU_RUNTIME_FAILED: " + ", ".join(failed)
        )


def discover_kaggle_source_roots(input_root: Path) -> dict[str, Path]:
    """Find each approved data root exactly once below a Kaggle input mount."""
    root = Path(input_root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Kaggle input root is unavailable: {root}")

    keys_by_directory = {
        directory_name: key
        for key, directory_name in SOURCE_DATA_ROOT_NAMES.items()
    }
    matches: dict[str, list[Path]] = {key: [] for key in SOURCE_ROOT_KEYS}
    for candidate in root.rglob("*"):
        key = keys_by_directory.get(candidate.name)
        if key is None or not candidate.is_dir():
            continue
        resolved = candidate.resolve()
        if resolved != root and root not in resolved.parents:
            raise ValueError(
                f"Discovered Kaggle source escapes the input root: {candidate}"
            )
        if any(
            marker in resolved.as_posix().casefold()
            for marker in FORBIDDEN_SOURCE_MARKERS
        ):
            raise ValueError(f"Locked TEST-like source root is forbidden: {resolved}")
        matches[key].append(resolved)

    resolved_roots: dict[str, Path] = {}
    for key in SOURCE_ROOT_KEYS:
        unique = sorted(set(matches[key]), key=lambda path: path.as_posix())
        expected = SOURCE_DATA_ROOT_NAMES[key]
        if not unique:
            raise ValueError(
                f"Missing Kaggle source root for {key}: expected one directory "
                f"named {expected!r} below {root}."
            )
        if len(unique) != 1:
            rendered = ", ".join(path.as_posix() for path in unique)
            raise ValueError(
                f"Ambiguous Kaggle source root for {key}: expected exactly one "
                f"directory named {expected!r}; found {len(unique)}: {rendered}"
            )
        resolved_roots[key] = unique[0]
    return resolved_roots


def build_execution_config(
    source_roots: Mapping[str, str | Path],
    *,
    batch_size: int = 32,
    start_training: bool = False,
    restart_interrupted_phase: bool = False,
) -> dict[str, object]:
    if set(source_roots) != set(SOURCE_ROOT_KEYS):
        raise ValueError("Exactly five approved Kaggle source roots are required.")
    if batch_size not in ALLOWED_BATCH_SIZES:
        raise ValueError("Kaggle batch size must be 32, 16, or 8.")
    normalized: dict[str, str] = {}
    for key in SOURCE_ROOT_KEYS:
        value = Path(source_roots[key]).as_posix()
        lowered = value.casefold()
        if not value.startswith("/kaggle/input/"):
            raise ValueError(f"{key} must point under /kaggle/input.")
        if any(marker in lowered for marker in FORBIDDEN_SOURCE_MARKERS):
            raise ValueError(f"Locked TEST-like source root is forbidden: {key}.")
        normalized[key] = value
    return {
        "experiment": "agri-diagnose-v2-exp-a",
        "source_roots": normalized,
        "batch_size": batch_size,
        "start_training": bool(start_training),
        "restart_interrupted_phase": bool(restart_interrupted_phase),
        "internal_test_loaded": False,
        "plantdoc_test_loaded": False,
    }


def write_execution_config(path: Path, payload: Mapping[str, object]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def load_execution_config(
    path: Path, *, allow_training: bool = False
) -> dict[str, object]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    validated = build_execution_config(
        payload.get("source_roots", {}),
        batch_size=int(payload.get("batch_size", 0)),
        start_training=bool(payload.get("start_training", False)),
        restart_interrupted_phase=bool(
            payload.get("restart_interrupted_phase", False)
        ),
    )
    if validated["start_training"] and not allow_training:
        raise RuntimeError("TRAINING_DISABLED_BY_USER")
    return validated


def isolated_entrypoint_command(
    layout: KaggleRuntimeLayout,
    action: str,
    *,
    config_path: Path | None = None,
    output_path: Path | None = None,
    preflight_report_path: Path | None = None,
    authorize_training: bool = False,
) -> list[str]:
    if action not in {"verify-runtime", "preflight", "train", "finalize-existing"}:
        raise ValueError(f"Unsupported isolated action: {action}")
    command = [
        str(layout.experiment_python),
        str(layout.project_root / "scripts/run_kaggle_model_v2_experiment_a.py"),
        action,
    ]
    if config_path is not None:
        command.extend(["--config", str(config_path)])
    if output_path is not None:
        command.extend(["--output", str(output_path)])
    if preflight_report_path is not None:
        command.extend(["--preflight-report", str(preflight_report_path)])
    if authorize_training:
        command.append("--authorize-training")
    return command


def is_full_git_revision(value: str) -> bool:
    return re.fullmatch(r"[0-9a-f]{40}", value) is not None


def command_lines(commands: Sequence[Sequence[str]]) -> list[str]:
    return [" ".join(command) for command in commands]
