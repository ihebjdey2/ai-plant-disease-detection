from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training.kaggle_runtime import (  # noqa: E402
    KaggleRuntimeLayout,
    UV_VERSION,
    bootstrap_commands,
    bootstrap_environment,
    command_lines,
    require_kaggle_working_root,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create an isolated Kaggle Python 3.11 / TensorFlow 2.15 runtime."
    )
    parser.add_argument(
        "--working-root",
        type=Path,
        default=Path("/kaggle/working/agridiagnose-tf215-runtime"),
    )
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    working_root = require_kaggle_working_root(args.working_root)
    project_root = args.project_root.expanduser().resolve()
    layout = KaggleRuntimeLayout(working_root, project_root)
    if not layout.requirements_path.is_file():
        raise FileNotFoundError(f"Missing isolated requirements: {layout.requirements_path}")
    commands = bootstrap_commands(layout)
    if args.dry_run:
        print(json.dumps({"commands": command_lines(commands)}, indent=2))
        return 0

    working_root.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment.update(bootstrap_environment(layout))
    for index, command in enumerate(commands):
        if index == 0 and layout.uv_bootstrap_python.is_file():
            continue
        if index == 3 and layout.experiment_python.is_file():
            continue
        subprocess.run(command, check=True, env=environment)

    if not layout.experiment_python.is_file():
        raise RuntimeError("Isolated Python 3.11 executable was not created.")
    installed = subprocess.run(
        [
            str(layout.uv_executable),
            "pip",
            "freeze",
            "--python",
            str(layout.experiment_python),
        ],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    ).stdout
    layout.installed_packages.write_text(installed, encoding="utf-8")
    requirements_hash = hashlib.sha256(layout.requirements_path.read_bytes()).hexdigest()
    payload = {
        "status": "BOOTSTRAP_COMPLETE_RUNTIME_GATE_PENDING",
        "host_python_version": platform.python_version(),
        "host_python_unchanged": True,
        "uv_version_requested": UV_VERSION,
        "python_request": "3.11",
        "isolated_python": str(layout.experiment_python),
        "requirements": layout.requirements_path.name,
        "requirements_sha256": requirements_hash,
        "installed_packages_report": layout.installed_packages.name,
        "training_performed": False,
    }
    report = working_root / "bootstrap.json"
    report.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
