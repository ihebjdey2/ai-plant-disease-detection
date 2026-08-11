from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

from app.taxonomy import CLASS_NAMES as APP_CLASS_NAMES
from training.taxonomy import CLASS_NAMES


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_TAXONOMY_SHA256 = (
    "2207c34ff2673bde7f36c53938cf5e6d97ca0652f21ef087be15680851ae87da"
)


def test_framework_neutral_taxonomy_contract_is_locked():
    assert len(CLASS_NAMES) == 39
    assert CLASS_NAMES[4] == "Background without leaves"
    serialized = json.dumps(
        CLASS_NAMES, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    assert hashlib.sha256(serialized).hexdigest() == EXPECTED_TAXONOMY_SHA256


def test_flask_taxonomy_module_is_a_compatibility_reexport():
    assert APP_CLASS_NAMES is CLASS_NAMES


def test_neutral_taxonomy_imports_without_any_site_packages():
    command = [
        sys.executable,
        "-S",
        "-c",
        (
            "from training.taxonomy import CLASS_NAMES; "
            "assert len(CLASS_NAMES) == 39; "
            "assert CLASS_NAMES[4] == 'Background without leaves'"
        ),
    ]
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def test_kaggle_training_module_does_not_import_app_or_flask():
    code = r"""
import importlib.abc
import sys

class BlockWebImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "app" or fullname.startswith("app."):
            raise AssertionError(f"Kaggle ML imported Flask app package: {fullname}")
        if fullname == "flask" or fullname.startswith("flask_"):
            raise AssertionError(f"Kaggle ML imported web dependency: {fullname}")
        return None

sys.meta_path.insert(0, BlockWebImports())
import training.kaggle_experiment_a
assert "app" not in sys.modules
assert "flask" not in sys.modules
"""
    environment = os.environ.copy()
    for name in ("PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV"):
        environment.pop(name, None)
    environment["PYTHONNOUSERSITE"] = "1"
    subprocess.run(
        [sys.executable, "-c", code],
        cwd=PROJECT_ROOT,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
