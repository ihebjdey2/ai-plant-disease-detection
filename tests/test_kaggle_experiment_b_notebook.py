from __future__ import annotations

import json
import re
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = PROJECT_ROOT / "notebooks/kaggle_model_v2_experiment_b.ipynb"
DOCUMENTATION_PATH = PROJECT_ROOT / "docs/kaggle-model-v2-experiment-b.md"
APPROVED_IMPLEMENTATION_REVISION = "1960e63d6eb8049d9b005bbbed377a1db085310d"


def load_notebook() -> dict[str, object]:
    return json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))


def test_experiment_b_notebook_is_clean_compilable_and_training_disabled():
    notebook = load_notebook()
    code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
    code = "\n".join("".join(cell["source"]) for cell in code_cells)

    for index, cell in enumerate(code_cells):
        assert cell["execution_count"] is None
        assert cell["outputs"] == []
        compile("".join(cell["source"]), f"experiment-b-cell-{index}", "exec")

    assert "START_TRAINING = False" in code
    assert "AUTHORIZE_TRAINING_CLI = False" in code
    assert "RUN_VALIDATION_COMPARISON = False" in code
    assert "BATCH_SIZE = 32" in code
    assert "INTERRUPTED_PHASE_ACTION = 'fail'" in code
    assert "interrupted_phase_action=INTERRUPTED_PHASE_ACTION" in code
    assert "RESTART_INTERRUPTED_PHASE" not in code
    assert "restart_interrupted_phase" not in code
    assert "build_execution_config_b" in code
    assert "run_kaggle_model_v2_experiment_b.py" in code
    assert "KAGGLE_TF215_GPU_EXPERIMENT_B_PREFLIGHT_PASSED" in code
    assert "--authorize-training" in code
    assert ".fit(" not in code

    preflight_position = code.index("'preflight'")
    training_position = code.index("'train'")
    assert preflight_position < training_position


def test_experiment_b_notebook_is_revision_pinned_or_fails_closed_until_pin():
    code = "\n".join(
        "".join(cell["source"])
        for cell in load_notebook()["cells"]
        if cell["cell_type"] == "code"
    )
    match = re.search(r"APPROVED_CODE_REVISION = '([^']+)'", code)
    assert match is not None
    revision = match.group(1)
    assert revision == APPROVED_IMPLEMENTATION_REVISION
    assert re.fullmatch(r"[0-9a-f]{40}", revision)
    assert "APPROVED_CODE_REVISION_NOT_PINNED" in code


def test_experiment_b_notebook_keeps_test_data_out_of_executable_cells():
    code = "\n".join(
        "".join(cell["source"])
        for cell in load_notebook()["cells"]
        if cell["cell_type"] == "code"
    ).casefold()

    assert "dataset-v2-test.csv" not in code
    assert "plantdoc-test" not in code
    assert "plantdoc_test/" not in code
    assert "internal-test" not in code
    assert "internal_test" in code  # Assertions inspect false safety metadata only.
    assert "internal_test_loaded'] is false" in code
    assert "plantdoc_test_loaded'] is false" in code


def test_experiment_b_documentation_records_control_and_safety_contract():
    documentation = DOCUMENTATION_PATH.read_text(encoding="utf-8")

    for required in (
        "TRAIN-only",
        "58,857",
        "7,362",
        "block_13_expand",
        "52 BatchNormalization",
        "Class weights remain `None`",
        "VALIDATION-only",
        "paired, true-class-stratified bootstrap",
        "start_training=true",
        "--authorize-training",
        "MPLBACKEND=Agg",
        "exact-enough",
        "initial_epoch=5",
        "interrupted_phase_action",
    ):
        assert required in documentation


def test_experiment_b_local_outputs_are_gitignored():
    patterns = set((PROJECT_ROOT / ".gitignore").read_text(encoding="utf-8").splitlines())

    assert "agridiagnose-exp-b-results/" in patterns
    assert "agridiagnose-exp-b-results.zip" in patterns
    assert "agridiagnose-exp-a-vs-b-validation/" in patterns
