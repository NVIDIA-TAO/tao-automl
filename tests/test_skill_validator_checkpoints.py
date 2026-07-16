import importlib.util
import sys
from pathlib import Path


def _load_validator_module():
    path = Path(__file__).parents[1] / "scripts" / "validate_skill_automl_model.py"
    spec = importlib.util.spec_from_file_location("skill_automl_validator_for_test", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_checkpoint_selection_prefers_highest_explicit_epoch_and_step():
    validator = _load_validator_module()
    checkpoints = [
        "/results/train/model_epoch_000_step_00010.pth",
        "/results/train/model_epoch_001_step_00020.pth",
        "/results/train/model_latest.pth",
    ]

    assert validator._prefer_epoch_or_step_checkpoint(checkpoints) == checkpoints[1]


def test_nvdinov2_checkpoint_selection_prefers_latest_student_checkpoint():
    validator = _load_validator_module()
    checkpoints = [
        "/results/train/model_epoch_003_step_00100.pth",
        "/results/train/teacher_epoch_003_step_00100.pth",
        "/results/train/student_epoch_002_step_00200.pth",
        "/results/train/student_epoch_003_step_00100.pth",
    ]

    assert (
        validator._prefer_epoch_or_step_checkpoint(checkpoints, model="nvdinov2")
        == checkpoints[3]
    )
