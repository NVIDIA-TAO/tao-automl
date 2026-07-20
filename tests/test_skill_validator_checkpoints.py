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


def test_pbt_best_job_selection_uses_archived_job_id_not_reused_member_id():
    validator = _load_validator_module()
    generation_one = {
        "rec_id": 0,
        "job_id": "best-generation-one",
        "checkpoint_paths": ["/results/best-generation-one/model_epoch_000.pth"],
    }
    generation_two = {
        "rec_id": 0,
        "job_id": "latest-generation-two",
        "checkpoint_paths": ["/results/latest-generation-two/model_epoch_001.pth"],
    }

    selected = validator._select_best_job(
        [generation_one, generation_two],
        {"rec_id": 0, "job_id": "best-generation-one"},
        {0: generation_two},
    )

    assert selected is generation_one


def test_depth_d1_direction_is_model_specific():
    validator = _load_validator_module()

    assert validator._direction("val/d1", model="depth-net-mono") == "maximize"
    assert validator._direction("val/d1", model="depth-net-stereo") == "minimize"
    assert validator._direction("val/epe", model="depth-net-stereo") == "minimize"


def test_nvdinov2_pbt_keeps_checkpoint_neutral_worker_parameter():
    validator = _load_validator_module()

    assert validator._pbt_resume_safe_parameters(
        ["dataset.workers", "dataset.batch_size"], "nvdinov2"
    ) == ["dataset.workers"]


def test_pbt_rejects_structural_parameters_when_no_safe_fallback_exists():
    validator = _load_validator_module()

    assert validator._pbt_resume_safe_parameters(
        ["model.hidden_dim", "dataset.batch_size"], "dino"
    ) == []


def test_sparse4d_conversion_keeps_camera_group_generation_enabled(monkeypatch):
    validator = _load_validator_module()
    monkeypatch.setattr(validator, "_read_yaml", lambda _path: {})
    monkeypatch.setattr(validator, "_model_profile_key", lambda _path: "sparse4d")
    monkeypatch.setattr(
        validator,
        "_schema_keys",
        lambda _path, _action: {
            "aicity.num_frames",
            "aicity.anchor_init_config.num_anchor",
            "aicity.camera_grouping_mode",
        },
    )
    monkeypatch.setattr(validator, "_add_data_source_overrides", lambda *_args: None)

    specs = validator._build_dataset_convert_specs(
        model_dir=Path("sparse4d"),
        skill_text="",
        profile=validator.MODEL_PROFILES["sparse4d"],
    )

    assert specs["aicity"]["camera_grouping_mode"] == "random"
