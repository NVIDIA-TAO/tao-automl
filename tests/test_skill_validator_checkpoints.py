import importlib.util
import json
import sys
import tarfile
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


def test_checkpoint_progress_detects_non_advancing_sparse4d_promotion():
    validator = _load_validator_module()

    parent = validator._checkpoint_progress("model_epoch_000_step_00003.pth")
    promoted = validator._checkpoint_progress("model_epoch_000_step_00003.pth")

    assert parent == ("step", 3)
    assert promoted == parent


def test_checkpoint_progress_accepts_advancing_sparse4d_promotion():
    validator = _load_validator_module()

    parent = validator._checkpoint_progress("model_epoch_000_step_00003.pth")
    promoted = validator._checkpoint_progress("model_epoch_001_step_00006.pth")

    assert parent == ("step", 3)
    assert promoted == ("step", 6)


def test_cosmos_checkpoint_actions_receive_adapter_directory(tmp_path):
    validator = _load_validator_module()
    checkpoint = (
        tmp_path / "job" / "safetensors" / "epoch_2" / "adapter_model.safetensors"
    )

    assert validator._checkpoint_action_container_path(
        str(checkpoint), tmp_path, "cosmos-rl"
    ) == "/results/job/safetensors/epoch_2"
    assert validator._checkpoint_action_container_path(
        str(checkpoint), tmp_path, "dino"
    ) == "/results/job/safetensors/epoch_2/adapter_model.safetensors"


def test_cosmos_inference_uses_referenced_nested_video(tmp_path):
    validator = _load_validator_module()
    out_dir = tmp_path / "cosmos-rl"
    eval_root = tmp_path / "datasets" / "cosmos-rl" / "eval"
    video = eval_root / "scene" / "clips" / "sample.webm"
    video.parent.mkdir(parents=True)
    video.write_bytes(b"video")
    (eval_root / "annotations.json").write_text(json.dumps([
        {"video": "scene/clips/sample.webm"}
    ]))

    assert validator._cosmos_inference_media_path(out_dir) == (
        "/data/automl_datasets/cosmos-rl/eval/scene/clips/sample.webm"
    )


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


def test_clip_uses_training_metric_for_selection_and_test_metric_for_checkpoint_eval():
    validator = _load_validator_module()

    assert validator._checkpoint_evaluation_metric("clip", "val/t2i_mAP") == "test/t2i_mAP"


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


def test_clip_caption_fallback_uses_sorted_coco_class_labels():
    validator = _load_validator_module()
    payload = {
        "images": [
            {"id": 7, "file_name": "sample.jpg"},
            {"id": 8, "file_name": "unannotated.jpg"},
        ],
        "categories": [
            {"id": 2, "name": "helmet"},
            {"id": 1, "name": "person"},
        ],
        "annotations": [
            {"image_id": 7, "category_id": 1},
            {"image_id": 7, "category_id": 2},
            {"image_id": 7, "category_id": 1},
        ],
    }

    assert validator._clip_captions_from_coco(payload) == {
        "sample.jpg": "a photo containing helmet, person",
    }


def test_cosmos_staging_extracts_referenced_video_and_measures_fps(tmp_path, monkeypatch):
    validator = _load_validator_module()
    source_video = tmp_path / "source.mp4"
    source_video.write_bytes(b"real-video-bytes")
    archive = tmp_path / "videos.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        tar.add(source_video, arcname="videos/scene/clips/sample.mp4")
    annotations = tmp_path / "source_annotations.json"
    annotations.write_text(json.dumps([
        {
            "id": "sample",
            "video": "scene/clips/sample.mp4",
            "conversations": [],
        }
    ]))
    monkeypatch.setattr(validator, "_video_fps", lambda _path: 29.97)
    monkeypatch.setattr(validator, "_ensure_cosmos_video_codec", lambda path: path)

    target = tmp_path / "staged"
    assert validator._stage_cosmos_split(annotations, archive, target) == 1

    staged = json.loads((target / "annotations.json").read_text())
    assert staged[0]["video_fps"] == 29.97
    assert (target / "scene/clips/sample.mp4").read_bytes() == b"real-video-bytes"
