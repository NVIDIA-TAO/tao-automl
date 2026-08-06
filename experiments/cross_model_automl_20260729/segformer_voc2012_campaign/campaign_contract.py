#!/usr/bin/env python3

"""Frozen SegFormer/VOC2012 three-mode campaign policy.

This module contains experiment intent only.  It never qualifies a checkpoint,
changes registry status, or chooses a winner.  Runtime eligibility is granted
only when a repository-supported PTM also has immutable evidence from a real
one-node/eight-GPU full-dataset training and evaluation workflow.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from tao_automl.ptm_registry import canonical_sha256, load_ptm_registry


MODES = ("accuracy", "latency", "multi_objective")
AGENT_FLAGS = (
    "agent_selected_candidate",
    "agent_injected_candidate",
    "agent_modified_search_space_after_results",
    "agent_changed_seed_after_results",
    "agent_changed_budget_after_results",
    "agent_changed_threshold_after_results",
    "agent_changed_ptm_after_results",
    "agent_overrode_winner",
)
SELECTION_FLAGS = (
    "selector_invoked_on_matched_measurements",
    "selection_time_objectives_replaced",
    "measurements_feed_selection",
    "measurements_feed_reselection",
    "algorithm_selected_candidate_overridden",
)

# PTM identity is a hierarchical categorical outer arm.  These are the two
# minimal, inference-invariant training parameters exposed by the packaged
# SegFormer train schema and useful across every FAN arm.
SEARCH_PARAMETERS = (
    "train.optim.lr",
    "train.optim.weight_decay",
)
SEARCH_SPACE = {
    "train.optim.lr": {
        "type": "float",
        "minimum": 2.0e-5,
        "maximum": 6.0e-4,
        "scale": "linear",
    },
    "train.optim.weight_decay": {
        "type": "float",
        "minimum": 1.0e-4,
        "maximum": 0.10,
        "scale": "linear",
    },
}

FROZEN_CANDIDATE_BUDGET = 30
# The already-preregistered AutoML search remains a ten-epoch experiment.
# Qualification v5 is a distinct, evidence-bound recovery boundary and must not
# silently mutate the search budget.
FROZEN_TRAINING_EPOCHS = 10
FROZEN_QUALIFICATION_TRAINING_EPOCHS = 50
FROZEN_SEARCH_SEED = 271828
FROZEN_TRAINING_SEED = 1234
FROZEN_CALIBRATION_POINTS_PER_ARM = 2
FROZEN_INVALID_RECOVERY_ISSUES_PER_ARM = 1
FROZEN_LATENCY_RETENTION = 0.90
FROZEN_LATENCY_TOLERANCE_MS = 0.73553775
FROZEN_VALIDATION_SANITY_MIN_MIOU = 0.10
FROZEN_SLURM_RETRY_CAP = 10
FROZEN_SLURM_PARTITION = "polar3"
FROZEN_SLURM_TIME_HOURS = 4.0
FROZEN_SLURM_TIMEOUT_HOURS = 3.8
FROZEN_IMAGE_SIZE = 512
FROZEN_BATCH_SIZE_PER_REPLICA = 4
FROZEN_HARDWARE = {
    "gpu_name": "NVIDIA A100-SXM4-80GB",
    "compute_capability": "8.0",
    "total_memory_bytes": 85174583296,
}
FROZEN_SQSH = {
    "path": (
        "/lustre/fsw/portfolios/edgeai/users/rarunachalam/"
        "nvcr.io_nvstaging_tao_tao-toolkit-pyt_7.1.0-rc-245-multiarch.sqsh"
    ),
    "sha256": (
        "e36640f9ae7a03bc80828cf7de93bd6bdbbb0fecf509a71a243be0ab5b497fc2"
    ),
    "image_reference": (
        "nvcr.io/nvstaging/tao/tao-toolkit-pyt:"
        "7.1.0-rc-245-multiarch"
    ),
}
QUALIFICATION_REVISION = 5
QUALIFICATION_CAMPAIGN_ID = (
    "segformer-voc2012-direct-full-ptm-qualification-v5"
)
FROZEN_QUALIFICATION_FIDELITY = {
    "source_recipe": (
        "nvidia_tao_pytorch/cv/segformer/experiment_specs/"
        "experiment_multi-class.yaml"
    ),
    "source_recipe_sha256": (
        "210b6b6c4952289e3dbc1f025b3f0b8f17a073702290cb565796ed6c6ea36b21"
    ),
    "training_epochs": FROZEN_QUALIFICATION_TRAINING_EPOCHS,
    "checkpoint_interval": FROZEN_QUALIFICATION_TRAINING_EPOCHS,
    "validation_interval": 1,
    "optimizer": "adamw",
    "learning_rate": 1.0e-4,
    "weight_decay": 5.0e-4,
    "random_color_enabled": False,
    "random_blur_enabled": False,
    "use_distributed_sampler": True,
}
FROZEN_QUALIFICATION_RUNTIME_OVERLAY = {
    "artifact_type": "tao_pytorch_source_overlay",
    "scope": "segformer_ptm_loading_and_global_ddp_metrics",
    "archive_path": (
        "/lustre/fsw/portfolios/edgeai/projects/"
        "edgeai_tao-ptm_image-foundation-model-clip/users/rarunachalam/"
        "tao-pytorch-overlays/segformer-product-fixes/"
        "2681dea4c876b759f8a0446491b3619e6120b531/"
        "tao-pytorch-segformer-product-fixes-2681dea4c876.tar"
    ),
    "archive_sha256": (
        "a7d5316816710b258c52001f979a22723c88fca5101a05ca3a48838ce81d1ee4"
    ),
    "archive_size_bytes": 61440,
    "installer_path": (
        "/lustre/fsw/portfolios/edgeai/projects/"
        "edgeai_tao-ptm_image-foundation-model-clip/users/rarunachalam/"
        "tao-pytorch-overlays/segformer-product-fixes/"
        "2681dea4c876b759f8a0446491b3619e6120b531/"
        "install_segformer_source_overlay.py"
    ),
    "installer_sha256": (
        "44535093217b74a7ad82e3bb58ecb45f28538491160de81be5e8e175518dc640"
    ),
    "installer_size_bytes": 8583,
    "receipt_path": (
        "/tmp/segformer-product-fixes-overlay-receipt.json"
    ),
    "source_repository": "tao-pytorch",
    "base_commit": "99741bc8229617d0d3dd52e30540111d55efd1af",
    "source_commits": [
        "eacc0c0e2e59776266bb07f0be205c71bd0830c3",
        "3b1e073571f3bbf3702b0ae837e9279ad12f4286",
        "2681dea4c876b759f8a0446491b3619e6120b531",
    ],
    "source_commit": "2681dea4c876b759f8a0446491b3619e6120b531",
    "combined_commit": "2681dea4c876b759f8a0446491b3619e6120b531",
    "file_count": 5,
    "required_actions": ["train", "evaluate"],
    "remediates": [
        "trainable_ptm_loaded_as_lightning_checkpoint",
        "ddp_metrics_not_globally_reduced",
        "nonzero_rank_status_kpi_writes",
        "prefixed_backbone_ptm_loaded_into_bare_backbone",
        "zero_compatible_tensor_load_fails_closed",
        "positive_pretrained_load_receipt",
    ],
}
FROZEN_QUALIFICATION_INFRASTRUCTURE_POLICY = {
    "schema_version": 2,
    "container_cuda_toolkit_version": "13.2",
    "cuda_runtime_probe": "torch_cuda_allocate_and_synchronize",
    "node_preflight_failure_exit_code": 92,
    "node_preflight_failure_marker": (
        "SEGFORMER_INFRASTRUCTURE_PREFLIGHT_FAILURE "
        "CUDA runtime probe failed"
    ),
    "node_preflight_success_marker": (
        "SEGFORMER_INFRASTRUCTURE_PREFLIGHT_OK"
    ),
    "maximum_job_attempts_per_phase": 2,
    "maximum_submission_attempts_per_job": 2,
    "retry_delay_seconds": 10,
    "retryable_submission_exception_type": "RuntimeError",
    "retryable_submission_message": (
        "SLURM cluster did not provide a stable identity; refusing to "
        "launch an unrecoverable job"
    ),
    "retryable_terminal_status": "Error",
    "sdk_failure_analysis_match": "CUDA runtime probe failed",
    "retry_scope": [
        "pre_submission_stable_identity",
        "pre_import_cuda_driver_runtime_compatibility",
    ],
    "non_infrastructure_failure_retry_allowed": False,
    "successful_job_replacement_allowed": False,
}
RUNTIME_LOCAL_ELIGIBILITY_KIND = (
    "segformer_positive_load_runtime_local_v1"
)
FROZEN_RUNTIME_LOCAL_CHECKPOINT_SPEC_FILE = {
    "source": "repository",
    "path": (
        "data/ptm_specs/segformer/"
        "segformer.runtime-local-qualified.v1.yaml"
    ),
    "sha256": (
        "67edb37c140aee9c465a56f0d713e22018f46b3877395d853c4cf4bd95d2731e"
    ),
    "provenance": {
        "source": "sealed SegFormer direct-full-run qualification",
        "evidence": (
            "Neutral packaged sidecar; exact backbone and checkpoint target "
            "remain bound by registry identity and campaign profile"
        ),
    },
}
FROZEN_V1_QUALIFICATION_EVIDENCE = {
    "campaign_id": "segformer-voc2012-direct-full-ptm-qualification-v1",
    "completion_path": (
        "/localhome/local-rarunachalam/.tao/artifacts/"
        "cross_model_automl_20260729/"
        "segformer_voc2012_ptm_qualification_v1/completion.json"
    ),
    "completion_whole_file_sha256": (
        "e7d604d63b2e79a54e21f7cac708ad6b1ff12ea8ad24556f911d662876412661"
    ),
    "evidence_sha256": (
        "23669ac5a091ed3b9b6841b9c06218e4bdaa01515860dd3c771f3d43eb34a08f"
    ),
    "ptm_stage_manifest_path": (
        "/localhome/local-rarunachalam/.tao/artifacts/"
        "cross_model_automl_20260729/"
        "segformer_voc2012_ptm_qualification_v1/ptm_stage_manifest.json"
    ),
    "ptm_stage_manifest_whole_file_sha256": (
        "f4f4d5a165f70cf4d67570143825c74406b4303224ed4cffad6e1577025a067a"
    ),
    "ptm_stage_manifest_sha256": (
        "06de8d1618739a358ca7bdc3912ff2d05bf262a8e702310cf25a381fe1a1393e"
    ),
    "status": "terminal_with_failures",
    "successful_workflows": 0,
    "failed_workflows": 13,
    "preserve_immutable": True,
    "reuse_for_v2": False,
}
FROZEN_V2_QUALIFICATION_EVIDENCE = {
    "campaign_id": "segformer-voc2012-direct-full-ptm-qualification-v2",
    "completion_path": (
        "/localhome/local-rarunachalam/.tao/artifacts/"
        "cross_model_automl_20260729/"
        "segformer_voc2012_ptm_qualification_v2/completion.json"
    ),
    "completion_whole_file_sha256": (
        "0e74b0016711bf614b40382c7a28b512c87971951dff6295cf107f44a901cf1a"
    ),
    "evidence_sha256": (
        "4faab6fd5d9f6bda96c1f6e5cfd73e83ff70dc99ef7aaeb990a68fea4f5ac422"
    ),
    "ptm_stage_manifest_path": (
        "/localhome/local-rarunachalam/.tao/artifacts/"
        "cross_model_automl_20260729/"
        "segformer_voc2012_ptm_qualification_v2/ptm_stage_manifest.json"
    ),
    "ptm_stage_manifest_whole_file_sha256": (
        "2ebb382ec62805caae24681ec1228bb248bdf86a94478c0c92c95cc9c0d9beec"
    ),
    "ptm_stage_manifest_sha256": (
        "dfd9eff35ab9b8ea5480606dc266512d3cccf6949791ef5372f2c0ee88eb1906"
    ),
    "launch_preflight_path": (
        "/localhome/local-rarunachalam/.tao/artifacts/"
        "cross_model_automl_20260729/"
        "segformer_voc2012_ptm_qualification_v2/"
        "qualification_launch_preflight.json"
    ),
    "launch_preflight_whole_file_sha256": (
        "322fd61f382ec1fcccab68b819098c75a49f61b7f829c361d4631b884c3ce2e6"
    ),
    "automatic_handoff_path": (
        "/localhome/local-rarunachalam/.tao/artifacts/"
        "cross_model_automl_20260729/"
        "segformer_voc2012_ptm_qualification_v2/automatic_handoff.json"
    ),
    "automatic_handoff_whole_file_sha256": (
        "a6f1c4e841e444266715a3a90243700527d71c89f98394ecfa15c3082eaa31bc"
    ),
    "source_commit": "aa4cecd4e2a5f1d5c8fc28b9a438fdd244425201",
    "status": "terminal_with_failures",
    "successful_workflows": 0,
    "failed_workflows": 13,
    "controller_failure_workflows": 12,
    "runtime_failure_workflows": 1,
    "preserve_immutable": True,
    "reuse_for_v3": False,
}
FROZEN_V3_QUALIFICATION_EVIDENCE = {
    "campaign_id": "segformer-voc2012-direct-full-ptm-qualification-v3",
    "contract_path": (
        "/localhome/local-rarunachalam/.tao/artifacts/"
        "cross_model_automl_20260729/"
        "segformer_voc2012_three_mode/campaign.v3.json"
    ),
    "contract_whole_file_sha256": (
        "cd448e8eabd5c23e6af6c73e3878a0b6a17e7952ef653dd0213a41d31f3548ef"
    ),
    "contract_sha256": (
        "320f857ad95747f4d0eab08da8703f00afc4eb84d7704bc214640544cd78dd17"
    ),
    "completion_path": (
        "/localhome/local-rarunachalam/.tao/artifacts/"
        "cross_model_automl_20260729/"
        "segformer_voc2012_ptm_qualification_v3/completion.json"
    ),
    "completion_whole_file_sha256": (
        "b8279dd87df2389c56a02db69dc8038f8bd84dcebe93cd1d3d66d04cb3fdfabc"
    ),
    "evidence_sha256": (
        "fe9fb7b93e19834ff56d12ad19c9733e01b56a873986ef20ed1ceabaad5190fc"
    ),
    "ptm_stage_manifest_path": (
        "/localhome/local-rarunachalam/.tao/artifacts/"
        "cross_model_automl_20260729/"
        "segformer_voc2012_ptm_qualification_v3/ptm_stage_manifest.json"
    ),
    "ptm_stage_manifest_whole_file_sha256": (
        "f432760feea6680b902e86eef002c56c887d8c87d512e1c27f1cf3faef5ab78f"
    ),
    "ptm_stage_manifest_sha256": (
        "9dc1fc9b38b4645e3095e33cf07f6811a5eaffa6df7bd08c8da9566d1ae541be"
    ),
    "launch_preflight_path": (
        "/localhome/local-rarunachalam/.tao/artifacts/"
        "cross_model_automl_20260729/"
        "segformer_voc2012_ptm_qualification_v3/"
        "qualification_launch_preflight.json"
    ),
    "launch_preflight_whole_file_sha256": (
        "a8891a8ec11bb36778013329825051d0a486b355a60758b95c9ec5e1a6c3097f"
    ),
    "automatic_handoff_path": (
        "/localhome/local-rarunachalam/.tao/artifacts/"
        "cross_model_automl_20260729/"
        "segformer_voc2012_ptm_qualification_v3/automatic_handoff.json"
    ),
    "automatic_handoff_whole_file_sha256": (
        "ff6d9a2b43963586016f7b5df1b91c3e4fc22cd24aa85ea04ce1605e2b5e99b0"
    ),
    "source_commit": "2a82adc9d90218359d1318dc2ad88062b71f40e3",
    "status": "terminal_with_failures",
    "successful_workflows": 0,
    "failed_workflows": 13,
    "controller_template_failure_workflows": 13,
    "preserve_immutable": True,
    "reuse_for_v4": False,
}
FROZEN_V4_QUALIFICATION_EVIDENCE = {
    "campaign_id": "segformer-voc2012-direct-full-ptm-qualification-v4",
    "contract_path": (
        "/localhome/local-rarunachalam/.tao/artifacts/"
        "cross_model_automl_20260729/"
        "segformer_voc2012_three_mode/campaign.v4.json"
    ),
    "contract_whole_file_sha256": (
        "acc23f910130abd3fd0be223077322b8845383a14223ecb7edc55496871945d2"
    ),
    "contract_sha256": (
        "aa0f89f62a35d7e8ffb20ed8f8ae1eb1bee35cbe54dc8df747997e68e7347274"
    ),
    "completion_path": (
        "/localhome/local-rarunachalam/.tao/artifacts/"
        "cross_model_automl_20260729/"
        "segformer_voc2012_ptm_qualification_v4/completion.json"
    ),
    "completion_whole_file_sha256": (
        "673c35a2f109be9cbd926d4e76a8042272379bc9c3fe3749935fca1bea77cc25"
    ),
    "evidence_sha256": (
        "0b73c7af76a615412fba007cc5d1926a720803eafe0f0b66976d114215f3369d"
    ),
    "ptm_stage_manifest_path": (
        "/localhome/local-rarunachalam/.tao/artifacts/"
        "cross_model_automl_20260729/"
        "segformer_voc2012_ptm_qualification_v4/ptm_stage_manifest.json"
    ),
    "ptm_stage_manifest_whole_file_sha256": (
        "48e7a6ae0820781f3b8ea6956d0b75decaedef69765a41dcd2c97ab00e53bad1"
    ),
    "ptm_stage_manifest_sha256": (
        "75201d653f7d60a115d27b4ccb9ff4bc073a9b8f6ce854261f3a427e77ee15f6"
    ),
    "launch_preflight_path": (
        "/localhome/local-rarunachalam/.tao/artifacts/"
        "cross_model_automl_20260729/"
        "segformer_voc2012_ptm_qualification_v4/"
        "qualification_launch_preflight.json"
    ),
    "launch_preflight_whole_file_sha256": (
        "349eeacf62f0fecbd3e88400707d4975098c818e76ec3f07cec0224cca752a5c"
    ),
    "automatic_handoff_path": (
        "/localhome/local-rarunachalam/.tao/artifacts/"
        "cross_model_automl_20260729/"
        "segformer_voc2012_ptm_qualification_v4/automatic_handoff.json"
    ),
    "automatic_handoff_whole_file_sha256": (
        "e9924b26183e63d1af136df8e97c6bfd12f2cd6a1380b2c136b6fcb9151d53c3"
    ),
    "ptm_load_audit_path": (
        "/localhome/local-rarunachalam/.tao/artifacts/"
        "cross_model_automl_20260729/"
        "segformer_voc2012_ptm_qualification_v4/ptm_load_audit.v2.json"
    ),
    "ptm_load_audit_whole_file_sha256": (
        "f9d6999279cb5619890308c1abfd030f0bdae33bb99d961fdbc417b81b279c87"
    ),
    "ptm_load_audit_sha256": (
        "cbf79f8bf38c53f3747bd95378d8e6a71905c1c4a1ad246b011574d85e5ba1ed"
    ),
    "ptm_load_audit_source_sha256": (
        "cc4d634efd413a19b7956d95edf6154e726e8135134ca192626f4ab95f49d202"
    ),
    "source_commit": "88a53144650ea895a2dcfa896828412a858f659b",
    "status": "terminal_with_failures",
    "successful_workflows": 0,
    "failed_workflows": 13,
    "positive_load_train_workflows": 4,
    "backbone_prefix_load_failure_workflows": 9,
    "preserve_immutable": True,
    "reuse_for_v5": "exact_positive_load_terminal_train_phase_only",
}
FROZEN_FAILED_V5_QUALIFICATION_CONTRACT = {
    "path": (
        "/localhome/local-rarunachalam/.tao/artifacts/"
        "cross_model_automl_20260729/segformer_voc2012_three_mode/"
        "campaign.v5.json"
    ),
    "whole_file_sha256": (
        "1e5bae36930e5eb50e945fbe8bb0f50af030bd2fea2031f23e8b2a6c27f9b60e"
    ),
    "contract_sha256": (
        "983af104b8eb6ffc4e17378205421107f769e5f1102f5fc2269b1f3ab4d9e762"
    ),
    "campaign_id": (
        "segformer-voc2012-objective-aware-three-mode-20260801-v5"
    ),
    "source_commit": "02ee7822c4479f08bbcabaf3a9b7cffd53164c50",
    "wheel_sha256": (
        "a5e78903aa7c540a7c13b9b413ed5daf64534df04cc91a21d9480875e7d16f3e"
    ),
    "sdk_commit": "a2e50d0930c3e3785b4b39fa8c3da88b39ff89e5",
    "skills_commit": "2e9c1b25f3c7cb1ae444c75652e36c47eace8229",
    "registry_version": "1.5.0",
    "registry_sha256": (
        "8d40ebde0eec2b7c53f4c698285146c44056d3cc2560ce481cc57b6375b25f74"
    ),
    "qualification_campaign_id": QUALIFICATION_CAMPAIGN_ID,
    "qualification_controller_sha256": (
        "9457bad3840a252241ca3170a5a84b79d5ca70d2059bcd7be9b4c1561d926f88"
    ),
    "qualification_gate_sha256": (
        "e230b882b3200af237aa40cb84a538d2a3d1b3accad60384c046c8bcef1501a1"
    ),
    "qualification_evidence_path": (
        "/localhome/local-rarunachalam/.tao/artifacts/"
        "cross_model_automl_20260729/"
        "segformer_voc2012_ptm_qualification_v5/completion.json"
    ),
    "ptm_stage_manifest_path": (
        "/localhome/local-rarunachalam/.tao/artifacts/"
        "cross_model_automl_20260729/"
        "segformer_voc2012_ptm_qualification_v5/ptm_stage_manifest.json"
    ),
    "ptm_stage_manifest_whole_file_sha256": (
        "7c8e70ae256fdb1f4d3d62841724cdee501a2097f20cbd0d5e2a5f8e0e293c07"
    ),
    "ptm_stage_manifest_sha256": (
        "299f92133e024501a34e0744764b7120b6d2f19517907543070695c7d274ad29"
    ),
}
FROZEN_V5_QUALIFICATION_CONTRACT = {
    "path": (
        "/localhome/local-rarunachalam/.tao/artifacts/"
        "cross_model_automl_20260729/"
        "segformer_voc2012_ptm_qualification_cuda_probe_recovery_v2/"
        "campaign.cuda_probe_recovery_v2.json"
    ),
    "whole_file_sha256": (
        "4a4753053a45dc8f5bdc9c408775960b8f12ba5f703447e72dc9f542ad1f0bdf"
    ),
    "contract_sha256": (
        "63846c8f54ad58bd4fe5c0c631e71b06c28bf83b7567743a10ad001959d24d89"
    ),
    "campaign_id": (
        "segformer-voc2012-direct-full-ptm-qualification-"
        "cuda-probe-recovery-v2"
    ),
    "source_commit": "ad627ecc1a55d88ba577cda786b5a83d53d05123",
    "wheel_sha256": (
        "a5e78903aa7c540a7c13b9b413ed5daf64534df04cc91a21d9480875e7d16f3e"
    ),
    "sdk_commit": "acb3c206bc0e04517ab852914efa9b6c2f7c656c",
    "skills_commit": "6d22b2efcd355454448f075945d318f0608ed480",
    "registry_version": "1.5.0",
    "registry_sha256": (
        "8d40ebde0eec2b7c53f4c698285146c44056d3cc2560ce481cc57b6375b25f74"
    ),
    "qualification_campaign_id": QUALIFICATION_CAMPAIGN_ID,
    "qualification_controller_sha256": (
        "58cb2342d5e6b0813a647ab6eac7476741ae20fd33bf8e321da2a1a6abd9290d"
    ),
    "qualification_gate_sha256": (
        "d5274323511bab2ac1a11e801f1cf76547885dd5a89e051c3f554a6018d9c8ed"
    ),
    "qualification_evidence_path": (
        "/localhome/local-rarunachalam/.tao/artifacts/"
        "cross_model_automl_20260729/"
        "segformer_voc2012_ptm_qualification_cuda_probe_recovery_v2/"
        "completion.json"
    ),
    "qualification_evidence_whole_file_sha256": (
        "56eb69bdbda184c3cc6fd6d3b30a9e4ded0efb5005641bed0aedbc309f957560"
    ),
    "qualification_evidence_sha256": (
        "763f964c05c3424419922ae98c85b6c73536e70ed23c20fe0098375a86567e02"
    ),
    "ptm_stage_manifest_path": (
        "/localhome/local-rarunachalam/.tao/artifacts/"
        "cross_model_automl_20260729/"
        "segformer_voc2012_ptm_qualification_cuda_probe_recovery_v2/"
        "ptm_stage_manifest.json"
    ),
    "ptm_stage_manifest_whole_file_sha256": (
        "524011fb3528427c30b60eea324cec8790ffc8ebd4a2742a71601d2990171504"
    ),
    "ptm_stage_manifest_sha256": (
        "fa44a890e2741be5f88d3341cf51aa71fa4ebe6a9268a76bcc239766e07e0b63"
    ),
}
FROZEN_V6_SUCCESSOR_CONTRACT_PATH = (
    "/localhome/local-rarunachalam/.tao/artifacts/"
    "cross_model_automl_20260729/segformer_voc2012_three_mode/"
    "campaign.v8.json"
)
FROZEN_V6_SUCCESSOR_RUNTIME_ROOT = (
    "/localhome/local-rarunachalam/.tao/artifacts/"
    "cross_model_automl_20260729/segformer_voc2012_three_mode_v8"
)
FROZEN_PRIOR_QUALIFICATION_EVIDENCE = [
    copy.deepcopy(FROZEN_V1_QUALIFICATION_EVIDENCE),
    copy.deepcopy(FROZEN_V2_QUALIFICATION_EVIDENCE),
    copy.deepcopy(FROZEN_V3_QUALIFICATION_EVIDENCE),
    copy.deepcopy(FROZEN_V4_QUALIFICATION_EVIDENCE),
]
FROZEN_V4_REUSABLE_TRAIN_CHECKPOINT_IDS = (
    "segformer.cityscapes.fan_base_hybrid.trainable.v1.0",
    "segformer.cityscapes.fan_large_hybrid.trainable.v1.0",
    "segformer.cityscapes.fan_small_hybrid.trainable.v1.0",
    "segformer.cityscapes.fan_tiny_hybrid.trainable.v1.0",
)
FROZEN_V5_FRESH_TRAIN_CHECKPOINT_IDS = (
    "segformer.imagenet.fan_small_hybrid",
    "segformer.imagenet.fan_tiny_hybrid",
    "segformer.imagenet22k.fan_base_hybrid",
    "segformer.imagenet22k.fan_base_hybrid.imagenet1k",
    "segformer.imagenet22k.fan_base_hybrid.imagenet1k.384",
    "segformer.imagenet22k.fan_large_hybrid",
    "segformer.imagenet22k.fan_large_hybrid.384",
    "segformer.imagenet22k.fan_large_hybrid.imagenet1k",
    "segformer.imagenet22k.fan_large_hybrid.imagenet1k.384",
)
FROZEN_QUALIFICATION_PHASE_RECOVERY_POLICY = {
    "schema_version": 1,
    "kind": "segformer_v4_terminal_train_phase_reuse_v1",
    "predecessor_campaign_id": FROZEN_V4_QUALIFICATION_EVIDENCE[
        "campaign_id"
    ],
    "predecessor_completion_whole_file_sha256": (
        FROZEN_V4_QUALIFICATION_EVIDENCE["completion_whole_file_sha256"]
    ),
    "predecessor_load_audit_whole_file_sha256": (
        FROZEN_V4_QUALIFICATION_EVIDENCE[
            "ptm_load_audit_whole_file_sha256"
        ]
    ),
    "reused_train_checkpoint_ids": list(
        FROZEN_V4_REUSABLE_TRAIN_CHECKPOINT_IDS
    ),
    "fresh_train_checkpoint_ids": list(FROZEN_V5_FRESH_TRAIN_CHECKPOINT_IDS),
    "execution_plan_sha256_by_checkpoint_id": {
        "segformer.cityscapes.fan_base_hybrid.trainable.v1.0": (
            "2c6477ad2eda63fc8fc8f5b5bfe1cf95e923bc05f6f603ed68f9efdbbfc53ed3"
        ),
        "segformer.cityscapes.fan_large_hybrid.trainable.v1.0": (
            "aac288876c2d9c618492f63161fce122fb2c923596e1c829b0e061df9754e844"
        ),
        "segformer.cityscapes.fan_small_hybrid.trainable.v1.0": (
            "666d99f614dff7b7e676261bbaab45e7337f363ee65ea2b4dc5b76aa98db24b1"
        ),
        "segformer.cityscapes.fan_tiny_hybrid.trainable.v1.0": (
            "9676f0e82397c5e31a179feb5b09c9f8e019ac42eeb1f9cf7eb1ee6c53be1b3a"
        ),
        "segformer.imagenet.fan_small_hybrid": (
            "37b88ef6caeda9f21aa3446e4dcb9c1331ce58d5eb9d5f254b939040852f35e3"
        ),
        "segformer.imagenet.fan_tiny_hybrid": (
            "9583c8170e70b544494f560dc59d822d92298f5b4bd175f00ff89181f45cb0c5"
        ),
        "segformer.imagenet22k.fan_base_hybrid": (
            "c4bcb4c7874b30ffabe6e33c200f31f23b2f02cce78f06213e2aa8c4d1bd6782"
        ),
        "segformer.imagenet22k.fan_base_hybrid.imagenet1k": (
            "51e6a0941400946ab5308f93313798bc0c22bf33420dce91e2c7601d38486bba"
        ),
        "segformer.imagenet22k.fan_base_hybrid.imagenet1k.384": (
            "29ae19d61c11c6a92d3ea15be15edb4be5d6865a3d87c7eee62e6089d35f6a7b"
        ),
        "segformer.imagenet22k.fan_large_hybrid": (
            "4426ad7c5a3d4e2594808d253dca16f5a2204d70263ef174adc7a94a12c99fd8"
        ),
        "segformer.imagenet22k.fan_large_hybrid.384": (
            "fbaa4dde903801e68901b73ed0aaf8fcc0a2ca3a8e4dc110e7870d835c2ea4ee"
        ),
        "segformer.imagenet22k.fan_large_hybrid.imagenet1k": (
            "c66c75219a7a17f98be2046b4a1c7ee7755096b685f174e85fbef6c3fdfed5ea"
        ),
        "segformer.imagenet22k.fan_large_hybrid.imagenet1k.384": (
            "a4bb98f728a7bec66a5302ab0adbb6b846150c815478a64dddb7a4235f8ed63c"
        ),
    },
    "new_full_train_job_count": 9,
    "new_standalone_evaluation_job_count": 13,
    "reuse_requires_exact_positive_load_receipt": True,
    "reuse_requires_exact_epoch_49_checkpoint_identity": True,
    "reuse_requires_full_50_epoch_validation_evidence": True,
    "reuse_failed_or_ambiguous_train_allowed": False,
    "fallback_checkpoint_allowed": False,
    "successful_train_reexecution_allowed": False,
}
LATENCY_PROTOCOL = {
    "warmup_iterations": 50,
    "timed_iterations": 100,
    "repeated_rounds": 5,
    "preloaded_batches": 16,
    "benchmark_seed": 20260727,
    "tail_percentile": 95.0,
    "bootstrap_resamples": 5000,
    "bootstrap_confidence_level": 0.95,
    "bootstrap_seed": 424242,
    "batch_size_per_replica": 1,
    "expected_replicas": 8,
    "precision": "fp32",
    "timed_scope": "segformer_model_forward",
    "excluded_scope": [
        "checkpoint_load",
        "disk_io",
        "image_decode",
        "resize_normalize",
        "host_to_device_transfer",
        "argmax_and_mask_serialization",
        "metric_accumulation",
        "distributed_gather",
    ],
    "synchronization": "accelerator_sync_before_and_after_each_sample",
    "replica_alignment": "nccl_barrier_before_each_timed_sample",
    "measurement_role": "selection_time",
    "raw_samples_per_candidate": 4000,
    "validity_thresholds": {
        "max_robust_cv": 0.10,
        "max_round_median_range_fraction": 0.05,
        "max_absolute_round_drift_fraction": 0.05,
        "max_device_median_range_fraction": 0.05,
        "max_bootstrap_ci_width_fraction": 0.03,
    },
}

VOC_CLASS_NAMES = (
    "background",
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
)


class CampaignContractError(ValueError):
    """The SegFormer campaign contract is inconsistent."""


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _finite_fraction(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise CampaignContractError(f"{name} must be finite in (0, 1]")
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise CampaignContractError(
            f"{name} must be finite in (0, 1]"
        ) from exc
    if not math.isfinite(number) or not 0.0 < number <= 1.0:
        raise CampaignContractError(f"{name} must be finite in (0, 1]")
    return number


def voc_palette() -> list[dict[str, Any]]:
    """Return the loss/metric-preserving grayscale VOC palette."""
    records = [
        {
            "label_id": label_id,
            "mapping_class": name,
            "rgb": [label_id],
            "seg_class": name,
        }
        for label_id, name in enumerate(VOC_CLASS_NAMES)
    ]
    records.append(
        {
            "label_id": 255,
            "mapping_class": "ignore",
            "rgb": [255],
            "seg_class": "ignore",
        }
    )
    return records


def segformer_registry_snapshot() -> dict[str, Any]:
    """Snapshot every official repository-owned SegFormer PTM record."""
    registry = load_ptm_registry()
    model = registry.to_dict()["models"]["segformer"]
    records = []
    for record in model["checkpoints"]:
        if (
            record.get("source", {}).get("official") is not True
            or record.get("model_family") != "segformer"
            or "semantic_segmentation"
            not in record.get("task_compatibility", ())
        ):
            raise CampaignContractError(
                f"invalid official SegFormer registry record: {record.get('id')}"
            )
        records.append(
            {
                "id": record["id"],
                "status": record["status"],
                "status_reason": record.get("status_reason"),
                "source": copy.deepcopy(record["source"]),
                "expected_size_bytes": record["expected_size_bytes"],
                "checkpoint_target": record["checkpoint_target"],
                "architecture": record["architecture"],
                "backbone": record["backbone"],
                "input_contract": copy.deepcopy(record["input_contract"]),
                "registry_record_sha256": canonical_sha256(record),
            }
        )
    if len(records) != 13 or len({item["id"] for item in records}) != 13:
        raise CampaignContractError(
            "the frozen repository inventory must contain 13 SegFormer PTMs"
        )
    records.sort(key=lambda item: item["id"])
    return {
        "registry_version": registry.registry_version,
        "registry_sha256": registry.document_sha256,
        "default_ptm": model["default_ptm"],
        "records": records,
        "record_count": len(records),
        "supported_ids": [
            item["id"] for item in records if item["status"] == "supported"
        ],
        "unverified_ids": [
            item["id"] for item in records if item["status"] == "unverified"
        ],
    }


def validate_packaged_train_schema(skill_dir: str | Path) -> dict[str, Any]:
    root = Path(skill_dir)
    info_path = root / "references/skill_info.yaml"
    schema_path = root / "schemas/train.schema.json"
    template_path = root / "references/spec_template_train.yaml"
    for path in (info_path, schema_path, template_path):
        if not path.is_file():
            raise CampaignContractError(f"missing packaged skill artifact: {path}")
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    if schema.get("x_tao_schema", {}).get("network_arch") != "segformer":
        raise CampaignContractError("packaged train schema is not SegFormer")
    defaults = schema.get("automl_default_parameters")
    if not isinstance(defaults, list):
        raise CampaignContractError(
            "packaged train schema lacks AutoML default parameters"
        )
    missing = sorted(set(SEARCH_PARAMETERS) - set(defaults))
    if missing:
        raise CampaignContractError(
            "frozen search parameters are not AutoML enabled: "
            + ", ".join(missing)
        )
    return {
        "skill_info_path": str(info_path),
        "skill_info_sha256": sha256_file(info_path),
        "schema_path": str(schema_path),
        "schema_sha256": sha256_file(schema_path),
        "template_path": str(template_path),
        "template_sha256": sha256_file(template_path),
        "explicit_search_parameters": list(SEARCH_PARAMETERS),
        "non_train_fields_excluded": True,
    }


def mode_objective(mode: str) -> dict[str, Any]:
    objectives = [
        {"metric": "val_miou", "direction": "maximize", "role": "accuracy"},
        {"metric": "latency_ms", "direction": "minimize", "role": "latency"},
    ]
    if mode == "accuracy":
        return {
            "selection_mode": mode,
            "objectives": objectives,
            "acquisition": "expected_improvement",
            "latency_accuracy_retention": None,
            "multi_objective_min_accuracy": None,
            "selection_policy": "highest_valid_accuracy",
        }
    if mode == "latency":
        return {
            "selection_mode": mode,
            "objectives": objectives,
            "acquisition": "constrained_expected_improvement",
            "latency_accuracy_retention": {
                "type": "relative",
                "retained_fraction": FROZEN_LATENCY_RETENTION,
                "reference": "accuracy_winner",
            },
            "multi_objective_min_accuracy": None,
            "selection_policy": "equivalent_fastest_accuracy_tiebreak",
        }
    if mode == "multi_objective":
        return {
            "selection_mode": mode,
            "objectives": objectives,
            "acquisition": "parego_expected_improvement",
            "latency_accuracy_retention": None,
            "multi_objective_min_accuracy": None,
            "selection_policy": "normalized_augmented_chebyshev",
        }
    raise CampaignContractError(f"unsupported mode: {mode!r}")


def mode_settings(campaign_id: str, mode: str) -> dict[str, Any]:
    objective = mode_objective(mode)
    settings = {
        "algorithm": "bayesian",
        "automl_max_recommendations": FROZEN_CANDIDATE_BUDGET,
        "automl_max_concurrent": 1,
        "campaign_id": campaign_id,
        "job_id": f"{campaign_id}-{mode}",
        "session_id": f"{campaign_id}-{mode}",
        "experiment_id": f"{campaign_id}-{mode}-observations",
        "random_seed": FROZEN_SEARCH_SEED,
        "objectives": [
            {"metric": item["metric"], "direction": item["direction"]}
            for item in objective["objectives"]
        ],
        "selection_mode": mode,
        "accuracy_metric": "val_miou",
        "latency_metric": "latency_ms",
        "objective_acquisition": {
            "calibration_points": FROZEN_CALIBRATION_POINTS_PER_ARM,
            "augmentation_rho": 1.0e-6,
        },
        "objective_normalization": "pareto_front",
        "augmentation_rho": 1.0e-6,
        "accuracy_tolerance": 1.0e-12,
        "latency_tolerance": FROZEN_LATENCY_TOLERANCE_MS,
        "selection_score_tolerance": 1.0e-12,
        "latency_ci_low_metric": "latency_ci95_low_ms",
        "latency_ci_high_metric": "latency_ci95_high_ms",
        "multi_objective_min_accuracy": None,
        "run_baseline": False,
        "run_final_evaluation": False,
        "require_eval_fn_success": True,
        "automl_delete_intermediate_ckpt": False,
        "automl_checkpoint_retention_strategy": "terminal",
    }
    if mode == "latency":
        settings["latency_accuracy_retention"] = copy.deepcopy(
            objective["latency_accuracy_retention"]
        )
    return settings


def custom_ranges() -> dict[str, dict[str, Any]]:
    return {
        name: {
            "valid_min": SEARCH_SPACE[name]["minimum"],
            "valid_max": SEARCH_SPACE[name]["maximum"],
        }
        for name in SEARCH_PARAMETERS
    }


def profile_overrides(dataset_root: str) -> dict[str, Any]:
    """Return the identical full-dataset spec applied to every mode/PTM arm."""
    if not isinstance(dataset_root, str) or not dataset_root.startswith(
        "/lustre/"
    ):
        raise CampaignContractError("dataset root must be an absolute Lustre path")
    return {
        "model_name": "segformer_voc2012",
        "results_dir": "",
        "wandb": {"enable": False},
        "dataset": {
            "segment": {
                "root_dir": dataset_root,
                "dataset": "SFDataset",
                "num_classes": 21,
                "img_size": FROZEN_IMAGE_SIZE,
                "batch_size": FROZEN_BATCH_SIZE_PER_REPLICA,
                "workers": 8,
                "shuffle": True,
                "train_split": "train",
                "validation_split": "val",
                "test_split": "val",
                "predict_split": "val",
                "label_transform": "None",
                "palette": voc_palette(),
            }
        },
        "train": {
            "num_gpus": 8,
            "gpu_ids": list(range(8)),
            "num_nodes": 1,
            "seed": FROZEN_TRAINING_SEED,
            "num_epochs": FROZEN_TRAINING_EPOCHS,
            "checkpoint_interval": FROZEN_TRAINING_EPOCHS,
            "checkpoint_interval_unit": "epoch",
            "validation_interval": 1,
            "resume_training_checkpoint_path": "",
            "results_dir": "",
            "tensorboard": {"enabled": False},
            "use_distributed_sampler": False,
            "sync_batchnorm": False,
            "cudnn": {"benchmark": False, "deterministic": True},
        },
    }


def qualification_profile_overrides(dataset_root: str) -> dict[str, Any]:
    """Return the v5 official multi-class fidelity for every PTM arm."""
    value = profile_overrides(dataset_root)
    fidelity = FROZEN_QUALIFICATION_FIDELITY
    value["dataset"]["segment"]["augmentation"] = {
        "random_color": {
            "enable": fidelity["random_color_enabled"],
        },
        "with_random_blur": fidelity["random_blur_enabled"],
    }
    value["train"].update(
        {
            "num_epochs": fidelity["training_epochs"],
            "checkpoint_interval": fidelity["checkpoint_interval"],
            "validation_interval": fidelity["validation_interval"],
            "optim": {
                "optim": fidelity["optimizer"],
                "lr": fidelity["learning_rate"],
                "weight_decay": fidelity["weight_decay"],
            },
            "use_distributed_sampler": fidelity[
                "use_distributed_sampler"
            ],
        }
    )
    return value


def validate_dataset_record(dataset: Mapping[str, Any]) -> dict[str, Any]:
    required = {
        "id": "pascal_voc_2012_full_semantic_segmentation",
        "train_image_count": 1464,
        "train_mask_count": 1464,
        "validation_image_count": 1449,
        "validation_mask_count": 1449,
        "num_classes": 21,
        "ignore_label": 255,
        "official_archive_sha256": (
            "e14f763270cf193d0b5f74b169f44157a4b0c6efa708f4dd0ff78ee691763bcb"
        ),
        "content_sha256": (
            "815b5d01b625238b449c4bca828bf96107b367f0f4d5d8a31d2f97c6161a5de0"
        ),
        "manifest_sha256": (
            "051ab20215b8e6976763ac82a3db20a68264759edef3d62fd0c8553c501123ff"
        ),
        "file_manifest_entry_count": 5827,
        "stage_manifest_sha256": (
            "437ff12490637950707b9b951d820ea34d38b926080a478a5d182c2d284a0c5d"
        ),
        "remote_read_only": True,
        "remote_writable_entries_after_lock": 0,
    }
    for key, expected in required.items():
        if dataset.get(key) != expected:
            raise CampaignContractError(
                f"VOC2012 dataset field {key!r} changed"
            )
    root = dataset.get("prepared_root")
    if not isinstance(root, str) or not root.startswith("/lustre/"):
        raise CampaignContractError("VOC2012 prepared_root must be on Lustre")
    if (
        not isinstance(dataset.get("manifest_path"), str)
        or not Path(dataset["manifest_path"]).is_absolute()
        or not isinstance(dataset.get("stage_manifest_path"), str)
        or not Path(dataset["stage_manifest_path"]).is_absolute()
        or not isinstance(dataset.get("stage_manifest_lustre_path"), str)
        or not dataset["stage_manifest_lustre_path"].startswith("/lustre/")
        or not isinstance(dataset.get("remote_file_manifest_path"), str)
        or not dataset["remote_file_manifest_path"].startswith("/lustre/")
    ):
        raise CampaignContractError(
            "VOC2012 local and Lustre provenance paths are invalid"
        )
    return copy.deepcopy(dict(dataset))


def build_preregistered_contract(
    *,
    campaign_id: str,
    dataset: Mapping[str, Any],
    skill_dir: str,
    runtime: Mapping[str, Any],
) -> dict[str, Any]:
    """Build immutable intent without granting launch authorization."""
    _finite_fraction(FROZEN_LATENCY_RETENTION, "latency retention")
    dataset_record = validate_dataset_record(dataset)
    schema = validate_packaged_train_schema(skill_dir)
    ptm_inventory = segformer_registry_snapshot()
    value = {
        "schema_version": 1,
        "campaign_id": campaign_id,
        "model": "segformer",
        "network_arch": "segformer",
        "task": "semantic_segmentation",
        "primary_accuracy_metric": "val_miou",
        "dataset": dataset_record,
        "runtime": copy.deepcopy(dict(runtime)),
        "sqsh": copy.deepcopy(FROZEN_SQSH),
        "schema": schema,
        "ptm_inventory": ptm_inventory,
        "qualification_policy": {
            "revision": QUALIFICATION_REVISION,
            "campaign_id": QUALIFICATION_CAMPAIGN_ID,
            "kind": (
                "selective_v4_train_phase_reuse_or_fresh_full_gpu_train_"
                "then_new_standalone_evaluation"
            ),
            "cpu_model_runs": 0,
            "smoke_model_runs": 0,
            "mini_step_runs": 0,
            "nodes_per_job": 1,
            "gpus_per_job": 8,
            "full_dataset": True,
            "training_epochs": FROZEN_QUALIFICATION_TRAINING_EPOCHS,
            "standalone_evaluation": True,
            "registry_bypass_allowed": False,
            "recipe_fidelity": copy.deepcopy(
                FROZEN_QUALIFICATION_FIDELITY
            ),
            "runtime_overlay": copy.deepcopy(
                FROZEN_QUALIFICATION_RUNTIME_OVERLAY
            ),
            "infrastructure_retry_policy": copy.deepcopy(
                FROZEN_QUALIFICATION_INFRASTRUCTURE_POLICY
            ),
            "phase_recovery_policy": copy.deepcopy(
                FROZEN_QUALIFICATION_PHASE_RECOVERY_POLICY
            ),
            "prior_revision_evidence": copy.deepcopy(
                FROZEN_PRIOR_QUALIFICATION_EVIDENCE
            ),
            "qualification_evidence_path": runtime[
                "qualification_evidence_path"
            ],
            "ptm_stage_manifest_path": runtime["ptm_stage_manifest_path"],
        },
        "execution": {
            "kind": "objective_aware_three_mode_search",
            "cpu_runs": 0,
            "smoke_runs": 0,
            "local_model_runs": 0,
            "independent_mode_jobs": True,
            "shared_archive": False,
            "first_candidate_gate": True,
            "automatic_remaining_budget_release": True,
            "automatic_trigger": True,
            "nodes_per_child": 1,
            "gpus_per_child": 8,
            "container_mode": "pinned_sqsh",
        },
        "search": {
            "algorithm": "bayesian",
            "implementation": (
                "hierarchical_ptm_objective_aware_bayesian_v1"
            ),
            "candidate_budget_per_mode": FROZEN_CANDIDATE_BUDGET,
            "search_seed": FROZEN_SEARCH_SEED,
            "training_seed": FROZEN_TRAINING_SEED,
            "training_epochs": FROZEN_TRAINING_EPOCHS,
            "calibration_points_per_arm": (
                FROZEN_CALIBRATION_POINTS_PER_ARM
            ),
            "invalid_recovery_issues_per_arm": (
                FROZEN_INVALID_RECOVERY_ISSUES_PER_ARM
            ),
            "parameters": list(SEARCH_PARAMETERS),
            "space": copy.deepcopy(SEARCH_SPACE),
            "space_sha256": canonical_sha256(SEARCH_SPACE),
            "latency_accuracy_retention": FROZEN_LATENCY_RETENTION,
            "latency_practical_tolerance_ms": (
                FROZEN_LATENCY_TOLERANCE_MS
            ),
            "ptm_representation": "hierarchical_nonordinal_arms",
            "ptm_policy_by_mode": {
                "accuracy": "all_runtime_supported",
                "latency": "all_runtime_supported",
                "multi_objective": "all_runtime_supported",
            },
        },
        "validation_sanity_gate": {
            "metric": "val_miou",
            "minimum": FROZEN_VALIDATION_SANITY_MIN_MIOU,
            "role": "experiment_correctness_gate_not_product_selection",
            "rationale": (
                "For 21-class VOC semantic segmentation, a value below 0.10 "
                "requires data, label, optimization, fidelity, and metric "
                "root-cause analysis before a campaign can continue."
            ),
            "low_finite_metric_automatically_accepted": False,
        },
        "latency_protocol": copy.deepcopy(LATENCY_PROTOCOL),
        "modes": [
            {
                "mode": mode,
                "observation_namespace": (
                    f"{campaign_id}-{mode}-observations"
                ),
                "observation_sharing": False,
                "initial_observation_ids": [],
                "objective": mode_objective(mode),
                "settings": mode_settings(campaign_id, mode),
            }
            for mode in MODES
        ],
        "agent_intervention_flags": {name: False for name in AGENT_FLAGS},
        "selection_isolation_flags": {
            name: False for name in SELECTION_FLAGS
        },
    }
    runtime_local = runtime.get("runtime_local_eligibility")
    if runtime_local is not None:
        value["qualification_policy"]["runtime_local_eligibility"] = (
            copy.deepcopy(runtime_local)
        )
    value["contract_sha256"] = canonical_sha256(value)
    return value


def validate_contract(document: Mapping[str, Any]) -> dict[str, Any]:
    value = copy.deepcopy(dict(document))
    observed = value.pop("contract_sha256", None)
    if observed != canonical_sha256(value):
        raise CampaignContractError("campaign contract integrity failed")
    if (
        value.get("model") != "segformer"
        or value.get("network_arch") != "segformer"
        or value.get("task") != "semantic_segmentation"
        or value.get("primary_accuracy_metric") != "val_miou"
        or value.get("execution", {}).get("cpu_runs") != 0
        or value.get("execution", {}).get("smoke_runs") != 0
        or value.get("execution", {}).get("gpus_per_child") != 8
        or value.get("execution", {}).get("container_mode")
        != "pinned_sqsh"
        or value.get("runtime", {}).get("partition")
        != FROZEN_SLURM_PARTITION
        or value.get("runtime", {}).get("time_hours")
        != FROZEN_SLURM_TIME_HOURS
        or value.get("runtime", {}).get("timeout_hours")
        != FROZEN_SLURM_TIMEOUT_HOURS
        or value.get("search", {}).get("space") != SEARCH_SPACE
        or tuple(item.get("mode") for item in value.get("modes", ()))
        != MODES
    ):
        raise CampaignContractError("campaign execution policy changed")
    validate_dataset_record(value["dataset"])
    if value.get("sqsh") != FROZEN_SQSH:
        raise CampaignContractError("pinned SQSH identity changed")
    qualification = value.get("qualification_policy", {})
    if (
        qualification.get("revision") != QUALIFICATION_REVISION
        or qualification.get("campaign_id") != QUALIFICATION_CAMPAIGN_ID
        or qualification.get("kind")
        != (
            "selective_v4_train_phase_reuse_or_fresh_full_gpu_train_"
            "then_new_standalone_evaluation"
        )
        or qualification.get("training_epochs")
        != FROZEN_QUALIFICATION_TRAINING_EPOCHS
        or qualification.get("recipe_fidelity")
        != FROZEN_QUALIFICATION_FIDELITY
        or qualification.get("runtime_overlay")
        != FROZEN_QUALIFICATION_RUNTIME_OVERLAY
        or qualification.get("infrastructure_retry_policy")
        != FROZEN_QUALIFICATION_INFRASTRUCTURE_POLICY
        or qualification.get("phase_recovery_policy")
        != FROZEN_QUALIFICATION_PHASE_RECOVERY_POLICY
        or qualification.get("prior_revision_evidence")
        != FROZEN_PRIOR_QUALIFICATION_EVIDENCE
    ):
        raise CampaignContractError(
            "qualification v5 fidelity or provenance changed"
        )
    runtime_local = qualification.get("runtime_local_eligibility")
    if runtime_local is not None:
        required_false = (
            "repository_registry_mutation_allowed",
            "missing_license_normalization_allowed",
            "failed_arm_promotion_allowed",
            "unsupported_arm_promotion_allowed",
            "agent_override_allowed",
        )
        snapshot = segformer_registry_snapshot()
        expected_keys = {
            "schema_version",
            "kind",
            "enabled",
            "scope",
            "model",
            "task",
            "tao_version",
            "container_sha256",
            "base_registry_version",
            "base_registry_sha256",
            "qualification_evidence_path",
            "qualification_file_sha256",
            "qualification_evidence_sha256",
            "qualification_contract_sha256",
            "qualification_controller_sha256",
            "eligibility_gate_sha256",
            "runtime_resolver_sha256",
            "eligibility_source_commit",
            "wheel_sha256",
            "sdk_commit",
            "skills_commit",
            "license_policy",
            "checkpoint_spec_file",
            *required_false,
        }
        hashes = (
            "base_registry_sha256",
            "qualification_file_sha256",
            "qualification_evidence_sha256",
            "qualification_contract_sha256",
            "qualification_controller_sha256",
            "eligibility_gate_sha256",
            "runtime_resolver_sha256",
            "wheel_sha256",
        )
        commits = (
            "eligibility_source_commit",
            "sdk_commit",
            "skills_commit",
        )
        if (
            not isinstance(runtime_local, Mapping)
            or set(runtime_local) != expected_keys
            or runtime_local.get("schema_version") != 1
            or runtime_local.get("kind") != RUNTIME_LOCAL_ELIGIBILITY_KIND
            or runtime_local.get("enabled") is not True
            or runtime_local.get("scope")
            != "campaign_local_in_memory_projection"
            or runtime_local.get("model") != "segformer"
            or runtime_local.get("task") != "semantic_segmentation"
            or runtime_local.get("tao_version") != "7.1.0"
            or runtime_local.get("container_sha256") != FROZEN_SQSH["sha256"]
            or runtime_local.get("base_registry_version")
            != snapshot["registry_version"]
            or runtime_local.get("base_registry_sha256")
            != snapshot["registry_sha256"]
            or runtime_local.get("qualification_contract_sha256")
            != FROZEN_V5_QUALIFICATION_CONTRACT["contract_sha256"]
            or runtime_local.get("qualification_evidence_path")
            != qualification.get("qualification_evidence_path")
            or runtime_local.get("qualification_controller_sha256")
            != value.get("launcher_integrity", {}).get(
                "qualification_campaign_sha256"
            )
            or runtime_local.get("eligibility_gate_sha256")
            != value.get("launcher_integrity", {}).get(
                "qualification_gate_sha256"
            )
            or runtime_local.get("eligibility_source_commit")
            != value.get("runtime", {}).get("source_commit")
            or runtime_local.get("wheel_sha256")
            != value.get("runtime", {}).get("wheel_sha256")
            or runtime_local.get("sdk_commit")
            != value.get("runtime", {}).get("sdk_commit")
            or runtime_local.get("skills_commit")
            != value.get("runtime", {}).get("skills_commit")
            or runtime_local.get("license_policy")
            != "complete_existing_registry_metadata_only"
            or runtime_local.get("checkpoint_spec_file")
            != FROZEN_RUNTIME_LOCAL_CHECKPOINT_SPEC_FILE
            or any(runtime_local.get(name) is not False for name in required_false)
            or any(
                not isinstance(runtime_local.get(name), str)
                or len(runtime_local[name]) != 64
                or any(
                    character not in "0123456789abcdef"
                    for character in runtime_local[name]
                )
                for name in hashes
            )
            or any(
                not isinstance(runtime_local.get(name), str)
                or len(runtime_local[name]) != 40
                or any(
                    character not in "0123456789abcdef"
                    for character in runtime_local[name]
                )
                for name in commits
            )
            or value.get("runtime", {}).get("runtime_local_eligibility")
            != runtime_local
            or value.get("runtime", {}).get(
                "automatic_successor_contract_path"
            )
            != FROZEN_V6_SUCCESSOR_CONTRACT_PATH
            or value.get("runtime", {}).get(
                "automatic_successor_runtime_root"
            )
            != FROZEN_V6_SUCCESSOR_RUNTIME_ROOT
        ):
            raise CampaignContractError(
                "runtime-local SegFormer eligibility seal is invalid"
            )
    if any(value["agent_intervention_flags"].values()):
        raise CampaignContractError("agent intervention flags must remain false")
    if any(value["selection_isolation_flags"].values()):
        raise CampaignContractError("selection isolation flags must remain false")
    value["contract_sha256"] = observed
    return value


__all__ = [
    "AGENT_FLAGS",
    "CampaignContractError",
    "FROZEN_BATCH_SIZE_PER_REPLICA",
    "FROZEN_CALIBRATION_POINTS_PER_ARM",
    "FROZEN_CANDIDATE_BUDGET",
    "FROZEN_HARDWARE",
    "FROZEN_IMAGE_SIZE",
    "FROZEN_LATENCY_RETENTION",
    "FROZEN_LATENCY_TOLERANCE_MS",
    "FROZEN_QUALIFICATION_FIDELITY",
    "FROZEN_QUALIFICATION_INFRASTRUCTURE_POLICY",
    "FROZEN_QUALIFICATION_PHASE_RECOVERY_POLICY",
    "FROZEN_QUALIFICATION_RUNTIME_OVERLAY",
    "FROZEN_RUNTIME_LOCAL_CHECKPOINT_SPEC_FILE",
    "FROZEN_QUALIFICATION_TRAINING_EPOCHS",
    "FROZEN_PRIOR_QUALIFICATION_EVIDENCE",
    "FROZEN_SEARCH_SEED",
    "FROZEN_SLURM_RETRY_CAP",
    "FROZEN_SLURM_PARTITION",
    "FROZEN_SLURM_TIME_HOURS",
    "FROZEN_SLURM_TIMEOUT_HOURS",
    "FROZEN_SQSH",
    "FROZEN_TRAINING_EPOCHS",
    "FROZEN_VALIDATION_SANITY_MIN_MIOU",
    "FROZEN_V1_QUALIFICATION_EVIDENCE",
    "FROZEN_V2_QUALIFICATION_EVIDENCE",
    "FROZEN_V3_QUALIFICATION_EVIDENCE",
    "FROZEN_V4_QUALIFICATION_EVIDENCE",
    "FROZEN_V4_REUSABLE_TRAIN_CHECKPOINT_IDS",
    "FROZEN_V5_QUALIFICATION_CONTRACT",
    "FROZEN_FAILED_V5_QUALIFICATION_CONTRACT",
    "FROZEN_V6_SUCCESSOR_CONTRACT_PATH",
    "FROZEN_V6_SUCCESSOR_RUNTIME_ROOT",
    "FROZEN_V5_FRESH_TRAIN_CHECKPOINT_IDS",
    "LATENCY_PROTOCOL",
    "MODES",
    "QUALIFICATION_CAMPAIGN_ID",
    "QUALIFICATION_REVISION",
    "RUNTIME_LOCAL_ELIGIBILITY_KIND",
    "SEARCH_PARAMETERS",
    "SEARCH_SPACE",
    "SELECTION_FLAGS",
    "VOC_CLASS_NAMES",
    "build_preregistered_contract",
    "custom_ranges",
    "mode_objective",
    "mode_settings",
    "profile_overrides",
    "qualification_profile_overrides",
    "segformer_registry_snapshot",
    "sha256_file",
    "validate_contract",
    "validate_dataset_record",
    "validate_packaged_train_schema",
    "voc_palette",
]
