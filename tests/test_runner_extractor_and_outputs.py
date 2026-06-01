# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for tao_automl.runner audit-bug fixes.

Covers:
  - audit #3: ``_extract_metric_from_logs`` must no longer return None for
    every non-cosmos-RL metric whose name happens to contain "val".
    Generic ``val_loss: 0.12`` lines must match.
  - audit #4: ``_auto_suffix_output_dirs`` rewrites hardcoded local
    ``results_dir`` / ``output_dir`` / ``save_dir`` values to ``/rec_<id>``
    subdirs, but leaves SDK-routed declared outputs and remote URIs alone.
"""
import pytest

from tao_automl.runner import (
    MetricExtractorError,
    _auto_suffix_output_dirs,
    _extract_metric_from_logs,
)


# --- audit #3 ---------------------------------------------------------------

def test_val_loss_in_metric_name_no_longer_hijacks_to_cosmos_only():
    """Previously, any metric_name containing 'val' was routed to a
    cosmos-RL-only regex and returned None for every PL-based TAO model.
    After the fix, a generic ``val_loss: 0.12`` line should match.
    """
    logs = "Epoch 5\nval_loss: 0.1234\nEpoch 6\nval_loss: 0.0987\n"
    assert _extract_metric_from_logs(logs, "val_loss") == pytest.approx(0.0987)


def test_val_acc_matches_generic_pattern():
    logs = "step 100 val_acc: 0.876 train_loss: 0.5\n"
    assert _extract_metric_from_logs(logs, "val_acc") == pytest.approx(0.876)


def test_cosmos_rl_sft_validation_still_wins_when_present():
    """If the cosmos-RL [SFT] Validation loss line IS in the logs, it
    should be returned in preference to whatever else."""
    logs = (
        "loss: 99.9\n"
        "[SFT] Validation loss: 0.42 for train step 100\n"
        "loss: 88.8\n"
    )
    assert _extract_metric_from_logs(logs, "val_loss") == pytest.approx(0.42)


def test_empty_logs_returns_none():
    assert _extract_metric_from_logs("", "loss") is None
    assert _extract_metric_from_logs(None, "loss") is None  # type: ignore


def test_no_match_returns_none_not_crash():
    assert _extract_metric_from_logs("hello world", "loss") is None


def test_loss_pattern_picks_last_occurrence():
    logs = "loss: 0.5\nepoch 2\nloss: 0.3\n"
    assert _extract_metric_from_logs(logs, "loss") == pytest.approx(0.3)


# --- audit #4 ---------------------------------------------------------------

def test_auto_suffix_rewrites_hardcoded_local_results_dir():
    specs = {"train": {"results_dir": "/workspace/baseline/train"}}
    rewritten = _auto_suffix_output_dirs(specs, rec_id=7, declared_outputs=set())
    assert rewritten == ["train.results_dir"]
    assert specs["train"]["results_dir"] == "/workspace/baseline/train/rec_7"


def test_auto_suffix_leaves_declared_outputs_alone():
    """If the skill declares train.results_dir as an output, the SDK routes
    it via env vars per-job. The auto-suffix safety net must not interfere."""
    specs = {"train": {"results_dir": "/workspace/baseline/train"}}
    rewritten = _auto_suffix_output_dirs(
        specs, rec_id=7, declared_outputs={"train.results_dir"})
    assert rewritten == []
    assert specs["train"]["results_dir"] == "/workspace/baseline/train"


def test_auto_suffix_leaves_remote_uris_alone():
    """A user-supplied s3:// URI is an explicit destination, not a bug —
    leave it. (Even though it can still cause overwrite; that's the user's
    call and the warning surfaces in _apply_output_destinations / docs.)"""
    specs = {"train": {"results_dir": "s3://my-bucket/baseline/train"}}
    rewritten = _auto_suffix_output_dirs(specs, rec_id=7, declared_outputs=set())
    assert rewritten == []
    assert specs["train"]["results_dir"] == "s3://my-bucket/baseline/train"


def test_auto_suffix_handles_multiple_output_dir_keys():
    specs = {
        "train": {"results_dir": "/work/train"},
        "evaluate": {"output_dir": "/work/eval"},
        "export": {"save_dir": "/work/export"},
        "dataset": {"csv_path": "/work/data.csv"},  # NOT a *_dir; skip
    }
    rewritten = _auto_suffix_output_dirs(specs, rec_id=3, declared_outputs=set())
    assert set(rewritten) == {
        "train.results_dir", "evaluate.output_dir", "export.save_dir"}
    assert specs["train"]["results_dir"] == "/work/train/rec_3"
    assert specs["evaluate"]["output_dir"] == "/work/eval/rec_3"
    assert specs["export"]["save_dir"] == "/work/export/rec_3"
    assert specs["dataset"]["csv_path"] == "/work/data.csv"


def test_auto_suffix_skips_empty_and_non_string():
    specs = {
        "train": {
            "results_dir": "",  # empty → skip
            "output_dir": None,  # not a string → skip
            "save_dir": "/x/y",  # rewrite
        }
    }
    rewritten = _auto_suffix_output_dirs(specs, rec_id=1, declared_outputs=set())
    assert rewritten == ["train.save_dir"]
    assert specs["train"]["results_dir"] == ""
    assert specs["train"]["output_dir"] is None
    assert specs["train"]["save_dir"] == "/x/y/rec_1"


def test_auto_suffix_strips_trailing_slash():
    specs = {"train": {"results_dir": "/work/train/"}}
    _auto_suffix_output_dirs(specs, rec_id=2, declared_outputs=set())
    assert specs["train"]["results_dir"] == "/work/train/rec_2"


# --- audit #3 fail-loud (MetricExtractorError class is exported) ----------

def test_metric_extractor_error_is_runtime_error():
    """Existing call sites that catch RuntimeError continue to catch
    MetricExtractorError too — gives users a safe upgrade path."""
    assert issubclass(MetricExtractorError, RuntimeError)
    err = MetricExtractorError("test")
    assert "test" in str(err)
