# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Comprehensive test for the nvidia-tao-automl wheel."""

import tempfile
import threading
import subprocess

import pytest


# ---------------------------------------------------------------
# 1. Import tests
# ---------------------------------------------------------------

def test_import_top_level():
    from tao_automl import AutoML
    assert AutoML is not None


def test_import_types():
    from tao_automl.types import Recommendation, ResumeRecommendation, JobStates, AutoMLContext
    assert JobStates.success == "success"


def test_import_state_store():
    from tao_automl.state.state_store import StateStore
    assert StateStore is not None


def test_import_controller():
    from tao_automl.controller.controller import Controller
    assert Controller is not None


def test_import_brain_factory():
    from tao_automl.brain.factory import BrainFactory, AlgorithmParams
    assert BrainFactory is not None


def test_import_brain_algorithms():
    from tao_automl.brain.bayesian import Bayesian
    from tao_automl.brain.hyperband import HyperBand
    from tao_automl.brain.bohb import BOHB
    from tao_automl.brain.asha import ASHA
    from tao_automl.brain.bfbo import BFBO
    from tao_automl.brain.dehb import DEHB
    from tao_automl.brain.pbt import PBT
    from tao_automl.brain.hyperband_es import HyperBandES
    assert all([Bayesian, HyperBand, BOHB, ASHA, BFBO, DEHB, PBT, HyperBandES])


def test_import_utils():
    from tao_automl.utils.math_utils import fix_input_dimension, clamp_value
    from tao_automl.utils.spec_utils import get_flatten_specs
    from tao_automl.utils.network_constants import gpu_mapper
    assert callable(fix_input_dimension)


# ---------------------------------------------------------------
# 2. Types tests
# ---------------------------------------------------------------

def test_recommendation_create():
    from tao_automl.types import Recommendation, JobStates
    rec = Recommendation(identifier=0, specs={"lr": 0.01}, metric="loss")
    assert rec.id == 0
    assert rec.status == JobStates.pending
    assert rec.result == 0.0


def test_recommendation_update():
    from tao_automl.types import Recommendation, JobStates
    rec = Recommendation(identifier=1, specs={"lr": 0.01}, metric="loss")
    rec.update_result(0.5)
    assert rec.result == 0.5
    rec.update_status(JobStates.success)
    assert rec.status == "success"
    rec.assign_job_id("job-123")
    assert rec.job_id == "job-123"


def test_recommendation_type_checks():
    from tao_automl.types import Recommendation
    with pytest.raises(AssertionError):
        Recommendation(identifier="bad", specs={}, metric="loss")


def test_automl_context():
    from tao_automl.types import AutoMLContext
    ctx = AutoMLContext(id="t1", network="dino", metric="loss")
    assert ctx.action == "train"


# ---------------------------------------------------------------
# 3. StateStore tests
# ---------------------------------------------------------------

def test_state_store_roundtrip():
    from tao_automl.state.state_store import StateStore
    with tempfile.TemporaryDirectory() as d:
        store = StateStore(d)
        store.save_job_specs("j1", {"lr": 0.01})
        assert store.get_job_specs("j1") == {"lr": 0.01}

        store.save_brain_info("j1", {"Xs": [[0.1]], "ys": [0.5]})
        assert store.get_brain_info("j1")["ys"] == [0.5]

        store.save_controller_info("j1", [{"id": 0}])
        assert store.get_controller_info("j1") == [{"id": 0}]

        store.save_best_rec_info("j1", rec_number=0, rec_data={"m": 0.1})
        assert store.get_best_rec_info("j1")["rec_number"] == 0

        store.save_custom_param_ranges("e1", {"lr": {"min": 1e-5}})
        assert store.get_custom_param_ranges("e1")["lr"]["min"] == 1e-5


def test_state_store_missing_returns_none():
    from tao_automl.state.state_store import StateStore
    with tempfile.TemporaryDirectory() as d:
        store = StateStore(d)
        assert store.get_job_specs("nope") is None
        assert store.get_brain_info("nope") is None


# ---------------------------------------------------------------
# 4. Controller tests
# ---------------------------------------------------------------

class MockBrain:
    def __init__(self, limit=5):
        self.limit = limit
        self.n = 0

    def generate_recommendations(self, history):
        if len(history) < self.limit:
            self.n += 1
            return [{"lr": 0.01 * self.n}]
        return []

    def save_state(self):
        pass


def _make_ctrl(d, limit=5, metric="loss"):
    from tao_automl.controller.controller import Controller
    from tao_automl.state.state_store import StateStore
    from tao_automl.types import AutoMLContext
    store = StateStore(d)
    ctx = AutoMLContext(id="test", network="dino")
    settings = type("P", (), {"automl_max_recommendations": limit})()
    return Controller(
        brain=MockBrain(limit), context=ctx, state_store=store,
        settings=settings, metric=metric, algorithm="bayesian",
    ), store, ctx


def test_controller_full_loop():
    with tempfile.TemporaryDirectory() as d:
        ctrl, _, _ = _make_ctrl(d, 5)
        for i in range(5):
            recs = ctrl.next_recommendation()
            assert recs[0].id == i
            ctrl.report_result(recs[0].id, 0.5 - i * 0.1)
        assert ctrl.is_complete()
        assert abs(ctrl.get_best().result - 0.1) < 1e-9
        assert ctrl.get_progress()["completed"] == 5


def test_controller_persistence():
    from tao_automl.controller.controller import Controller
    with tempfile.TemporaryDirectory() as d:
        ctrl, store, ctx = _make_ctrl(d, 5)
        for i in range(3):
            recs = ctrl.next_recommendation()
            ctrl.report_result(recs[0].id, 0.5 - i * 0.1)

        settings = type("P", (), {"automl_max_recommendations": 5})()
        ctrl2 = Controller.load_state(
            brain=MockBrain(5), context=ctx, state_store=store,
            settings=settings, metric="loss", algorithm="bayesian",
        )
        assert len(ctrl2.history) == 3
        assert ctrl2._next_id == 3


def test_controller_higher_is_better():
    with tempfile.TemporaryDirectory() as d:
        ctrl, _, _ = _make_ctrl(d, 3, metric="accuracy")
        for m in [0.8, 0.95, 0.9]:
            recs = ctrl.next_recommendation()
            ctrl.report_result(recs[0].id, m)
        assert abs(ctrl.get_best().result - 0.95) < 1e-9


def test_controller_failure_skipped_in_best():
    with tempfile.TemporaryDirectory() as d:
        ctrl, _, _ = _make_ctrl(d, 2)
        recs = ctrl.next_recommendation()
        ctrl.report_result(recs[0].id, 0.0, status="failure")
        assert ctrl.get_best() is None


# ---------------------------------------------------------------
# 5. AlgorithmParams tests
# ---------------------------------------------------------------

def test_algorithm_params():
    from tao_automl.brain.factory import AlgorithmParams
    p = AlgorithmParams.from_dict({"automl_max_recommendations": 15})
    assert p.automl_max_recommendations == 15
    assert p.automl_reduction_factor == 3  # default


# ---------------------------------------------------------------
# 6. Concurrent report_result
# ---------------------------------------------------------------

def test_concurrent_reports():
    with tempfile.TemporaryDirectory() as d:
        ctrl, _, _ = _make_ctrl(d, 8)
        all_recs = []
        for _ in range(8):
            all_recs.extend(ctrl.next_recommendation())

        errors = []
        def report(rid, m):
            try:
                ctrl.report_result(rid, m)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=report, args=(r.id, 0.5 - r.id * 0.05)) for r in all_recs]
        for t in threads: t.start()
        for t in threads: t.join()

        assert not errors
        assert ctrl.is_complete()


# ---------------------------------------------------------------
# 7. Wheel metadata
# ---------------------------------------------------------------

def test_version():
    import tao_automl
    assert tao_automl.__version__ == "0.1.0"


def test_pip_show():
    r = subprocess.run(["pip", "show", "nvidia-tao-automl"], capture_output=True, text=True)
    assert r.returncode == 0
    assert "0.1.0" in r.stdout
