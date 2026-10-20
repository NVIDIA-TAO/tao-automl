# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Opt-in, bounded CPU-controller continuation; never a GPU retry policy.

SLURM requeues the same CPU batch job. A cooperative safe point checkpoints
the search before requeue; durable trial identities are adopted on restart.
"""

import datetime
import fcntl
import json
import math
import os
import re
import signal
import subprocess
import time
from contextlib import contextmanager
from contextvars import ContextVar

from automl_core._worker import atomic
from automl_core.contracts import fingerprint
from automl_core.slurm import duration

_session = ContextVar("automl_controller_session", default=None)
_termination_scope = ContextVar("automl_termination_scope", default=False)


class ControllerHandoff(BaseException):
    """Not a workload failure, user cancellation, or manual review pause."""


def bounded_deadline(deadline):
    session = _session.get()
    return min(deadline, session.deadline) if session else deadline


def checkpoint():
    """Call only after submission/request receipts are durably committed."""
    session = _session.get()
    if session:
        session.checkpoint()


def review(max_allocations, allocation_seconds, grace_seconds):
    if (
        max_allocations < 1
        or allocation_seconds <= grace_seconds + 60
        or grace_seconds < 60
    ):
        raise ValueError(
            "Controller needs positive allocation cap and >=60s grace plus useful runtime"
        )
    return {
        "max_allocations": max_allocations,
        "allocation_seconds": allocation_seconds,
        "grace_seconds": grace_seconds,
        "maximum_cpu_allocation_seconds": max_allocations * allocation_seconds,
        "mechanism": "same CPU job, bounded SLURM requeue",
        "additional_gpu_allocations": 0,
        "deadline_extended": False,
    }


def _native(job_id):
    result = subprocess.run(
        ["scontrol", "show", "job", "-o", job_id],
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
    )
    fields = dict(re.findall(r"(?:^|\s)(\w+)=([^\s]+)", result.stdout))
    if fields.get("JobId") != job_id or not fields.get("UserId", "").endswith(
        f"({os.getuid()})"
    ):
        raise ValueError("CPU controller native ownership could not be verified")
    if fields.get("BatchFlag", "0") == "0" or fields.get("JobState") != "RUNNING":
        raise ValueError("Controller handoff requires its own running SLURM batch job")
    for name in ("ReqTRES", "AllocTRES", "TresPerNode", "Gres"):
        if "gpu" in fields.get(name, "").lower():
            raise ValueError("Controller handoff is for CPU allocations, not GPU jobs")
    return fields


@contextmanager
def termination_signals():
    """Give cleanup one chance; repeated scheduler signals do not interrupt it."""
    if _termination_scope.get():
        # Prompt orchestration and its child Experiment share one cleanup scope.
        yield
        return
    token = _termination_scope.set(True)
    handlers = {}
    stopping = False

    def stop(signum, frame):
        nonlocal stopping
        if not stopping:
            stopping = True
            raise KeyboardInterrupt("Controller termination requested")

    import threading

    if threading.current_thread() is threading.main_thread():
        for signum in (signal.SIGTERM, signal.SIGHUP):
            handlers[signum] = signal.signal(signum, stop)
    try:
        yield
    finally:
        for signum, handler in handlers.items():
            signal.signal(signum, handler)
        _termination_scope.reset(token)


def stop_experiment(experiment, reason):
    """Persist observation, not an unproven claim that native jobs stopped.

    Caller must own the experiment lock. Cancellation markers and independent
    worker deadlines remain effective if the scheduler cannot be contacted.
    """
    prompt = (experiment.root / "prompt.json").exists()
    path = experiment.root / ("prompt.json" if prompt else "experiment.json")
    saved = json.loads(path.read_text())
    try:
        observed = experiment.cancel()
        trials = (
            [
                trial
                for batch in observed["batches"]
                for trial in batch.get("native", {}).get("trials", [])
            ]
            if prompt
            else observed["trials"]
        )
        confirmed = all(
            trial.get("observed_state", trial["state"])
            in {"COMPLETE", "ERROR", "CANCELED"}
            for trial in trials
        )
        saved["termination_observation"] = observed
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError):
        confirmed = False
    saved.update(
        state="CANCELED" if confirmed else "ERROR",
        stop_reason=reason
        + (
            ""
            if confirmed
            else "; native cancellation unconfirmed; reconcile before resume"
        ),
    )
    atomic(path, saved)
    return saved


class Session:
    def __init__(self, experiment, policy):
        if experiment.execution.kind != "slurm":
            raise ValueError(
                "CPU controller handoff currently requires the SLURM executor"
            )
        self.experiment, self.policy = experiment, policy
        self.root = experiment.root
        self.requested = False
        self.job_id = os.environ.get("SLURM_JOB_ID", "")
        if not re.fullmatch(r"[1-9][0-9]*", self.job_id):
            raise ValueError("Controller handoff must run inside a SLURM CPU batch job")

    def prepare(self):
        ledger_path = self.root / (
            "experiment.json" if hasattr(self.experiment, "search") else "prompt.json"
        )
        ledger = json.loads(ledger_path.read_text()) if ledger_path.exists() else None
        if ledger and ledger["fingerprint"] != fingerprint(self.experiment.identity):
            raise ValueError(
                "Experiment identity changed; do not retrofit a different controller build"
            )
        # Native identity is read-only but was another unhandled 30s timeout
        # before the search could even install its observation recovery loop.
        preparation_path = self.root / "_controller_preflight.json"
        preparation = (
            json.loads(preparation_path.read_text())
            if preparation_path.exists()
            else {}
        )
        generation = os.environ.get("SLURM_RESTART_COUNT", "0")
        key = {
            "job_id": self.job_id,
            "generation": generation,
            "experiment": fingerprint(self.experiment.identity),
        }
        if preparation.get("identity") != key:
            preparation = {"identity": key, "started_at": time.time()}
        created = ledger["created_at"] if ledger else preparation["started_at"]
        original = (
            self.experiment.search.budget.deadline(created)
            if hasattr(self.experiment, "search")
            else created + self.experiment.settings.total_timeout_seconds
        )
        if ledger:
            original = min(original, ledger.get("controller_deadline", original))
        until = min(
            original,
            preparation["started_at"]
            + self.policy["allocation_seconds"]
            - self.policy["grace_seconds"],
        )
        atomic(preparation_path, preparation)

        def read_native():
            while True:
                if (
                    time.time() >= until
                    or self.requested
                    or (self.root / "cancel.json").exists()
                ):
                    preparation.update(state="STOPPED", updated_at=time.time())
                    atomic(preparation_path, preparation)
                    raise ValueError(
                        "CPU identity verification exhausted its original time budget or was stopped"
                    )
                try:
                    native = _native(self.job_id)
                    preparation.update(state="COMPLETE", updated_at=time.time())
                    atomic(preparation_path, preparation)
                    return native
                except subprocess.TimeoutExpired:
                    preparation.update(
                        state="WAITING",
                        operation="cpu_native_identity",
                        updated_at=time.time(),
                    )
                    atomic(preparation_path, preparation)
                    wake = min(until, time.time() + 5)
                    while time.time() < wake and not self.requested:
                        time.sleep(max(0, min(1, wake - time.time())))

        native = read_native()
        from automl_core.migration import validate_controller

        validate_controller(self.root, self.policy, native)
        seconds = duration(native["TimeLimit"])
        if (
            not math.isfinite(seconds)
            or not self.policy["grace_seconds"] + 60
            < seconds
            <= self.policy["allocation_seconds"]
        ):
            raise ValueError(
                "Native CPU wall time does not fit the reviewed controller envelope"
            )
        self.generation = int(native["Restarts"])
        if (
            self.generation < self.policy["max_allocations"] - 1
            and native.get("Requeue") != "1"
        ):
            raise ValueError("Controller batch must explicitly enable --requeue")
        # These resource/owner fields must remain identical on every allocation.
        shape = {
            name: native.get(name)
            for name in (
                "JobId",
                "UserId",
                "Partition",
                "Account",
                "QOS",
                "NumCPUs",
                "NumNodes",
                "ReqTRES",
                "TimeLimit",
            )
        }
        self.path = self.root / "controller.json"
        identity = {
            "experiment": fingerprint(self.experiment.identity),
            "policy": self.policy,
            "native": shape,
        }
        if self.path.exists():
            self.saved = json.loads(self.path.read_text())
            if self.saved["identity"] != identity:
                raise ValueError(
                    "Controller job, build, resources or handoff policy changed"
                )
            previous = self.saved["generation"]
            if self.generation != previous + 1:
                raise ValueError(
                    "Controller allocation was already consumed or native restart identity changed"
                )
        else:
            if self.generation != 0:
                raise ValueError(
                    "Untracked controller restart; use a fresh reviewed CPU job"
                )
            if hasattr(self.experiment, "search"):
                created = ledger["created_at"] if ledger else preparation["started_at"]
                deadline = self.experiment.search.budget.deadline(created)
            else:
                created = ledger["created_at"] if ledger else preparation["started_at"]
                deadline = created + self.experiment.settings.total_timeout_seconds
            if ledger:
                deadline = min(deadline, ledger.get("controller_deadline", deadline))
            self.saved = {"identity": identity, "deadline": deadline, "allocations": []}
        self.deadline = self.saved["deadline"]
        end = datetime.datetime.fromisoformat(native["EndTime"]).timestamp()
        self.soft_deadline = end - self.policy["grace_seconds"]
        if (
            self.generation >= self.policy["max_allocations"] - 1
            and native.get("Requeue") != "0"
        ):
            # On the last permitted allocation, disable scheduler/node-failure
            # requeue as well as our own handoff. Site/admin overrides cannot
            # be bounded by an unprivileged controller.
            try:
                subprocess.run(
                    ["scontrol", "update", f"JobId={self.job_id}", "Requeue=0"],
                    capture_output=True,
                    timeout=30,
                    check=True,
                )
            except subprocess.TimeoutExpired:
                pass  # Lost acknowledgement: verify the effect; do not repeat the RPC.
            if read_native().get("Requeue") != "0":
                raise ValueError("Could not disable requeue on final CPU allocation")
        self.saved.update(generation=self.generation, state="RUNNING")
        self.saved["allocations"].append(
            {
                "generation": self.generation,
                "started_at": time.time(),
                "native_end": end,
            }
        )
        atomic(self.path, self.saved)

    def checkpoint(self):
        if time.time() >= self.deadline or (self.root / "cancel.json").exists():
            return  # Normal experiment cleanup owns overall expiry/cancellation.
        if self.requested or time.time() >= self.soft_deadline:
            raise ControllerHandoff()

    def _stop(self, reason):
        # No child controller remains: Handoff unwound and released its lock.
        with (self.root / ".controller.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            ledger = self.root / (
                "experiment.json"
                if hasattr(self.experiment, "search")
                else "prompt.json"
            )
            if ledger.exists():
                result = stop_experiment(self.experiment, reason)
                # A resource cap failure is distinct from explicit user cancellation.
                result["state"] = "ERROR"
                atomic(ledger, result)
            else:
                result = {"state": "ERROR", "stop_reason": reason}
        self.saved.update(state="ERROR", stop_reason=reason)
        atomic(self.path, self.saved)
        return result

    def handoff(self):
        if self.generation + 1 >= self.policy["max_allocations"]:
            return self._stop("CPU controller allocation cap exhausted")
        if (
            time.time() + self.policy["grace_seconds"] >= self.deadline
            or (self.root / "cancel.json").exists()
        ):
            return self._stop(
                "Insufficient original experiment time for CPU controller handoff or cancellation requested"
            )
        # Intent precedes RPC. Never issue it twice after lost acknowledgement.
        self.saved.update(state="REQUEUE_REQUESTED", requeue_requested_at=time.time())
        atomic(self.path, self.saved)
        old_term = signal.signal(signal.SIGTERM, lambda *args: None)
        old_hup = signal.signal(signal.SIGHUP, lambda *args: None)
        try:
            try:
                subprocess.run(
                    ["scontrol", "requeue", self.job_id],
                    capture_output=True,
                    timeout=30,
                    check=True,
                )
            except (OSError, subprocess.SubprocessError):
                # The RPC may have succeeded. Leave GPU identities and the
                # consumed intent intact; the next native generation can resume.
                self.saved.update(state="REQUEUE_UNCONFIRMED")
                atomic(self.path, self.saved)
                return {
                    "state": "ERROR",
                    "stop_reason": "CPU requeue unconfirmed; inspect native job; do not repeat submission",
                    "controller": self.saved,
                }
            return {"state": "SUSPENDED", "controller": self.saved}
        finally:
            signal.signal(signal.SIGTERM, old_term)
            signal.signal(signal.SIGHUP, old_hup)

    def run(self, resume=False):
        self.root.mkdir(parents=True, exist_ok=True)
        with (self.root / ".supervisor.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            previous = signal.signal(
                signal.SIGUSR1, lambda *args: setattr(self, "requested", True)
            )
            token = None
            prepared = False
            try:
                self.prepare()
                prepared = True
                if self.generation >= self.policy["max_allocations"]:
                    return self._stop(
                        "Native CPU restarts exceeded the approved allocation cap"
                    )
                token = _session.set(self)
                ledger = self.root / (
                    "experiment.json"
                    if hasattr(self.experiment, "search")
                    else "prompt.json"
                )
                # Only a verified native successor automatically switches run to resume.
                result = self.experiment.run(
                    resume=resume or (self.generation > 0 and ledger.exists())
                )
                self.saved.update(state=result["state"])
                atomic(self.path, self.saved)
                return result
            except ControllerHandoff:
                return self.handoff()
            except BaseException:
                if prepared:
                    self.saved.update(
                        state="ERROR",
                        stop_reason="Controller interrupted or failed; inspect experiment and native receipts",
                    )
                    atomic(self.path, self.saved)
                raise
            finally:
                signal.signal(signal.SIGUSR1, previous)
                if token is not None:
                    _session.reset(token)
