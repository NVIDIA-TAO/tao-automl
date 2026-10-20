# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Bounded sequential controller with atomic state and native job reconciliation."""

import copy
import fcntl
import re
import subprocess
import time
from contextlib import contextmanager
from pathlib import Path

from automl_core import __version__, build_fingerprint
from automl_core._worker import atomic
from automl_core.contracts import Execution, fingerprint, override
from automl_core.controller import (
    ControllerHandoff,
    bounded_deadline,
    checkpoint,
    stop_experiment,
    termination_signals,
)
from automl_core.execution import Executor
from automl_core.recovery import attempt_root, health, read_receipt, validate_policy
from automl_core.scheduling import next_trial, review
from automl_core.search import best, tell
from automl_core.slurm import SchedulerUnavailable


def manage(workspace, experiment_id, cancel=False):
    """Observe/stop recorded native execution without mutable source manifests."""
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", experiment_id):
        raise ValueError("Invalid experiment ID")
    root = Path(workspace).resolve() / experiment_id
    if (root / "prompt.json").exists():
        from automl_core.prompt_runner import manage_prompt

        return manage_prompt(root, cancel=cancel)
    ledger = read_receipt(root / "experiment.json", missing_ok=False)
    executor = Executor(Execution.model_validate(ledger["identity"]["execution"]))
    if cancel:
        atomic(root / "cancel.json", {"requested_at": time.time()})
    for trial in ledger["trials"]:
        trial_root = attempt_root(root, trial)
        if trial["state"] in {"PENDING", "RUNNING"}:
            try:
                if cancel:
                    executor.cancel(trial_root)
                trial["observed_state"] = executor.status(trial_root)
                trial.pop("observation_error", None)
            except SchedulerUnavailable as exc:
                trial["observed_state"] = "UNKNOWN"
                trial["observation_error"] = {
                    "reason": "scheduler_unavailable",
                    "queries": exc.errors,
                }
            except subprocess.TimeoutExpired:
                if not cancel:
                    raise
                # Cancellation may have reached the scheduler. Preserve its
                # marker and report ambiguity; never claim terminal success.
                trial["observed_state"] = "UNKNOWN"
                trial["observation_error"] = {"reason": "cancellation_unconfirmed"}
            if (
                cancel
                and trial["observed_state"] in {"ERROR", "CANCELED"}
                and hasattr(executor, "finalize")
            ):
                executor.finalize(trial_root, trial["observed_state"])
        trial["log"] = executor.logs(trial_root)
    ledger["cancel_requested"] = (root / "cancel.json").exists()
    return ledger


class Experiment:
    def __init__(self, search, workload, execution, workspace):
        self.search = search
        self.workload = workload
        self.execution = execution
        self.root = Path(workspace).resolve() / search.experiment_id
        self.executor = Executor(execution)
        if execution.managed:
            self.executor.runtime_timeout_seconds = search.budget.trial_timeout_seconds
        if execution.kind == "slurm":
            settings = execution.slurm
            if (
                settings.startup_timeout_seconds
                + settings.cleanup_seconds
                + search.budget.trial_timeout_seconds
                > settings.time_limit_seconds
            ):
                raise ValueError(
                    "SLURM allocation must cover startup, trial runtime and cleanup"
                )
        self.identity = {
            "schema_version": 1,
            "engine_version": __version__,
            "engine_build": build_fingerprint(),
            "search": search.model_dump(mode="json"),
            "workload": workload.identity(),
            "execution": execution.model_dump(mode="json"),
        }

    def plan(self):
        """No directories, imports of training modules, or execution side effects."""
        policy = self.search.recovery
        validate_policy(policy, self.execution)
        if hasattr(self.workload, "validate_search"):
            self.workload.validate_search(self.search)
        if self.search.fidelity and not hasattr(self.workload, "plan_stage"):
            raise ValueError(
                "This workload has no resource/checkpoint promotion contract"
            )
        self.workload.validate(self.search.base_config)
        for path, parameter in self.search.parameters.items():
            for value in (
                parameter.choices
                if parameter.kind == "categorical"
                else [parameter.decode(0), parameter.decode(1)]
            ):
                self.workload.validate(override(self.search.base_config, {path: value}))
        if self.search.algorithm == "autoresearch":
            from automl_core.samplers import encode

            encode(self.search, self.search.base_config)
        if self.search.fidelity:
            plan = self.workload.plan_stage(
                self.search.base_config,
                self.root / "trial-00000",
                self.execution,
                {"resource": self.search.fidelity.minimum},
                None,
            )
        else:
            plan = self.workload.plan(
                self.search.base_config, self.root / "trial-00000", self.execution
            )
        validate_policy(policy, self.execution, plan)
        if hasattr(self.executor, "validate_plan"):
            self.executor.validate_plan(plan, self.root / "trial-00000")
        return {
            "identity": self.identity,
            "fingerprint": fingerprint(self.identity),
            "example_trial": plan.model_dump(mode="json"),
            "workspace": str(self.root),
            "concurrency": 1,
            "approval_required": True,
            "algorithm_review": review(self.search),
            **(
                {"execution_review": self.executor.review()}
                if hasattr(self.executor, "review")
                else {}
            ),
            "recovery_review": {
                **policy.model_dump(),
                "maximum_allocations": self.search.budget.max_trials
                + min(
                    policy.max_retries_total,
                    self.search.budget.max_trials * policy.max_retries_per_trial,
                ),
                "retryable": [
                    "NODE_FAIL",
                    "BOOT_FAIL",
                    "PREEMPTED",
                    "confirmed unresponsive node",
                    "unobserved submission fenced before workload start",
                ]
                if self.execution.kind == "slurm"
                else [],
                "deadline_extended": False,
                "partial_checkpoint_resume": False,
                "submission_reconcile_seconds": (
                    getattr(
                        self.execution, self.execution.kind
                    ).submission_reconcile_seconds
                    if self.execution.managed
                    else None
                ),
            },
        }

    @contextmanager
    def _lock(self):
        self.root.mkdir(parents=True, exist_ok=True)
        with (self.root / ".controller.lock").open("a") as handle:
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise ValueError("Another controller owns this experiment") from exc
            try:
                yield
            except ControllerHandoff:
                if getattr(self, "_active_ledger_ready", False):
                    ledger = self._read()
                    ledger.update(
                        state="SUSPENDED",
                        stop_reason="CPU controller allocation handoff",
                    )
                    self._save(ledger)
                raise
            except KeyboardInterrupt:
                if (
                    getattr(self, "_active_ledger_ready", False)
                    and self.execution.managed
                ):
                    stop_experiment(self, "Controller interrupted")
                    self._active_ledger_ready = False
                raise
            except Exception as exc:
                if (
                    getattr(self, "_active_ledger_ready", False)
                    and self.execution.managed
                ):
                    # Persist cleanup under the same lock, rather than leaving a
                    # stale RUNNING ledger and canceling after releasing ownership.
                    stopped = stop_experiment(
                        self, "Controller failed: " + type(exc).__name__
                    )
                    stopped["state"] = "ERROR"
                    self._save(stopped)
                    self._active_ledger_ready = False
                raise
            finally:
                fcntl.flock(handle, fcntl.LOCK_UN)

    def _read(self):
        ledger = read_receipt(self.root / "experiment.json", missing_ok=False)
        if ledger["fingerprint"] != fingerprint(self.identity):
            raise ValueError(
                "Experiment identity changed; use the original manifest or a new experiment ID"
            )
        return ledger

    def _save(self, ledger):
        ledger["best"] = best(self.search, ledger["trials"])
        atomic(self.root / "experiment.json", ledger)
        atomic(
            self.root / "best.json",
            {
                "schema_version": 1,
                "experiment_id": self.search.experiment_id,
                "objective": self.search.objective.model_dump(),
                "trial": ledger["best"],
            },
        )

    def status(self):
        return manage(self.root.parent, self.search.experiment_id)

    def cancel(self):
        return manage(self.root.parent, self.search.experiment_id, cancel=True)

    def run(self, resume=False):
        self._active_ledger_ready = False
        if not self.execution.managed:
            return self._run(resume)
        with termination_signals():
            try:
                return self._run(resume)
            except ControllerHandoff:
                raise  # Preserve active jobs; the bounded CPU supervisor owns continuation.
            except BaseException:
                if self._active_ledger_ready:
                    # Native deadlines remain effective if the scheduler is unavailable.
                    try:
                        self.cancel()
                    except (OSError, ValueError, subprocess.SubprocessError):
                        pass
                raise

    def _deadline(self, ledger):
        return min(
            self.search.budget.deadline(ledger["created_at"]),
            ledger.get("controller_deadline", float("inf")),
        )

    def _run(self, resume=False):
        from automl_core.controller import _session
        from automl_core.migration import continuation

        migration = continuation(self.root)
        if migration and migration.get("controller") and _session.get() is None:
            raise ValueError(
                "Migrated search requires its bounded CPU controller session"
            )
        plan = self.plan()
        with self._lock():
            path = self.root / "experiment.json"
            if path.exists() != resume:
                raise ValueError(
                    "Use resume for an existing experiment; run requires a new ID"
                )
            if resume:
                ledger = self._read()
            if self.execution.kind == "slurm":
                # Persist the clock before a possibly long read-only preflight.
                # A restart must not receive a new elapsed-time allowance.
                preparation = read_receipt(self.root / "_preflight.json")
                if (
                    preparation
                    and not resume
                    and preparation["fingerprint"] != fingerprint(self.identity)
                ):
                    raise ValueError(
                        "Preflight identity changed; use the reviewed manifest/build"
                    )
                created = (
                    ledger["created_at"]
                    if resume
                    else preparation.get("created_at", time.time())
                )
                deadline = bounded_deadline(
                    self._deadline(ledger)
                    if resume
                    else self.search.budget.deadline(created)
                )
                if preparation:
                    deadline = min(deadline, preparation["deadline"])
                atomic(
                    self.root / "_preflight.json",
                    {
                        "fingerprint": fingerprint(self.identity),
                        "created_at": created,
                        "deadline": deadline,
                    },
                )
                self.executor.configure_checks(self.root, deadline)
            environment = self.executor.preflight(Path(plan["example_trial"]["cwd"]))
            if resume:
                if ledger["environment"] != environment:
                    raise ValueError(
                        "Execution environment changed; restore it or use a new experiment ID"
                    )
                if ledger["state"] == "PAUSED":
                    ledger.setdefault("acknowledged_pauses", []).append(
                        len(ledger["trials"])
                    )
                    ledger["state"] = "RUNNING"
                    self._save(ledger)
                if ledger["state"] == "SUSPENDED":
                    ledger["state"] = "RUNNING"
                    ledger.pop("stop_reason", None)
                    self._save(ledger)
            else:
                ledger = {
                    "environment": environment,
                    "identity": self.identity,
                    "fingerprint": fingerprint(self.identity),
                    "created_at": created
                    if self.execution.kind == "slurm"
                    else time.time(),
                    "trials": [],
                    "state": "RUNNING",
                }
                self._save(ledger)
            self._active_ledger_ready = True
            deadline = bounded_deadline(self._deadline(ledger))
            if deadline < self._deadline(ledger):
                ledger["controller_deadline"] = deadline
                self._save(ledger)
            while True:
                checkpoint()
                active = next(
                    (
                        trial
                        for trial in ledger["trials"]
                        if trial["state"] in {"PENDING", "RUNNING"}
                    ),
                    None,
                )
                if active:
                    if self._wait(ledger, active) is False:
                        break
                    continue
                if (self.root / "cancel.json").exists():
                    ledger["state"] = "CANCELED"
                    break
                if ledger["trials"] and self._retry(
                    ledger, ledger["trials"][-1], deadline
                ):
                    continue
                if (self.root / "cancel.json").exists():
                    ledger["state"] = "CANCELED"
                    break
                if (
                    ledger.get("stop_reason")
                    == "recovery blocked: retry reconciliation required"
                ):
                    ledger["state"] = "ERROR"
                    break
                if self.search.stop_on_failure and any(
                    t["state"] in {"ERROR", "CANCELED"} for t in ledger["trials"]
                ):
                    ledger["state"] = "ERROR"
                    break
                if len(ledger["trials"]) in self.search.pause_after_trials and len(
                    ledger["trials"]
                ) not in ledger.get("acknowledged_pauses", []):
                    ledger["state"] = "PAUSED"
                    break
                if (
                    time.time() >= deadline
                    or len(ledger["trials"]) >= self.search.budget.max_trials
                    or (
                        self.execution.kind == "slurm"
                        and deadline - time.time()
                        < self.execution.slurm.time_limit_seconds
                    )
                ):
                    ledger["state"] = (
                        "COMPLETE" if best(self.search, ledger["trials"]) else "ERROR"
                    )
                    ledger["stop_reason"] = (
                        "trial cap reached"
                        if len(ledger["trials"]) >= self.search.budget.max_trials
                        else "experiment elapsed deadline"
                        if time.time() >= deadline
                        else "insufficient original elapsed budget for another allocation"
                    )
                    break
                if self.search.llm:
                    from automl_core.advice import Advice

                    try:
                        candidate = Advice(self.search, self.root, deadline).propose(
                            ledger["trials"], self.workload.validate
                        )
                    except (OSError, ValueError, TypeError, KeyError):
                        ledger["state"] = "ERROR"
                        ledger["stop_reason"] = (
                            "proposal failed validation or request envelope; inspect advice receipts"
                        )
                        self._save(ledger)
                        raise
                else:
                    candidate = next_trial(self.search, ledger["trials"])
                if candidate is None:
                    ledger["state"] = (
                        "COMPLETE" if best(self.search, ledger["trials"]) else "ERROR"
                    )
                    ledger["stop_reason"] = "algorithm schedule exhausted"
                    break
                if time.time() >= deadline:
                    continue  # LLM time is included in the same experiment budget.
                trial = {
                    "id": f"trial-{len(ledger['trials']):05d}",
                    **candidate,
                    "state": "PENDING",
                    "result": None,
                }
                ledger["trials"].append(trial)
                self._save(ledger)  # Reserve before staging / submitting.
                self._submit_trial(ledger, trial, deadline)
            self._save(ledger)
            return ledger

    def _submit_trial(self, ledger, trial, deadline, retry_of=None):
        root = attempt_root(self.root, trial)
        root.mkdir(exist_ok=False)
        config = trial["config"]
        try:
            self.workload.validate(config)
            if trial.get("resource") is not None:
                parent = next(
                    (
                        t
                        for t in ledger["trials"]
                        if t["id"] == trial.get("source_trial")
                    ),
                    None,
                )
                if trial.get("source_trial") and (
                    parent is None
                    or parent["state"] != "COMPLETE"
                    or parent["resource"] >= trial["resource"]
                ):
                    raise ValueError("Invalid checkpoint lineage")
                trial_plan = self.workload.plan_stage(
                    config, root, self.execution, trial, parent
                )
            else:
                trial_plan = self.workload.plan(config, root, self.execution)
            validate_policy(self.search.recovery, self.execution, trial_plan)
        except (ValueError, TypeError, KeyError):
            tell(trial, "ERROR")
            trial["reason"] = "candidate failed workload validation"
            self._save(ledger)
            return
        if (self.root / "cancel.json").exists():
            tell(trial, "CANCELED")
            self._save(ledger)
            return
        # Intent ambiguity leaves PENDING; it is never eligible for a retry.
        self.executor.submit(
            root,
            trial_plan,
            deadline
            if self.execution.managed
            else min(deadline, time.time() + self.search.budget.trial_timeout_seconds),
            **({"retry_of": retry_of} if retry_of else {}),
        )
        trial["state"] = "PENDING" if self.execution.managed else "RUNNING"
        self._save(ledger)

    def _retry(self, ledger, trial, deadline):
        policy = self.search.recovery
        if trial["state"] not in {"ERROR", "CANCELED"} or not trial.get(
            "failure", {}
        ).get("retryable"):
            return False
        attempts = trial.setdefault("attempts", [])
        used = sum(len(t.get("attempts", [])) for t in ledger["trials"])
        if (
            len(attempts) >= policy.max_retries_per_trial
            or used >= policy.max_retries_total
        ):
            trial["recovery_decision"] = "retry budget exhausted or disabled"
            return False
        allocation = self.execution.slurm.time_limit_seconds
        ready = trial.setdefault(
            "retry_ready_at",
            time.time() + min(300, policy.backoff_seconds * 2 ** len(attempts)),
        )
        self._save(ledger)
        if max(time.time(), ready) + allocation > deadline:
            trial["recovery_decision"] = (
                "insufficient original elapsed budget for retry"
            )
            return False
        while time.time() < ready:
            checkpoint()
            if (self.root / "cancel.json").exists():
                return False
            time.sleep(max(0, min(1, ready - time.time())))
        previous = attempt_root(self.root, trial)
        evidence = self._read_failure(ledger, trial, previous)
        if evidence is None:
            return False
        if (
            not evidence["retryable"]
            or time.time() + allocation > deadline
            or (self.root / "cancel.json").exists()
        ):
            trial["recovery_decision"] = (
                "retry blocked: reconcile native state or budget"
            )
            ledger["stop_reason"] = "recovery blocked: retry reconciliation required"
            return False
        self.executor.finalize(previous, "ERROR")
        # Keep the logical trial/config/proposal/lineage. Only execution gets a new ID.
        attempts.append(
            copy.deepcopy({k: v for k, v in trial.items() if k != "attempts"})
        )
        trial["attempt_id"] = f"{trial['id']}-attempt-{len(attempts):02d}"
        trial["retry_of"] = previous.name
        trial["state"], trial["result"] = "PENDING", None
        for key in (
            "failure",
            "reason",
            "retry_ready_at",
            "recovery_decision",
            "observed_state",
            "observation_error",
            "research_decision",
            "recovery",
            "termination_started_at",
            "submission_reconciliation",
            "cancellation_reconciliation",
        ):
            trial.pop(key, None)
        ledger["state"] = "RUNNING"
        ledger.pop("stop_reason", None)
        self._save(ledger)  # Reserve attempt and consume retry budget before launch.
        self._submit_trial(ledger, trial, deadline, retry_of=previous)
        return True

    def _read_failure(self, ledger, trial, root):
        """An outage while classifying/rechecking a failure is not retry denial."""
        failures = 0
        while True:
            checkpoint()
            try:
                evidence = self.executor.failure(root)
                trial.pop("observation_error", None)
                return evidence
            except SchedulerUnavailable as exc:
                failures += 1
                trial["observation_error"] = {
                    "reason": "scheduler_unavailable",
                    "operation": "failure_reconciliation",
                    "queries": exc.errors,
                    "consecutive_failures": failures,
                    "observed_at": time.time(),
                }
                self._save(ledger)
                deadline = self._deadline(ledger)
                if (self.root / "cancel.json").exists() or time.time() >= deadline:
                    self._unconfirmed_cancel(ledger, trial, root, request=False)
                    return None
                wake = min(
                    deadline,
                    time.time()
                    + min(300, self.execution.poll_seconds * 2 ** min(failures - 1, 8)),
                )
                while time.time() < wake and not (self.root / "cancel.json").exists():
                    checkpoint()
                    time.sleep(max(0, min(1, wake - time.time())))

    def _wait(self, ledger, trial):
        root = attempt_root(self.root, trial)
        failures = 0
        deadline = self._deadline(ledger)
        try:
            while True:
                checkpoint()
                if self.execution.kind == "slurm":
                    self.executor.continue_submission(root)
                if self.execution.kind == "slurm" and hasattr(
                    self.executor, "reconcile_submission"
                ):
                    observation = self.executor.reconcile_submission(root)
                    if observation:
                        trial["observed_state"] = "UNKNOWN"
                        trial["submission_reconciliation"] = observation
                        self._save(ledger)
                        if (
                            self.root / "cancel.json"
                        ).exists() or time.time() >= deadline:
                            self._unconfirmed_cancel(ledger, trial, root)
                            return False
                        if observation["decision"] == "blocked":
                            ledger.update(
                                state="ERROR",
                                stop_reason="submission reconciliation blocked: "
                                + observation["reason"],
                            )
                            self._save(ledger)
                            return False
                        if observation["decision"] == "fenced":
                            state = "ERROR"
                            break
                        # Read-only re-observation; no extra trial/proposal/RPC.
                        until = min(
                            deadline, time.time() + self.execution.slurm.poll_seconds
                        )
                        while (
                            time.time() < until
                            and not (self.root / "cancel.json").exists()
                        ):
                            checkpoint()
                            time.sleep(max(0, min(1, until - time.time())))
                        continue
                try:
                    state = self.executor.status(root)
                except SchedulerUnavailable as exc:
                    failures += 1
                    trial["observed_state"] = "UNKNOWN"
                    trial["observation_error"] = {
                        "reason": "scheduler_unavailable",
                        "queries": exc.errors,
                        "consecutive_failures": failures,
                        "observed_at": time.time(),
                    }
                    self._save(ledger)
                    recovery = read_receipt(root / "_recovery.json")
                    if (
                        recovery
                        and time.time()
                        >= recovery["cancel_requested_at"]
                        + self.search.recovery.cancellation_timeout_seconds
                    ):
                        self._reconciling_cancellation(ledger, trial, recovery)
                    if (self.root / "cancel.json").exists() or time.time() >= deadline:
                        self._unconfirmed_cancel(ledger, trial, root)
                        return False
                    # Retry observations only. Never consume a trial/API budget,
                    # fabricate RUNNING, or reset the original elapsed deadline.
                    delay = min(
                        300,
                        self.execution.poll_seconds * 2 ** min(failures - 1, 8),
                    )
                    until = min(deadline, time.time() + delay)
                    while (
                        time.time() < until and not (self.root / "cancel.json").exists()
                    ):
                        checkpoint()
                        time.sleep(max(0, min(1, until - time.time())))
                    if (self.root / "cancel.json").exists() or time.time() >= deadline:
                        self._unconfirmed_cancel(ledger, trial, root)
                        return False
                    continue
                failures = 0
                if "submission_reconciliation" in trial:
                    trial["submission_reconciliation"] = read_receipt(
                        root / "_submission_reconciliation.json"
                    )
                    trial["observed_state"] = state
                    self._save(ledger)
                if "observation_error" in trial:
                    trial.pop("observation_error")
                    trial["observed_state"] = state
                    self._save(ledger)
                if state == "UNKNOWN":
                    if self.execution.managed:
                        trial["observed_state"] = "UNKNOWN"
                        ledger.update(
                            state="ERROR",
                            stop_reason="native identity or completion evidence unresolved; reconciliation required",
                        )
                        self._save(ledger)
                        return False
                    raise ValueError(
                        f"{trial['id']}: launch state UNKNOWN; inspect native execution; no automatic retry"
                    )
                if state not in {"PENDING", "RUNNING"}:
                    break
                if (
                    trial.get("termination_started_at") is not None
                    and time.time() - trial["termination_started_at"]
                    >= self.search.recovery.cancellation_timeout_seconds
                ):
                    self._unconfirmed_cancel(ledger, trial, root, request=False)
                    return False
                recovery = read_receipt(root / "_recovery.json")
                if (
                    not recovery
                    and state == "RUNNING"
                    and not (self.root / "cancel.json").exists()
                ):
                    issue = (
                        self.executor.health(root, self.search.recovery)
                        if self.execution.kind == "slurm"
                        else health(root, self.search.recovery)
                    )
                    if issue:
                        recovery = {**issue, "cancel_requested_at": time.time()}
                        atomic(root / "_recovery.json", recovery)
                if recovery:
                    # The scheduler can discover a node failure after the initial
                    # watchdog. Refresh only against this still-owned allocation;
                    # never infer retryability from a stale heartbeat alone.
                    if self.execution.kind == "slurm" and hasattr(
                        self.executor, "refresh_recovery"
                    ):
                        recovery = self.executor.refresh_recovery(root, recovery)
                        atomic(root / "_recovery.json", recovery)
                    trial["recovery"] = recovery
                    self._save(ledger)
                    if (
                        time.time() - recovery["cancel_requested_at"]
                        >= self.search.recovery.cancellation_timeout_seconds
                        and (root / "_cancel.json").exists()
                    ):
                        # Slow epilog/unkillable tasks can outlive the cancellation
                        # RPC window. Keep the brain and its handoff checkpoints
                        # alive, but do not resubmit or repeatedly send scancel.
                        self._reconciling_cancellation(ledger, trial, recovery)
                    else:
                        try:
                            self.executor.cancel(root)
                        except (
                            SchedulerUnavailable,
                            subprocess.SubprocessError,
                            ValueError,
                        ):
                            pass  # A successful RPC is not termination evidence.
                if self.execution.managed and trial["state"] != state:
                    trial["state"] = state
                    self._save(ledger)
                if (self.root / "cancel.json").exists():
                    trial.setdefault("termination_started_at", time.time())
                    self._save(ledger)
                    try:
                        self.executor.cancel(root)
                    except (SchedulerUnavailable, subprocess.TimeoutExpired):
                        self._unconfirmed_cancel(ledger, trial, root, request=False)
                        return False
                if self.execution.managed and time.time() >= deadline:
                    trial["reason"] = "experiment elapsed deadline"
                    trial.setdefault("termination_started_at", time.time())
                    self._save(ledger)
                    try:
                        self.executor.cancel(root)
                    except (SchedulerUnavailable, subprocess.TimeoutExpired):
                        self._unconfirmed_cancel(ledger, trial, root, request=False)
                        return False
                until = time.time() + self.execution.poll_seconds
                while time.time() < until:
                    checkpoint()
                    time.sleep(max(0, min(1, until - time.time())))
        except KeyboardInterrupt:
            if not self.execution.managed:
                self.cancel()
            raise
        result = None
        if "cancellation_reconciliation" in trial:
            trial["cancellation_reconciliation"].update(
                state="CONFIRMED", confirmed_at=time.time(), terminal_state=state
            )
            trial["observed_state"] = state
        recovery = read_receipt(root / "_recovery.json")
        if recovery and state != "COMPLETE":
            state = "ERROR"  # Recovery cancellation is not a user cancellation.
        if state in {"ERROR", "CANCELED"} and self.execution.kind == "slurm":
            evidence = self._read_failure(ledger, trial, root)
            if evidence is None:
                return False
            trial["failure"] = evidence
        if state == "COMPLETE":
            try:
                result = self.workload.collect(root, self.search.objective)
            except (OSError, ValueError, TypeError, KeyError):
                state = "ERROR"
                trial["reason"] = "missing or invalid objective/artifacts"
        incumbent = best(self.search, [t for t in ledger["trials"] if t is not trial])
        tell(trial, state, result)
        if self.search.algorithm == "autoresearch":
            improved = result is not None and (
                incumbent is None
                or (
                    result.metric > incumbent["result"]["metric"]
                    if self.search.objective.direction == "maximize"
                    else result.metric < incumbent["result"]["metric"]
                )
            )
            trial["research_decision"] = "keep" if improved else "discard"
        self._save(ledger)
        if hasattr(self.executor, "finalize"):
            self.executor.finalize(root, state)

    def _reconciling_cancellation(self, ledger, trial, recovery):
        """Waiting is not failure or permission for a replacement allocation."""
        trial.setdefault(
            "cancellation_reconciliation",
            {
                "state": "WAITING",
                "started_at": time.time(),
                "cancel_requested_at": recovery["cancel_requested_at"],
                "deadline": self._deadline(ledger),
                "reason": "awaiting native termination after cancellation window",
            },
        )
        self._save(ledger)

    def _unconfirmed_cancel(self, ledger, trial, root, request=True):
        """End bounded monitoring without inventing a terminal native job state."""
        if request:
            try:
                self.executor.cancel(root)
            except (SchedulerUnavailable, subprocess.TimeoutExpired, ValueError):
                # The SLURM cancel marker is durable before any scheduler RPC.
                # Batch/worker deadlines still bound queued and running work.
                pass
        trial["observed_state"] = "UNKNOWN"
        trial["reason"] = "Cancellation requested; native reconciliation required"
        ledger["state"] = "ERROR"
        ledger["stop_reason"] = (
            "User cancellation or elapsed deadline reached while scheduler unavailable; "
            "inspect native jobs before resuming; no automatic retry"
        )
        self._save(ledger)
