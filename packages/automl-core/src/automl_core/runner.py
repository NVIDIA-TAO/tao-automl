# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Bounded sequential controller with atomic state and native job reconciliation."""

import fcntl
import json
import re
import time
from contextlib import contextmanager
from pathlib import Path

from automl_core import __version__
from automl_core._worker import atomic
from automl_core.contracts import Execution, fingerprint, override
from automl_core.execution import Executor
from automl_core.search import ask, best, tell


def manage(workspace, experiment_id, cancel=False):
    """Observe/stop recorded native execution without mutable source manifests."""
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", experiment_id):
        raise ValueError("Invalid experiment ID")
    root = Path(workspace).resolve() / experiment_id
    ledger = json.loads((root / "experiment.json").read_text())
    executor = Executor(Execution.model_validate(ledger["identity"]["execution"]))
    if cancel:
        atomic(root / "cancel.json", {"requested_at": time.time()})
    for trial in ledger["trials"]:
        trial_root = root / trial["id"]
        if trial_root.resolve().parent != root.resolve():
            raise ValueError("Invalid recorded trial directory")
        if trial["state"] in {"PENDING", "RUNNING"}:
            if cancel:
                executor.cancel(trial_root)
            trial["observed_state"] = executor.status(trial_root)
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
        self.identity = {
            "schema_version": 1,
            "engine_version": __version__,
            "search": search.model_dump(mode="json"),
            "workload": workload.identity(),
            "execution": execution.model_dump(mode="json"),
        }

    def plan(self):
        """No directories, imports of training modules, or execution side effects."""
        self.workload.validate(self.search.base_config)
        for path, parameter in self.search.parameters.items():
            for value in (
                parameter.choices
                if parameter.kind == "categorical"
                else [parameter.decode(0), parameter.decode(1)]
            ):
                self.workload.validate(override(self.search.base_config, {path: value}))
        plan = self.workload.plan(
            self.search.base_config, self.root / "trial-00000", self.execution
        )
        return {
            "identity": self.identity,
            "fingerprint": fingerprint(self.identity),
            "example_trial": plan.model_dump(mode="json"),
            "workspace": str(self.root),
            "concurrency": 1,
            "approval_required": True,
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
            finally:
                fcntl.flock(handle, fcntl.LOCK_UN)

    def _read(self):
        ledger = json.loads((self.root / "experiment.json").read_text())
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
        plan = self.plan()
        environment = self.executor.preflight(Path(plan["example_trial"]["cwd"]))
        with self._lock():
            path = self.root / "experiment.json"
            if path.exists() != resume:
                raise ValueError(
                    "Use resume for an existing experiment; run requires a new ID"
                )
            if resume:
                ledger = self._read()
                if ledger["environment"] != environment:
                    raise ValueError(
                        "Execution environment changed; restore it or use a new experiment ID"
                    )
            else:
                ledger = {
                    "environment": environment,
                    "identity": self.identity,
                    "fingerprint": fingerprint(self.identity),
                    "created_at": time.time(),
                    "trials": [],
                    "state": "RUNNING",
                }
                self._save(ledger)
            deadline = ledger["created_at"] + self.search.budget.total_timeout_seconds
            while True:
                active = next(
                    (
                        trial
                        for trial in ledger["trials"]
                        if trial["state"] in {"PENDING", "RUNNING"}
                    ),
                    None,
                )
                if active:
                    self._wait(ledger, active)
                    continue
                if (self.root / "cancel.json").exists():
                    ledger["state"] = "CANCELED"
                    break
                if (
                    time.time() >= deadline
                    or len(ledger["trials"]) >= self.search.budget.max_trials
                ):
                    ledger["state"] = (
                        "COMPLETE" if best(self.search, ledger["trials"]) else "ERROR"
                    )
                    break
                point, config = ask(self.search, ledger["trials"])
                trial = {
                    "id": f"trial-{len(ledger['trials']):05d}",
                    "point": point,
                    "config": config,
                    "state": "PENDING",
                    "result": None,
                }
                ledger["trials"].append(trial)
                self._save(ledger)  # Reserve before staging / submitting.
                root = self.root / trial["id"]
                root.mkdir(exist_ok=False)
                try:
                    self.workload.validate(config)
                    trial_plan = self.workload.plan(config, root, self.execution)
                except (ValueError, TypeError, KeyError):
                    tell(trial, "ERROR")
                    trial["reason"] = "candidate failed workload validation"
                    self._save(ledger)
                    continue
                if (self.root / "cancel.json").exists():
                    tell(trial, "CANCELED")
                    self._save(ledger)
                    continue
                # Submission exceptions leave PENDING for explicit reconciliation.
                self.executor.submit(
                    root,
                    trial_plan,
                    min(
                        deadline, time.time() + self.search.budget.trial_timeout_seconds
                    ),
                )
                trial["state"] = "RUNNING"
                self._save(ledger)
            self._save(ledger)
            return ledger

    def _wait(self, ledger, trial):
        root = self.root / trial["id"]
        try:
            while True:
                state = self.executor.status(root)
                if state == "UNKNOWN":
                    raise ValueError(
                        f"{trial['id']}: launch state UNKNOWN; inspect native execution; no automatic retry"
                    )
                if state != "RUNNING":
                    break
                if (self.root / "cancel.json").exists():
                    self.executor.cancel(root)
                time.sleep(0.2)
        except KeyboardInterrupt:
            self.cancel()
            raise
        result = None
        if state == "COMPLETE":
            try:
                result = self.workload.collect(root, self.search.objective)
            except (OSError, ValueError, TypeError, KeyError):
                state = "ERROR"
                trial["reason"] = "missing or invalid objective/artifacts"
        tell(trial, state, result)
        self._save(ledger)
