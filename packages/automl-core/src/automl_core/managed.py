# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Durable shared-storage lifecycle for Kubernetes, Brev and Lepton.

No provisioning, data transfer, TAO imports or automatic training retries.
Platform transports implement native identity, observation and bounded cancel.
"""

import time
import uuid
from pathlib import Path

from automl_core import _submission, _worker
from automl_core.contracts import fingerprint
from automl_core.recovery import read_receipt
from automl_core.slurm import SchedulerUnavailable


class ManagedExecutor:
    def __init__(self, spec):
        self.spec = spec
        self.settings = getattr(spec, spec.kind)
        self.runtime_timeout_seconds = None

    def validate_plan(self, plan, trial):
        if not plan.cwd.is_dir():
            raise ValueError("Workload cwd must be an existing shared directory")
        self.storage_check([plan.cwd, *plan.read_only_mounts])
        if not any(
            trial.is_relative_to(v.path) and not v.read_only
            for v in self.settings.volumes
        ):
            raise ValueError(
                "The workspace must be under a writable persistent shared volume"
            )

    def review(self):
        return {
            "platform": self.spec.kind,
            "gpus_per_trial": self.settings.gpus,
            "storage": "same absolute paths and POSIX locks on controller and worker",
            "provisioning": False,
            "automatic_training_retries": False,
            "submission_reconcile_seconds": self.settings.submission_reconcile_seconds,
            "deadline": "worker enforces original absolute deadline and trial runtime; Kubernetes also sets activeDeadlineSeconds",
            "controller_required": "Keep controller alive for queue/startup cancellation; Lepton/Brev have no native allocation wall limit in this adapter",
        }

    def storage_check(self, paths, output=None):
        for path in paths:
            path = Path(path)
            if not path.is_absolute() or not path.exists():
                raise ValueError(
                    f"Shared path does not exist on the controller: {path}"
                )
            if not any(path.is_relative_to(v.path) for v in self.settings.volumes):
                raise ValueError(
                    f"Shared path is not covered by a declared volume: {path}"
                )
        for volume in self.settings.volumes:
            if not volume.path.is_dir():
                raise ValueError(f"Controller volume is not mounted: {volume.path}")
        if output and not any(
            output.is_relative_to(v.path) and not v.read_only
            for v in self.settings.volumes
        ):
            raise ValueError("The trial must be on a writable persistent shared volume")

    def command(self, trial, intent):
        return [
            self.spec.python,
            str(trial / "_managed_worker.py"),
            str(trial / "_execution.json"),
            intent["token"],
        ]

    def submit(self, trial, plan, deadline):
        self.storage_check([plan.cwd, trial, *plan.read_only_mounts], output=trial)
        if any(
            path.resolve().is_relative_to(trial.resolve())
            for path in plan.read_only_mounts
        ):
            raise ValueError("Stage inputs must be outside the new trial")
        if (trial / "_remote.json").exists() or (trial / "_execution.json").exists():
            raise ValueError("Managed launch intent exists; reconcile, never resubmit")
        for name, content in plan.files.items():
            target = trial / name
            if name.startswith("_") or not target.resolve().is_relative_to(trial):
                raise ValueError("Unsafe or reserved staged path")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content)
        for name in ("_worker.py", "_submission.py", "_managed_worker.py"):
            (trial / name).write_text(Path(__file__).with_name(name).read_text())
        token = uuid.uuid4().hex
        intent = {
            "token": token,
            "name": "automl-" + token,
            "deadline": deadline,
            "submitted_at": time.time(),
            "scope": self.scope(),
            "native_id": None,
        }
        _worker.atomic(trial / "_submission_gate.json", {"state": "open"})
        _worker.atomic(
            trial / "_execution.json",
            {
                "commands": plan.commands,
                "cwd": str(plan.cwd),
                "deadline": deadline,
                "runtime_timeout_seconds": self.runtime_timeout_seconds
                or max(0, deadline - time.time()),
                "progress_files": plan.progress_files,
                "environment": plan.environment,
            },
        )
        body = self.render(trial, plan, intent)
        intent["spec_hash"] = fingerprint(body)
        _worker.atomic(trial / "_native_spec.json", body)
        _worker.atomic(trial / "_remote.json", intent)  # Before any launch RPC.
        try:
            self.create(body)
        except SchedulerUnavailable as exc:
            # Possibly accepted. Retain safe transport metadata for diagnosis;
            # only read-only reconciliation follows, never another create.
            _worker.atomic(
                trial / "_submission_error.json",
                {"observed_at": time.time(), "queries": exc.errors},
            )

    def _intent(self, trial):
        intent = read_receipt(trial / "_remote.json")
        if intent and intent["scope"] != self.scope():
            raise ValueError(
                "Managed platform identity changed; refusing cross-platform operation"
            )
        return intent

    def owned(self, trial, intent):
        job = self.find(intent)
        if job is None:
            return None
        native_id, token = self.identity(job)
        if (
            not native_id
            or token != intent["token"]
            or (intent["native_id"] and intent["native_id"] != native_id)
        ):
            raise ValueError("Managed job ownership/immutable identity mismatch")
        if not intent["native_id"]:
            intent["native_id"] = native_id
            _worker.atomic(trial / "_remote.json", intent)
        return job

    def status(self, trial):
        intent = self._intent(trial)
        if not intent:
            return "UNKNOWN"
        job = self.owned(trial, intent)
        if job is None:
            if not intent["native_id"] and time.time() < min(
                intent["deadline"],
                intent["submitted_at"] + self.settings.submission_reconcile_seconds,
            ):
                return "PENDING"
            try:
                _submission.gate(trial, close=True)
            except (OSError, ValueError):
                pass
            return "UNKNOWN"  # Absence never authorizes a second launch.
        state = self.native_state(job)
        _worker.atomic(
            trial / "_native_status.json",
            {
                "observed_at": time.time(),
                "native_id": intent["native_id"],
                "state": state,
            },
        )
        if state in {"PENDING", "RUNNING", "UNKNOWN"}:
            return state
        receipt = read_receipt(trial / "_receipt.json") or {}
        if (trial / "_cancel.json").exists():
            return "CANCELED"  # Only after native termination confirmation.
        if state == "COMPLETE":
            return (
                receipt.get("state")
                if receipt.get("state") in {"COMPLETE", "ERROR", "CANCELED"}
                else "UNKNOWN"
            )
        return state

    def cancel(self, trial):
        _worker.atomic(trial / "_cancel.json", {"requested_at": time.time()})
        if (trial / "_submission_gate.json").exists():
            try:
                _submission.gate(trial, close=True)
            except (OSError, ValueError):
                pass  # Cancellation is still unconfirmed until native observation.
        intent = self._intent(trial)
        if not intent:
            return
        job = self.owned(trial, intent)
        if job and self.native_state(job) in {"PENDING", "RUNNING", "UNKNOWN"}:
            self.stop(job)

    def logs(self, trial):
        for name in (
            "trial.log",
            "worker.log",
            "_submission_error.json",
            "_native_status.json",
        ):
            path = trial / name
            if path.exists():
                return str(path)
        return None
