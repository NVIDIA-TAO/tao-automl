# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Explicit cold SLURM recovery migration, not an unrestricted resume override."""

import copy
import fcntl
import hashlib
import json
import os
import time
import zipfile
from contextlib import ExitStack
from pathlib import Path

from automl_core._submission import gate
from automl_core._worker import atomic
from automl_core.contracts import fingerprint
from automl_core.recovery import attempt_root, read_receipt
from automl_core.slurm import CANCELED, ERROR, timestamp

# Search/proposal/promotion/workload code must be byte-identical to the old wheel.
RECOVERY_FILES = {
    "_worker.py",
    "_slurm_batch.py",
    "contracts.py",
    "recovery.py",
    "runner.py",
    "slurm.py",
    "controller.py",
    "cli.py",
    "migration.py",
}


def source_build(wheel):
    with zipfile.ZipFile(wheel) as archive:
        sources = {
            name.removeprefix("automl_core/"): archive.read(name)
            for name in archive.namelist()
            if name.startswith("automl_core/") and name.endswith(".py")
        }
    digest = hashlib.sha256()
    for name, content in sorted(sources.items()):
        digest.update(name.encode() + b"\0")
        digest.update(hashlib.sha256(content).digest())
    installed = Path(__file__).parent
    current = {
        p.relative_to(installed).as_posix(): p.read_bytes()
        for p in installed.rglob("*.py")
    }
    changed = sorted(
        k for k in sources.keys() | current.keys() if sources.get(k) != current.get(k)
    )
    if not sources or set(changed) - RECOVERY_FILES:
        raise ValueError("Source wheel has incompatible search/proposal/workload code")
    return digest.hexdigest(), changed


def continuation(root):
    marker = read_receipt(root / "migration.json")
    if not marker:
        return None
    if marker.get("state") != "COMMITTED":
        raise ValueError(
            "Cold migration incomplete; rerun the approved migrate command"
        )
    return marker["review"]


def validate_controller(root, policy, native):
    review = continuation(root)
    if not review or not review.get("controller"):
        return
    budget = review["controller"]
    if (
        policy["max_allocations"] > budget["remaining_allocations"]
        or policy["allocation_seconds"] > budget["policy"]["allocation_seconds"]
        or policy["grace_seconds"] < budget["policy"]["grace_seconds"]
    ):
        raise ValueError("Continuation exceeds remaining CPU controller budget")
    for key, value in budget["native"].items():
        if key not in {"JobId", "UserId"} and native.get(key) != value:
            raise ValueError("Continuation CPU resource/partition identity changed")


def terminal_cpu(executor, controller, created):
    native = controller["identity"]["native"]
    job = native["JobId"]
    if not native["UserId"].endswith(f"({os.getuid()})"):
        raise ValueError("Old CPU controller owner mismatch")
    queue = executor._call(
        ["squeue", "-h", "--user", str(os.getuid()), "-o", "%i"]
    ).stdout
    if job in queue.split():
        raise ValueError("Old CPU controller is still scheduler-owned")
    reply = executor._call(
        [
            "sacct",
            "-X",
            "-n",
            "-P",
            "--user",
            str(os.getuid()),
            "--jobs",
            job,
            "--starttime",
            timestamp(created - 60),
            "--format=JobIDRaw,UID,State%40",
        ]
    ).stdout
    rows = [line.split("|")[:3] for line in reply.splitlines() if line.strip()]
    if (
        len(rows) != 1
        or len(rows[0]) != 3
        or rows[0][:2] != [job, str(os.getuid())]
        or not rows[0][2].split()
        or rows[0][2].split()[0].rstrip("+") not in ERROR | CANCELED | {"COMPLETED"}
    ):
        raise ValueError("Old CPU termination/ownership is unresolved")
    if controller["state"] not in {"ERROR", "PAUSED", "CANCELED", "COMPLETE"}:
        raise ValueError("Unacknowledged CPU handoff/restart requires reconciliation")
    generations = [a["generation"] for a in controller["allocations"]]
    if generations != list(range(controller["generation"] + 1)):
        raise ValueError("CPU allocation history is incomplete")
    remaining = controller["identity"]["policy"]["max_allocations"] - len(generations)
    if remaining <= 0:
        raise ValueError("Original CPU allocation budget exhausted")
    return {
        "native": native,
        "policy": controller["identity"]["policy"],
        "consumed_allocations": len(generations),
        "remaining_allocations": remaining,
        "terminal_state": rows[0][2],
    }


def migrate(experiment, wheel, *, approve=False, acknowledge_cancel=False):
    """Review by default; --approve commits a backed-up, restartable transaction."""
    if not hasattr(experiment, "search") or experiment.execution.kind != "slurm":
        raise ValueError("Cold migration currently supports SLURM SFT searches only")
    root = experiment.root
    old_build, changed = source_build(wheel)
    with ExitStack() as stack:
        for name in (".supervisor.lock", ".controller.lock"):
            handle = stack.enter_context((root / name).open("a"))
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        marker = read_receipt(root / "migration.json")
        if marker and marker.get("state") == "COMMITTED":
            if (
                marker["review"]["to_identity"] == fingerprint(experiment.identity)
                and marker["review"]["from_build"] == old_build
            ):
                return marker["review"]  # Idempotent, never reset later progress.
            raise ValueError(
                "A different migration already exists; chaining is not supported"
            )
        path = root / "migrations" / "recovery-v1.json"
        transaction = read_receipt(path) if marker else None
        before = (
            transaction["before"]
            if transaction
            else {
                "ledger": read_receipt(root / "experiment.json"),
                "controller": read_receipt(root / "controller.json"),
                "cancel": read_receipt(root / "cancel.json")
                if (root / "cancel.json").exists()
                else None,
            }
        )
        ledger = before["ledger"]
        old = ledger["identity"]
        if (
            ledger["fingerprint"] != fingerprint(old)
            or old["engine_build"] != old_build
        ):
            raise ValueError("Source wheel does not match the sealed experiment")
        if ledger["state"] not in {"ERROR", "PAUSED", "CANCELED", "RUNNING"} or (
            ledger["state"] == "RUNNING" and not before["controller"]
        ):
            raise ValueError("Only stopped, incomplete searches may migrate")
        expected = copy.deepcopy(old)
        expected["engine_build"] = experiment.identity["engine_build"]
        exclusions = expected["execution"]["slurm"].get("exclude_nodes", [])
        selected = experiment.execution.slurm.exclude_nodes
        if not set(exclusions) <= set(selected):
            raise ValueError("Migration cannot remove existing node exclusions")
        expected["execution"]["slurm"]["exclude_nodes"] = selected
        if expected != experiment.identity:
            raise ValueError(
                "Migration may change only the recovery build and add node exclusions"
            )
        if before["cancel"] is not None and not acknowledge_cancel:
            raise ValueError(
                "Explicit --acknowledge-cancel required to archive the old stop request"
            )
        deadline = experiment._deadline(ledger)
        if time.time() + experiment.execution.slurm.time_limit_seconds > deadline:
            raise ValueError("Insufficient original experiment time for continuation")
        plan = experiment.plan()
        environment = experiment.executor.preflight(Path(plan["example_trial"]["cwd"]))
        previous_environment = dict(ledger["environment"])
        previous_environment["batch_python"] = environment["batch_python"]
        if previous_environment != environment:
            raise ValueError("Cluster/image/supporting-file identity changed")
        controller = before["controller"]
        cpu = (
            terminal_cpu(experiment.executor, controller, ledger["created_at"])
            if controller
            else None
        )
        if controller and controller["identity"]["experiment"] != ledger["fingerprint"]:
            raise ValueError("Old CPU controller experiment identity mismatch")
        # Paid requests must be resolved. Receipts and counters stay in place.
        advice = {}
        for receipt in sorted((root / "advice").glob("*.json")):
            value = json.loads(receipt.read_text())
            if "state" in value and value["state"] != "COMPLETE":
                raise ValueError("Unresolved LLM receipt; never replay a paid request")
            advice[receipt.name] = hashlib.sha256(receipt.read_bytes()).hexdigest()
        after = copy.deepcopy(ledger)
        observations = []
        for trial in after["trials"]:
            for attempt in [*trial.get("attempts", []), trial]:
                folder = attempt_root(root, attempt)
                intent = read_receipt(folder / "_slurm.json")
                if not intent:
                    raise ValueError("Missing trial submission intent")
                if intent.get("owner_uid") != os.getuid():
                    raise ValueError("Trial submission owner mismatch")
                if intent.get("submission_attempted"):
                    if not intent.get("job_id"):
                        # Preserve already-consumed uncertain submissions. A
                        # closed unclaimed fence rejects any late workload; it
                        # is not permission to spend another attempt here.
                        if (
                            not intent.get("submission_guard")
                            or gate(folder)["state"] != "closed"
                        ):
                            raise ValueError(
                                "Unresolved submission identity blocks migration"
                            )
                        rows = experiment.executor._rows(intent)
                        if any(
                            len(row) != 4
                            or row[1] != intent["name"]
                            or not row[2].split()
                            or row[2].split()[0].rstrip("+")
                            not in ERROR | CANCELED | {"COMPLETED"}
                            for row in rows
                        ):
                            raise ValueError(
                                "Fenced submission still has an active or conflicting allocation"
                            )
                        if attempt["state"] != "ERROR":
                            raise ValueError(
                                "Fenced attempt must already be recorded as failed"
                            )
                        observations.append(
                            {
                                "attempt": folder.name,
                                "job_id": None,
                                "state": "FENCED",
                                "native_ids": [r[0] for r in rows],
                            }
                        )
                        continue
                    native = experiment.executor._native(folder)
                    if native is None or native[1] not in ERROR | CANCELED | {
                        "COMPLETED"
                    }:
                        raise ValueError(
                            "An old allocation is still active or unresolved"
                        )
                    state = "CANCELED" if native[1] in CANCELED else "ERROR"
                    if native[1] == "COMPLETED" and native[2] == "0:0":
                        receipts = [
                            read_receipt(folder / name)
                            for name in ("_receipt.json", "_batch_receipt.json")
                        ]
                        if not all(
                            r.get("state") == "COMPLETE"
                            and r.get("slurm_job_id") == native[0]
                            for r in receipts
                        ):
                            raise ValueError("Native completion evidence unresolved")
                        state = "COMPLETE"
                    if attempt["state"] == "COMPLETE" and state != "COMPLETE":
                        raise ValueError(
                            "Completed result disagrees with native evidence"
                        )
                else:
                    if intent.get("job_id"):
                        raise ValueError("Inconsistent unsubmitted trial identity")
                    state = "ERROR"
                observations.append(
                    {
                        "attempt": folder.name,
                        "job_id": intent.get("job_id"),
                        "state": state,
                    }
                )
                if attempt is trial and trial["state"] in {"PENDING", "RUNNING"}:
                    if state == "COMPLETE":
                        raise ValueError(
                            "Collect completed-but-unrecorded trial with original build first"
                        )
                    trial.update(
                        state="ERROR",
                        observed_state=state,
                        result=None,
                        reason="Native failure reconciled during approved cold migration",
                        failure={
                            "reason": "reconciled_previous_attempt",
                            "retryable": False,
                        },
                    )
        review = {
            "from_build": old_build,
            "to_build": experiment.identity["engine_build"],
            "from_identity": ledger["fingerprint"],
            "to_identity": fingerprint(experiment.identity),
            "changed_runtime_files": changed,
            "original_deadline": deadline,
            "controller": cpu,
            "trials": len(ledger["trials"]),
            "completed_trials": sum(t["state"] == "COMPLETE" for t in ledger["trials"]),
            "consumed_retries": sum(
                len(t.get("attempts", [])) for t in ledger["trials"]
            ),
            "advice_receipts": advice,
            "native_observations": observations,
            "exclude_nodes": selected,
            "acknowledged_cancel": before["cancel"] is not None,
            "source_wheel_sha256": hashlib.sha256(Path(wheel).read_bytes()).hexdigest(),
        }
        after.update(
            identity=experiment.identity,
            fingerprint=review["to_identity"],
            environment=environment,
            state="SUSPENDED",
            controller_deadline=deadline,
            stop_reason="Approved cold recovery migration; awaiting bounded continuation",
        )
        if not approve:
            return review
        if transaction:
            if transaction["after"] != after or transaction["review"] != review:
                raise ValueError(
                    "Interrupted migration evidence changed; investigate before committing"
                )
        else:
            path.parent.mkdir(exist_ok=True)
            transaction = {
                "before": before,
                "after": after,
                "review": review,
                "prepared_at": time.time(),
            }
            atomic(path, transaction)
            atomic(root / "migration.json", {"state": "PREPARED", "review": review})
        current = read_receipt(root / "experiment.json")
        if current not in (before["ledger"], after):
            raise ValueError("Experiment changed during migration")
        current_controller = read_receipt(root / "controller.json")
        if current_controller and current_controller != before["controller"]:
            raise ValueError("Controller changed during migration")
        atomic(root / "experiment.json", after)
        # Recoverable originals are retained verbatim in the transaction archive.
        (root / "controller.json").unlink(missing_ok=True)
        (root / "cancel.json").unlink(missing_ok=True)
        atomic(
            root / "migration.json",
            {"state": "COMMITTED", "review": review, "committed_at": time.time()},
        )
        return review
