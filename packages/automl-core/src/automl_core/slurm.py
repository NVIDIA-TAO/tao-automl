# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Single-node SLURM executor using native CLIs and shared storage, not SSH/SDKs.

Run the controller on the cluster. Input images are already-cached squashfs;
the consumer supplies its reviewed container template and optional bootstrap.
"""

import datetime
import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
import uuid
from pathlib import Path

from automl_core import _slurm_batch, _submission, _worker
from automl_core._worker import atomic
from automl_core.recovery import health, read_receipt

ACTIVE = {
    "PENDING",
    "CONFIGURING",
    "RUNNING",
    "COMPLETING",
    "SUSPENDED",
    "STOPPED",
    "RESIZING",
    "SIGNALING",
    "STAGE_OUT",
}
ERROR = {"FAILED", "TIMEOUT", "OUT_OF_MEMORY", "NODE_FAIL", "BOOT_FAIL", "DEADLINE"}
CANCELED = {"CANCELLED", "PREEMPTED", "REVOKED"}


class SchedulerUnavailable(RuntimeError):
    """Read-only scheduler observation failed; not evidence of job failure."""

    def __init__(self, errors):
        self.errors = errors
        super().__init__(
            "Platform observation temporarily unavailable; no replacement submission"
        )


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def timestamp(epoch):
    # SLURM interprets a timestamp in the controller's local timezone.
    return (
        datetime.datetime.fromtimestamp(epoch, datetime.timezone.utc)
        .astimezone()
        .strftime("%Y-%m-%dT%H:%M:%S")
    )


def duration(seconds):
    if seconds in {"UNLIMITED", "INFINITE"}:
        return float("inf")
    days, _, clock = seconds.rpartition("-")
    parts = [int(part) for part in clock.split(":")]
    if len(parts) != 3:
        raise ValueError("Unrecognized scheduler time format")
    return (int(days or 0) * 24 + parts[0]) * 3600 + parts[1] * 60 + parts[2]


class SlurmExecutor:
    def __init__(self, spec):
        self.spec = spec
        self.settings = spec.slurm

    def _env(self):
        # Do not inherit SBATCH_* overrides, controller PYTHONPATH, or credentials
        # not explicitly requested by name. Values never enter persisted payloads.
        names = {
            "PATH",
            "HOME",
            "USER",
            "LOGNAME",
            "LANG",
            "LC_ALL",
            "TZ",
            "SLURM_CONF",
            *self.spec.environment,
        }
        return {
            name: os.environ[name]
            for name in names
            if name in os.environ and not name.startswith("SBATCH_")
        }

    def _call(self, args, check=True):
        return subprocess.run(
            args,
            capture_output=True,
            text=True,
            timeout=30,
            check=check,
            env=self._env(),
        )

    def configure_checks(self, root, deadline):
        """Bound repeatable checks to the caller's original experiment clock."""
        self.check_root, self.check_deadline = Path(root), deadline

    def _checked_call(self, args, operation, *, root=None, deadline=None):
        """Retry timed-out *read-only* checks, never a submit/cancel/record hook.

        Nonzero exits remain actionable configuration errors. Persist only the
        operation name and numeric diagnostics, not argv, stderr or environment.
        A plain preflight outside a running experiment remains a one-shot check.
        """
        from automl_core.controller import checkpoint

        root = root or getattr(self, "check_root", None)
        until = min(
            deadline if deadline is not None else float("inf"),
            getattr(self, "check_deadline", float("inf")),
        )
        started, failures = time.time(), 0

        def save(state, **details):
            if root is not None:
                atomic(
                    root / "_scheduler_check.json",
                    dict(
                        operation=operation,
                        state=state,
                        started_at=started,
                        updated_at=time.time(),
                        failures=failures,
                        deadline=until if until != float("inf") else None,
                        **details,
                    ),
                )

        while True:
            checkpoint()  # No uncommitted submission or paid request here.
            if (
                hasattr(self, "check_root")
                and (self.check_root / "cancel.json").exists()
            ):
                save("STOPPED", reason="cancellation_requested")
                raise InterruptedError("Read-only scheduler check canceled")
            if time.time() >= until:
                save("STOPPED", reason="original_time_budget_exhausted")
                raise TimeoutError(
                    "Original time budget exhausted during scheduler check"
                )
            try:
                result = self._call(args)
            except subprocess.TimeoutExpired as exc:
                failures += 1
                save(
                    "WAITING",
                    error_type=type(exc).__name__,
                    timeout_seconds=exc.timeout,
                )
                if not hasattr(self, "check_deadline"):
                    raise
                delay = min(300, self.settings.poll_seconds * 2 ** min(failures - 1, 8))
                wake = min(until, time.time() + delay)
                while time.time() < wake:
                    checkpoint()
                    if (self.check_root / "cancel.json").exists():
                        break
                    time.sleep(max(0, min(1, wake - time.time())))
                continue
            except (OSError, subprocess.CalledProcessError) as exc:
                save(
                    "ERROR",
                    error_type=type(exc).__name__,
                    returncode=getattr(exc, "returncode", None),
                    errno=getattr(exc, "errno", None),
                )
                raise
            save("COMPLETE")
            return result

    def preflight(self, cwd):
        for name in ("sbatch", "squeue", "sacct", "scancel", "scontrol", "srun"):
            if not shutil.which(name):
                raise ValueError(
                    f"SLURM controller requires native {name} on the cluster"
                )
        if not self.spec.image and (
            not cwd.is_dir() or not os.access(self.spec.python, os.X_OK)
        ):
            raise ValueError(
                "Native SLURM cwd and executable Python must exist on shared storage"
            )
        missing = [name for name in self.spec.environment if not os.environ.get(name)]
        if missing:
            raise ValueError(
                f"Missing environment variable names: {', '.join(missing)}"
            )
        for path, expected in self.settings.supporting_files.items():
            if digest(path) != expected:
                raise ValueError(f"Supporting file SHA256 mismatch: {path}")
        if self.spec.image:
            if digest(self.spec.image) != self.settings.image_sha256:
                raise ValueError("Cached SLURM image SHA256 mismatch")
            with open(self.spec.image, "rb") as stream:
                if stream.read(4) != b"hsqs":
                    raise ValueError("SLURM image is not a squashfs file")
            for path in self.spec.mounts:
                if not path.is_absolute() or not path.exists():
                    raise ValueError("SLURM mounts must be existing absolute paths")
        config = self._checked_call(
            ["scontrol", "show", "config"], "cluster_identity"
        ).stdout
        cluster = re.search(r"^\s*ClusterName\s*=\s*(\S+)", config, re.MULTILINE)
        if not cluster:
            raise ValueError("Unable to identify the SLURM cluster")
        if self.settings.exclude_nodes:
            inventory = self._checked_call(
                ["scontrol", "show", "nodes", "-o"], "node_inventory"
            ).stdout
            nodes = set(re.findall(r"(?:^|\s)NodeName=([^\s]+)", inventory))
            missing = sorted(set(self.settings.exclude_nodes) - nodes)
            if missing:
                raise ValueError(
                    "Excluded SLURM nodes are absent from live inventory: "
                    + ", ".join(missing)
                    + "; reconcile the private launch configuration before running"
                )
        for partition in self.settings.partition.split(","):
            output = self._checked_call(
                ["scontrol", "show", "partition", partition, "-o"], "partition_limits"
            ).stdout
            fields = dict(re.findall(r"(\w+)=([^\s]+)", output))
            if fields.get("State") != "UP":
                raise ValueError(f"Partition is not UP: {partition}")
            if duration(fields["MaxTime"]) < self.settings.time_limit_seconds:
                raise ValueError(
                    f"Requested allocation exceeds partition MaxTime: {partition}"
                )
        return {
            "cluster": cluster.group(1),
            "image_sha256": self.settings.image_sha256,
            "supporting_files": self.settings.supporting_files,
            "batch_python": str(Path(sys.executable).resolve()),
        }

    def _hook(self, command, values):
        result = self._call([arg.format_map(values) for arg in command])
        return result.stdout.strip()

    def _intent(self, trial):
        path = trial / "_slurm.json"
        return json.loads(path.read_text()) if path.exists() else None

    def _record(self, trial, state, job_id, message):
        if not self.settings.record_mark:
            return
        intent = self._intent(trial)
        if intent.get("record_state") == state:
            return
        self._hook(
            self.settings.record_mark,
            {
                "id": intent["name"],
                "state": state,
                "backend_ref": job_id or "",
                "message": message,
                "trial": str(trial),
                "image": self.spec.image or "native-python",
            },
        )
        intent["record_state"] = state
        atomic(trial / "_slurm.json", intent)

    def _render(self, trial, plan, intent):
        settings = self.settings
        for path in (
            trial,
            plan.cwd,
            *(self.spec.mounts or []),
            *plan.read_only_mounts,
            *settings.supporting_files,
        ):
            if any(char.isspace() or char in ",:\"'" for char in str(path)):
                raise ValueError(
                    "SLURM shared paths cannot contain whitespace, quotes, commas or colons"
                )
        seconds = settings.time_limit_seconds
        wall = f"{seconds // 3600:02d}:{seconds % 3600 // 60:02d}:00"
        extra = [
            f"#SBATCH --partition={settings.partition}",
            f"#SBATCH --account={settings.account}",
            "#SBATCH --no-requeue",
            "#SBATCH --chdir=/tmp",
            f"#SBATCH --output={trial}/slurm-%j.out",
            f"#SBATCH --error={trial}/slurm-%j.err",
            "#SBATCH --signal=B:TERM@60",
        ]
        if settings.qos:
            extra.append(f"#SBATCH --qos={settings.qos}")
        if settings.exclude_nodes:
            extra.append("#SBATCH --exclude=" + ",".join(settings.exclude_nodes))
        if settings.exclusive:
            extra.append("#SBATCH --exclusive")
        if settings.memory_mb:
            extra.append(f"#SBATCH --mem={settings.memory_mb}M")
        command = [
            self.spec.python,
            str(trial / "_worker.py"),
            str(trial / "_execution.json"),
        ]
        if settings.bootstrap:
            command = ["/bin/bash", str(settings.bootstrap), *command]
        cpu_env = f"""allocated_cpus=$(scontrol show job "$SLURM_JOB_ID" -o | sed -n 's/.* NumCPUs=\\([0-9]*\\).*/\\1/p')
test -n "$allocated_cpus"
test "$allocated_cpus" -ge {settings.cpus_per_task}
export SRUN_CPUS_PER_TASK="$allocated_cpus"
printf '{{"requested":{settings.cpus_per_task},"allocated":%s,"step":%s}}\\n' "$allocated_cpus" "$allocated_cpus" > {shlex.quote(str(trial / "cpu-allocation.json"))}
export AUTOML_TRIAL_DIR={shlex.quote(str(trial))}
export AUTOML_GPUS={settings.gpus}
export TMPDIR=/tmp
export ENROOT_TEMP_PATH=/tmp/enroot-automl-${{SLURM_JOB_ID}}
export SLURM_ENROOT_TEMP_PATH="$ENROOT_TEMP_PATH"
mkdir -p "$ENROOT_TEMP_PATH"
ulimit -n 65536"""
        if self.spec.image:
            mounts = {str(p): "ro" for p in [*self.spec.mounts, *plan.read_only_mounts]}
            mounts.update({path: "ro" for path in settings.supporting_files})
            mounts[str(trial)] = "rw"
            mount_text = ",".join(
                f"{path}:{path}:{mode}" for path, mode in mounts.items()
            )
            forwarding = sorted(
                set(self.spec.environment) | {"AUTOML_TRIAL_DIR", "AUTOML_GPUS"}
            )
            container = [
                "--no-container-mount-home",
                "--no-container-remap-root",
                f"--container-workdir={plan.cwd}",
                "--container-env=" + ",".join(forwarding),
            ]
            fields = {
                "JOB_NAME": intent["name"],
                "NUM_GPUS": str(settings.gpus),
                "CPUS_PER_TASK": str(settings.cpus_per_task),
                "TIME": wall,
                "LOG_DIR": str(trial),
                "IMAGE": shlex.quote(self.spec.image),
                "CONTAINER_MOUNTS": shlex.quote(mount_text),
                "COMMAND": '--cpus-per-task="$allocated_cpus" '
                + shlex.join(container + command),
                "SBATCH_EXTRA": "\n".join(extra),
                "ENV_FILE": "",
                "EXTRA_ENV": cpu_env,
            }
            step = settings.template.read_text()
            for key, value in fields.items():
                step = step.replace(f"@@{key}@@", value)
            if re.search(r"@@[A-Z_]+@@", step):
                raise ValueError("Unresolved SLURM template placeholder")
        else:
            step = (
                "#!/bin/bash\nset -euo pipefail\n"
                + cpu_env
                + '\nsrun --cpus-per-task="$allocated_cpus" '
                + shlex.join(command)
                + "\n"
            )
        # Only typed resource settings reach sbatch. The consumer template is
        # executed as a shell step, so its historical #SBATCH defaults cannot
        # override partition, budget, identity, or no-requeue policy.
        directives = [
            f"#SBATCH --job-name={intent['name']}",
            "#SBATCH --nodes=1",
            "#SBATCH --ntasks=1",
            f"#SBATCH --cpus-per-task={settings.cpus_per_task}",
            f"#SBATCH --time={wall}",
            *extra,
        ]
        if settings.gpus:
            directives.append(f"#SBATCH --gres=gpu:{settings.gpus}")
        # The existing consumer template remains the step envelope. An outer
        # stdlib supervisor bounds image startup, workload time, and orphan life.
        batch = "#!/bin/bash\n" + "\n".join(directives) + "\nset -euo pipefail\nexec "
        batch += (
            shlex.join([sys.executable, str(trial / "_slurm_batch.py"), str(trial)])
            + "\n"
        )
        return step, batch

    def submit(self, trial, plan, deadline, retry_of=None):
        for mount in plan.read_only_mounts:
            if (
                not mount.is_absolute()
                or not mount.exists()
                or mount.resolve() == trial.resolve()
            ):
                raise ValueError(
                    "Stage input mounts must be existing absolute paths distinct from the trial"
                )
        if self._intent(trial) or (trial / "_execution.json").exists():
            raise ValueError(
                "SLURM launch intent exists; reconcile instead of resubmitting"
            )
        settings = self.settings
        if deadline - time.time() < settings.time_limit_seconds:
            raise ValueError("Insufficient experiment time for a full SLURM allocation")
        name = "automl-" + uuid.uuid4().hex
        previous = self._intent(retry_of) if retry_of else None
        if settings.record_open:
            name = self._hook(
                settings.record_retry_open if previous else settings.record_open,
                {
                    "trial": str(trial),
                    "image": self.spec.image or "native-python",
                    "retry_of": previous["name"] if previous else "",
                },
            )
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", name):
            raise ValueError("Invalid native job name from record writer")
        intent = {
            "name": name,
            "submitted_at": time.time(),
            "deadline": deadline,
            "time_limit_seconds": settings.time_limit_seconds,
            "startup_timeout_seconds": settings.startup_timeout_seconds,
            "cleanup_seconds": settings.cleanup_seconds,
            "job_id": None,
            "submission_attempted": False,
            "submission_guard": True,
            "owner_uid": os.getuid(),
            "supporting_files": settings.supporting_files,
            "exclude_nodes": settings.exclude_nodes,
            "retry_of": previous["name"] if previous else None,
            "retry_of_job_id": previous.get("job_id") if previous else None,
        }
        atomic(trial / "_slurm.json", intent)
        atomic(trial / "_submission_gate.json", {"state": "open"})
        for name, content in plan.files.items():
            target = trial / name
            if name.startswith("_") or not target.resolve().is_relative_to(trial):
                raise ValueError("Unsafe or reserved staged path")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content)
        for module in (_worker, _slurm_batch, _submission):
            (trial / Path(module.__file__).name).write_text(
                Path(module.__file__).read_text()
            )
        atomic(
            trial / "_execution.json",
            {
                "commands": plan.commands,
                "cwd": str(plan.cwd),
                "deadline": deadline,
                "runtime_timeout_seconds": self.runtime_timeout_seconds,
                "environment": plan.environment,
                "progress_files": plan.progress_files,
            },
        )
        step, batch = self._render(trial, plan, intent)
        for name, content in (("_slurm_step.sh", step), ("_slurm.sbatch", batch)):
            path = trial / name
            path.write_text(content)
        # Only a completely staged, hashed intent is restartable before sbatch.
        # Never overwrite execution files after the submit boundary is crossed.
        intent["staged_files"] = {
            str(path.relative_to(trial)): digest(path)
            for path in (
                *(trial / name for name in plan.files),
                *(
                    trial / Path(module.__file__).name
                    for module in (_worker, _slurm_batch, _submission)
                ),
                trial / "_execution.json",
                trial / "_slurm_step.sh",
                trial / "_slurm.sbatch",
            )
        }
        atomic(trial / "_slurm.json", intent)
        self.continue_submission(trial)

    def continue_submission(self, trial):
        """Continue the SAME reserved, provably unsubmitted staged attempt.

        Actual sbatch uncertainty still goes exclusively through reconciliation.
        Missing/old/incomplete staging, stop markers and closed/claimed gates
        cannot be turned into a fresh submission by restarting the controller.
        """
        intent = self._intent(trial)
        if not intent or intent.get("submission_attempted") or intent.get("job_id"):
            return
        files = intent.get("staged_files")
        if (
            not files
            or intent.get("owner_uid") != os.getuid()
            or not intent.get("submission_guard")
            or (trial / "_cancel.json").exists()
            or (trial / "_receipt.json").exists()
            or (trial / "_batch_receipt.json").exists()
            or _submission.gate(trial)["state"] != "open"
        ):
            raise ValueError(
                "Unsubmitted intent is not safe to continue; reconcile staging"
            )
        for name, expected in files.items():
            path = trial / name
            if (
                not path.resolve().is_relative_to(trial.resolve())
                or digest(path) != expected
            ):
                raise ValueError("Staged submission identity changed")
        deadline = intent["deadline"]
        latest = deadline - self.settings.time_limit_seconds
        for name in ("_slurm_step.sh", "_slurm.sbatch"):
            self._checked_call(
                ["/bin/bash", "-n", str(trial / name)],
                "batch_syntax",
                root=trial,
                deadline=latest,
            )
        argv = [
            "sbatch",
            "--parsable",
            "--export=ALL",
            "--deadline=" + timestamp(deadline),
            str(trial / "_slurm.sbatch"),
        ]
        self._checked_call(
            ["sbatch", "--test-only", *argv[2:]],
            "submission_preflight",
            root=trial,
            deadline=latest,
        )
        if (
            time.time() >= latest
            or (getattr(self, "check_root", trial.parent) / "cancel.json").exists()
        ):
            raise TimeoutError(
                "No approved time remains for submission, or cancellation requested"
            )
        intent["submission_attempted"] = True
        atomic(trial / "_slurm.json", intent)  # Durable reservation precedes the RPC.
        try:
            response = self._call(argv)
            job_id = response.stdout.strip().split(";")[0]
            if not re.fullmatch(r"[0-9]+", job_id):
                raise ValueError("Ambiguous submission response")
        except (OSError, subprocess.SubprocessError, ValueError) as exc:
            # The RPC may have succeeded. Preserve the same candidate and let
            # the controller reconcile; never retry sbatch here or log stderr.
            intent.update(
                submission_error=type(exc).__name__,
                reconciliation_started_at=time.time(),
            )
            atomic(trial / "_slurm.json", intent)
            return
        intent["job_id"] = job_id
        atomic(trial / "_slurm.json", intent)
        self._record(trial, "PENDING", job_id, "Submitted; consult native scheduler")

    def reconcile_submission(self, trial):
        """One bounded observation step, called by the controller, not status CLI."""
        intent = self._intent(trial)
        if not intent or intent.get("job_id") or not intent["submission_attempted"]:
            return None
        started = intent.setdefault("reconciliation_started_at", time.time())
        atomic(trial / "_slurm.json", intent)
        until = min(
            intent["deadline"], started + self.settings.submission_reconcile_seconds
        )

        def result(decision, reason, **details):
            value = {
                "decision": decision,
                "reason": reason,
                "deadline": until,
                "observed_at": time.time(),
                **details,
            }
            atomic(trial / "_submission_reconciliation.json", value)
            return value

        if not intent.get("submission_guard") or intent.get("owner_uid") != os.getuid():
            return result(
                "blocked", "unguarded legacy submission requires reconciliation"
            )
        try:
            guard = _submission.gate(trial)
        except (OSError, ValueError):
            return result("blocked", "submission fence unavailable")
        if guard["state"] == "closed":
            return result("fenced", "submission fenced before workload start")

        errors, found = {}, {}
        for command in (
            [
                "squeue",
                "--noheader",
                "--user",
                str(os.getuid()),
                "--name",
                intent["name"],
                "--format=%i|%j|%T",
            ],
            [
                "sacct",
                "-X",
                "-n",
                "-P",
                "--user",
                str(os.getuid()),
                "--name",
                intent["name"],
                "--starttime",
                timestamp(intent["submitted_at"] - 60),
                "--format=JobIDRaw,JobName%128,State%40,ExitCode",
            ],
        ):
            reply = self._query(command, errors)
            if reply is None:
                continue
            for line in reply.stdout.splitlines():
                if not line.strip():
                    continue
                row = line.strip().split("|")
                if (
                    len(row) < 3
                    or row[1] != intent["name"]
                    or not re.fullmatch(r"[0-9]+", row[0])
                ):
                    return result("blocked", "invalid submission identity evidence")
                found[row[0]] = row[2]
        if len(found) > 1:
            return result(
                "blocked",
                "multiple jobs match submission identity",
                job_ids=sorted(found),
            )
        if guard["state"] == "claimed":
            claimed = guard.get("job_id")
            if (
                not isinstance(claimed, str)
                or not re.fullmatch(r"[0-9]+", claimed)
                or (found and claimed not in found)
            ):
                return result(
                    "blocked", "batch claim conflicts with scheduler identity"
                )
            if not found:
                # Claim proves work may exist, not that scheduler observation is
                # healthy. Wait for visibility; never fence an accepted batch.
                return result(
                    "blocked" if time.time() >= until else "wait",
                    "batch claimed; waiting for scheduler visibility",
                    job_id=claimed,
                    queries=errors,
                )
        if found:
            intent["job_id"] = next(iter(found))
            atomic(trial / "_slurm.json", intent)
            result("adopted", "existing submission adopted", job_id=intent["job_id"])
            return None
        if errors:
            # Unavailable observations never establish absence or permit fencing.
            intent.pop("absence_since", None)
            intent.pop("absence_observations", None)
            atomic(trial / "_slurm.json", intent)
            return result(
                "blocked" if time.time() >= until else "wait",
                "submission scheduler observations unavailable",
                queries=errors,
            )
        intent.setdefault("absence_since", time.time())
        intent["absence_observations"] = intent.get("absence_observations", 0) + 1
        atomic(trial / "_slurm.json", intent)
        if time.time() < until:
            return result("wait", "waiting for submission visibility")
        if intent["absence_observations"] < 2:
            return result("blocked", "insufficient confirmed-absence observations")
        if (trial / "_receipt.json").exists() or (
            trial / "_batch_receipt.json"
        ).exists():
            return result(
                "blocked", "worker evidence exists without scheduler identity"
            )
        try:
            guard = _submission.gate(trial, close=True)
        except (OSError, ValueError):
            return result("blocked", "submission fence could not be closed")
        if guard["state"] == "claimed":
            # Batch won the race. Re-observe/adopt it; never replace it.
            return result("wait", "batch started during reconciliation")
        return result("fenced", "submission fenced before workload start")

    def _query(self, args, errors):
        # Discard partial stdout and raw stderr on errors. A timed-out status
        # query is safe to repeat; sbatch/scancel retain their separate contract.
        try:
            result = self._call(args, check=False)
        except subprocess.TimeoutExpired:
            errors[args[0]] = {"kind": "timeout"}
            return None
        except OSError as exc:
            errors[args[0]] = {"kind": "unavailable", "errno": exc.errno}
            return None
        if result.returncode:
            errors[args[0]] = {"kind": "exit", "returncode": result.returncode}
            return None
        return result

    def _rows(self, intent):
        errors = {}
        query = (
            ["--jobs", intent["job_id"]]
            if intent.get("job_id")
            else ["--name", intent["name"]]
        )
        queue = self._query(
            [
                "squeue",
                "--noheader",
                "--user",
                str(os.getuid()),
                *query,
                "--format=%i|%j|%T",
            ],
            errors,
        )
        rows = [
            line.strip().split("|")
            for line in (queue.stdout.splitlines() if queue is not None else [])
            if line.strip()
        ]
        if rows and all(row[-1] in ACTIVE for row in rows):
            return [row + [""] for row in rows]
        accounting = self._query(
            [
                "sacct",
                "-X",
                "-n",
                "-P",
                "--user",
                str(os.getuid()),
                *query,
                "--starttime",
                timestamp(intent["submitted_at"] - 60),
                "--format=JobIDRaw,JobName%128,State%40,ExitCode",
            ],
            errors,
        )
        if accounting is None or (queue is None and not accounting.stdout.strip()):
            raise SchedulerUnavailable(errors)
        return [
            line.strip().split("|")[:4]
            for line in accounting.stdout.splitlines()
            if line.strip()
        ]

    def _native(self, trial):
        intent = self._intent(trial)
        if not intent or not intent["submission_attempted"]:
            return None
        rows = [
            row
            for row in self._rows(intent)
            if len(row) == 4
            and row[1] == intent["name"]
            and re.fullmatch(r"[0-9]+", row[0])
        ]
        if len(rows) != 1:
            return None
        job_id, _, state, exit_code = rows[0]
        if intent.get("job_id") and job_id != intent["job_id"]:
            return None
        if not intent.get("job_id"):
            intent["job_id"] = job_id
            atomic(trial / "_slurm.json", intent)
        return job_id, state.split()[0].rstrip("+"), exit_code

    def status(self, trial):
        try:
            native = self._native(trial)
        except SchedulerUnavailable as exc:
            previous = read_receipt(trial / "_scheduler_status.json")
            atomic(
                trial / "_scheduler_status.json",
                {
                    "observed_state": "UNKNOWN",
                    "reason": "scheduler_unavailable",
                    "queries": exc.errors,
                    "observed_at": time.time(),
                    "active_since": previous.get("active_since"),
                },
            )
            raise
        observation = trial / "_scheduler_status.json"
        previous = read_receipt(observation)
        atomic(
            observation,
            {
                "native_state": native[1] if native else None,
                "job_id": native[0] if native else None,
                "reason": (
                    "scheduler_recovered"
                    if native
                    and previous.get("reason")
                    in {"scheduler_unavailable", "scheduler_recovered"}
                    else "scheduler_observed"
                    if native
                    else "unresolved_identity"
                ),
                "observed_at": time.time(),
                "active_since": (
                    previous.get("active_since") or time.time()
                    if native and native[1] in ACTIVE - {"PENDING", "CONFIGURING"}
                    else None
                ),
            },
        )
        if native is None:
            return "UNKNOWN"
        job_id, state, exit_code = native
        if state in ACTIVE:
            result = "PENDING" if state in {"PENDING", "CONFIGURING"} else "RUNNING"
            self._record(trial, result, job_id, f"Native SLURM state: {state}")
            return result
        if state in CANCELED:
            return "CANCELED"
        if state in ERROR:
            # A cooperative worker may stop on our marker before scancel reaches
            # the scheduler. Preserve that verified cancellation, not a false
            # training failure from the batch's nonzero exit.
            batch = trial / "_batch_receipt.json"
            if (trial / "_cancel.json").exists() and batch.exists():
                receipt = json.loads(batch.read_text())
                if (
                    receipt.get("slurm_job_id") == job_id
                    and receipt.get("state") == "CANCELED"
                ):
                    return "CANCELED"
            return "ERROR"
        if state != "COMPLETED" or exit_code != "0:0":
            return "UNKNOWN" if state == "COMPLETED" and not exit_code else "ERROR"
        for name in ("_receipt.json", "_batch_receipt.json"):
            path = trial / name
            if not path.exists():
                return "UNKNOWN"
            receipt = json.loads(path.read_text())
            if receipt.get("slurm_job_id") != job_id:
                return "UNKNOWN"
            if receipt.get("state") != "COMPLETE":
                return "ERROR"
        return "COMPLETE"

    def health(self, trial, policy):
        observation = read_receipt(trial / "_scheduler_status.json")
        # Queue time is not a workload stall. Only inspect confirmed active jobs.
        if observation.get("native_state") in {None, "PENDING", "CONFIGURING"}:
            return None
        job_id = observation["job_id"]
        issue = health(trial, policy, job_id)
        if (
            not issue
            and policy.heartbeat_timeout_seconds
            and not (trial / "_batch_receipt.json").exists()
            and time.time() - (observation.get("active_since") or time.time())
            >= self.settings.startup_timeout_seconds
        ):
            issue = {
                "reason": "batch_start_unacknowledged",
                "retryable": False,
                "observed_at": time.time(),
            }
        if issue:
            issue["job_id"] = job_id
            return self._node_health(job_id, issue)
        return issue

    def refresh_recovery(self, trial, recovery):
        """Retain later native infrastructure evidence while cleanup is pending."""
        observation = read_receipt(trial / "_scheduler_status.json")
        intent = self._intent(trial)
        job_id = recovery.get("job_id")
        if (
            recovery.get("retryable")
            or not job_id
            or not intent
            or intent.get("job_id") != job_id
            or observation.get("job_id") != job_id
            or observation.get("native_state")
            not in ACTIVE - {"PENDING", "CONFIGURING"}
        ):
            return recovery
        return self._node_health(job_id, dict(recovery))

    def _node_health(self, job_id, issue):
        # DRAIN alone (maintenance, job error, etc.) is not infrastructure proof.
        # A fresh job query prevents attributing a later fault on a reused node
        # to an allocation that has already terminated.
        errors = {}
        job = self._query(["scontrol", "show", "job", job_id, "-o"], errors)
        fields = dict(re.findall(r"(\w+)=([^\s]+)", job.stdout)) if job else {}
        node = fields.get("NodeList", "")
        if (
            fields.get("JobId") == job_id
            and fields.get("JobState") in ACTIVE - {"PENDING", "CONFIGURING"}
            and re.fullmatch(r"[A-Za-z0-9_.-]+", node)
        ):
            reply = self._query(["scontrol", "show", "node", node, "-o"], errors)
            values = dict(re.findall(r"(\w+)=([^\s]+)", reply.stdout)) if reply else {}
            flags = set(values.get("State", "").split("+"))
            if values.get("NodeName") == node and flags & {"DOWN", "NOT_RESPONDING"}:
                issue.update(
                    retryable=True,
                    node=node,
                    reason="node_unresponsive",
                    health_reason=issue["reason"],
                    node_evidence_at=time.time(),
                )
        return issue

    def failure(self, trial):
        """Replacement needs terminal evidence or a closed, unclaimed launch fence."""
        intent = self._intent(trial)
        if intent and intent.get("submission_guard") and not intent.get("job_id"):
            try:
                guard = _submission.gate(trial)
            except (OSError, ValueError):
                return {"reason": "submission_fence_unavailable", "retryable": False}
            if guard["state"] == "closed":
                return {"reason": "submission_fenced", "retryable": True}
        native = self._native(trial)
        if native is None or native[1] not in ERROR | CANCELED | {"COMPLETED"}:
            return {"reason": "termination_unconfirmed", "retryable": False}
        job_id, state, _ = native
        if state in {"NODE_FAIL", "BOOT_FAIL", "PREEMPTED"}:
            return {"reason": state, "retryable": True, "job_id": job_id}
        if state in {"OUT_OF_MEMORY", "TIMEOUT", "DEADLINE"}:
            return {"reason": state, "retryable": False, "job_id": job_id}
        recovery = read_receipt(trial / "_recovery.json")
        if recovery.get("job_id") == job_id and recovery.get("retryable"):
            return recovery
        worker = read_receipt(trial / "_receipt.json")
        return {
            "reason": recovery.get("reason") or worker.get("reason") or state,
            "retryable": False,
            "job_id": job_id,
        }

    def finalize(self, trial, state):
        intent = self._intent(trial)
        if intent:
            self._record(
                trial,
                state,
                intent.get("job_id"),
                "Trial reconciled with native state and workload artifacts",
            )

    def logs(self, trial):
        # Native logs exist even if container/bootstrap never reaches the worker.
        path = trial / "trial.log"
        intent = self._intent(trial)
        return str(
            path
            if path.exists()
            else trial / f"slurm-{intent.get('job_id', '*') if intent else '*'}.err"
        )

    def cancel(self, trial):
        atomic(trial / "_cancel.json", {"requested_at": time.time()})
        native = self._native(trial)
        if native is None:
            raise ValueError(
                "SLURM launch UNKNOWN; cannot safely identify a job to cancel"
            )
        job_id, state, _ = native
        if state in ACTIVE:
            self._call(["scancel", job_id])
