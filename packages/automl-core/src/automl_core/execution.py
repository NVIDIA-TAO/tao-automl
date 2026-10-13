# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Native venv and Docker submit/status/logs/cancel; no SDK or registry pulls."""

import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

from automl_core import _worker
from automl_core.contracts import fingerprint


class Executor:
    def __new__(cls, spec):
        if cls is Executor and spec.kind in {"kubernetes", "brev", "lepton"}:
            from automl_core.brev import BrevExecutor
            from automl_core.kubernetes import KubernetesExecutor
            from automl_core.lepton import LeptonExecutor

            return {
                "kubernetes": KubernetesExecutor,
                "brev": BrevExecutor,
                "lepton": LeptonExecutor,
            }[spec.kind](spec)
        if cls is Executor and spec.kind == "slurm":
            from automl_core.slurm import SlurmExecutor

            return SlurmExecutor(spec)
        return super().__new__(cls)

    def __init__(self, spec):
        self.spec = spec
        self.children = {}

    def preflight(self, cwd):
        if sys.platform != "linux":
            raise ValueError(
                "The local execution adapter currently requires Linux process identities"
            )
        if not cwd.is_dir():
            raise ValueError(f"Workload directory does not exist: {cwd}")
        missing = [name for name in self.spec.environment if not os.environ.get(name)]
        if missing:
            raise ValueError(
                f"Missing environment variable names: {', '.join(missing)}"
            )
        if self.spec.kind == "venv":
            if not Path(self.spec.python).is_absolute() or not os.access(
                self.spec.python, os.X_OK
            ):
                raise ValueError(
                    "venv.python must be an absolute executable interpreter path"
                )
            inventory = subprocess.run(
                [
                    self.spec.python,
                    "-c",
                    "import importlib.metadata as m,json,sys; print(json.dumps({'python':sys.version,'packages':sorted((d.metadata['Name'],d.version) for d in m.distributions())}))",
                ],
                check=True,
                capture_output=True,
                text=True,
                timeout=15,
            )
            return {
                "interpreter": str(Path(self.spec.python).resolve()),
                "inventory": json.loads(inventory.stdout),
            }
        else:
            if not shutil.which("docker"):
                raise ValueError("Docker CLI is required")
            image = subprocess.run(
                ["docker", "image", "inspect", self.spec.image],
                check=True,
                capture_output=True,
                text=True,
                timeout=15,
            )
            if any(
                not path.is_absolute() or not path.exists() for path in self.spec.mounts
            ):
                raise ValueError("Docker mounts must be existing absolute paths")
            return {"image_id": json.loads(image.stdout)[0]["Id"]}

    def container_name(self, trial):
        return "automl-" + fingerprint(str(trial))[:24]

    def docker_create(self, trial, plan):
        args = [
            "docker",
            "create",
            "--pull=never",
            "--user",
            f"{os.getuid()}:{os.getgid()}",
            "--env",
            f"HOME={trial / '_home'}",
            "--name",
            self.container_name(trial),
            "--label",
            f"automl.trial={fingerprint(str(trial))}",
        ]
        mounts = {
            plan.cwd.resolve(): "ro",
            **{p.absolute(): "ro" for p in [*self.spec.mounts, *plan.read_only_mounts]},
            trial: "rw",
        }
        for path, mode in mounts.items():
            if "," in str(path) or ":" in str(path):
                raise ValueError("Docker mount paths cannot contain commas or colons")
            args += [
                "--mount",
                f"type=bind,src={path},dst={path}"
                + (",readonly" if mode == "ro" else ""),
            ]
        for name in self.spec.environment:
            args += ["--env", name]
        for name, value in plan.environment.items():
            # Internal, non-secret variables only (e.g. per-trial output root).
            args += ["--env", f"{name}={value}"]
        if self.spec.gpu_devices:
            args += ["--gpus", f'"device={self.spec.gpu_devices}"']
        return args + [
            "--entrypoint",
            self.spec.python,
            self.spec.image,
            str(trial / "_worker.py"),
            str(trial / "_execution.json"),
        ]

    def submit(self, trial, plan, deadline):
        if any(
            not p.is_absolute() or not p.exists() or p.resolve() == trial.resolve()
            for p in plan.read_only_mounts
        ):
            raise ValueError(
                "Stage input mounts must be existing absolute read-only paths outside the new trial"
            )
        if (trial / "_execution.json").exists():
            raise ValueError(
                "A launch intent already exists; reconcile it instead of resubmitting"
            )
        for name, content in plan.files.items():
            target = trial / name
            if name.startswith("_") or not target.resolve().is_relative_to(trial):
                raise ValueError(f"Reserved or unsafe staged path: {name}")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content)
        (trial / "_worker.py").write_text(Path(_worker.__file__).read_text())
        payload = {
            "commands": plan.commands,
            "cwd": str(plan.cwd),
            "deadline": deadline,
            "progress_files": plan.progress_files,
        }
        _worker.atomic(trial / "_execution.json", payload)  # Intent precedes launch.
        if self.spec.kind == "docker":
            (trial / "_home").mkdir()
            subprocess.run(
                self.docker_create(trial, plan),
                check=True,
                capture_output=True,
                timeout=30,
            )
            subprocess.run(
                ["docker", "start", self.container_name(trial)],
                check=True,
                capture_output=True,
                timeout=30,
            )
        else:
            names = {
                "PATH",
                "HOME",
                "TMPDIR",
                "LANG",
                "LC_ALL",
                "LD_LIBRARY_PATH",
                *self.spec.environment,
            }
            env = {name: os.environ[name] for name in names if name in os.environ}
            env.update(plan.environment)
            if self.spec.gpu_devices:
                env["CUDA_VISIBLE_DEVICES"] = self.spec.gpu_devices
            with (trial / "worker.log").open("ab") as log:
                self.children[str(trial)] = subprocess.Popen(
                    [
                        sys.executable,
                        str(trial / "_worker.py"),
                        str(trial / "_execution.json"),
                    ],
                    env=env,
                    stdout=log,
                    stderr=log,
                    start_new_session=True,
                )
        # Wait briefly for durable launch acknowledgement, never launch again on timeout.
        for _ in range(100):
            if (trial / "_receipt.json").exists():
                return
            time.sleep(0.05)

    def _container(self, trial):
        result = subprocess.run(
            ["docker", "inspect", self.container_name(trial)],
            check=False,
            capture_output=True,
            text=True,
            timeout=15,
        )
        if result.returncode:
            return None
        data = json.loads(result.stdout)[0]
        if data["Config"].get("Labels", {}).get("automl.trial") != fingerprint(
            str(trial)
        ):
            raise ValueError("Container ownership label mismatch")
        return data["State"]

    def status(self, trial):
        child = self.children.get(str(trial))
        if child:
            child.poll()  # Reap exited local children.
        receipt_path = trial / "_receipt.json"
        receipt = json.loads(receipt_path.read_text()) if receipt_path.exists() else {}
        if self.spec.kind == "venv" and receipt.get("cleanup_confirmed") is False:
            return "UNKNOWN"
        if self.spec.kind == "docker":
            native = self._container(trial)
            if native is None:
                return "UNKNOWN"
            if native.get("Running"):
                return "RUNNING"
            if native.get("Status") == "created":
                return "UNKNOWN"  # Creation may have preceded a controller crash.
            if native.get("ExitCode", 1) != 0:
                return "CANCELED" if (trial / "_cancel.json").exists() else "ERROR"
            if receipt.get("state") not in {"COMPLETE", "ERROR", "CANCELED"}:
                return "UNKNOWN"  # Native exit zero is not a terminal worker receipt.
        elif receipt.get("state") == "RUNNING":
            if receipt.get("host") != _worker.host_identity():
                return "UNKNOWN"
            return (
                "RUNNING"
                if _worker.process_token(receipt["pid"]) == receipt["token"]
                else "UNKNOWN"
            )
        return receipt.get("state", "UNKNOWN")

    def logs(self, trial):
        path = trial / "trial.log"
        return str(path) if path.exists() else None

    def cancel(self, trial):
        if trial.is_dir():
            _worker.atomic(trial / "_cancel.json", {"requested_at": time.time()})
        if self.spec.kind == "docker":
            native = self._container(trial)
            if native and native.get("Running"):
                subprocess.run(
                    ["docker", "stop", "--time", "5", self.container_name(trial)],
                    check=True,
                    capture_output=True,
                    timeout=15,
                )
            return
        receipt_path = trial / "_receipt.json"
        if not receipt_path.exists():
            return  # Queued for the worker; status remains UNKNOWN until acknowledged.
        receipt = json.loads(receipt_path.read_text())
        if receipt.get("state") != "RUNNING":
            return
        if receipt.get("host") != _worker.host_identity():
            raise ValueError(
                "Local receipt belongs to a different host/boot; cancel on the recorded host"
            )
        if _worker.process_token(receipt["pid"]) != receipt["token"]:
            if json.loads(receipt_path.read_text()).get("state") in {
                "COMPLETE",
                "ERROR",
                "CANCELED",
            }:
                return  # Completion raced the process identity check.
            # A dead worker may have left its separately grouped command alive.
            pid = receipt.get("child_pid")
            if pid and _worker.process_token(pid) == receipt.get("child_token"):
                os.killpg(pid, signal.SIGTERM)
            raise ValueError(
                "Worker is absent; inspect its child process before resuming"
            )
        try:
            os.kill(receipt["pid"], signal.SIGTERM)
        except ProcessLookupError:
            if json.loads(receipt_path.read_text()).get("state") not in {
                "COMPLETE",
                "ERROR",
                "CANCELED",
            }:
                raise ValueError(
                    "Worker disappeared without a terminal receipt; inspect native execution"
                ) from None
