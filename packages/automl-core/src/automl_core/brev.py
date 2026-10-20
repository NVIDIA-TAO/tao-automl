# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Containers on an existing Brev instance; provisioning remains explicit."""

import json
import os
import shlex
import subprocess

from automl_core.managed import ManagedExecutor
from automl_core.slurm import SchedulerUnavailable


class BrevExecutor(ManagedExecutor):
    def cli(self, args):
        try:
            result = subprocess.run(
                ["brev", "exec", self.settings.instance, shlex.join(args)],
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
            if result.returncode:
                raise SchedulerUnavailable(
                    [{"platform": "brev", "operation": args[:2]}]
                )
            return result.stdout.strip()
        except (OSError, subprocess.TimeoutExpired):
            raise SchedulerUnavailable(
                [{"platform": "brev", "operation": args[:2]}]
            ) from None

    def scope(self):
        daemon = self.cli(["docker", "info", "--format", "{{.ID}}"])
        if daemon != self.settings.daemon_id:
            raise ValueError("Brev instance Docker daemon identity mismatch")
        return {"instance": self.settings.instance, "daemon_id": daemon}

    def preflight(self, cwd):
        self.storage_check([cwd])
        identity = self.scope()
        image = json.loads(self.cli(["docker", "image", "inspect", self.spec.image]))[0]
        # This is a read-only check. Bind data is not uploaded, instances are not
        # started, and images are never pulled by the executor.
        for volume in self.settings.volumes:
            self.cli(["test", "-d", str(volume.path)])
        if self.settings.gpus:
            devices = self.cli(
                ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"]
            ).splitlines()
            if not set(map(str, self.settings.gpu_devices)).issubset(devices):
                raise ValueError("Brev GPU device indices are unavailable")
        return {
            **identity,
            "image_id": image["Id"],
            "storage": "same filesystem required; no transfer",
        }

    def render(self, trial, plan, intent):
        args = [
            "docker",
            "run",
            "--detach",
            "--pull=never",
            "--restart=no",
            "--user",
            f"{os.getuid()}:{os.getgid()}",
            "--name",
            intent["name"],
            "--label",
            "automl-owner=" + intent["token"],
            "--env",
            "HOME=" + str(trial),
            "--cpus",
            str(self.settings.cpus),
            "--memory",
            f"{self.settings.memory_gib}g",
            "--shm-size",
            f"{self.settings.shared_memory_gib}g",
        ]
        mounts = {v.path: v.read_only for v in self.settings.volumes}
        mounts.update({path: True for path in plan.read_only_mounts})
        for path, read_only in mounts.items():
            if any(c in str(path) for c in (",", ":", "\n")):
                raise ValueError(
                    "Brev bind paths cannot contain commas, colons or newlines"
                )
            args += [
                "--mount",
                f"type=bind,src={path},dst={path}" + (",readonly" if read_only else ""),
            ]
        for name in self.spec.environment:
            args += ["--env", name]  # Values must already exist on the instance.
        if self.settings.gpus:
            args += [
                "--gpus",
                '"device=' + ",".join(map(str, self.settings.gpu_devices)) + '"',
            ]
        else:
            args += ["--env", "NVIDIA_VISIBLE_DEVICES=void"]
        return args + [
            "--entrypoint",
            self.spec.python,
            self.spec.image,
            *self.command(trial, intent)[1:],
        ]

    def create(self, body):
        self.cli(body)

    def find(self, intent):
        ids = self.cli(
            [
                "docker",
                "container",
                "ls",
                "--all",
                "--no-trunc",
                "--filter",
                "name=^/" + intent["name"] + "$",
                "--format",
                "{{.ID}}",
            ]
        )
        if not ids:
            return None
        if len(ids.splitlines()) != 1:
            raise ValueError("Ambiguous Brev container identity")
        return json.loads(self.cli(["docker", "inspect", ids]))[0]

    def identity(self, job):
        return job.get("Id"), job["Config"].get("Labels", {}).get("automl-owner")

    def native_state(self, job):
        state = job["State"]
        if state.get("Running"):
            return "RUNNING"
        if state.get("Status") in {"created", "restarting", "removing"}:
            return "UNKNOWN"
        if state.get("Status") not in {"exited", "dead"}:
            return "UNKNOWN"
        return "COMPLETE" if state.get("ExitCode") == 0 else "ERROR"

    def stop(self, job):
        self.cli(["docker", "stop", "--time", "10", job["Id"]])
