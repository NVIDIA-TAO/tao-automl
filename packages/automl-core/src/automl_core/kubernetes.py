# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Single-pod Kubernetes Jobs through an explicitly selected kubectl context."""

import json
import math
import os
import subprocess
import time
from pathlib import Path

from automl_core.managed import ManagedExecutor
from automl_core.slurm import SchedulerUnavailable


class KubernetesExecutor(ManagedExecutor):
    def cli(self, args, body=None):
        try:
            result = subprocess.run(
                [
                    "kubectl",
                    "--context",
                    self.settings.context,
                    "--namespace",
                    self.settings.namespace,
                    "--request-timeout=20s",
                    *args,
                ],
                input=json.dumps(body) if body is not None else None,
                text=True,
                capture_output=True,
                timeout=30,
                check=False,
            )
            if result.returncode:
                raise SchedulerUnavailable(
                    [{"platform": "kubernetes", "operation": args[0]}]
                )
            return json.loads(result.stdout) if result.stdout.strip() else None
        except (OSError, subprocess.TimeoutExpired, json.JSONDecodeError):
            raise SchedulerUnavailable(
                [{"platform": "kubernetes", "operation": args[0]}]
            ) from None

    def scope(self):
        obj = self.cli(["get", "namespace", self.settings.namespace, "-o", "json"])
        return {
            "namespace_uid": obj["metadata"]["uid"],
            "context": self.settings.context,
        }

    def preflight(self, cwd):
        self.storage_check([cwd])
        identity = self.scope()
        for v in self.settings.volumes:
            pvc = self.cli(["get", "pvc", v.source, "-o", "json"])
            if pvc.get("status", {}).get("phase") != "Bound":
                raise ValueError("Kubernetes volume claims must already be Bound")
        self.cli(["get", "serviceaccount", self.settings.service_account, "-o", "json"])
        nodes = self.cli(["get", "nodes", "-o", "json"])["items"]
        if not any(
            all(
                node["metadata"].get("labels", {}).get(k) == v
                for k, v in self.settings.node_selector.items()
            )
            and int(
                node.get("status", {})
                .get("allocatable", {})
                .get(self.settings.gpu_resource, 0)
            )
            >= self.settings.gpus
            for node in nodes
        ):
            raise ValueError(
                "No matching Kubernetes node has the requested per-node GPU capacity"
            )
        return {
            **identity,
            "image": self.spec.image,
            "storage": "shared PVC; worker verifies launch receipt",
        }

    def render(self, trial, plan, intent):
        settings = self.settings
        volumes = [
            {"name": f"data-{i}", "persistentVolumeClaim": {"claimName": v.source}}
            for i, v in enumerate(settings.volumes)
        ]
        mounts = [
            {
                "name": f"data-{i}",
                "mountPath": str(v.path),
                "readOnly": v.read_only,
                **({"subPath": v.sub_path} if v.sub_path else {}),
            }
            for i, v in enumerate(settings.volumes)
        ]
        for path in plan.read_only_mounts:
            i, volume = next(
                (i, v)
                for i, v in enumerate(settings.volumes)
                if path.is_relative_to(v.path)
            )
            mounts.append(
                {
                    "name": f"data-{i}",
                    "mountPath": str(path),
                    "readOnly": True,
                    "subPath": str(
                        Path(volume.sub_path) / path.relative_to(volume.path)
                    ),
                }
            )
        volumes.append(
            {
                "name": "shm",
                "emptyDir": {
                    "medium": "Memory",
                    "sizeLimit": f"{settings.shared_memory_gib}Gi",
                },
            }
        )
        mounts.append({"name": "shm", "mountPath": "/dev/shm"})
        resources = {"cpu": str(settings.cpus), "memory": f"{settings.memory_gib}Gi"}
        if settings.gpus:
            resources[settings.gpu_resource] = str(settings.gpus)
        return {
            "apiVersion": "batch/v1",
            "kind": "Job",
            "metadata": {
                "name": intent["name"],
                "namespace": settings.namespace,
                "labels": {"automl-owner": intent["token"]},
            },
            "spec": {
                "backoffLimit": 0,
                "completions": 1,
                "parallelism": 1,
                "activeDeadlineSeconds": max(
                    1, math.ceil(intent["deadline"] - time.time())
                ),
                "template": {
                    "metadata": {"labels": {"automl-owner": intent["token"]}},
                    "spec": {
                        "restartPolicy": "Never",
                        "terminationGracePeriodSeconds": 30,
                        "automountServiceAccountToken": False,
                        "serviceAccountName": settings.service_account,
                        "securityContext": {
                            "runAsUser": os.getuid(),
                            "runAsGroup": os.getgid(),
                        },
                        "nodeSelector": settings.node_selector,
                        "imagePullSecrets": [
                            {"name": name} for name in settings.image_pull_secrets
                        ],
                        "volumes": volumes,
                        "containers": [
                            {
                                "name": "trial",
                                "image": self.spec.image,
                                "imagePullPolicy": "IfNotPresent",
                                "command": self.command(trial, intent),
                                "resources": {
                                    "requests": resources,
                                    "limits": resources,
                                },
                                "volumeMounts": mounts,
                                "env": [
                                    {"name": "HOME", "value": str(trial)},
                                    *[
                                        {
                                            "name": name,
                                            "valueFrom": {"secretKeyRef": ref},
                                        }
                                        for name, ref in settings.secrets.items()
                                    ],
                                ],
                            }
                        ],
                    },
                },
            },
        }

    def create(self, body):
        self.cli(["create", "-f", "-", "-o", "json"], body)

    def find(self, intent):
        return self.cli(
            ["get", "job", intent["name"], "--ignore-not-found", "-o", "json"]
        )

    def identity(self, job):
        return job["metadata"].get("uid"), job["metadata"].get("labels", {}).get(
            "automl-owner"
        )

    def native_state(self, job):
        uid = job["metadata"]["uid"]
        pods = self.cli(
            ["get", "pods", "-l", "automl-owner=" + self.identity(job)[1], "-o", "json"]
        )["items"]
        owned = [
            pod
            for pod in pods
            if any(
                ref.get("uid") == uid
                for ref in pod["metadata"].get("ownerReferences", [])
            )
        ]
        active = any(
            pod.get("status", {}).get("phase") not in {"Succeeded", "Failed"}
            for pod in owned
        )
        conditions = {
            c["type"]
            for c in job.get("status", {}).get("conditions", [])
            if c.get("status") == "True"
        }
        if job["spec"].get("suspend"):
            # An accepted spec patch is only a request. Wait for the Job
            # controller to acknowledge it AND for all owned pods to terminate.
            return (
                "CANCELED"
                if "Suspended" in conditions
                and not active
                and not job.get("status", {}).get("active")
                else "RUNNING"
            )
        if not active and "Complete" in conditions:
            return "COMPLETE"
        if not active and "Failed" in conditions:
            return "ERROR"
        return (
            "RUNNING"
            if any(p.get("status", {}).get("phase") == "Running" for p in owned)
            else "PENDING"
        )

    def stop(self, job):
        # UID test makes a name-reuse race fail instead of touching another job.
        patch = [
            {"op": "test", "path": "/metadata/uid", "value": job["metadata"]["uid"]},
            {"op": "add", "path": "/spec/suspend", "value": True},
        ]
        self.cli(
            [
                "patch",
                "job",
                job["metadata"]["name"],
                "--type=json",
                "--patch-file=/dev/stdin",
                "-o",
                "json",
            ],
            patch,
        )
