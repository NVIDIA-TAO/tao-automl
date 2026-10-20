# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Lepton workspace v1 Jobs API, without a TAO SDK or mutable CLI workspace.

Wire contract checked against NVIDIA leptonai 0.27.3's api/v1/job.py and types.
The caller supplies the complete workspace API URL and a token *variable name*.
"""

import json
import os
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode
from urllib.request import HTTPRedirectHandler, Request, build_opener

from automl_core.managed import ManagedExecutor
from automl_core.slurm import SchedulerUnavailable


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None  # Never forward a workspace credential to another endpoint.


class LeptonExecutor(ManagedExecutor):
    def api(self, method, path, body=None, missing=False):
        token = os.environ.get(self.settings.token_env)
        if not token:
            raise ValueError(
                f"Missing credential environment name: {self.settings.token_env}"
            )
        request = Request(
            self.settings.api_url + path,
            method=method,
            data=json.dumps(body).encode() if body is not None else None,
            headers={
                "Authorization": "Bearer " + token,
                "Content-Type": "application/json",
            },
        )
        try:
            with build_opener(NoRedirect()).open(request, timeout=20) as response:
                raw = response.read(16 * 1024 * 1024 + 1)
                if len(raw) > 16 * 1024 * 1024:
                    raise ValueError("Lepton response exceeded the bounded reader")
                return json.loads(raw) if raw else None
        except HTTPError as exc:
            if exc.code == 404 and missing:
                return None
            raise SchedulerUnavailable(
                [{"platform": "lepton", "method": method, "http_status": exc.code}]
            ) from None
        except (URLError, OSError, json.JSONDecodeError):
            raise SchedulerUnavailable(
                [{"platform": "lepton", "method": method}]
            ) from None

    def scope(self):
        return {"api_url": self.settings.api_url}

    def preflight(self, cwd):
        self.storage_check([cwd])
        info = self.api("GET", "/workspace")
        shape = info.get("supported_shapes", {}).get(self.settings.resource_shape)
        if shape is None or shape.get("accelerator_num", 0) != self.settings.gpus:
            raise ValueError(
                "Lepton resource shape is unavailable or GPU count differs from the reviewed request"
            )
        for group in self.settings.node_groups:
            self.api("GET", "/dedicated-node-groups/" + quote(group, safe=""))
        return {
            **self.scope(),
            "resource_shape": shape,
            "image": self.spec.image,
            "storage": "shared node-nfs; worker verifies launch receipt",
        }

    def render(self, trial, plan, intent):
        mounts = [
            {
                "from": v.source,
                "path": "/" + v.sub_path,
                "mount_path": str(v.path),
                "mount_options": {"read_only": v.read_only},
            }
            for v in self.settings.volumes
        ]
        for path in plan.read_only_mounts:
            volume = next(
                v for v in self.settings.volumes if path.is_relative_to(v.path)
            )
            mounts.append(
                {
                    "from": volume.source,
                    "path": str(
                        Path("/") / volume.sub_path / path.relative_to(volume.path)
                    ),
                    "mount_path": str(path),
                    "mount_options": {"read_only": True},
                }
            )
        return {
            "metadata": {
                # The v1 create contract takes the requested display name in
                # `id`; responses have a generated ID and a separate `name`.
                "id": intent["name"],
                "visibility": "private",
                "labels": {"automl-owner": intent["token"]},
            },
            "spec": {
                "resource_shape": self.settings.resource_shape,
                "affinity": {
                    "allowed_dedicated_node_groups": self.settings.node_groups
                },
                "container": {
                    "image": self.spec.image,
                    "command": self.command(trial, intent),
                },
                "completions": 1,
                "parallelism": 1,
                "max_failure_retry": 0,
                "max_job_failure_retry": 0,
                "shared_memory_size": self.settings.shared_memory_mib,
                "ttl_seconds_after_finished": 604800,
                "image_pull_secrets": self.settings.image_pull_secrets,
                "mounts": mounts,
                "envs": [
                    {"name": name, "value_from": {"secret_name_ref": secret}}
                    for name, secret in self.settings.secrets.items()
                ],
            },
        }

    def create(self, body):
        self.api("POST", "/jobs", body)

    def find(self, intent):
        if intent["native_id"]:
            return self.api(
                "GET",
                "/jobs/"
                + quote(intent["native_id"], safe="")
                + "?job_query_mode=alive_and_archive",
                missing=True,
            )
        found = []
        # Bounded exact-name reconciliation over alive AND archived jobs. Never
        # adopt just the newest name match, nor assume a server-generated ID.
        for page in range(1, 21):
            result = self.api(
                "GET",
                "/jobs?"
                + urlencode(
                    {
                        "q": intent["name"],
                        "job_query_mode": "alive_and_archive",
                        "page": page,
                        "page_size": 100,
                    }
                ),
            )
            rows = result if isinstance(result, list) else result["jobs"]
            found.extend(
                job for job in rows if job["metadata"].get("name") == intent["name"]
            )
            if len(found) > 1:
                raise ValueError("Conflicting Lepton job identities; no replacement")
            if not rows:
                return found[0] if found else None
        raise SchedulerUnavailable(
            [{"platform": "lepton", "reason": "reconciliation pagination limit"}]
        )

    def identity(self, job):
        return job["metadata"].get("id"), (job["metadata"].get("labels") or {}).get(
            "automl-owner"
        )

    def native_state(self, job):
        status = job.get("status") or {}
        state = status.get("state")
        if state in {"Starting", "Queueing", "Awaiting"}:
            return "PENDING"
        if state in {"Running", "Stopping", "Deleting"}:
            return "RUNNING"
        if state in {"Completed", "Failed", "Stopped"} and not status.get("active", 0):
            return {"Completed": "COMPLETE", "Failed": "ERROR", "Stopped": "CANCELED"}[
                state
            ]
        # Archived/Deleted/unknown and unexpected retries are not success proof.
        return "UNKNOWN"

    def stop(self, job):
        self.api(
            "PATCH",
            "/jobs/" + quote(job["metadata"]["id"], safe=""),
            {"spec": {"stopped": True}},
        )
