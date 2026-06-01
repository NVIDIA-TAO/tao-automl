# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""JSON-file based persistence for AutoML state, replacing MongoDB.

Concurrency safety
------------------
All reads and writes are protected by file-level locks using
``fcntl.flock()``.  Each data file has a corresponding ``.lock`` file;
reads take a shared lock and writes take an exclusive lock.

A global workspace lock (``state_store.lock()``) is also available for
compound read-modify-write sequences (e.g. ``report_result`` reads
history, updates a rec, then writes back).

``fcntl.flock()`` is safe for threads and processes on the **same local
filesystem**.  It does NOT work on NFS — keep the workspace on local disk.
"""

import fcntl
import json
import logging
import os
import threading

logger = logging.getLogger(__name__)


class _FileLock:
    """Context manager for an exclusive file lock using fcntl.flock."""

    def __init__(self, lock_path: str):
        self._lock_path = lock_path

    def __enter__(self):
        os.makedirs(os.path.dirname(self._lock_path), exist_ok=True)
        self._fd = open(self._lock_path, "w")
        fcntl.flock(self._fd, fcntl.LOCK_EX)
        return self

    def __exit__(self, *exc):
        fcntl.flock(self._fd, fcntl.LOCK_UN)
        self._fd.close()
        return False


class StateStore:
    """File-based state store using JSON files under workspace/.automl/.

    All state is persisted as JSON files in a ``.automl`` subdirectory
    within the provided workspace path.  Each logical entity (specs,
    brain info, controller info, etc.) is stored as a separate file
    keyed by job or experiment ID.

    Thread- and process-safe via ``fcntl.flock`` per-file locks.
    """

    def __init__(self, workspace_path: str):
        self._root = os.path.join(workspace_path, ".automl")
        os.makedirs(self._root, exist_ok=True)
        self._global_lock_path = os.path.join(self._root, ".global.lock")
        # In-process mutex so threads don't interleave within the same
        # Python process (fcntl.flock is per-fd, not per-thread).
        self._thread_lock = threading.Lock()

    # ------------------------------------------------------------------
    # Locking
    # ------------------------------------------------------------------

    def lock(self):
        """Acquire an exclusive lock for compound read-modify-write ops.

        Usage::

            with state_store.lock():
                data = state_store.get_controller_info(job_id)
                data.append(new_rec)
                state_store.save_controller_info(job_id, data)
        """
        return _FileLock(self._global_lock_path)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _path(self, *parts: str) -> str:
        """Build a file path under the .automl directory."""
        return os.path.join(self._root, *parts)

    def _lock_path_for(self, *parts: str) -> str:
        """Build a lock file path corresponding to a data file."""
        return self._path(*parts) + ".lock"

    def _read_json(self, *parts: str):
        """Read and return parsed JSON from a file, or None if missing.

        Takes a shared (read) lock so concurrent reads don't block each
        other, but a concurrent write will wait.
        """
        path = self._path(*parts)
        lock_path = self._lock_path_for(*parts)
        os.makedirs(os.path.dirname(lock_path), exist_ok=True)
        with self._thread_lock:
            try:
                with open(lock_path, "w") as lf:
                    fcntl.flock(lf, fcntl.LOCK_SH)
                    try:
                        with open(path, "r", encoding="utf-8") as fh:
                            return json.load(fh)
                    except (FileNotFoundError, json.JSONDecodeError):
                        return None
                    finally:
                        fcntl.flock(lf, fcntl.LOCK_UN)
            except OSError:
                return None

    def _write_json(self, data, *parts: str) -> None:
        """Atomically write *data* as JSON to a file.

        Takes an exclusive (write) lock.  Uses a temp file +
        ``os.replace()`` for atomicity.
        """
        path = self._path(*parts)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        lock_path = self._lock_path_for(*parts)
        # Per-thread unique tmp file to avoid collisions
        tmp_path = path + f".tmp.{threading.get_ident()}"
        with self._thread_lock:
            with open(lock_path, "w") as lf:
                fcntl.flock(lf, fcntl.LOCK_EX)
                try:
                    with open(tmp_path, "w", encoding="utf-8") as fh:
                        json.dump(data, fh, indent=2, default=str)
                    os.replace(tmp_path, path)
                except Exception:
                    try:
                        os.remove(tmp_path)
                    except OSError:
                        pass
                    raise
                finally:
                    fcntl.flock(lf, fcntl.LOCK_UN)

    # ------------------------------------------------------------------
    # Job specs
    # ------------------------------------------------------------------

    def get_job_specs(self, job_id: str):
        """Return the saved specs dict for *job_id*, or None."""
        return self._read_json("specs", f"{job_id}.json")

    def save_job_specs(self, job_id: str, specs: dict) -> None:
        """Persist specs for *job_id*."""
        self._write_json(specs, "specs", f"{job_id}.json")

    # ------------------------------------------------------------------
    # Brain info
    # ------------------------------------------------------------------

    def get_brain_info(self, job_id: str):
        """Return brain state for *job_id*, or None."""
        return self._read_json("brain", f"{job_id}.json")

    def save_brain_info(self, job_id: str, state: dict) -> None:
        """Persist brain state for *job_id*."""
        self._write_json(state, "brain", f"{job_id}.json")

    # ------------------------------------------------------------------
    # Controller info (list of recommendation dicts)
    # ------------------------------------------------------------------

    def get_controller_info(self, job_id: str):
        """Return controller recommendation list for *job_id*, or None."""
        return self._read_json("controller", f"{job_id}.json")

    def save_controller_info(self, job_id: str, recs) -> None:
        """Persist controller recommendations for *job_id*."""
        self._write_json(recs, "controller", f"{job_id}.json")

    # ------------------------------------------------------------------
    # Current recommendation pointer
    # ------------------------------------------------------------------

    def get_current_rec(self, job_id: str):
        """Return the current recommendation ID for *job_id*, or None."""
        data = self._read_json("current_rec", f"{job_id}.json")
        if data is not None:
            return data.get("rec_id")
        return None

    def save_current_rec(self, job_id: str, rec_id) -> None:
        """Persist current recommendation ID for *job_id*."""
        self._write_json({"rec_id": rec_id}, "current_rec", f"{job_id}.json")

    # ------------------------------------------------------------------
    # Custom parameter ranges (keyed by experiment ID)
    # ------------------------------------------------------------------

    def get_custom_param_ranges(self, experiment_id: str):
        """Return custom parameter ranges for *experiment_id*, or None."""
        return self._read_json("custom_ranges", f"{experiment_id}.json")

    def save_custom_param_ranges(self, experiment_id: str, ranges: dict) -> None:
        """Persist custom parameter ranges for *experiment_id*."""
        self._write_json(ranges, "custom_ranges", f"{experiment_id}.json")

    # ------------------------------------------------------------------
    # Best recommendation info
    # ------------------------------------------------------------------

    def get_best_rec_info(self, job_id: str):
        """Return best recommendation info for *job_id*, or None.

        Returns a dict with keys ``rec_number`` and ``rec_data``, or None.
        """
        return self._read_json("best_rec", f"{job_id}.json")

    def save_best_rec_info(self, job_id: str, rec_number, rec_data) -> None:
        """Persist best recommendation info for *job_id*."""
        self._write_json(
            {"rec_number": rec_number, "rec_data": rec_data},
            "best_rec",
            f"{job_id}.json",
        )
