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
import hashlib
import json
import logging
import os
import threading

from tao_automl.utils.value_utils import normalize_json_value

logger = logging.getLogger(__name__)

STATE_TRANSACTION_SCHEMA_VERSION = 1


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
                    normalized = normalize_json_value(
                        data,
                        path="persisted_state",
                    )
                    with open(tmp_path, "w", encoding="utf-8") as fh:
                        json.dump(normalized, fh, indent=2, allow_nan=False)
                    os.replace(tmp_path, path)
                except Exception:
                    try:
                        os.remove(tmp_path)
                    except OSError:
                        pass
                    raise
                finally:
                    fcntl.flock(lf, fcntl.LOCK_UN)

    @staticmethod
    def _canonical_sha256(data) -> str:
        """Return a deterministic digest for one normalized JSON payload."""
        normalized = normalize_json_value(data, path="persisted_state")
        encoded = json.dumps(
            normalized,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def _state_transaction_parts(self, job_id: str) -> tuple[str, str]:
        return "state_transactions", f"{job_id}.json"

    def _read_state_transaction(self, job_id: str):
        parts = self._state_transaction_parts(job_id)
        transaction = self._read_json(*parts)
        if transaction is None and os.path.exists(self._path(*parts)):
            raise RuntimeError(
                "AutoML state transaction record is unreadable; refusing "
                f"to resume workspace {job_id!r}"
            )
        return transaction

    @staticmethod
    def _validate_transaction_record(
        transaction,
        *,
        job_id: str,
        expected_status: str | None = None,
    ) -> None:
        if not isinstance(transaction, dict):
            raise RuntimeError(
                f"AutoML state transaction for {job_id!r} is not a mapping"
            )
        if transaction.get("schema_version") != STATE_TRANSACTION_SCHEMA_VERSION:
            raise RuntimeError(
                f"AutoML state transaction for {job_id!r} has an unsupported "
                "schema version"
            )
        generation = transaction.get("generation")
        if (
            isinstance(generation, bool)
            or not isinstance(generation, int)
            or generation <= 0
        ):
            raise RuntimeError(
                f"AutoML state transaction for {job_id!r} has an invalid "
                "generation"
            )
        status = transaction.get("status")
        if status not in {"pending", "committed"}:
            raise RuntimeError(
                f"AutoML state transaction for {job_id!r} has an invalid status"
            )
        if expected_status is not None and status != expected_status:
            raise RuntimeError(
                f"AutoML state transaction for {job_id!r} is {status!r}; "
                f"expected {expected_status!r}"
            )

    def validate_state_transaction(self, job_id: str) -> None:
        """Validate the last compound brain/controller persistence generation.

        Workspaces created before this protocol have no transaction record and
        remain readable. Once a record exists, resume fails closed if a process
        stopped between component writes or either committed component changed
        independently afterward.
        """
        transaction = self._read_state_transaction(job_id)
        if transaction is None:
            return
        self._validate_transaction_record(transaction, job_id=job_id)
        if transaction["status"] != "committed":
            raise RuntimeError(
                "Incomplete AutoML state transaction detected for workspace "
                f"{job_id!r} at generation {transaction['generation']}; "
                "brain and controller state may belong to different "
                "recommendation decisions"
            )

        for component, parts in (
            ("brain", ("brain", f"{job_id}.json")),
            ("controller", ("controller", f"{job_id}.json")),
        ):
            value = self._read_json(*parts)
            present = value is not None
            expected_present = transaction.get(f"{component}_present")
            expected_sha256 = transaction.get(f"{component}_sha256")
            if expected_present is not present:
                raise RuntimeError(
                    "Committed AutoML state transaction component presence "
                    f"mismatch for {component!r} in workspace {job_id!r}"
                )
            actual_sha256 = (
                self._canonical_sha256(value) if present else None
            )
            if expected_sha256 != actual_sha256:
                raise RuntimeError(
                    "Committed AutoML state transaction integrity mismatch "
                    f"for {component!r} in workspace {job_id!r}"
                )

    def begin_state_transaction(
        self,
        job_id: str,
        *,
        operation: str,
    ) -> int:
        """Mark the start of one compound brain/controller state write."""
        if not isinstance(operation, str) or not operation.strip():
            raise ValueError("state transaction operation must be non-empty")
        previous = self._read_state_transaction(job_id)
        if previous is None:
            generation = 1
        else:
            self.validate_state_transaction(job_id)
            generation = int(previous["generation"]) + 1
        self._write_json(
            {
                "schema_version": STATE_TRANSACTION_SCHEMA_VERSION,
                "generation": generation,
                "status": "pending",
                "operation": operation.strip(),
            },
            *self._state_transaction_parts(job_id),
        )
        return generation

    def commit_state_transaction(self, job_id: str, generation: int) -> None:
        """Commit a compound write after hashing both persisted components."""
        transaction = self._read_state_transaction(job_id)
        self._validate_transaction_record(
            transaction,
            job_id=job_id,
            expected_status="pending",
        )
        if (
            isinstance(generation, bool)
            or not isinstance(generation, int)
            or transaction["generation"] != generation
        ):
            raise RuntimeError(
                f"AutoML state transaction generation mismatch for {job_id!r}"
            )

        components = {}
        for component, parts in (
            ("brain", ("brain", f"{job_id}.json")),
            ("controller", ("controller", f"{job_id}.json")),
        ):
            value = self._read_json(*parts)
            components[f"{component}_present"] = value is not None
            components[f"{component}_sha256"] = (
                self._canonical_sha256(value) if value is not None else None
            )
        self._write_json(
            {
                **transaction,
                **components,
                "status": "committed",
            },
            *self._state_transaction_parts(job_id),
        )
        self.validate_state_transaction(job_id)

    def get_state_transaction(self, job_id: str):
        """Return a copy of the transaction record for diagnostics/tests."""
        return self._read_state_transaction(job_id)

    # ------------------------------------------------------------------
    # Job specs
    # ------------------------------------------------------------------

    def get_job_specs(self, job_id: str):
        """Return the saved specs dict for *job_id*, or None."""
        return self._read_json("specs", f"{job_id}.json")

    def save_job_specs(self, job_id: str, specs: dict) -> None:
        """Persist specs for *job_id*."""
        self._write_json(specs, "specs", f"{job_id}.json")

    def list_job_spec_ids(self) -> tuple[str, ...]:
        """Return persisted job-spec identities in deterministic order.

        AutoML uses this only to resolve an omitted session ID during resume.
        Lock and temporary files are intentionally excluded.  A caller must
        still read and validate the selected spec before constructing a brain.
        """
        specs_dir = self._path("specs")
        try:
            names = os.listdir(specs_dir)
        except OSError:
            return ()
        return tuple(
            sorted(
                name[:-5]
                for name in names
                if name.endswith(".json")
                and os.path.isfile(os.path.join(specs_dir, name))
            )
        )

    # ------------------------------------------------------------------
    # Brain info
    # ------------------------------------------------------------------

    def get_brain_info(self, job_id: str):
        """Return brain state for *job_id*, or None."""
        self.validate_state_transaction(job_id)
        return self._read_json("brain", f"{job_id}.json")

    def get_brain_info_for_update(self, job_id: str):
        """Read brain state while constructing the same pending transaction.

        This narrow internal hook supports composite brain implementations
        whose ``save_state`` extends a base brain payload in place. Resume and
        all public reads must use :meth:`get_brain_info`, which validates the
        committed brain/controller generation first.
        """
        return self._read_json("brain", f"{job_id}.json")

    def save_brain_info(self, job_id: str, state: dict) -> None:
        """Persist brain state for *job_id*."""
        self._write_json(state, "brain", f"{job_id}.json")

    # ------------------------------------------------------------------
    # Controller info (list of recommendation dicts)
    # ------------------------------------------------------------------

    def get_controller_info(self, job_id: str):
        """Return controller recommendation list for *job_id*, or None."""
        self.validate_state_transaction(job_id)
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
