# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Standalone stdlib worker: survives controller restarts; also runs in containers."""

import json
import os
import signal
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path


def atomic(path, value):
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w") as stream:
            json.dump(value, stream, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


def process_token(pid):
    try:
        # comm can contain spaces or parentheses; starttime is field 22.
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        return None if fields[0] == "Z" else fields[19]
    except (FileNotFoundError, ProcessLookupError):
        return None


def host_identity():
    return {
        "hostname": socket.gethostname(),
        "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
    }


def stop_child(child):
    try:
        os.killpg(child.pid, signal.SIGTERM)
        child.wait(timeout=2)
    except subprocess.TimeoutExpired:
        os.killpg(child.pid, signal.SIGKILL)
        child.wait()
    except ProcessLookupError:
        pass


def main():
    plan_path = Path(sys.argv[1])
    root = plan_path.parent
    plan = json.loads(plan_path.read_text())
    canceled = False

    def request_stop(signum, frame):
        nonlocal canceled
        canceled = True

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    receipt = {
        "host": host_identity(),
        "pid": os.getpid(),
        "token": process_token(os.getpid()),
        "state": "RUNNING",
    }
    atomic(root / "_receipt.json", receipt)
    child = None
    try:
        with (root / "trial.log").open("ab", buffering=0) as log:
            for command in plan["commands"]:
                canceled = (
                    canceled
                    or (root / "_cancel.json").exists()
                    or (root.parent / "cancel.json").exists()
                )
                if canceled or time.time() >= plan["deadline"]:
                    receipt["state"] = "CANCELED" if canceled else "ERROR"
                    receipt["reason"] = "cancel requested" if canceled else "timeout"
                    break
                child = subprocess.Popen(
                    command,
                    cwd=plan["cwd"],
                    stdout=log,
                    stderr=log,
                    start_new_session=True,
                )
                receipt.update(
                    child_pid=child.pid, child_token=process_token(child.pid)
                )
                atomic(root / "_receipt.json", receipt)
                while child.poll() is None:
                    canceled = (
                        canceled
                        or (root / "_cancel.json").exists()
                        or (root.parent / "cancel.json").exists()
                    )
                    if canceled or time.time() >= plan["deadline"]:
                        stop_child(child)
                        receipt["state"] = "CANCELED" if canceled else "ERROR"
                        receipt["reason"] = (
                            "cancel requested" if canceled else "timeout"
                        )
                        break
                    time.sleep(0.1)
                if receipt["state"] != "RUNNING":
                    break
                if child.returncode:
                    receipt.update(
                        state="ERROR",
                        exit_code=child.returncode,
                        reason="command failed",
                    )
                    break
            else:
                receipt.update(state="COMPLETE", exit_code=0)
    except BaseException as exc:  # noqa: BLE001 -- persist terminal evidence even for worker interruption
        if child and child.poll() is None:
            stop_child(child)
        # Exception text could contain credentials supplied by a user command.
        receipt.update(state="ERROR", reason=type(exc).__name__)
    finally:
        receipt["finished_at"] = time.time()
        atomic(root / "_receipt.json", receipt)


if __name__ == "__main__":
    main()
