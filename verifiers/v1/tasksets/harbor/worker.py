# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""Runtime-side Worker transport and execution; never holds provider credentials."""

import base64
import json
import os
import runpy
import sys
import tarfile
import tempfile
import threading
from pathlib import Path


def install() -> None:
    from rewardkit.sandbox import (
        Worker,
        WorkerError,
        WorkerResult,
        WorkerSetupError,
        _copy_in,
        _validate,
    )

    # Reserve stdout for requests, including when criterion code writes directly
    # to fd 1. Serialize Worker calls so each response belongs to its caller.
    sys.stdout.flush()
    wire = os.fdopen(os.dup(1), "w", buffering=1)
    os.dup2(2, 1)
    lock = threading.Lock()

    def run(self, script, *, files=(), stdin=b"", outputs=(), timeout_sec=10):
        _validate(script, files, outputs, timeout_sec)
        with lock, tempfile.TemporaryDirectory(prefix="vf-worker-") as directory:
            root = Path(directory)
            inputs = root / "files"
            inputs.mkdir()
            try:
                _copy_in(files, inputs, os.geteuid())
            except (OSError, ValueError) as exc:
                raise WorkerError(f"Cannot copy in submission: {exc}") from exc
            (root / "stdin").write_bytes(stdin)
            (root / "request.json").write_text(
                json.dumps(
                    {
                        "script": script,
                        "outputs": list(outputs),
                        "timeout_sec": timeout_sec,
                        "env": dict(self.env),
                        "private_paths": [str(p) for p in self.private_paths],
                        "max_output_bytes": self.max_output_bytes,
                    }
                )
            )
            with tarfile.open(root / "request.tar", "w") as archive:
                for name in ("files", "stdin", "request.json"):
                    archive.add(root / name, arcname=name)
            wire.write(json.dumps({"path": directory}) + "\n")
            if not sys.stdin.readline():
                raise WorkerSetupError("Native Worker executor disconnected")
            response = json.loads((root / "result.json").read_text())
            if "error" in response:
                error = {
                    "WorkerError": WorkerError,
                    "WorkerSetupError": WorkerSetupError,
                }[response["type"]](response["error"])
                if isinstance(error, WorkerError):
                    error.stderr = response["stderr"]
                raise error
            return WorkerResult(
                response["returncode"],
                base64.b64decode(response["stdout"], validate=True),
                response["stderr"],
                {
                    name: base64.b64decode(data, validate=True)
                    for name, data in response["files"].items()
                },
            )

    Worker.run = run


def main() -> None:
    # Use the pinned upstream implementation for command semantics, safe copying,
    # output reads and process cleanup, inside an otherwise disposable Prime VM.
    root = Path(sys.argv[1])
    sandbox = runpy.run_path(str(root / "sandbox.py"))
    with tarfile.open(root / "request.tar") as archive:
        archive.extractall(root / "input", filter="data")
    request = json.loads((root / "input/request.json").read_text())
    try:
        result = sandbox["Worker"](
            env=request["env"],
            private_paths=tuple(Path(p) for p in request["private_paths"]),
            max_output_bytes=request["max_output_bytes"],
        ).run(
            request["script"],
            files=list((root / "input/files").iterdir()),
            stdin=(root / "input/stdin").read_bytes(),
            outputs=request["outputs"],
            timeout_sec=request["timeout_sec"],
        )
        response = {
            "returncode": result.returncode,
            "stdout": base64.b64encode(result.stdout).decode(),
            "stderr": result.stderr,
            "files": {
                name: base64.b64encode(data).decode()
                for name, data in result.files.items()
            },
        }
    except (sandbox["WorkerError"], sandbox["WorkerSetupError"]) as exc:
        response = {"type": type(exc).__name__, "error": str(exc)}
        if isinstance(exc, sandbox["WorkerError"]):
            response["stderr"] = exc.stderr
    (root / "result.json").write_text(json.dumps(response))


if __name__ == "__main__":
    main()
