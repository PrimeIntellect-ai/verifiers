"""The Harbor verifier command returns bounded output and the script's exit code."""

import subprocess

import pytest

pytest.importorskip("harbor")

from verifiers.v1.tasksets.harbor.taskset import (
    VERIFIER_OUTPUT_TAIL_BYTES,
    verifier_argv,
)


def run(tmp_path, body: str) -> subprocess.CompletedProcess:
    script = tmp_path / "test.sh"
    script.write_text(body)
    return subprocess.run(
        verifier_argv(str(script)), check=False, capture_output=True, cwd=tmp_path
    )


def test_small_output_passes_through(tmp_path):
    result = run(tmp_path, "echo out\necho err >&2\nexit 0\n")

    assert (result.returncode, result.stdout, result.stderr) == (0, b"out\n", b"err\n")


def test_large_output_returns_only_the_tail(tmp_path):
    result = run(
        tmp_path,
        "head -c 20000000 /dev/zero | tr '\\0' x; echo; echo last-out\n"
        "head -c 3000000 /dev/zero | tr '\\0' y >&2; echo last-err >&2\n"
        "exit 3\n",
    )

    assert result.returncode == 3
    assert len(result.stdout) == len(result.stderr) == VERIFIER_OUTPUT_TAIL_BYTES
    assert result.stdout.endswith(b"x\nlast-out\n")
    assert result.stderr.endswith(b"ylast-err\n")


def test_script_runs_in_the_callers_working_directory(tmp_path):
    result = run(tmp_path, "pwd\n")

    assert (result.returncode, result.stdout) == (0, f"{tmp_path}\n".encode())
