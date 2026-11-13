"""The local bash tool returns when the command exits, even if a child it started with `&`
still holds the output; it kills the whole process group on timeout; and `background`
returns at once with a PID and a log file."""

import os
import re
import time

from verifiers.v1.harnesses.utils.core import run_bash


def alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def test_returns_combined_output():
    assert run_bash("echo out; echo err >&2") == "out\nerr\n"


def test_ampersand_child_does_not_block():
    start = time.monotonic()
    output = run_bash("sleep 30 & echo $!")
    assert time.monotonic() - start < 5
    child = int(output.strip())
    assert alive(child)
    os.kill(child, 9)


def test_timeout_kills_process_group():
    start = time.monotonic()
    output = run_bash("sleep 30 & echo $!; sleep 30", timeout=1)
    assert time.monotonic() - start < 5
    assert "killed after 1s timeout" in output
    child = int(output.splitlines()[0])
    for _ in range(50):
        if not alive(child):
            break
        time.sleep(0.1)
    assert not alive(child)


def test_background_returns_pid_and_log():
    start = time.monotonic()
    output = run_bash("echo ready; sleep 30", background=True)
    assert time.monotonic() - start < 5
    match = re.search(r"pid (\d+), output in (\S+)\.", output)
    assert match is not None
    pid, log = int(match.group(1)), match.group(2)
    for _ in range(50):
        if open(log).read() == "ready\n":
            break
        time.sleep(0.1)
    assert open(log).read() == "ready\n"
    os.killpg(pid, 9)
    os.unlink(log)
