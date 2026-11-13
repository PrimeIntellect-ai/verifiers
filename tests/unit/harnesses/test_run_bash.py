"""The local bash tool returns when its command exits, even if a child started with `&`
still holds the output; a command still running at `yield_after` keeps running and the
call returns its output so far, its PID and its log file."""

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
    assert run_bash("echo out; echo err >&2", yield_after=30) == "out\nerr\n"


def test_ampersand_child_does_not_block():
    start = time.monotonic()
    output = run_bash("sleep 30 & echo $!", yield_after=30)
    assert time.monotonic() - start < 5
    child = int(output.strip())
    assert alive(child)
    os.kill(child, 9)


def test_yield_keeps_command_running():
    start = time.monotonic()
    output = run_bash("echo ready; sleep 30; echo done", yield_after=1)
    assert time.monotonic() - start < 5
    assert output.startswith("ready\n")
    match = re.search(r"pid (\d+), output continues in (\S+)\.", output)
    assert match is not None
    pid, log = int(match.group(1)), match.group(2)
    assert alive(pid)
    assert open(log).read() == "ready\n"
    os.killpg(pid, 9)
    os.unlink(log)
