"""The bash tool returns when bash exits, even if a background job holds its pipes."""

import subprocess
import time
import uuid

import pytest

from verifiers.v1.harnesses.utils import compaction, core
from verifiers.v1.harnesses.utils.core import BACKGROUND_NOTE, run_bash


@pytest.fixture(autouse=True)
def bundled(monkeypatch):
    """The chat program bundles compaction's names into core; a plain import does not."""
    monkeypatch.setattr(
        core, "TOOL_OUTPUT_MAX_BYTES", compaction.TOOL_OUTPUT_MAX_BYTES, raising=False
    )


@pytest.fixture
def length():
    """A sleep length unique to the test, to find and stop its background job."""
    value = f"{300 + uuid.uuid4().int % 1000 / 1000:.3f}"
    yield value
    subprocess.run(["pkill", "-f", f"sleep {value}"], check=False)


def timed(command: str) -> tuple[float, str]:
    start = time.monotonic()
    out = run_bash(command)
    return time.monotonic() - start, out


@pytest.mark.parametrize(
    "command",
    [
        # `&` puts the whole && list in the background, and bash's subshell for the
        # list keeps the pipes although the job's own output is redirected.
        "cd /tmp && nohup sleep {s} >/dev/null 2>&1 &",
        "sleep {s} &",
        "nohup sleep {s} &",
        "sleep {s} >/dev/null &",
    ],
)
def test_background_job_does_not_hold_the_call_open(command, length):
    elapsed, out = timed(command.format(s=length))
    assert elapsed < 2
    assert out == BACKGROUND_NOTE
    running = subprocess.run(
        ["pgrep", "-f", f"sleep {length}"], capture_output=True, check=False
    )
    assert running.returncode == 0


def test_output_written_before_bash_exits_is_complete(length):
    size = 300_000
    out = run_bash(
        f"head -c {size} /dev/zero | tr '\\0' a; head -c {size} /dev/zero | tr '\\0' b >&2;"
        f" sleep {length} &"
    )
    assert f"[... bash output truncated; {2 * size} bytes total ...]" in out
    assert out.startswith("a") and out.endswith("b\n" + BACKGROUND_NOTE)


def test_output_written_after_bash_exits_is_not_shown():
    elapsed, out = timed("echo fg; (sleep 1; echo bg) &")
    assert out == "fg\n" + BACKGROUND_NOTE
    assert elapsed < 1


def test_wait_still_waits_for_background_jobs():
    elapsed, out = timed("(sleep 1; echo late) & wait; echo done")
    assert out == "late\ndone\n"
    assert elapsed >= 1


def test_foreground_output_and_order_are_unchanged():
    assert run_bash("echo a; echo b >&2; echo c") == "a\nc\nb\n"
