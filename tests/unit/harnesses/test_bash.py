"""The bash tool returns when bash exits, even if a background job holds its pipes."""

import os
import resource
import subprocess
import threading
import time
import uuid

import pytest

from verifiers.v1.harnesses.utils.launch import CHAT_PROGRAM_SOURCE

# The real chat program, with compaction bundled into core as the harness runs it.
PROGRAM = {"__name__": "chat_program"}
exec(  # noqa: S102 - the bundled program is this repo's own code
    compile(CHAT_PROGRAM_SOURCE, "chat_program.py", "exec"), PROGRAM
)
NOTE = PROGRAM["BACKGROUND_NOTE"]


@pytest.fixture
def length():
    """A sleep length unique to the test, to find and stop its background job."""
    value = f"{300 + uuid.uuid4().int % 1000 / 1000:.3f}"
    yield value
    subprocess.run(["pkill", "-f", f"sleep {value}"], check=False)


def running(pattern: str) -> bool:
    found = subprocess.run(["pgrep", "-f", pattern], capture_output=True, check=False)
    return found.returncode == 0


def timed(command: str) -> tuple[float, str]:
    start = time.monotonic()
    out = PROGRAM["run_bash"](command)
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
    assert 0.5 <= elapsed < 2
    assert out == NOTE
    assert running(f"sleep {length}")


@pytest.mark.parametrize(
    "command, expected",
    [
        ("echo hello > >(sleep 0.2; cat)", "hello\n"),
        ("exec > >(sleep 0.2; cat) 2>&1; echo ok", "ok\n"),
        ("echo now; (sleep 0.2; echo later) &", "now\nlater\n"),
    ],
)
def test_output_of_processes_bash_does_not_wait_for_is_kept(command, expected):
    elapsed, out = timed(command)
    assert out == expected
    assert elapsed < 0.5


def test_output_bounded_with_the_note(length):
    size = 300_000
    out = PROGRAM["run_bash"](
        f"head -c {size} /dev/zero | tr '\\0' a; head -c {size} /dev/zero | tr '\\0' b >&2;"
        f" sleep {length} &"
    )
    assert f"[... bash output truncated; {2 * size} bytes total ...]" in out
    assert out.startswith("a") and out.endswith("b\n" + NOTE)
    message = {"role": "tool", "content": out}
    assert PROGRAM["bound_tool_message"](message) == message


def test_job_writing_after_the_call_runs_to_completion(tmp_path):
    rc = tmp_path / "rc"
    elapsed, out = timed(
        f"echo fg; (sleep 0.8; head -c 3000000 /dev/zero; echo $? > {rc}) &"
    )
    assert out == "fg\n" + NOTE
    assert elapsed < 0.8
    for _ in range(100):
        if rc.exists() and rc.read_text():
            break
        time.sleep(0.1)
    assert (
        rc.read_text() == "0\n"
    )  # neither blocked on a full pipe nor killed by SIGPIPE


def test_timeout_kills_the_command(monkeypatch, length):
    marker = f"y{uuid.uuid4().hex[:8]}"
    writer = f"import os; os.setsid(); os.execvp('yes', ['yes', '{marker}'])"
    monkeypatch.setitem(PROGRAM, "BASH_TIMEOUT", 1)
    try:
        _, out = timed(f'python3 -c "{writer}" & sleep {length}; true')
        assert out.endswith("timed out after 1 seconds")
        time.sleep(0.5)
        # The foreground child dies with bash's process group; a writer that left the
        # group dies on the closed pipe instead of being drained.
        assert not running(f"sleep {length}")
        assert not running(f"yes {marker}")
    finally:
        subprocess.run(["pkill", "-f", f"yes {marker}"], check=False)


def test_job_holding_fds_above_1024(length):
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (min(4096, hard), hard))
    held = [os.open(os.devnull, os.O_RDONLY) for _ in range(1100)]
    try:
        assert PROGRAM["run_bash"](f"echo hi; sleep {length} &") == "hi\n" + NOTE
    finally:
        for fd in held:
            os.close(fd)
        resource.setrlimit(resource.RLIMIT_NOFILE, (soft, hard))


def test_failed_cleanup_keeps_the_result(monkeypatch, length):
    def fail(self):
        raise RuntimeError("can't start new thread")

    monkeypatch.setattr(threading.Thread, "start", fail)
    assert PROGRAM["run_bash"](f"echo hi; sleep {length} &") == "hi\n" + NOTE


def test_wait_still_waits_for_background_jobs():
    elapsed, out = timed("(sleep 1; echo late) & wait; echo done")
    assert out == "late\ndone\n"
    assert elapsed >= 1


def test_foreground_output_and_order_are_unchanged():
    elapsed, out = timed("echo a; echo b >&2; echo c")
    assert out == "a\nc\nb\n"
    assert elapsed < 0.4
