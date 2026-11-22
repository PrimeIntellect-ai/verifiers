"""The browser tool returns when the model's program exits, even if a child process it
left running holds the program's pipes."""

import os
import resource
import subprocess
import sys
import time
import uuid

import pytest

from verifiers.v1.harnesses.browser_use import program
from verifiers.v1.harnesses.browser_use.program import (
    EXIT_GRACE,
    HELD_OUTPUT_NOTE,
    MAX_KEPT_BYTES,
    run_browser,
)


@pytest.fixture
def env(tmp_path):
    """A stand-in for browser-harness's CLI that runs the code on stdin, so no browser
    is needed."""
    package = tmp_path / "browser_harness"
    package.mkdir()
    (package / "__init__.py").write_text("")
    (package / "run.py").write_text("import sys\nexec(sys.stdin.read())\n")
    return {**os.environ, "PYTHONPATH": str(tmp_path)}


@pytest.fixture
def length():
    """A sleep length unique to the test, to find and stop its leftover child."""
    value = f"{300 + uuid.uuid4().int % 1000 / 1000:.3f}"
    yield value
    subprocess.run(["pkill", "-f", f"sleep {value}"], check=False)


def timed(code: str, env: dict[str, str]) -> tuple[float, str]:
    start = time.monotonic()
    out = run_browser(code, env)
    return time.monotonic() - start, out


def running(pattern: str) -> bool:
    found = subprocess.run(["pgrep", "-f", pattern], capture_output=True, check=False)
    return found.returncode == 0


HELD = {HELD_OUTPUT_NOTE}


@pytest.mark.parametrize(
    "code, results",
    [
        ('import os; os.system("sleep {s} &")', HELD),
        ('import subprocess; subprocess.Popen(["sleep", "{s}"])', HELD),
        # bash's subshell for the && list keeps the pipes; dash execs `nohup` in it,
        # and the redirects then close them.
        (
            'import os; os.system("cd /tmp && nohup sleep {s} >/dev/null 2>&1 &")',
            HELD | {"(no output)"},
        ),
    ],
)
def test_leftover_child_does_not_hold_the_call_open(code, results, env, length):
    elapsed, out = timed(code.format(s=length), env)
    assert out in results
    # A held pipe is given EXIT_GRACE to close before the call returns.
    assert (EXIT_GRACE if out == HELD_OUTPUT_NOTE else 0) <= elapsed < 5
    assert running(f"sleep {length}")


def test_output_written_before_the_program_exits_is_complete(env, length):
    size = 300_000
    code = (
        "import os, sys\n"
        f"sys.stdout.write('a' * {size})\n"
        f"sys.stderr.write('b' * {size})\n"
        f"os.system('sleep {length} &')\n"
    )
    assert run_browser(code, env) == "a" * size + "b" * size + "\n" + HELD_OUTPUT_NOTE


def test_output_written_after_the_program_exits_is_not_shown(env):
    code = "import os\nprint('fg', flush=True)\nos.system('(sleep 2; echo bg) &')\n"
    elapsed, out = timed(code, env)
    assert out == "fg\n" + HELD_OUTPUT_NOTE
    assert elapsed < 2


def test_leftover_child_that_writes_after_the_call_finishes_cleanly(env, tmp_path):
    status = tmp_path / "status"
    # Writes far more than a pipe holds, after the call has returned, then records
    # whether the write succeeded (a closed pipe fails it, an unread one blocks it).
    child = (
        "import sys, time\n"
        "time.sleep(1.5)\n"
        "try:\n"
        "    sys.stdout.write('x' * 3_000_000); sys.stdout.flush(); result = 0\n"
        "except OSError as e:\n"
        "    result = e.errno\n"
        f"open({str(status)!r}, 'w').write(str(result))\n"
    )
    code = (
        f"import subprocess, sys\nsubprocess.Popen([sys.executable, '-c', {child!r}])\n"
    )
    elapsed, out = timed(code, env)
    assert out == HELD_OUTPUT_NOTE
    assert elapsed < 1.5
    deadline = time.monotonic() + 10
    while not status.exists() and time.monotonic() < deadline:
        time.sleep(0.05)
    assert status.read_text() == "0"


@pytest.mark.parametrize(
    "code",
    [
        "import subprocess\nsubprocess.Popen(['sh', '-c', 'sleep 0.05; echo late'])\n",
        # bash exits before the process substitution has written.
        "import os\nos.system(\"bash -c 'echo late > >(sleep 0.05; cat)'\")\n",
    ],
)
def test_short_lived_child_output_is_included(code, env):
    assert run_browser(code, env) == "late\n"


def test_ordinary_calls_return_their_output(env):
    assert (
        run_browser(
            "import sys; print('a'); print('b', file=sys.stderr); print('c')", env
        )
        == "a\nc\nb\n"
    )
    assert run_browser("pass", env) == "(no output)"
    # Code larger than a pipe's capacity reaches the program whole.
    assert run_browser(f"print(len({'x' * 200_000!r}))", env) == "200000\n"
    assert run_browser("raise ValueError('boom')", env).endswith("ValueError: boom\n")


CLOSE_OUTPUT = (
    "import os, time\n"
    "devnull = os.open(os.devnull, os.O_WRONLY)\n"
    "os.dup2(devnull, 1); os.dup2(devnull, 2)\n"
)
"""Code that points the program's stdout and stderr away from the pipes."""


def test_program_that_closes_its_output_still_runs_to_the_end(env, tmp_path):
    done = tmp_path / "done"
    code = CLOSE_OUTPUT + f"time.sleep(1)\nopen({str(done)!r}, 'w').close()\n"
    assert run_browser(code, env) == "(no output)"
    assert done.exists()


def test_stdout_and_stderr_newlines_are_translated_separately(env):
    # A CR ending stdout and an LF starting stderr stay two line breaks.
    code = (
        "import sys\n"
        "sys.stdout.write('50%\\r'); sys.stdout.flush()\n"
        "sys.stderr.write('\\nERR')\n"
    )
    assert run_browser(code, env) == "50%\n\nERR"


@pytest.mark.parametrize(
    "code",
    [
        "import time; time.sleep(5)",
        CLOSE_OUTPUT + "time.sleep(5)\n",
    ],
)
def test_program_running_past_the_timeout_is_an_error(code, env, monkeypatch):
    monkeypatch.setattr(program, "BROWSER_TOOL_TIMEOUT", 1)
    args = [sys.executable, "-m", "browser_harness.run"]
    elapsed, out = timed(code, env)
    assert out == f"error: Command '{args}' timed out after 1 seconds"
    assert elapsed < 3


def test_timeout_kills_the_programs_children(env, length, monkeypatch):
    monkeypatch.setattr(program, "BROWSER_TOOL_TIMEOUT", 1)
    code = (
        "import subprocess, time\n"
        f"subprocess.Popen(['sleep', '{length}'])\n"
        "time.sleep(5)\n"
    )
    elapsed, out = timed(code, env)
    assert out.startswith("error: Command ") and out.endswith(
        "timed out after 1 seconds"
    )
    assert elapsed < 3
    deadline = time.monotonic() + 2
    while running(f"sleep {length}") and time.monotonic() < deadline:
        time.sleep(0.05)
    assert not running(f"sleep {length}")


def test_pipes_on_fds_above_1024(env, length):
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    needed = 1200
    if hard != resource.RLIM_INFINITY and hard < needed:
        pytest.skip(f"RLIMIT_NOFILE hard limit {hard} < {needed}")
    resource.setrlimit(resource.RLIMIT_NOFILE, (max(soft, needed), hard))
    fds = [os.open(os.devnull, os.O_RDONLY) for _ in range(1024)]
    try:
        assert max(fds) >= 1024
        out = run_browser(
            f"import os; print('hi', flush=True); os.system('sleep {length} &')", env
        )
    finally:
        for fd in fds:
            os.close(fd)
        resource.setrlimit(resource.RLIMIT_NOFILE, (soft, hard))
    assert out == "hi\n" + HELD_OUTPUT_NOTE


def test_cleanup_failure_still_returns_the_result(env, length, monkeypatch):
    def fail(self):
        raise RuntimeError("can't start new thread")

    monkeypatch.setattr(program.threading.Thread, "start", fail)
    before = len(os.listdir("/dev/fd"))
    out = run_browser(
        f"import os; print('hi', flush=True); os.system('sleep {length} &')", env
    )
    assert out == "hi\n" + HELD_OUTPUT_NOTE
    # The fd duplicated for the drain thread is closed again.
    assert len(os.listdir("/dev/fd")) == before


def test_output_up_to_the_bound_is_kept_whole(env):
    code = (
        "import sys\n"
        f"sys.stdout.write('a' * {MAX_KEPT_BYTES})\n"
        f"sys.stderr.write('b' * {MAX_KEPT_BYTES})\n"
    )
    assert run_browser(code, env) == "a" * MAX_KEPT_BYTES + "b" * MAX_KEPT_BYTES


def test_output_above_the_bound_keeps_its_head_and_tail(env):
    half = MAX_KEPT_BYTES // 2
    code = (
        "import sys\n"
        f"sys.stdout.write('h' * {half} + 'm' * 10 + 't' * {MAX_KEPT_BYTES - half})\n"
        "sys.stderr.write('err')\n"
    )
    assert run_browser(code, env) == (
        "h" * half
        + "\n[... 10 bytes of output omitted ...]\n"
        + "t" * (MAX_KEPT_BYTES - half)
        + "err"
    )


def test_child_flooding_its_output_during_the_grace_is_bounded(env):
    word = f"flood{uuid.uuid4().hex}"
    try:
        elapsed, out = timed(
            f"import subprocess; subprocess.Popen(['yes', '{word}'])", env
        )
    finally:
        subprocess.run(["pkill", "-f", f"yes {word}"], check=False)
    assert elapsed < 2
    assert out.startswith(f"{word}\n")
    assert " bytes of output omitted ...]\n" in out
    assert out.endswith(HELD_OUTPUT_NOTE)
    assert len(out.encode()) <= MAX_KEPT_BYTES + 100 + len(HELD_OUTPUT_NOTE)
