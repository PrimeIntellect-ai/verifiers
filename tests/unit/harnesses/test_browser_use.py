"""The browser tool returns when the model's program exits, even if a child process it
left running holds the program's pipes."""

import os
import subprocess
import time
import uuid

import pytest

from verifiers.v1.harnesses.browser_use.program import HELD_OUTPUT_NOTE, run_browser


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
    assert elapsed < 5
    assert out in results
    running = subprocess.run(
        ["pgrep", "-f", f"sleep {length}"], capture_output=True, check=False
    )
    assert running.returncode == 0


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


def test_leftover_child_keeps_running_after_the_call(env, tmp_path):
    done = tmp_path / "done"
    code = (
        "import subprocess\n"
        f"subprocess.Popen(['sh', '-c', 'sleep 0.5; echo late; touch {done}'])\n"
    )
    assert run_browser(code, env) == HELD_OUTPUT_NOTE
    # Its pipes are read to the end after the call, so its late write succeeds.
    deadline = time.monotonic() + 5
    while not done.exists() and time.monotonic() < deadline:
        time.sleep(0.05)
    assert done.exists()


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


def test_program_that_closes_its_output_still_runs_to_the_end(env, tmp_path):
    done = tmp_path / "done"
    code = (
        "import os, time\n"
        "devnull = os.open(os.devnull, os.O_WRONLY)\n"
        "os.dup2(devnull, 1); os.dup2(devnull, 2)\n"
        "time.sleep(1)\n"
        f"open({str(done)!r}, 'w').close()\n"
    )
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
