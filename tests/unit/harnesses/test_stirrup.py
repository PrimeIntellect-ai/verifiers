"""The stirrup shell tool returns when bash exits, even if a background job holds its
pipes."""

import importlib.util
import subprocess
import sys
import time
import types
import uuid
from dataclasses import dataclass
from pathlib import Path

import pytest

from verifiers.v1.harnesses.stirrup import harness

PROGRAM = Path(harness.__file__).with_name("program.py")
DEPENDENCIES = (
    "acp.schema",
    "stirrup.clients.chat_completions_client",
    "stirrup.core.agent",
    "stirrup.core.exceptions",
    "stirrup.core.models",
    "stirrup.tools.code_backends.base",
    "stirrup.tools.mcp",
)


class Dependency(types.ModuleType):
    """Stands in for a module the program installs with uv; any name is a class."""

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        return type(name, (), {})


@dataclass
class CommandResult:
    exit_code: int
    stdout: str
    stderr: str
    error_kind: str | None = None


@pytest.fixture
def shell(monkeypatch, tmp_path):
    """The program's shell tool, loaded without the dependencies uv installs for it."""
    for name in DEPENDENCIES:
        parts = name.split(".")
        for i in range(1, len(parts) + 1):
            module = ".".join(parts[:i])
            monkeypatch.setitem(sys.modules, module, Dependency(module))
    sys.modules["stirrup.tools.code_backends.base"].CommandResult = CommandResult
    spec = importlib.util.spec_from_file_location("stirrup_program", PROGRAM)
    program = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(program)
    provider = program.RuntimeCodeExecToolProvider(str(tmp_path))
    provider._shell_timeout = 30  # Set by Stirrup's base class.
    return provider


@pytest.fixture
def length():
    """A sleep length unique to the test, to find and stop its background job."""
    value = f"{300 + uuid.uuid4().int % 1000 / 1000:.3f}"
    yield value
    subprocess.run(["pkill", "-f", f"sleep {value}"], check=False)


def running(length: str) -> bool:
    pgrep = ["pgrep", "-f", f"sleep {length}"]
    return subprocess.run(pgrep, capture_output=True, check=False).returncode == 0


async def timed(shell, command: str, timeout=None) -> tuple[float, CommandResult]:
    start = time.monotonic()
    result = await shell.run_command(command, timeout=timeout)
    return time.monotonic() - start, result


@pytest.mark.parametrize(
    "command",
    [
        # `&` puts the whole && list in the background, and bash's subshell for the
        # list keeps the pipes although the job's own output is redirected.
        "cd /tmp && nohup sleep {s} >/dev/null 2>&1 &",
        "sleep {s} &",
        "nohup sleep {s} &",
        "echo hi; sleep {s} >/dev/null &",
    ],
)
async def test_background_job_does_not_hold_the_call_open(shell, command, length):
    elapsed, result = await timed(shell, command.format(s=length))
    assert elapsed < 2
    assert result.exit_code == 0 and result.error_kind is None
    assert running(length)


async def test_output_written_before_bash_exits_is_complete(shell, length):
    size = 300_000
    result = await shell.run_command(
        f"head -c {size} /dev/zero | tr '\\0' a; head -c {size} /dev/zero | tr '\\0' b >&2;"
        f" sleep {length} &"
    )
    assert result.stdout == "a" * size
    assert result.stderr == "b" * size


async def test_output_written_after_bash_exits_is_not_shown(shell):
    elapsed, result = await timed(shell, "echo fg; (sleep 1; echo bg) &")
    assert result.stdout == "fg\n"
    assert elapsed < 1


async def test_wait_still_waits_for_background_jobs(shell):
    elapsed, result = await timed(shell, "(sleep 1; echo late) & wait; echo done")
    assert result.stdout == "late\ndone\n"
    assert elapsed >= 1


async def test_timeout_keeps_the_output_so_far_and_stops_the_command(shell, length):
    elapsed, result = await timed(
        shell, f"echo partial; echo oops >&2; sleep {length}", timeout=1
    )
    assert elapsed < 3
    assert result.exit_code == 124
    assert result.stdout == "partial\n"
    assert result.stderr == "oops\nCommand timed out after 1 seconds"
    assert not running(length)


async def test_foreground_result_is_unchanged(shell):
    result = await shell.run_command("echo a; echo b >&2; exit 3")
    assert (result.exit_code, result.stdout, result.stderr) == (3, "a\n", "b\n")
