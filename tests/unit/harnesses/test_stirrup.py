"""The stirrup shell tool returns when bash exits, even if a background job holds its
pipes."""

import asyncio
import importlib.util
import os
import re
import resource
import subprocess
import sys
import threading
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
def program(monkeypatch):
    """The program, loaded without the dependencies uv installs for it."""
    for name in DEPENDENCIES:
        parts = name.split(".")
        for i in range(1, len(parts) + 1):
            module = ".".join(parts[:i])
            monkeypatch.setitem(sys.modules, module, Dependency(module))
    sys.modules["stirrup.tools.code_backends.base"].CommandResult = CommandResult
    spec = importlib.util.spec_from_file_location("stirrup_program", PROGRAM)
    program = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(program)
    return program


@pytest.fixture
def shell(program, tmp_path):
    """The program's shell tool."""
    provider = program.RuntimeCodeExecToolProvider(str(tmp_path))
    provider._shell_timeout = 30  # Set by Stirrup's base class.
    return provider


@pytest.fixture
def length():
    """A sleep length unique to the test run, to find and stop its background job."""
    value = f"{1000 + uuid.uuid4().int % 10**8 / 10**6:.6f}"
    yield value
    subprocess.run(["pkill", "-f", sleep_pattern(value)], check=False)


def sleep_pattern(length: str) -> str:
    # The whole command line, not a prefix.
    return f"^sleep {re.escape(length)}$"


def running(length: str) -> bool:
    pgrep = ["pgrep", "-f", sleep_pattern(length)]
    return subprocess.run(pgrep, capture_output=True, check=False).returncode == 0


def open_fds() -> int:
    return len(os.listdir("/proc/self/fd" if sys.platform == "linux" else "/dev/fd"))


async def timed(shell, command: str, timeout=None) -> tuple[float, CommandResult]:
    start = time.monotonic()
    result = await shell.run_command(command, timeout=timeout)
    return time.monotonic() - start, result


@pytest.mark.parametrize(
    "command",
    [
        # bash forks a subshell for the backgrounded && list, which keeps the pipes.
        "cd /tmp && nohup sleep {s} >/dev/null 2>&1 &",
        "sleep {s} &",
        "nohup sleep {s} &",
        "echo hi; sleep {s} >/dev/null &",
    ],
)
async def test_background_job_does_not_hold_the_call_open(
    program, shell, command, length
):
    elapsed, result = await timed(shell, command.format(s=length))
    assert program.OUTPUT_GRACE <= elapsed < 2
    assert result.exit_code == 0 and result.error_kind is None
    assert result.stderr == program.BACKGROUND_NOTE
    assert running(length)


async def test_output_written_before_bash_exits_is_complete(program, shell, length):
    size = 300_000
    result = await shell.run_command(
        f"head -c {size} /dev/zero | tr '\\0' a; head -c {size} /dev/zero | tr '\\0' b >&2;"
        f" sleep {length} &"
    )
    assert result.stdout == "a" * size
    assert result.stderr == "b" * size + "\n" + program.BACKGROUND_NOTE


async def test_output_written_after_bash_exits_is_not_shown(program, shell):
    elapsed, result = await timed(shell, "echo fg; (sleep 2; echo bg) &")
    assert result.stdout == "fg\n"
    assert result.stderr == program.BACKGROUND_NOTE
    assert program.OUTPUT_GRACE <= elapsed < 1.5


@pytest.mark.parametrize(
    ("command", "stdout"),
    [
        ("echo hello > >(cat)", "hello\n"),
        ("seq 1 2000 > >(grep -c .)", "2000\n"),
        ("exec > >(tee -a run.log) 2>&1; echo A", "A\n"),
        ("(sleep 0.1; echo late) &", "late\n"),
    ],
)
async def test_output_of_a_process_that_ends_just_after_bash_is_kept(
    program, shell, command, stdout
):
    elapsed, result = await timed(shell, command)
    assert (result.exit_code, result.stdout, result.stderr) == (0, stdout, "")
    assert elapsed < program.OUTPUT_GRACE


async def test_a_plain_command_does_not_wait_for_the_grace(program, shell):
    elapsed, result = await timed(shell, "echo hi")
    assert result.stdout == "hi\n"
    assert elapsed < program.OUTPUT_GRACE / 2


async def test_a_process_writing_after_the_call_is_not_blocked_or_killed(
    program, shell, tmp_path
):
    # A closed pipe would fail the late write, and an unread one would block it.
    result = await shell.run_command(
        "echo fg; (sleep 0.8; head -c 3000000 /dev/zero; echo $? > rc) &"
    )
    assert result.stdout == "fg\n"
    assert result.stderr == program.BACKGROUND_NOTE
    for _ in range(100):
        if (tmp_path / "rc").exists():
            break
        await asyncio.sleep(0.1)
    assert (tmp_path / "rc").read_text() == "0\n"


async def test_wait_still_waits_for_background_jobs(shell):
    elapsed, result = await timed(shell, "(sleep 1; echo late) & wait; echo done")
    assert result.stdout == "late\ndone\n"
    assert result.stderr == ""
    assert elapsed >= 1


async def test_closing_both_pipes_does_not_end_the_call(shell, tmp_path):
    elapsed, result = await timed(
        shell, "echo early; exec >/dev/null 2>&1; sleep 1; touch marker"
    )
    assert elapsed >= 1
    assert (tmp_path / "marker").exists()
    assert (result.exit_code, result.stdout, result.stderr) == (0, "early\n", "")


async def test_timeout_keeps_the_output_so_far_and_stops_the_command(shell, length):
    elapsed, result = await timed(
        shell, f"echo partial; echo oops >&2; sleep {length}", timeout=1
    )
    assert elapsed < 3
    assert result.exit_code == 124
    assert result.stdout == "partial\n"
    assert result.stderr == "oops\nCommand timed out after 1 seconds"
    assert not running(length)


async def test_timeout_closes_a_pipe_an_escaped_process_holds(
    program, shell, length, monkeypatch
):
    # setsid escapes the timeout's group kill; its pipe is closed, not drained.
    drained = []
    monkeypatch.setattr(program, "_discard_until_eof", drained.append)
    escape = f"import os; os.setsid(); os.execvp('sleep', ['sleep', '{length}'])"
    result = await shell.run_command(
        f'echo partial; python3 -c "{escape}" & sleep {length}', timeout=1
    )
    assert result.exit_code == 124
    assert result.stdout == "partial\n"
    assert drained == []


def test_a_non_blocking_pipe_is_read_to_its_end(program):
    # An event loop such as uvloop can leave the fd it watched non-blocking.
    read_fd, write_fd = os.pipe()
    os.set_blocking(read_fd, False)
    reader = threading.Thread(target=program._discard_until_eof, args=(read_fd,))
    reader.start()
    time.sleep(0.2)
    with open(write_fd, "wb", buffering=0) as pipe:
        pipe.write(b"x" * 300_000)  # More than a pipe holds; EPIPE once it is closed.
    reader.join(5)
    assert not reader.is_alive()


async def test_running_out_of_file_descriptors_is_a_tool_result(
    program, shell, monkeypatch
):
    pipe = os.pipe
    pipes = []

    def second_pipe_fails():
        if pipes:
            raise OSError(24, "Too many open files")
        pipes.append(pipe())
        return pipes[-1]

    fds = open_fds()
    monkeypatch.setattr(program.os, "pipe", second_pipe_fails)
    result = await shell.run_command("echo hi")
    monkeypatch.undo()
    assert result.error_kind == "execution_error"
    assert "Too many open files" in result.stderr
    assert open_fds() == fds


async def test_a_thread_that_fails_to_start_still_returns_the_result(
    program, shell, length, monkeypatch
):
    class Thread(threading.Thread):
        def start(self):
            raise RuntimeError("can't start new thread")

    fds = open_fds()
    # Only the program's threads: asyncio may need one to watch the process.
    monkeypatch.setattr(program, "threading", types.SimpleNamespace(Thread=Thread))
    result = await shell.run_command(f"echo hi; sleep {length} &")
    assert (result.exit_code, result.stdout) == (0, "hi\n")
    assert result.stderr == program.BACKGROUND_NOTE
    assert open_fds() == fds


async def test_cancelling_twice_kills_the_command_and_closes_its_pipes(shell, length):
    fds = open_fds()
    for _ in range(5):
        task = asyncio.create_task(
            shell.run_command(f"sleep {length} & sleep {length}")
        )
        await asyncio.sleep(0.1)
        task.cancel()
        await asyncio.sleep(0)  # The second cancel lands in the cleanup.
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    await asyncio.sleep(0.2)
    assert open_fds() == fds
    assert not running(length)


async def test_cancelling_during_the_grace_kills_a_job_the_command_left(
    program, shell, length
):
    fds, threads = open_fds(), threading.active_count()
    task = asyncio.create_task(shell.run_command(f"echo hi; sleep {length} &"))
    await asyncio.sleep(program.OUTPUT_GRACE / 2)  # Bash has exited by now.
    assert running(length) and not task.done()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    await asyncio.sleep(0.2)
    assert not running(length)
    assert open_fds() == fds
    assert threading.active_count() == threads


@pytest.fixture
def flood():
    """A word unique to the test, for `yes` to flood a pipe with until it is stopped."""
    value = f"flood-{uuid.uuid4().hex}"
    yield value
    subprocess.run(["pkill", "-f", f"yes {value}"], check=False)


def kept(program, size: int) -> str:
    """What a call keeps of `size` bytes of `seq` output on a stream."""
    half = program.MAX_KEPT_BYTES // 2
    output = "".join(f"{i}\n" for i in range(1, 300_000))[:size]
    if size <= program.MAX_KEPT_BYTES:
        return output
    head, tail = output[:half], output[-half:]
    # The marker is on a line of its own; a newline at the cut is not doubled.
    before = "" if head.endswith("\n") else "\n"
    after = "" if tail.startswith("\n") else "\n"
    omitted = size - 2 * half
    unit = "byte" if omitted == 1 else "bytes"
    return f"{head}{before}[... {omitted} {unit} of output omitted ...]{after}{tail}"


@pytest.mark.parametrize("over", [-1, 0, 1, 2, 100_000])
async def test_output_beyond_the_bound_keeps_its_head_and_tail(program, shell, over):
    size = program.MAX_KEPT_BYTES + over
    result = await shell.run_command(
        f"seq 1 300000 | head -c {size}; seq 1 300000 | head -c {size} >&2"
    )
    assert result.stdout == kept(program, size)
    assert result.stderr == kept(program, size)


async def test_a_character_across_the_middle_of_the_bound_is_kept(program, shell):
    # Within the bound, the first and last halves are decoded together.
    n = program.MAX_KEPT_BYTES // 2 - 1
    result = await shell.run_command(
        f"head -c {n} /dev/zero | tr '\\0' a; printf '\\xc3\\xa9';"
        f" head -c {n} /dev/zero | tr '\\0' b"
    )
    assert result.stdout == "a" * n + "\u00e9" + "b" * n


async def test_invalid_utf8_is_replaced_not_dropped(shell):
    result = await shell.run_command("printf 'a\\377b'; printf 'c\\376d' >&2")
    assert (result.stdout, result.stderr) == ("a\ufffdb", "c\ufffdd")


HALF = 1 << 19  # MAX_KEPT_BYTES // 2, checked below.


@pytest.mark.parametrize(
    ("output", "text"),
    [
        (
            b"a" * (2 * HALF + 1),
            f"{'a' * HALF}\n[... 1 byte of output omitted ...]\n{'a' * HALF}",
        ),
        (
            b"a" * (HALF - 1) + "\u00e9".encode() + b"b" * HALF,
            f"{'a' * (HALF - 1)}\n[... 2 bytes of output omitted ...]\n{'b' * HALF}",
        ),
        (
            b"a" * HALF + b"c" + "\u20ac".encode() + b"b" * (HALF - 2),
            f"{'a' * HALF}\n[... 4 bytes of output omitted ...]\n{'b' * (HALF - 2)}",
        ),
        (
            b"a" * (HALF - 2) + "\U0001f600".encode() + b"b" * (HALF + 1),
            f"{'a' * (HALF - 2)}\n[... 5 bytes of output omitted ...]\n{'b' * HALF}",
        ),
        (
            b"\xff" + b"a" * (2 * HALF),
            f"\ufffd{'a' * (HALF - 1)}\n[... 1 byte of output omitted ...]\n{'a' * HALF}",
        ),
        (
            b"a" * (HALF - 2) + b"\r\n-\r\n" + b"b" * (HALF - 2),
            f"{'a' * (HALF - 2)}\r\n[... 1 byte of output omitted ...]\r\n{'b' * (HALF - 2)}",
        ),
        (
            b"a" * HALF + b"\r\n" + b"b" * (HALF - 1),
            f"{'a' * HALF}\n[... 1 byte of output omitted ...]\n{'b' * (HALF - 1)}",
        ),
    ],
    # Short ids: generated ones would hold the megabyte inputs.
    ids=[
        "one-byte-over",
        "character-at-head-cut",
        "character-at-tail-cut",
        "emoji-at-head-cut",
        "invalid-byte-replaced",
        "line-breaks-at-cut",
        "crlf-split-at-cut",
    ],
)
async def test_the_cut_keeps_whole_characters_and_lines(program, output, text):
    assert program.MAX_KEPT_BYTES == 2 * HALF
    pipe = program.OutputPipe()
    for i in range(0, len(output), program.PIPE_READ_SIZE):
        pipe.keep(output[i : i + program.PIPE_READ_SIZE])
    assert pipe.finish(drain=False) == text


async def test_finish_stops_reading_a_pipe_still_held_open(program):
    loop = asyncio.get_running_loop()
    pipe = program.OutputPipe()
    fd = pipe.fd
    held = os.dup(pipe.write_fd)  # As by a process the command left running.
    pipe.close_write_end()
    try:
        pipe.finish(drain=False)
        assert pipe.held_open
        assert not loop.remove_reader(fd)  # The loop no longer watches the fd.
    finally:
        os.close(held)


async def test_a_pipe_holds_a_bounded_amount_of_output(program):
    pipe = program.OutputPipe()
    for _ in range(64):
        pipe.keep(b"x" * program.PIPE_READ_SIZE)
    assert len(pipe.head) + len(pipe.tail) <= 2 * program.MAX_KEPT_BYTES
    assert len(pipe.finish()) <= program.MAX_KEPT_BYTES + 100


@pytest.mark.parametrize(
    ("command", "timeout"),
    [
        # Left running, `yes` writes to the pipes as fast as they are read.
        ("yes {w} & yes {w} >&2 & echo started", None),
        ("echo started; yes {w}", 1),
    ],
)
async def test_a_flood_of_output_is_not_all_kept(
    program, shell, flood, command, timeout
):
    elapsed, result = await timed(shell, command.format(w=flood), timeout=timeout)
    bound = program.MAX_KEPT_BYTES + 100  # With the omission marker.
    assert elapsed < 2 + (timeout or 0)
    assert result.stdout.startswith("started\n")
    assert "bytes of output omitted" in result.stdout
    assert len(result.stdout) <= bound
    if timeout is None:
        assert result.exit_code == 0
        # The flood is cut off mid-line, and the note follows on a line of its own.
        assert result.stderr.startswith(flood)
        assert result.stderr.endswith(f"\n{program.BACKGROUND_NOTE}")
        assert len(result.stderr) <= bound + len(program.BACKGROUND_NOTE)
    else:
        assert result.exit_code == 124
        assert result.stderr == "Command timed out after 1 seconds"


async def test_foreground_result_is_unchanged(shell):
    result = await shell.run_command("echo a; echo b >&2; exit 3")
    assert (result.exit_code, result.stdout, result.stderr) == (3, "a\n", "b\n")


async def test_background_job_works_with_many_open_files(program, shell, length):
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    if hard != resource.RLIM_INFINITY and hard < 1200:
        pytest.skip("the hard limit on open files is below 1200")
    resource.setrlimit(resource.RLIMIT_NOFILE, (max(soft, 1200), hard))
    fds = [os.open(os.devnull, os.O_RDONLY) for _ in range(1100)]
    try:
        result = await shell.run_command(f"echo fg; sleep {length} &")
        read_fd, write_fd = os.pipe()
        os.close(write_fd)
        try:
            closed = program._at_end(read_fd)
        finally:
            os.close(read_fd)
    finally:
        for fd in fds:
            os.close(fd)
        resource.setrlimit(resource.RLIMIT_NOFILE, (soft, hard))
    assert result.stdout == "fg\n"
    assert result.stderr == program.BACKGROUND_NOTE
    assert read_fd >= 1024 and closed
