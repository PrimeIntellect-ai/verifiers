"""The bash tool returns when bash exits, even if a background job holds its pipes."""

import os
import resource
import signal
import subprocess
import sys
import threading
import time
import uuid

import pytest

from verifiers.v1.harnesses._utils.launch import CHAT_PROGRAM_SOURCE

# The real chat program, with compaction bundled into core as the harness runs it.
PROGRAM = {"__name__": "chat_program"}
exec(  # noqa: S102 - the bundled program is this repo's own code
    compile(CHAT_PROGRAM_SOURCE, "chat_program.py", "exec"), PROGRAM
)
NOTE = PROGRAM["BACKGROUND_NOTE"]


@pytest.fixture
def new_length():
    """Sleep lengths unique to the test, to find and stop its processes."""
    made = []

    def make() -> str:
        made.append(f"{300 + uuid.uuid4().int % 10**6 / 10**6:.6f}")
        return made[-1]

    yield make
    for value in made:
        subprocess.run(["pkill", "-f", value], check=False)


@pytest.fixture
def length(new_length):
    return new_length()


def running(pattern: str) -> bool:
    found = subprocess.run(["pgrep", "-f", pattern], capture_output=True, check=False)
    return found.returncode == 0


def gone(pattern: str, within: float = 3) -> bool:
    deadline = time.monotonic() + within
    while running(pattern):
        if time.monotonic() > deadline:
            return False
        time.sleep(0.05)
    return True


def timed(command: str) -> tuple[float, str]:
    start = time.monotonic()
    out = PROGRAM["run_bash"](command)
    return time.monotonic() - start, out


@pytest.mark.parametrize(
    "command",
    [
        # bash forks a subshell for the backgrounded && list, which keeps the pipes.
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
    try:
        elapsed, out = timed(
            f"echo fg; (sleep 0.8; head -c 3000000 /dev/zero; echo $? > {rc}) &"
        )
        assert out == "fg\n" + NOTE
        assert elapsed < 0.8
        for _ in range(100):
            if rc.exists() and rc.read_text():
                break
            time.sleep(0.1)
        assert rc.read_text() == "0\n"
    finally:
        subprocess.run(["pkill", "-f", str(rc)], check=False)


def test_timeout_closes_the_pipes(monkeypatch, length):
    # `; true` keeps bash from exec'ing the writer.
    ticker = f"import time\nwhile True: print({length}, flush=True); time.sleep(0.05)"
    monkeypatch.setitem(PROGRAM, "BASH_TIMEOUT", 1)
    elapsed, out = timed(f'python3 -c "{ticker}"; true')
    assert out.endswith("timed out after 1 seconds")
    assert elapsed < 2
    assert gone(length)


def test_runtime_reaping_the_program_stops_its_jobs(tmp_path, new_length):
    # The subprocess runtime kills the chat program's process group on cancel.
    job, in_flight = new_length(), new_length()
    program = tmp_path / "program.py"
    program.write_text(
        f"program = {{'__name__': 'chat_program'}}\n"
        f"exec(compile({CHAT_PROGRAM_SOURCE!r}, 'chat_program.py', 'exec'), program)\n"
        f"print(repr(program['run_bash']('echo started; sleep {job} &')), flush=True)\n"
        f"program['run_bash']('sleep {in_flight}; echo done')\n"
    )
    proc = subprocess.Popen(
        [sys.executable, str(program)],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        start_new_session=True,
    )
    try:
        assert proc.stdout is not None
        assert proc.stdout.readline().decode() == repr("started\n" + NOTE) + "\n"
        for _ in range(100):
            if running(f"sleep {in_flight}"):
                break
            time.sleep(0.05)
        assert running(f"sleep {job}") and running(f"sleep {in_flight}")
        os.killpg(proc.pid, signal.SIGKILL)
        proc.wait()
        assert gone(f"sleep {job}") and gone(f"sleep {in_flight}")
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()
        proc.stdout.close()


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


def test_character_cut_at_the_hand_off_is_replaced(length):
    writer = (
        "import os, time; os.write(1, ('\u00e9' * 3000).encode()[:4095]);"
        f" time.sleep({length})"
    )
    out = PROGRAM["run_bash"](f'echo start; python3 -c "{writer}" &')
    assert out == "start\n" + "\u00e9" * 2047 + "\ufffd\n" + NOTE


def test_at_end_tells_a_closed_pipe_from_a_held_one():
    read, write = os.pipe()
    try:
        assert not PROGRAM["_at_end"](read)
        os.write(write, b"late")
        assert not PROGRAM["_at_end"](read)
        os.close(write)
        write = None
        assert PROGRAM["_at_end"](read)
    finally:
        os.close(read)
        if write is not None:
            os.close(write)


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
