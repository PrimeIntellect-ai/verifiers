"""An ACP turn never outlives its process stream: the in-box runner beats a heartbeat,
the host fails the turn once the stream carries nothing for `stall_timeout`, and an answer
the stream dropped while it re-attached is asked for again.

Prime's live process stream re-attaches after a dropped connection without replaying what
the box wrote meanwhile, and a re-attached stream to a box whose network path died stays
open with nothing on it until its 24-hour bound: OpenCode rollouts held their slot for
2-21 hours after their last tool call (2026-10-27)."""

import asyncio
import contextlib
import sys
from pathlib import Path

import pytest

import verifiers.v1.acp as acp_harness
from verifiers.v1.acp import ACPHarnessSession, _PacketReader
from verifiers.v1.errors import SandboxError
from verifiers.v1.runtimes import RuntimeProcess

ACP_STUB = """
PROTOCOL_VERSION = 1
class Client: pass
class RequestError(Exception): pass
def image_block(*a, **k): pass
def spawn_agent_process(*a, **k): pass
def text_block(*a, **k): pass
"""
SCHEMA_STUB = "def __getattr__(name):\n    return type(name, (), {})\n"


class RunnerProcess(RuntimeProcess):
    """The real runner.py in a local subprocess, its stdout behind a stream the test can
    cut (`drop_answer`: the next answer is lost, as output written during a re-attach is)
    or freeze (`stall`: nothing more arrives, nothing fails)."""

    def __init__(self, process: asyncio.subprocess.Process) -> None:
        self._process = process
        self.drop_answer = False
        self.stall = False
        self.stdout = self._stdout()
        self.stderr = self._stderr()
        self.written: list[bytes] = []

    async def _stdout(self):
        assert self._process.stdout is not None
        while chunk := await self._process.stdout.read(65536):
            if self.stall:
                await asyncio.Event().wait()
            if self.drop_answer and b'"result"' in chunk:
                self.drop_answer = False
                continue
            yield chunk

    async def _stderr(self):
        assert self._process.stderr is not None
        while chunk := await self._process.stderr.read(65536):
            yield chunk

    async def write(self, data: bytes) -> None:
        assert self._process.stdin is not None
        self.written.append(data)
        self._process.stdin.write(data)
        await self._process.stdin.drain()

    async def wait(self) -> int:
        return await self._process.wait()

    async def poll(self) -> int | None:
        return self._process.returncode

    async def terminate(self) -> None:
        with contextlib.suppress(ProcessLookupError):
            self._process.terminate()

    async def kill(self) -> None:
        with contextlib.suppress(ProcessLookupError):
            self._process.kill()


@contextlib.asynccontextmanager
async def runner_session(tmp_path: Path, heartbeat: float, stall: float):
    stubs = tmp_path / "stubs" / "acp"
    stubs.mkdir(parents=True)
    (stubs / "__init__.py").write_text(ACP_STUB)
    (stubs / "schema.py").write_text(SCHEMA_STUB)
    script = Path(acp_harness.__file__).parent / "runner.py"
    process = await asyncio.create_subprocess_exec(
        sys.executable,
        str(script),
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env={
            "PYTHONPATH": str(tmp_path / "stubs"),
            "VF_ACP_HEARTBEAT_SECONDS": str(heartbeat),
        },
    )
    proc = RunnerProcess(process)
    session = ACPHarnessSession(*(None,) * 4, "", "", {}, None, None)  # type: ignore[arg-type]
    session.stall_timeout = stall
    session._process = proc
    session._reader = _PacketReader(proc.stdout)
    session._response_task = asyncio.create_task(
        session._read_responses(session._reader)
    )
    try:
        yield session, proc
    finally:
        await proc.kill()
        await proc.wait()
        session._response_task.cancel()
        with contextlib.suppress(BaseException):
            await session._response_task


async def test_heartbeats_keep_a_quiet_turn_alive(tmp_path):
    """A turn whose stream carries no answer for longer than `stall_timeout` (a long tool
    call) lives on while the runner's heartbeats arrive."""
    async with runner_session(tmp_path, heartbeat=0.05, stall=0.5) as (session, _):
        await asyncio.sleep(1.5)
        assert not session._response_task.done()
        response = await asyncio.wait_for(session._request("steer", message="x"), 5)
        assert response["ok"] and response["result"]["reason"] == "noRunningTurn"


async def test_a_stalled_stream_fails_the_turn_as_a_sandbox_io_error(tmp_path):
    """A stream that stops carrying anything (the box's network path died; the SDK
    re-attached to nothing) fails the awaited request instead of holding it for 24 h."""
    async with runner_session(tmp_path, heartbeat=0.05, stall=0.5) as (session, proc):
        await asyncio.sleep(0.2)
        proc.stall = True
        with pytest.raises(SandboxError, match="stalled") as raised:
            await asyncio.wait_for(session._request("steer", message="x"), 5)
        assert raised.value.kind == "io"


async def test_an_answer_lost_on_the_stream_is_asked_for_again(tmp_path):
    """An answer written while the stream re-attached never arrives; the next heartbeat
    names it as sent, and the host asks the runner to send it again."""
    async with runner_session(tmp_path, heartbeat=0.1, stall=5) as (session, proc):
        proc.drop_answer = True
        response = await asyncio.wait_for(session._request("steer", message="x"), 5)
        assert not proc.drop_answer
        assert response["ok"] and response["result"]["reason"] == "noRunningTurn"
        assert any(b'"resend"' in packet for packet in proc.written)
