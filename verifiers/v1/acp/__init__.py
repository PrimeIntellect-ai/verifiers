"""Public Agent Client Protocol support for harness programs."""

import asyncio
import contextlib
import json
import secrets
from collections.abc import AsyncIterator
from pathlib import Path

from verifiers.v1.clients import ModelContext
from verifiers.v1.dialects.chat import message_to_wire
from verifiers.v1.errors import HarnessError
from verifiers.v1.harness import Harness, HarnessSession
from verifiers.v1.runtimes import ProgramResult, Runtime, RuntimeProcess
from verifiers.v1.task import TaskData
from verifiers.v1.trace import Trace
from verifiers.v1.types import Messages
from verifiers.v1.utils.aio import run_shielded

ACP_SOURCE = (Path(__file__).resolve().parent / "runner.py").read_text()
MAX_PACKET_BYTES = 128 * 1024 * 1024
# Keep in sync with runner.py: sync responses are framed with this marker plus
# the request nonce so a reattached host can re-align packet framing.
RESYNC_MAGIC = b"\xff\x00vf-acp-resync\x00\xff"
RECONNECT_ATTEMPTS = 4
RECONNECT_BACKOFF_SECONDS = 2.0
# The runner emits keepalives every 15s even while a prompt runs, so a healthy
# stream is never silent for long. Longer silence means the stream (or runner)
# is dead: surface it so the turn can reattach or fail instead of hanging.
IDLE_TIMEOUT_SECONDS = 120.0

__all__ = ["ACP"]


class ACP:
    """Run one-shot ACP agents or create rollout-scoped ACP sessions."""

    async def setup(self, harness: Harness, runtime: Runtime) -> None:
        await runtime.prepare_uv_script(
            ACP_SOURCE, {**harness.config.resolved_env, "UV_FROZEN": "false"}
        )

    def session(
        self,
        harness: Harness,
        ctx: ModelContext,
        trace: Trace,
        runtime: Runtime,
        endpoint: str,
        secret: str,
        mcp_urls: dict[str, str],
        data: TaskData,
        *,
        env: dict[str, str],
        command: list[str],
        prompt: str | Messages | None,
        system_prompt: str | None = None,
        session_meta: dict | None = None,
    ) -> "ACPHarnessSession":
        """Create a persistent ACP-backed handle owned by one rollout."""
        return ACPHarnessSession(
            harness,
            ctx,
            trace,
            runtime,
            endpoint,
            secret,
            mcp_urls,
            data,
            env=env,
            command=command,
            prompt=prompt,
            system_prompt=system_prompt,
            session_meta=session_meta,
        )

    async def run(
        self,
        runtime: Runtime,
        env: dict[str, str],
        command: list[str],
        prompt: str | Messages | None,
        *,
        mcp_urls: dict[str, str] | None = None,
        system_prompt: str | None = None,
        session_path: str | None = None,
        session_meta: dict | None = None,
        allow_empty_tool_reply: bool = False,
    ) -> ProgramResult:
        """Run one ACP segment without retaining its process."""
        return await self._run(
            runtime,
            env,
            command,
            prompt,
            mcp_urls=mcp_urls,
            system_prompt=system_prompt,
            session_path=session_path,
            session_meta=session_meta,
            allow_empty_tool_reply=allow_empty_tool_reply,
        )

    async def _run(
        self,
        runtime: Runtime,
        env: dict[str, str],
        command: list[str],
        prompt: str | Messages | None,
        *,
        mcp_urls: dict[str, str] | None = None,
        system_prompt: str | None = None,
        session_path: str | None = None,
        session_meta: dict | None = None,
        allow_empty_tool_reply: bool = False,
    ) -> ProgramResult:
        if prompt is None:
            raise ValueError("ACP requires a prompt")
        messages = (
            [{"role": "user", "content": prompt}]
            if isinstance(prompt, str)
            else [message_to_wire(message) for message in prompt]
        )
        config = {
            "command": command,
            "messages": messages,
            "mcp_urls": mcp_urls or {},
            "system_prompt": system_prompt or "",
            "session_path": session_path,
            "session_meta": session_meta or {},
            "allow_empty_tool_reply": allow_empty_tool_reply,
        }
        program = await runtime.prepare_uv_script(
            ACP_SOURCE,
            {**env, "UV_FROZEN": "false"},
            activate=False,
        )
        directory = f".vf-acp-{secrets.token_hex(8)}"
        created = await runtime.run(["mkdir", "-m", "700", directory], {})
        if created.exit_code != 0:
            raise RuntimeError(f"ACP config directory failed: {created.stderr.strip()}")
        path = f"{directory}/config.json"
        try:
            await runtime.write(path, json.dumps(config).encode())
            return await runtime.run_program([*program, "once", path], env)
        finally:
            await run_shielded(runtime.run(["rm", "-rf", directory], {}))


def _packet(value: dict) -> bytes:
    data = json.dumps(value, ensure_ascii=False).encode()
    if len(data) > MAX_PACKET_BYTES:
        raise ValueError(f"ACP session packet is too large: {len(data)} bytes")
    return len(data).to_bytes(8, "big") + data


class _PacketReader:
    def __init__(
        self,
        source: AsyncIterator[bytes],
        idle_timeout: float | None = None,
    ) -> None:
        self._source = source.__aiter__()
        self._buffer = bytearray()
        self._idle_timeout = idle_timeout

    async def _next_chunk(self) -> None:
        try:
            chunk = await asyncio.wait_for(anext(self._source), self._idle_timeout)
        except StopAsyncIteration as e:
            raise EOFError("ACP process closed its stdout") from e
        except TimeoutError as e:
            raise EOFError(
                f"ACP process stream was silent for {self._idle_timeout}s "
                "(keepalives stopped)"
            ) from e
        self._buffer.extend(chunk)

    async def _readexactly(self, size: int) -> bytes:
        while len(self._buffer) < size:
            await self._next_chunk()
        data = bytes(self._buffer[:size])
        del self._buffer[:size]
        return data

    async def read(self) -> dict:
        while True:
            size = int.from_bytes(await self._readexactly(8), "big")
            if size > MAX_PACKET_BYTES:
                raise ValueError(f"ACP session packet is too large: {size} bytes")
            packet = json.loads((await self._readexactly(size)).decode())
            if packet.get("type") != "keepalive":
                return packet

    async def resync(self, nonce: str) -> dict:
        """Discard bytes until this nonce's resync marker, then read the sync
        response that follows it. Realigns framing after a stream reattach.
        The runner only answers sync between requests, so this may legitimately
        wait out the rest of an in-flight prompt; its keepalives keep the
        stream from tripping the idle timeout meanwhile."""
        marker = RESYNC_MAGIC + nonce.encode() + RESYNC_MAGIC
        while True:
            index = self._buffer.find(marker)
            if index >= 0:
                del self._buffer[: index + len(marker)]
                return await self.read()
            # Keep a tail in case the marker arrives split across chunks.
            if len(self._buffer) > len(marker):
                del self._buffer[: len(self._buffer) - len(marker)]
            await self._next_chunk()


class ACPHarnessSession(HarnessSession):
    """A live ACP process, connection, and native session for one rollout."""

    def __init__(
        self,
        harness: Harness,
        ctx: ModelContext,
        trace: Trace,
        runtime: Runtime,
        endpoint: str,
        secret: str,
        mcp_urls: dict[str, str],
        data: TaskData,
        env: dict[str, str],
        command: list[str],
        prompt: str | Messages | None,
        system_prompt: str | None,
        session_meta: dict | None,
    ) -> None:
        super().__init__(harness, ctx, trace, runtime, endpoint, secret, mcp_urls, data)
        self.env = env
        self.command = command
        self.prompt = prompt
        self.system_prompt = system_prompt
        self.session_meta = session_meta or {}
        self._process: RuntimeProcess | None = None
        self._reader: _PacketReader | None = None
        self._stderr_tail = bytearray()
        self._stderr_task: asyncio.Task[None] | None = None
        self._lock = asyncio.Lock()
        self._seq = 0

    async def _start(self) -> None:
        self._stderr_tail.clear()
        program = await self.runtime.prepare_uv_script(
            ACP_SOURCE,
            {**self.env, "UV_FROZEN": "false"},
            activate=False,
        )
        process = await self.runtime.open_process([*program, "stream"], self.env)
        self._process = process
        self._reader = _PacketReader(process.stdout, idle_timeout=IDLE_TIMEOUT_SECONDS)
        self._stderr_task = asyncio.create_task(self._drain_stderr(process.stderr))

    async def _drain_stderr(self, stream: AsyncIterator[bytes]) -> None:
        async for chunk in stream:
            self._stderr_tail.extend(chunk)
            if len(self._stderr_tail) > 4000:
                del self._stderr_tail[:-4000]

    def _stderr(self) -> str:
        return self._stderr_tail.decode(errors="replace").strip()

    async def _run(self, messages: Messages | None) -> ProgramResult:
        prompt = self.prompt if messages is None else messages
        if prompt is None:
            raise ValueError("ACP requires a prompt")
        wire_messages = (
            [{"role": "user", "content": prompt}]
            if isinstance(prompt, str)
            else [message_to_wire(message) for message in prompt]
        )
        config = {
            "command": self.command,
            "messages": wire_messages,
            "mcp_urls": self.mcp_urls,
            "system_prompt": self.system_prompt or "",
            "session_path": None,
            "session_meta": self.session_meta,
        }
        async with self._lock:
            if self._closed:
                raise HarnessError(
                    f"harness {self.harness.config.id!r} session is already closed"
                )
            if self._process is None:
                await self._start()
            assert self._process is not None
            assert self._reader is not None
            self._seq += 1
            seq = self._seq
            request = _packet({"operation": "prompt", "config": config, "seq": seq})
            try:
                try:
                    await self._process.write(request)
                    response = await self._read_response(seq)
                except Exception as error:  # noqa: BLE001 - any stream fault
                    # The process stream died mid-turn — on Prime this is the
                    # gateway dropping the Start RPC ("process stream RPC
                    # failed"), not the runner exiting. Reattach to the
                    # still-running process, realign framing, and recover or
                    # resend the turn. Re-raises `error` if unsupported.
                    response = await self._recover_turn(request, seq, error)
            except BaseException:
                await run_shielded(self._stop(graceful=False))
                raise
        if not response.get("ok"):
            detail = response.get("error") or "ACP session request failed"
            if stderr := self._stderr():
                detail = f"{detail}\n\nACP process stderr:\n{stderr}"
            raise RuntimeError(detail)
        return ProgramResult(exit_code=0, stdout=response.get("reply", ""), stderr="")

    async def _read_response(self, seq: int) -> dict:
        assert self._reader is not None
        while True:
            response = await self._reader.read()
            # Skip responses to earlier requests replayed after a reattach.
            if response.get("seq") in (None, seq):
                return response

    async def _recover_turn(self, request: bytes, seq: int, error: Exception) -> dict:
        """Reattach to the runner after a transport failure and finish the turn.

        The sandbox keeps the runner alive when the host's process stream dies;
        bytes written while detached are dropped. After reattaching, a sync
        exchange realigns packet framing and reports the runner's last answered
        seq: if it already answered this turn the cached response is used,
        otherwise the request is resent (the runner dedupes by seq)."""
        assert self._process is not None
        for attempt in range(1, RECONNECT_ATTEMPTS + 1):
            await asyncio.sleep(RECONNECT_BACKOFF_SECONDS * attempt)
            if self._closed:
                raise error
            try:
                if not await self._process.reattach():
                    raise error  # runtime cannot reattach: original failure
            except Exception as reattach_error:
                if reattach_error is error:
                    raise
                continue  # process stream not reachable yet; retry
            self._reader = _PacketReader(
                self._process.stdout, idle_timeout=IDLE_TIMEOUT_SECONDS
            )
            if self._stderr_task is not None and not self._stderr_task.done():
                self._stderr_task.cancel()
            self._stderr_task = asyncio.create_task(
                self._drain_stderr(self._process.stderr)
            )
            try:
                nonce = secrets.token_hex(8)
                await self._process.write(
                    _packet({"operation": "sync", "nonce": nonce})
                )
                # The runner answers sync after finishing any in-flight prompt,
                # which may take a while; its keepalives hold off the reader's
                # idle timeout until then.
                sync = await self._reader.resync(nonce)
                if sync.get("last_seq") == seq and sync.get("last_response"):
                    return sync["last_response"]
                await self._process.write(request)
                return await self._read_response(seq)
            except Exception:  # noqa: BLE001, S112 - retry until attempts end
                continue
        raise error

    async def _stop(self, *, graceful: bool) -> None:
        process, self._process = self._process, None
        reader, self._reader = self._reader, None
        stderr_task, self._stderr_task = self._stderr_task, None
        if process is None:
            return
        failure: BaseException | None = None
        try:
            if graceful and reader is not None:
                try:
                    await process.write(_packet({"operation": "shutdown"}))
                    response = await asyncio.wait_for(reader.read(), timeout=10)
                    if not response.get("ok"):
                        raise RuntimeError(
                            response.get("error") or "ACP session shutdown failed"
                        )
                except BaseException as error:  # noqa: BLE001 - finish teardown if cancelled
                    failure = error
            try:
                await asyncio.wait_for(process.wait(), timeout=10 if graceful else 0.1)
            except BaseException:  # noqa: BLE001 - cancellation still requires termination
                with contextlib.suppress(Exception):
                    await asyncio.wait_for(process.terminate(), timeout=5)
                try:
                    await asyncio.wait_for(process.wait(), timeout=5)
                except BaseException:  # noqa: BLE001 - cancellation still requires a kill
                    with contextlib.suppress(Exception):
                        await asyncio.wait_for(process.kill(), timeout=5)
                    with contextlib.suppress(BaseException):
                        await asyncio.wait_for(process.wait(), timeout=5)
        finally:
            if stderr_task is not None:
                if not stderr_task.done():
                    stderr_task.cancel()
                with contextlib.suppress(BaseException):
                    await stderr_task
            # A process abandoned without its exit event still holds its remote
            # stream open (on Prime, one gateway HTTP/2 stream slot plus its
            # dedicated connection) until the sandbox dies. Always release it.
            closer = getattr(process, "aclose", None)
            if closer is not None:
                with contextlib.suppress(BaseException):
                    await closer()
        if failure is not None:
            detail = str(failure)
            if stderr := self._stderr():
                detail = f"{detail}\n\nACP process stderr:\n{stderr}"
            raise RuntimeError(detail) from failure

    async def close(self) -> None:
        if self._closed:
            return
        # Publish closure before waiting for the process lock. A turn that
        # already passed HarnessSession.turn()'s fast check rechecks under the
        # same lock in _run(), so it cannot restart after teardown.
        await super().close()

        async def close_process() -> None:
            async with self._lock:
                await self._stop(graceful=True)

        await run_shielded(close_process())
