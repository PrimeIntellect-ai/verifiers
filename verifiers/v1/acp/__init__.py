"""Public Agent Client Protocol support for harness programs."""

import asyncio
import contextlib
import json
import secrets
from abc import abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeVar

from verifiers.v1.clients import ModelContext
from verifiers.v1.configs.harness import HarnessConfig
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

__all__ = ["ACP", "ACPConfig", "ACPHarness"]

ConfigT = TypeVar("ConfigT", bound=HarnessConfig)


@dataclass
class ACPConfig:
    """One harness's prepared ACP process and prompt."""

    env: dict[str, str]
    command: list[str]
    prompt: str | Messages | None
    mcp_urls: dict[str, str] | None = None
    system_prompt: str | None = None
    session_path: str | None = None
    session_meta: dict | None = None
    allow_empty_tool_reply: bool = False


def _record_acp_meta(trace: Trace, meta: dict[str, list[Any]]) -> None:
    if not meta:
        return
    recorded = trace.info.setdefault("acp_meta", {})
    for namespace, events in meta.items():
        recorded.setdefault(namespace, []).extend(events)


class ACP:
    """Run one-shot ACP agents or create rollout-scoped ACP sessions."""

    def __init__(self, *, metadata_expected: bool = False) -> None:
        # ACP 0.11 has no standard extension-metadata capability advertisement.
        self.metadata_expected = metadata_expected

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
        allow_empty_tool_reply: bool = False,
    ) -> "ACPHarnessSession":
        """Create a live ACP-backed handle owned by one rollout."""
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
            allow_empty_tool_reply=allow_empty_tool_reply,
            metadata_expected=self.metadata_expected,
        )

    async def run(
        self,
        runtime: Runtime,
        env: dict[str, str],
        command: list[str],
        prompt: str | Messages | None,
        *,
        trace: Trace,
        mcp_urls: dict[str, str] | None = None,
        system_prompt: str | None = None,
        session_path: str | None = None,
        session_meta: dict | None = None,
        allow_empty_tool_reply: bool = False,
    ) -> ProgramResult:
        """Run one ACP segment without retaining its process."""
        calls_before = len(trace.calls)
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
            "metadata_expected": self.metadata_expected,
        }
        program = await runtime.prepare_uv_script(
            ACP_SOURCE, {**env, "UV_FROZEN": "false"}, activate=False
        )
        directory = f".vf-acp-{secrets.token_hex(8)}"
        created = await runtime.run(["mkdir", "-m", "700", directory], {})
        if created.exit_code != 0:
            raise RuntimeError(f"ACP config directory failed: {created.stderr.strip()}")
        path = f"{directory}/config.json"
        meta_path = f"{directory}/meta.json"
        config["meta_path"] = meta_path
        try:
            await runtime.write(path, json.dumps(config).encode())
            result = await runtime.run_program([*program, "once", path], env)
            with contextlib.suppress(Exception):
                meta = json.loads((await runtime.read(meta_path)).decode())
                _record_acp_meta(trace, meta)
        finally:
            await run_shielded(runtime.run(["rm", "-rf", directory], {}))
        _require_model_turn(trace, calls_before, result)
        return result


class ACPHarness(Harness[ConfigT]):
    """Harness whose execution is completely described by an ACP configuration."""

    acp = ACP()

    @abstractmethod
    async def prepare_acp(
        self,
        ctx: ModelContext,
        trace: Trace,
        runtime: Runtime,
        endpoint: str,
        secret: str,
        mcp_urls: dict[str, str],
        data: TaskData,
    ) -> ACPConfig:
        pass

    async def session(
        self,
        ctx: ModelContext,
        trace: Trace,
        runtime: Runtime,
        endpoint: str,
        secret: str,
        mcp_urls: dict[str, str],
        data: TaskData,
    ) -> HarnessSession:
        if not runtime.supports_live_processes:
            return HarnessSession(
                self, ctx, trace, runtime, endpoint, secret, mcp_urls, data
            )
        config = await self.prepare_acp(
            ctx, trace, runtime, endpoint, secret, mcp_urls, data
        )
        return self.acp.session(
            self,
            ctx,
            trace,
            runtime,
            endpoint,
            secret,
            mcp_urls if config.mcp_urls is None else config.mcp_urls,
            data,
            env=config.env,
            command=config.command,
            prompt=config.prompt,
            system_prompt=config.system_prompt,
            session_meta=config.session_meta,
            allow_empty_tool_reply=config.allow_empty_tool_reply,
        )

    async def launch(
        self,
        ctx: ModelContext,
        trace: Trace,
        runtime: Runtime,
        endpoint: str,
        secret: str,
        mcp_urls: dict[str, str],
        data: TaskData,
    ) -> ProgramResult:
        config = await self.prepare_acp(
            ctx, trace, runtime, endpoint, secret, mcp_urls, data
        )
        return await self.acp.run(
            runtime,
            config.env,
            config.command,
            config.prompt,
            trace=trace,
            mcp_urls=mcp_urls if config.mcp_urls is None else config.mcp_urls,
            system_prompt=config.system_prompt,
            session_path=config.session_path,
            session_meta=config.session_meta,
            allow_empty_tool_reply=config.allow_empty_tool_reply,
        )


def _packet(value: dict) -> bytes:
    data = json.dumps(value, ensure_ascii=False).encode()
    if len(data) > MAX_PACKET_BYTES:
        raise ValueError(f"ACP session packet is too large: {len(data)} bytes")
    return len(data).to_bytes(8, "big") + data


def _require_model_turn(trace: Trace, calls_before: int, result: ProgramResult) -> None:
    if (
        result.exit_code
        or trace.stop_condition is not None
        or any(call.node is not None for call in trace.calls[calls_before:])
    ):
        return
    detail = (result.stderr or result.stdout).strip()[-500:] or "<no output>"
    raise RuntimeError("ACP agent completed without committing a model turn: " + detail)


class _PacketReader:
    def __init__(self, source: AsyncIterator[bytes]) -> None:
        self._source = source.__aiter__()
        self._buffer = bytearray()

    async def _readexactly(self, size: int) -> bytes:
        while len(self._buffer) < size:
            try:
                self._buffer.extend(await anext(self._source))
            except StopAsyncIteration as e:
                raise EOFError("ACP process closed its stdout") from e
        data = bytes(self._buffer[:size])
        del self._buffer[:size]
        return data

    async def read(self) -> dict:
        size = int.from_bytes(await self._readexactly(8), "big")
        if size > MAX_PACKET_BYTES:
            raise ValueError(f"ACP session packet is too large: {size} bytes")
        return json.loads((await self._readexactly(size)).decode())


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
        allow_empty_tool_reply: bool,
        metadata_expected: bool,
    ) -> None:
        super().__init__(harness, ctx, trace, runtime, endpoint, secret, mcp_urls, data)
        self.env = env
        self.command = command
        self.prompt = prompt
        self.system_prompt = system_prompt
        self.session_meta = session_meta or {}
        self.allow_empty_tool_reply = allow_empty_tool_reply
        self.metadata_expected = metadata_expected
        self._process: RuntimeProcess | None = None
        self._reader: _PacketReader | None = None
        self._stderr_tail = bytearray()
        self._stderr_task: asyncio.Task[None] | None = None
        self._lock = asyncio.Lock()

    async def _start(self) -> None:
        self._stderr_tail.clear()
        program = await self.runtime.prepare_uv_script(
            ACP_SOURCE,
            {**self.env, "UV_FROZEN": "false"},
            activate=False,
        )
        process = await self.runtime.open_process([*program, "stream"], self.env)
        self._process = process
        self._reader = _PacketReader(process.stdout)
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
            "allow_empty_tool_reply": self.allow_empty_tool_reply,
            "metadata_expected": self.metadata_expected,
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
            calls_before = len(self.trace.calls)
            try:
                await self._process.write(
                    _packet({"operation": "prompt", "config": config})
                )
                response = await self._reader.read()
            except BaseException:
                await run_shielded(self._stop(graceful=False))
                raise
        if meta := response.get("meta"):
            _record_acp_meta(self.trace, meta)
        if not response.get("ok"):
            detail = response.get("error") or "ACP session request failed"
            if stderr := self._stderr():
                detail = f"{detail}\n\nACP process stderr:\n{stderr}"
            raise RuntimeError(detail)
        result = ProgramResult(exit_code=0, stdout=response.get("reply", ""), stderr="")
        _require_model_turn(self.trace, calls_before, result)
        return result

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
