"""Public Agent Client Protocol support for harness programs."""

import asyncio
import contextlib
import json
from abc import abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeAlias, TypeVar, cast

from pydantic import BaseModel, ConfigDict, Field

from verifiers.v1.clients import ModelContext
from verifiers.v1.configs.harness import HarnessConfig
from verifiers.v1.errors import HarnessError
from verifiers.v1.harness import Harness, HarnessSession
from verifiers.v1.lineage import ACP_LINEAGE_METADATA_KEY, LineageManifest
from verifiers.v1.runtimes import ProgramResult, Runtime, RuntimeProcess
from verifiers.v1.task import TaskData
from verifiers.v1.trace import Trace
from verifiers.v1.types import Messages
from verifiers.v1.utils.aio import run_shielded

ACP_SOURCE = (Path(__file__).resolve().parent / "runner.py").read_text()
MAX_PACKET_BYTES = 128 * 1024 * 1024

__all__ = ["ACPConfig", "ACPHarness", "ACPTurn"]

ConfigT = TypeVar("ConfigT", bound=HarnessConfig)
JsonValue: TypeAlias = (
    str | int | float | bool | None | list["JsonValue"] | dict[str, "JsonValue"]
)
JsonObject: TypeAlias = dict[str, JsonValue]


class ACPTurn(BaseModel):
    """One completed ACP prompt and its extension metadata."""

    model_config = ConfigDict(extra="forbid", strict=True)

    reply: str
    stop_reason: str | None = None
    response_metadata: dict[str, Any] = Field(default_factory=dict)
    update_metadata: list[dict[str, Any]] = Field(default_factory=list)


@dataclass
class ACPConfig:
    """One harness's ACP process and initial prompt."""

    env: dict[str, str]
    command: list[str]
    prompt: str | Messages | None
    mcp_urls: dict[str, str] | None = None
    system_prompt: str | None = None
    session_meta: JsonObject | None = None


class ACPHarness(Harness[ConfigT]):
    """Harness backed by one live ACP process and native session per rollout."""

    async def setup(self, runtime: Runtime) -> None:
        await runtime.prepare_uv_script(
            ACP_SOURCE, {**self.config.resolved_env, "UV_FROZEN": "false"}
        )

    def acp_turn_result(self, trace: Trace, result: ACPTurn) -> None:
        """Consume the typed result of one ACP prompt."""

    def acp_close_result(self, trace: Trace, response_metadata: dict[str, Any]) -> None:
        """Consume extension metadata returned by `session/close`, when supported."""

    def _consume_protocol_metadata(
        self, trace: Trace, response_metadata: dict[str, Any]
    ) -> None:
        """Attach optional protocol extensions understood by every ACP harness."""
        if ACP_LINEAGE_METADATA_KEY not in response_metadata:
            return
        manifest = LineageManifest.model_validate(
            response_metadata[ACP_LINEAGE_METADATA_KEY]
        )
        trace.reconcile_lineage(manifest)

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
        tool_interception_url: str | None = None,
    ) -> HarnessSession:
        if not runtime.supports_live_processes:
            raise HarnessError(
                f"harness {self.config.id!r} requires a runtime with live process support"
            )
        config = await self.prepare_acp(
            ctx, trace, runtime, endpoint, secret, mcp_urls, data
        )
        return ACPHarnessSession(
            self,
            ctx,
            trace,
            runtime,
            endpoint,
            secret,
            mcp_urls if config.mcp_urls is None else config.mcp_urls,
            data,
            config,
            tool_interception_url,
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
        raise HarnessError(
            f"harness {self.config.id!r} requires a rollout-scoped session"
        )


def _packet(value: JsonObject) -> bytes:
    data = json.dumps(value, ensure_ascii=False).encode()
    if len(data) > MAX_PACKET_BYTES:
        raise ValueError(f"ACP session packet is too large: {len(data)} bytes")
    return len(data).to_bytes(8, "big") + data


def _turn_result(response: JsonObject) -> ACPTurn:
    value = response.get("result")
    return ACPTurn.model_validate(value)


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

    async def read(self) -> JsonObject:
        size = int.from_bytes(await self._readexactly(8), "big")
        if size > MAX_PACKET_BYTES:
            raise ValueError(f"ACP session packet is too large: {size} bytes")
        return json.loads((await self._readexactly(size)).decode())


class ACPHarnessSession(HarnessSession):
    """A live ACP process, connection, and native session for one rollout."""

    def __init__(
        self,
        harness: ACPHarness,
        ctx: ModelContext,
        trace: Trace,
        runtime: Runtime,
        endpoint: str,
        secret: str,
        mcp_urls: dict[str, str],
        data: TaskData,
        config: ACPConfig,
        tool_interception_url: str | None = None,
    ) -> None:
        super().__init__(
            harness,
            ctx,
            trace,
            runtime,
            endpoint,
            secret,
            mcp_urls,
            data,
            tool_interception_url,
        )
        self.config = config
        self._process: RuntimeProcess | None = None
        self._reader: _PacketReader | None = None
        self._stderr_tail = bytearray()
        self._stderr_task: asyncio.Task[None] | None = None
        self._lock = asyncio.Lock()

    async def _start(self) -> None:
        self._stderr_tail.clear()
        program = await self.runtime.prepare_uv_script(
            ACP_SOURCE,
            {**self.config.env, "UV_FROZEN": "false"},
            activate=False,
        )
        process = await self.runtime.open_process(program, self.config.env)
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
        prompt = self.config.prompt if messages is None else messages
        if prompt is None:
            raise ValueError("ACP requires a prompt")
        if not isinstance(prompt, str) and (
            not prompt or any(message.role != "user" for message in prompt)
        ):
            raise ValueError("an ACP turn must contain user messages only")
        user_contents = (
            [prompt]
            if isinstance(prompt, str)
            else [
                message.model_dump(mode="json", include={"content"})["content"]
                for message in prompt
            ]
        )
        config = {
            "command": self.config.command,
            "user_contents": user_contents,
            "mcp_urls": self.mcp_urls,
            "system_prompt": self.config.system_prompt or "",
            "session_meta": self.config.session_meta or {},
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
        turn = _turn_result(response)
        self.trace.root_reply = turn.reply.strip()
        if not response.get("ok"):
            detail = response.get("error") or "ACP session request failed"
            if stderr := self._stderr():
                detail = f"{detail}\n\nACP process stderr:\n{stderr}"
            raise RuntimeError(detail)
        harness = cast(ACPHarness, self.harness)
        harness._consume_protocol_metadata(self.trace, turn.response_metadata)
        harness.acp_turn_result(self.trace, turn)
        result = ProgramResult(exit_code=0, stdout=turn.reply, stderr="")
        _require_model_turn(self.trace, calls_before, result)
        return result

    async def _stop(self, *, graceful: bool) -> dict[str, Any]:
        process, self._process = self._process, None
        reader, self._reader = self._reader, None
        stderr_task, self._stderr_task = self._stderr_task, None
        if process is None:
            return {}
        response_metadata: dict[str, Any] = {}
        try:
            if graceful and reader is not None:
                with contextlib.suppress(BaseException):
                    await process.write(_packet({"operation": "shutdown"}))
                    response = await asyncio.wait_for(reader.read(), timeout=10)
                    result = response.get("result")
                    if response.get("ok") and isinstance(result, dict):
                        metadata = result.get("response_metadata")
                        if isinstance(metadata, dict):
                            response_metadata = metadata
            for timeout, stop in (
                (10 if graceful else 0.1, None),
                (5, process.terminate),
                (5, process.kill),
            ):
                if stop is not None:
                    with contextlib.suppress(BaseException):
                        await stop()
                try:
                    await asyncio.wait_for(process.wait(), timeout)
                    break
                except TimeoutError:
                    continue
        finally:
            if stderr_task is not None:
                if not stderr_task.done():
                    stderr_task.cancel()
                with contextlib.suppress(BaseException):
                    await stderr_task
        return response_metadata

    async def close(self) -> None:
        if self._closed:
            return
        # Publish closure before waiting for the process lock. A turn that
        # already passed HarnessSession.turn()'s fast check rechecks under the
        # same lock in _run(), so it cannot restart after teardown.
        await super().close()

        async def close_process() -> None:
            async with self._lock:
                response_metadata = await self._stop(graceful=True)
                if response_metadata:
                    harness = cast(ACPHarness, self.harness)
                    harness._consume_protocol_metadata(self.trace, response_metadata)
                    harness.acp_close_result(self.trace, response_metadata)

        await run_shielded(close_process())
