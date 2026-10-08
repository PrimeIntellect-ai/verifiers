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
from verifiers.v1.errors import HarnessError, InterceptionError
from verifiers.v1.harness import Harness, HarnessSession
from verifiers.v1.runtimes import ProgramResult, Runtime, RuntimeProcess
from verifiers.v1.semantic import (
    ACP_SEMANTIC_EDGES_METADATA_KEY,
    SemanticEdgeSet,
)
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
    client_capabilities: JsonObject | None = None


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
        if ACP_SEMANTIC_EDGES_METADATA_KEY in response_metadata:
            edge_set = SemanticEdgeSet.model_validate(
                response_metadata[ACP_SEMANTIC_EDGES_METADATA_KEY]
            )
            trace.add_semantic_edges(edge_set)

    async def gate_tools(
        self, config: ACPConfig, runtime: Runtime, url: str, secret: str
    ) -> None:
        """Configure the agent to ask before every tool call, so each request reaches the
        runner's gate (`/tool` at `url`, keyed by `secret`). A harness that advertises
        `SUPPORTS_TOOL_INTERCEPTION` implements this with the agent's own config."""

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
        if tool_interception_url is not None:
            await self.gate_tools(config, runtime, tool_interception_url, secret)
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
        self._write_lock = asyncio.Lock()
        self._responses: dict[int, asyncio.Future] = {}
        self._request_id = 0
        self._response_task: asyncio.Task | None = None
        self._stopping = False
        self._lost: asyncio.Future[Exception] = (
            asyncio.get_running_loop().create_future()
        )

    async def lost(self) -> Exception:
        return await asyncio.shield(self._lost)

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
        self._response_task = asyncio.create_task(self._read_responses(self._reader))
        self._stderr_task = asyncio.create_task(self._drain_stderr(process.stderr))

    async def _read_responses(self, reader: _PacketReader) -> None:
        try:
            while True:
                response = await reader.read()
                future = self._responses.get(response.get("id"))
                if future is not None and not future.done():
                    future.set_result(response)
        except BaseException as error:  # noqa: BLE001 - settle callers on disconnect or shutdown
            for future in self._responses.values():
                if not future.done():
                    future.set_exception(
                        RuntimeError(f"ACP connection closed: {error}")
                    )
            # Its stream ending unasked means the process (or its box) is gone,
            # which a caller waiting between turns learns only through `lost()`.
            if not (
                self._closed
                or self._stopping
                or isinstance(error, asyncio.CancelledError)
                or self._lost.done()
            ):
                self._lost.set_result(
                    HarnessError(f"the ACP process stream ended: {error}")
                )

    async def _request(self, operation: str, **payload: Any) -> JsonObject:
        process = self._process
        if process is None or self._response_task is None or self._response_task.done():
            raise RuntimeError("ACP process is not running")
        self._request_id += 1
        request_id = self._request_id
        future = asyncio.get_running_loop().create_future()
        self._responses[request_id] = future
        try:
            async with self._write_lock:
                await process.write(
                    _packet({"id": request_id, "operation": operation, **payload})
                )
            return await future
        finally:
            self._responses.pop(request_id, None)
            if not future.done():
                future.cancel()

    async def steer(self, message: str, *, message_id: str | None = None) -> dict:
        if self._closed:
            raise RuntimeError("ACP session is closed")
        if not isinstance(message, str) or not message.strip():
            raise ValueError("steering requires a nonempty user message")
        if self._process is None:
            return {"outcome": "promptRequired", "reason": "noRunningTurn"}
        response = await self._request("steer", message=message, message_id=message_id)
        if not response.get("ok"):
            raise RuntimeError(response.get("error") or "ACP steering failed")
        result = response.get("result")
        if not isinstance(result, dict):
            raise TypeError("invalid ACP steering receipt")
        return result

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
            "mcp_headers": self.harness.config.resolve_mcp_headers(self.mcp_urls),
            "system_prompt": self.config.system_prompt or "",
            "session_meta": self.config.session_meta or {},
            "client_capabilities": self.config.client_capabilities or {},
            "tool_interception": {
                "url": self.tool_interception_url,
                "secret": self.secret,
            }
            if self.tool_interception_url
            else None,
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
                response = await self._request("prompt", config=config)
            except BaseException:
                await run_shielded(self._stop(graceful=False))
                raise
        turn = _turn_result(response)
        self.trace.root_reply = turn.reply.strip()
        if not response.get("ok"):
            detail = response.get("error") or "ACP session request failed"
            if stderr := self._stderr():
                detail = f"{detail}\n\nACP process stderr:\n{stderr}"
            error_data = response.get("error_data")
            if (
                isinstance(error_data, dict)
                and error_data.get("kind") == "model_transport"
            ):
                raise InterceptionError(detail)
            raise RuntimeError(detail)
        harness = cast(ACPHarness, self.harness)
        harness._consume_protocol_metadata(self.trace, turn.response_metadata)
        harness.acp_turn_result(self.trace, turn)
        result = ProgramResult(exit_code=0, stdout=turn.reply, stderr="")
        _require_model_turn(self.trace, calls_before, result)
        return result

    async def _stop(self, *, graceful: bool) -> dict[str, Any]:
        process = self._process
        response_task = self._response_task
        stderr_task, self._stderr_task = self._stderr_task, None
        if process is None:
            return {}
        self._stopping = True
        response_metadata: dict[str, Any] = {}
        try:
            if graceful:
                with contextlib.suppress(BaseException):
                    response = await asyncio.wait_for(
                        self._request("shutdown"), timeout=10
                    )
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
            self._stopping = False
            self._process = None
            self._reader = None
            self._response_task = None
            if response_task is not None:
                response_task.cancel()
                with contextlib.suppress(BaseException):
                    await response_task
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
