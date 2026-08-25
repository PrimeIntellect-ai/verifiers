"""Public Agent Client Protocol support for harness programs."""

import asyncio
import contextlib
import json
import logging
from abc import abstractmethod
from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TypeAlias, TypeVar

from verifiers.v1.clients import ModelContext
from verifiers.v1.configs.harness import HarnessConfig
from verifiers.v1.errors import HarnessError, HarnessFinalizationError
from verifiers.v1.harness import Harness, HarnessSession
from verifiers.v1.runtimes import ProgramResult, Runtime, RuntimeProcess
from verifiers.v1.task import TaskData
from verifiers.v1.trace import Trace
from verifiers.v1.types import Messages
from verifiers.v1.utils.aio import run_shielded

ACP_SOURCE = (Path(__file__).resolve().parent / "runner.py").read_text()
MAX_PACKET_BYTES = 128 * 1024 * 1024
logger = logging.getLogger(__name__)

__all__ = [
    "ACPCloseResult",
    "ACPConfig",
    "ACPHarness",
    "ACPTurnResult",
]

ConfigT = TypeVar("ConfigT", bound=HarnessConfig)
JsonValue: TypeAlias = (
    str | int | float | bool | None | list["JsonValue"] | dict[str, "JsonValue"]
)
JsonObject: TypeAlias = dict[str, JsonValue]


@dataclass(frozen=True)
class ACPTurnResult:
    """One prompt's visible reply and ACP lifecycle metadata."""

    reply: str
    stop_reason: str | None
    update_metadata: tuple[JsonObject, ...]


@dataclass(frozen=True)
class ACPCloseResult:
    """Metadata returned when an ACP session closes."""

    metadata: JsonObject


@dataclass
class ACPConfig:
    """One harness's ACP process and initial prompt."""

    env: dict[str, str]
    command: list[str]
    prompt: str | Messages | None
    mcp_urls: dict[str, str] | None = None
    system_prompt: str | None = None
    session_meta: JsonObject | None = None
    required_agent_meta: JsonObject | None = None


class ACPHarness(Harness[ConfigT]):
    """Harness backed by one live ACP process and native session per rollout."""

    async def setup(self, runtime: Runtime) -> None:
        await runtime.prepare_uv_script(
            ACP_SOURCE, {**self.config.resolved_env, "UV_FROZEN": "false"}
        )

    def acp_turn_result(self, trace: Trace, result: ACPTurnResult) -> None:
        """Consume the typed result of one ACP prompt."""

    def acp_close_metrics(
        self, trace: Trace, result: ACPCloseResult
    ) -> Mapping[str, float]:
        """Extract harness metrics from a successful ACP session/close response."""
        return {}

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


def _turn_result(response: JsonObject) -> ACPTurnResult:
    value = response.get("result")
    if not isinstance(value, dict):
        raise TypeError("ACP prompt result must be an object")
    reply = value.get("reply")
    if not isinstance(reply, str):
        raise TypeError("ACP prompt reply must be a string")
    stop_reason = value.get("stop_reason")
    if stop_reason is not None and not isinstance(stop_reason, str):
        raise TypeError("ACP stop reason must be a string or null")
    update_metadata = value.get("update_metadata")
    if not isinstance(update_metadata, list) or any(
        not isinstance(item, dict) for item in update_metadata
    ):
        raise TypeError("ACP update metadata must be a list of objects")
    return ACPTurnResult(
        reply=reply,
        stop_reason=stop_reason,
        update_metadata=tuple(dict(item) for item in update_metadata),
    )


def _close_result(response: JsonObject) -> ACPCloseResult:
    value = response.get("result")
    if not isinstance(value, dict):
        raise TypeError("ACP close result must be an object")
    metadata = value.get("metadata")
    if not isinstance(metadata, dict):
        raise TypeError("ACP close metadata must be an object")
    return ACPCloseResult(metadata=dict(metadata))


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
        self._acp_harness = harness
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
            "required_agent_meta": self.config.required_agent_meta or {},
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
        try:
            self._acp_harness.acp_turn_result(self.trace, turn)
        except Exception:
            if response.get("ok"):
                raise
            logger.warning(
                "ACP failed-turn result could not be consumed", exc_info=True
            )
        if not response.get("ok"):
            detail = response.get("error") or "ACP session request failed"
            if stderr := self._stderr():
                detail = f"{detail}\n\nACP process stderr:\n{stderr}"
            raise RuntimeError(detail)
        result = ProgramResult(exit_code=0, stdout=turn.reply, stderr="")
        _require_model_turn(self.trace, calls_before, result)
        self.trace.primary_reply = turn.reply.strip()
        return result

    async def _stop(self, *, graceful: bool) -> None:
        process, self._process = self._process, None
        reader, self._reader = self._reader, None
        stderr_task, self._stderr_task = self._stderr_task, None
        if process is None:
            return
        finalization_error: HarnessFinalizationError | None = None
        finalization_cause: BaseException | None = None
        shutdown_error: BaseException | None = None
        shutdown_detail: str | None = None
        try:
            close_result = ACPCloseResult(metadata={})
            if graceful and reader is not None:
                response: JsonObject | None = None
                try:
                    await process.write(_packet({"operation": "shutdown"}))
                    response = await asyncio.wait_for(reader.read(), timeout=10)
                    if not response.get("ok"):
                        raise RuntimeError(
                            response.get("error") or "ACP session shutdown failed"
                        )
                    close_result = _close_result(response)
                except BaseException as error:
                    logger.warning("ACP session shutdown failed", exc_info=True)
                    shutdown_error = error
                    shutdown_detail = f"{type(error).__name__}: {error}"
                    if stderr := self._stderr():
                        shutdown_detail = (
                            f"{shutdown_detail}\n\nACP process stderr:\n{stderr}"
                        )
                    if response is not None and response.get("ok"):
                        finalization_error = HarnessFinalizationError(
                            f"ACP close result is invalid: {shutdown_detail}"
                        )
                        finalization_cause = error
            if graceful and finalization_error is None:
                try:
                    metrics = self._acp_harness.acp_close_metrics(
                        self.trace, close_result
                    )
                    self.trace.record_metrics(metrics)
                except Exception as error:  # noqa: BLE001 - extension boundary
                    if shutdown_error is not None:
                        finalization_error = HarnessFinalizationError(
                            "ACP session shutdown failed and required finalization "
                            f"could not complete: {shutdown_detail}; "
                            f"{type(error).__name__}: {error}"
                        )
                        finalization_cause = shutdown_error
                    else:
                        finalization_error = HarnessFinalizationError(
                            "ACP session finalization failed: "
                            f"{type(error).__name__}: {error}"
                        )
                        finalization_cause = error
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
            if finalization_error is not None:
                raise finalization_error from finalization_cause
        finally:
            if stderr_task is not None:
                if not stderr_task.done():
                    stderr_task.cancel()
                with contextlib.suppress(BaseException):
                    await stderr_task

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
