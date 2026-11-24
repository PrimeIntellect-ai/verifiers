"""Public Agent Client Protocol support for harness programs."""

import asyncio
import contextlib
import json
import shlex
import sys
import uuid
from abc import abstractmethod
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeAlias, TypeVar, cast

from pydantic import BaseModel, ConfigDict, Field

from verifiers.v1.clients import ModelContext
from verifiers.v1.configs.harness import HarnessConfig
from verifiers.v1.errors import HarnessError, InterceptionError, SandboxError
from verifiers.v1.harness import Harness, HarnessSession
from verifiers.v1.runtimes import ProgramResult, Runtime, RuntimeProcess
from verifiers.v1.runtimes.base import _ENSURE_UV
from verifiers.v1.semantic import (
    ACP_SEMANTIC_EDGES_METADATA_KEY,
    SemanticEdgeSet,
)
from verifiers.v1.task import TaskData
from verifiers.v1.trace import Trace
from verifiers.v1.types import Messages
from verifiers.v1.utils.aio import run_shielded

_ACP_PKG = "agent-client-protocol==0.12.1"
_RUNNER_VENV = "/var/tmp/vf-acp-venv"
_MAX_FRAME = 32 * 1024 * 1024
_RUNNER_SOURCE = (Path(__file__).resolve().parent / "runner.py").read_text()

_STOP_CONDITIONS = {
    "max_tokens": "max_output_tokens",
    "max_turn_requests": "max_turns",
    "refusal": "refusal",
    "cancelled": "cancelled",
}

__all__ = ["ACPConfig", "ACPHarness", "ACPTurn"]

ConfigT = TypeVar("ConfigT", bound=HarnessConfig)
JsonValue: TypeAlias = (
    str | int | float | bool | None | list["JsonValue"] | dict[str, "JsonValue"]
)
JsonObject: TypeAlias = dict[str, JsonValue]


async def _prepare_acp_python(runtime: Runtime) -> list[str]:
    """Resolve an interpreter in the harness sandbox with `agent-client-protocol` installed."""
    if getattr(runtime, "type", "") not in ("docker", "prime"):
        return [
            "uv",
            "run",
            "--quiet",
            "--no-project",
            "--python",
            sys.executable,
            "--with",
            _ACP_PKG,
            "--with",
            "httpx>=0.27.0",
            "python",
        ]
    root_rt = runtime.backend.bind(env=runtime.env).with_user(None)
    cache = root_rt.backend.cache
    if "acp-python" in cache.uv_interpreters:
        return [cache.uv_interpreters["acp-python"]]
    async with cache.setup_lock:
        if "acp-python" in cache.uv_interpreters:
            return [cache.uv_interpreters["acp-python"]]
        check = "import importlib.metadata as m, httpx; assert m.version('agent-client-protocol') == '0.12.1'"
        venv_q = shlex.quote(_RUNNER_VENV)
        cmd = (
            "set -e; "
            "sys_py=$(command -v python3 || command -v python); "
            f'if [ -n "$sys_py" ] && "$sys_py" -I -c "{check}" >/dev/null 2>&1; then printf "%s\\n" "$sys_py"; exit 0; fi; '
            f'if [ -x {venv_q}/bin/python ] && {venv_q}/bin/python -I -c "{check}" >/dev/null 2>&1; then printf "%s\\n" "{_RUNNER_VENV}/bin/python"; exit 0; fi; '
            f"{{ {_ENSURE_UV}; }}; "
            f"uv venv --allow-existing {venv_q} -q; "
            f"uv pip install --python {venv_q} {shlex.quote(_ACP_PKG)} 'httpx>=0.27.0' -q; "
            f'printf "%s\\n" "{_RUNNER_VENV}/bin/python"'
        )
        res = await root_rt.run(["sh", "-c", cmd], {})
        if res.exit_code != 0 or not res.stdout.strip():
            raise HarnessError(
                f"ACP runner setup failed: {(res.stderr or res.stdout).strip()[-1000:]}"
            )
        python = res.stdout.strip().splitlines()[-1]
        cache.uv_interpreters["acp-python"] = python
        return [python]


class _ACPRunnerRPC:
    """Host-side framed stdio client to one `acp/runner.py` process in the harness sandbox."""

    def __init__(self, process: RuntimeProcess) -> None:
        self.process = process
        self._pending: dict[str, asyncio.Future] = {}
        self._write_lock = asyncio.Lock()
        self._error: BaseException | None = None
        self._stderr = bytearray()
        self._tasks = [
            asyncio.create_task(self._receive()),
            asyncio.create_task(self._drain()),
        ]

    async def _receive(self) -> None:
        buffer = bytearray()
        try:
            async for chunk in self.process.stdout:
                buffer.extend(chunk)
                while len(buffer) >= 4:
                    size = int.from_bytes(buffer[:4], "big")
                    if size > _MAX_FRAME:
                        raise RuntimeError("ACP runner frame exceeds 32 MiB")
                    if len(buffer) < size + 4:
                        break
                    message = json.loads(buffer[4 : 4 + size])
                    del buffer[: 4 + size]
                    future = self._pending.get(message["id"])
                    if future is not None and not future.done():
                        if "error" in message:
                            future.set_exception(SandboxError(message["error"]))
                        else:
                            future.set_result(message["result"])
            raise SandboxError(
                f"ACP runner closed - {self._stderr.decode(errors='replace')}"
            )
        except BaseException as error:  # noqa: BLE001
            self._error = error
            for future in self._pending.values():
                if not future.done():
                    future.set_exception(error)

    async def _drain(self) -> None:
        async for chunk in self.process.stderr:
            self._stderr.extend(chunk)
            del self._stderr[:-4000]

    async def call(self, operation: str, **values: Any) -> Any:
        if self._error is not None:
            raise self._error
        request_id = uuid.uuid4().hex
        future = asyncio.get_running_loop().create_future()
        self._pending[request_id] = future
        data = json.dumps({**values, "id": request_id, "operation": operation}).encode()
        if len(data) > _MAX_FRAME:
            raise RuntimeError("ACP runner frame exceeds 32 MiB")
        try:
            async with self._write_lock:
                await self.process.write(len(data).to_bytes(4, "big") + data)
            return await future
        finally:
            self._pending.pop(request_id, None)

    async def close(self, *, abandon: bool = False) -> None:
        try:
            if abandon:
                with contextlib.suppress(Exception):
                    await self.process.abandon()
            else:
                with contextlib.suppress(Exception):
                    await self.process.kill()
        finally:
            for task in self._tasks:
                task.cancel()
            await asyncio.gather(*self._tasks, return_exceptions=True)


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
    cwd: str | None = None
    """The session's working directory, such as the worker sandbox's workdir."""
    session_meta: JsonObject | None = None
    client_capabilities: JsonObject | None = None
    required_agent_metadata: str | None = None
    ready_file: str | None = None
    """A trusted native plugin must create this file before the first prompt."""
    session_title: str | None = None


class ACPHarness(Harness[ConfigT]):
    """Harness backed by one live ACP process and native session per rollout."""

    async def setup(self, runtime: Runtime) -> None:
        await _prepare_acp_python(runtime)

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
        tool_interception_url: str | None = None,
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
        config = await self.prepare_acp(
            ctx,
            trace,
            runtime,
            endpoint,
            secret,
            mcp_urls,
            data,
            tool_interception_url=tool_interception_url,
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


def _require_model_turn(trace: Trace, calls_before: int, result: ProgramResult) -> None:
    if (
        result.exit_code
        or trace.stop_condition is not None
        or any(call.node is not None for call in trace.calls[calls_before:])
    ):
        return
    detail = (result.stderr or result.stdout).strip()[-500:] or "<no output>"
    raise RuntimeError("ACP agent completed without committing a model turn: " + detail)


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
        self._lock = asyncio.Lock()
        self._rpc: _ACPRunnerRPC | None = None
        self._graceful_close: bool = True

    async def _ensure_runner(self) -> _ACPRunnerRPC:
        if self._rpc is None:
            root_rt = self.runtime.backend.bind(env={}).with_user(None)
            command = await _prepare_acp_python(root_rt)
            process = await root_rt.exec(
                [*command, "-I", "-u", "-c", _RUNNER_SOURCE], {}
            )
            rpc = _ACPRunnerRPC(process)
            try:
                await asyncio.wait_for(rpc.call("ready"), timeout=45)
            except BaseException:
                await rpc.close()
                raise
            self._rpc = rpc
        return self._rpc

    async def steer(self, message: str, *, message_id: str | None = None) -> dict:
        if self._closed:
            raise RuntimeError("ACP session is closed")
        if not isinstance(message, str) or not message.strip():
            raise ValueError("steering requires a nonempty user message")
        if self._rpc is None:
            return {"outcome": "promptRequired", "reason": "noRunningTurn"}
        res = await self._rpc.call("acp_steer", message=message, message_id=message_id)
        if (
            isinstance(res, dict)
            and res.get("ok") is False
            and "not_implemented" in res
        ):
            raise NotImplementedError(res["not_implemented"])
        result = res["result"] if isinstance(res, dict) and "result" in res else res
        if not isinstance(result, dict):
            raise TypeError("invalid ACP steering receipt")
        return result

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
            "cwd": self.config.cwd or getattr(self.runtime.info, "workdir", None),
            "ready_file": self.config.ready_file,
            "user_contents": user_contents,
            "mcp_urls": self.mcp_urls,
            "mcp_headers": self.harness.config.resolve_mcp_headers(self.mcp_urls),
            "system_prompt": self.config.system_prompt or "",
            "session_meta": self.config.session_meta or {},
            "client_capabilities": self.config.client_capabilities or {},
            "required_agent_metadata": self.config.required_agent_metadata,
            "session_title": self.config.session_title,
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
            rpc = await self._ensure_runner()
            calls_before = len(self.trace.calls)
            try:
                response = await rpc.call(
                    "acp_prompt",
                    env=self.config.env,
                    config=config,
                )
            except BaseException:
                await run_shielded(self._stop(graceful=False))
                raise
        turn = ACPTurn.model_validate(response.get("result"))
        self.trace.root_reply = turn.reply.strip()
        self.trace.info["acp_stop_reason"] = turn.stop_reason
        if not response.get("ok"):
            detail = response.get("error") or "ACP session request failed"
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
        if condition := _STOP_CONDITIONS.get(turn.stop_reason or ""):
            self.trace.stop(condition)
        result = ProgramResult(exit_code=0, stdout=turn.reply, stderr="")
        _require_model_turn(self.trace, calls_before, result)
        return result

    async def _stop(self, *, graceful: bool) -> dict[str, Any]:
        if self._rpc is None:
            return {}
        rpc, self._rpc = self._rpc, None
        meta: dict[str, Any] = {}
        try:
            if rpc._error is None:
                with suppress(Exception):
                    meta = await rpc.call(
                        "acp_close", graceful=graceful and self._graceful_close
                    )
        finally:
            await rpc.close(abandon=not graceful or not self._graceful_close)
        return meta

    async def close(self) -> None:
        if self._closed:
            return
        await super().close()

        async def close_process() -> None:
            async with self._lock:
                response_metadata = await self._stop(graceful=True)
                if response_metadata:
                    harness = cast(ACPHarness, self.harness)
                    harness._consume_protocol_metadata(self.trace, response_metadata)
                    harness.acp_close_result(self.trace, response_metadata)

        await run_shielded(close_process())
