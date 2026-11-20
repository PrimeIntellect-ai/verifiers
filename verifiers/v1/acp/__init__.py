"""Public Agent Client Protocol support for harness programs."""

import asyncio
from abc import abstractmethod
from dataclasses import dataclass
from typing import Any, TypeAlias, TypeVar, cast

from pydantic import BaseModel, ConfigDict, Field

from verifiers.v1.clients import ModelContext
from verifiers.v1.configs.harness import HarnessConfig
from verifiers.v1.errors import HarnessError, InterceptionError
from verifiers.v1.harness import Harness, HarnessSession
from verifiers.v1.runtimes import ProgramResult, Runtime
from verifiers.v1.runtimes.harness_daemon import (
    HarnessDaemonClient,
    ensure_backend_daemon,
)
from verifiers.v1.semantic import (
    ACP_SEMANTIC_EDGES_METADATA_KEY,
    SemanticEdgeSet,
)
from verifiers.v1.task import TaskData
from verifiers.v1.trace import Trace
from verifiers.v1.types import Messages
from verifiers.v1.utils.aio import run_shielded

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
    """The session's working directory, when it is not the process's own, such as a
    split deployment's workspace."""
    session_meta: JsonObject | None = None
    client_capabilities: JsonObject | None = None
    required_agent_metadata: str | None = None
    ready_file: str | None = None
    """A trusted native plugin must create this file before the first prompt."""
    session_title: str | None = None


class ACPHarness(Harness[ConfigT]):
    """Harness backed by one live ACP process and native session per rollout."""

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
        self._daemon: HarnessDaemonClient | None = None
        self._daemon_key = f"{trace.id}:{id(self)}"
        self._graceful_close: bool = True

    async def _ensure_daemon(self) -> HarnessDaemonClient:
        if self._daemon is None or not self._daemon.alive:
            daemon = await ensure_backend_daemon(self.runtime.backend)
            if daemon is None:
                raise HarnessError("harness runtime does not support ACP daemon RPC")
            self._daemon = daemon
        return self._daemon

    async def warm(self) -> None:
        async with self._lock:
            if not self._closed:
                await self._ensure_daemon()

    async def steer(self, message: str, *, message_id: str | None = None) -> dict:
        if self._closed:
            raise RuntimeError("ACP session is closed")
        if not isinstance(message, str) or not message.strip():
            raise ValueError("steering requires a nonempty user message")
        if self._daemon is None:
            return {"outcome": "promptRequired", "reason": "noRunningTurn"}
        result = await self._daemon.acp_steer(
            self._daemon_key, message, message_id=message_id
        )
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
            "cwd": self.config.cwd or self.runtime.workdir,
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
            daemon = await self._ensure_daemon()
            calls_before = len(self.trace.calls)
            try:
                response = await daemon.acp_prompt(
                    session_key=self._daemon_key,
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
        if self._daemon is None:
            return {}
        daemon, self._daemon = self._daemon, None
        res = await daemon.acp_close(
            self._daemon_key,
            graceful=graceful and self._graceful_close,
        )
        meta = res.get("response_metadata") if isinstance(res, dict) else None
        return meta if isinstance(meta, dict) else {}

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
