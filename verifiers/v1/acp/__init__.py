"""Public Agent Client Protocol support for harness programs."""

import asyncio
import contextlib
import json
import logging
from abc import abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any, TypeAlias, TypeVar, cast

import httpx
from acp import (
    PROTOCOL_VERSION,
    Client,
    RequestError,
    connect_to_agent,
    image_block,
    text_block,
)
from acp.client import ClientSideConnection
from acp.helpers import ContentBlock
from acp.schema import (
    AgentCapabilities,
    AgentMessageChunk,
    AllowedOutcome,
    ClientCapabilities,
    DeniedOutcome,
    HttpHeader,
    HttpMcpServer,
    PermissionOption,
    PromptResponse,
    RequestPermissionResponse,
    TextContentBlock,
    ToolCallUpdate,
)
from pydantic import BaseModel, ConfigDict, Field

from verifiers.v1.acp.transport import ProcessTransport
from verifiers.v1.clients import ModelContext
from verifiers.v1.configs.harness import HarnessConfig
from verifiers.v1.errors import HarnessError
from verifiers.v1.harness import Harness, HarnessSession
from verifiers.v1.runtimes import ProgramResult, Runtime, RuntimeProcess
from verifiers.v1.semantic import ACP_SEMANTIC_EDGES_METADATA_KEY, SemanticEdgeSet
from verifiers.v1.task import TaskData
from verifiers.v1.trace import Trace
from verifiers.v1.types import Messages
from verifiers.v1.utils.aio import run_shielded

__all__ = ["ACPConfig", "ACPHarness", "ACPTurn"]

logger = logging.getLogger(__name__)
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
        """Configure the agent to gate every tool through ACP permissions or its native
        hooks. `url` is the `/tool` endpoint as reached from the agent's runtime."""

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
            await self.gate_tools(
                config, runtime, runtime.host_url(tool_interception_url), secret
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


def user_content_blocks(contents: list, supports_images: bool) -> list[ContentBlock]:
    """Render one user turn's ordered VF contents as ACP prompt blocks."""
    blocks: list[ContentBlock] = []
    for index, content in enumerate(contents):
        if index:
            blocks.append(text_block("\n\n"))
        content = content or ""
        parts: list[dict[str, Any]] = (
            [{"type": "text", "text": content}] if isinstance(content, str) else content
        )
        for part in parts:
            if part["type"] == "text":
                blocks.append(text_block(part["text"]))
                continue
            if not supports_images:
                raise ValueError("ACP agent does not support image prompts")
            url = part["image_url"]["url"]
            metadata, separator, data = url.partition(",")
            media_type, *parameters = metadata.removeprefix("data:").split(";")
            if (
                not separator
                or not metadata.startswith("data:image/")
                or not any(value.lower() == "base64" for value in parameters)
            ):
                raise ValueError("ACP image prompts require base64 data:image URLs")
            blocks.append(image_block(data, media_type))
    return blocks


class ACPHarnessSession(HarnessSession, Client):
    """Own the process, SDK connection, native session, and callbacks for a rollout."""

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
        self._connection: ClientSideConnection | None = None
        self._capabilities: AgentCapabilities | None = None
        self._session_id: str | None = None
        self._is_new = True
        self._turn = ACPTurn(reply="")
        self._message_id: str | None = None
        self._prompt_task: asyncio.Task[PromptResponse] | None = None
        self._gate: httpx.AsyncClient | None = None
        self._stderr_tail = bytearray()
        self._stderr_task: asyncio.Task[None] | None = None
        self._lock = asyncio.Lock()

    async def _start(self) -> None:
        self._stderr_tail.clear()
        cwd = await self.runtime.run(["pwd", "-P"], {})
        if cwd.exit_code:
            raise RuntimeError(f"ACP working directory lookup failed: {cwd.stderr}")
        self._process = await self.runtime.open_process(
            self.config.command, self.config.env
        )
        self._stderr_task = asyncio.create_task(
            self._drain_stderr(self._process.stderr)
        )
        self._connection = connect_to_agent(self, ProcessTransport(self._process))
        if self.tool_interception_url is not None:
            self._gate = httpx.AsyncClient(
                headers={"Authorization": f"Bearer {self.secret}"},
                timeout=httpx.Timeout(120, connect=5),
            )
        initialized = await self._connection.initialize(
            protocol_version=PROTOCOL_VERSION,
            client_capabilities=ClientCapabilities.model_validate(
                self.config.client_capabilities or {}
            ),
        )
        self._capabilities = initialized.agent_capabilities
        headers = self.harness.config.resolve_mcp_headers(self.mcp_urls)
        metadata: dict[str, Any] = self.config.session_meta or {}
        session = await self._connection.new_session(
            cwd=cwd.stdout.removesuffix("\n"),
            mcp_servers=[
                HttpMcpServer(
                    type="http",
                    name=name,
                    url=url,
                    headers=[
                        HttpHeader(name=key, value=value)
                        for key, value in headers.get(name, {}).items()
                    ],
                )
                for name, url in self.mcp_urls.items()
            ],
            **metadata,
        )
        self._session_id = session.session_id
        self._is_new = True

    async def _drain_stderr(self, stream: AsyncIterator[bytes]) -> None:
        async for chunk in stream:
            self._stderr_tail.extend(chunk)
            if len(self._stderr_tail) > 4000:
                del self._stderr_tail[:-4000]

    async def session_update(self, session_id: str, update: Any, **kwargs: Any) -> None:
        metadata = dict(kwargs)
        if isinstance(field_meta := getattr(update, "field_meta", None), dict):
            metadata.update(field_meta)
        if metadata:
            self._turn.update_metadata.append(metadata)
        if isinstance(update, AgentMessageChunk) and isinstance(
            update.content, TextContentBlock
        ):
            message_id = getattr(update, "message_id", None)
            if message_id is not None and message_id != self._message_id:
                self._turn.reply = ""
                self._message_id = message_id
            self._turn.reply += update.content.text

    async def request_permission(
        self,
        session_id: str,
        tool_call: ToolCallUpdate,
        options: list[PermissionOption],
        **kwargs: Any,
    ) -> RequestPermissionResponse:
        """Deny rejects one call; stop cancels the turn while retaining its partial reply."""
        decision = "allow"
        if self._gate is not None:
            assert self.tool_interception_url is not None
            tool_call_id, arguments = tool_call.tool_call_id, tool_call.raw_input
            try:
                # pi-acp wraps extension confirmations in a separate UI permission call.
                if (
                    tool_call_id.startswith("pi-ui-")
                    and isinstance(arguments, dict)
                    and arguments.get("method") == "confirm"
                ):
                    tool_call_id, arguments = (
                        arguments["title"],
                        json.loads(arguments["message"]),
                    )
                response = await self._gate.post(
                    self.tool_interception_url,
                    json={"tool_call_id": tool_call_id, "arguments": arguments},
                )
                response.raise_for_status()
                decision = response.json()["action"]
            except Exception as error:  # noqa: BLE001 - an unreachable gate lets nothing run
                logger.warning("Tool gate denied %s: %s", tool_call_id, error)
                decision = "deny"
        if decision == "stop":
            self._turn.stop_reason = "cancelled"
            if self._prompt_task is not None:
                self._prompt_task.cancel()
            return RequestPermissionResponse(outcome=DeniedOutcome(outcome="cancelled"))
        kinds = (
            ("allow_once", "allow_always")
            if decision == "allow"
            else ("reject_once", "reject_always")
        )
        option = next(
            (item for kind in kinds for item in options if item.kind == kind), None
        )
        return RequestPermissionResponse(
            outcome=AllowedOutcome(outcome="selected", option_id=option.option_id)
            if option
            else DeniedOutcome(outcome="cancelled")
        )

    async def _run(self, messages: Messages | None) -> ProgramResult:
        prompt = self.config.prompt if messages is None else messages
        if prompt is None:
            raise ValueError("ACP requires a prompt")
        if not isinstance(prompt, str) and (
            not prompt or any(message.role != "user" for message in prompt)
        ):
            raise ValueError("an ACP turn must contain user messages only")
        contents = (
            [prompt]
            if isinstance(prompt, str)
            else [
                message.model_dump(mode="json", include={"content"})["content"]
                for message in prompt
            ]
        )
        async with self._lock:
            if self._closed:
                raise HarnessError(
                    f"harness {self.harness.config.id!r} session is already closed"
                )
            self._turn = ACPTurn(reply="")
            self._message_id = None
            calls_before = len(self.trace.calls)
            try:
                if self._process is None:
                    await self._start()
                    self._turn = ACPTurn(reply="")
                    self._message_id = None
                assert self._connection is not None
                assert self._session_id is not None
                capabilities = (
                    self._capabilities and self._capabilities.prompt_capabilities
                )
                blocks = user_content_blocks(
                    contents, bool(capabilities and capabilities.image)
                )
                if self._is_new and self.config.system_prompt:
                    blocks.insert(
                        0,
                        text_block(
                            f"(system)\n{self.config.system_prompt}\n\n[user]\n"
                        ),
                    )
                if not blocks:
                    raise ValueError("ACP prompt has no content")
                self._prompt_task = asyncio.create_task(
                    self._connection.prompt(session_id=self._session_id, prompt=blocks)
                )
                try:
                    response = await self._prompt_task
                    self._turn.stop_reason = response.stop_reason
                    self._turn.response_metadata = dict(response.field_meta or {})
                except asyncio.CancelledError:
                    # A gate stop is a completed partial turn; external cancellation tears down.
                    task = asyncio.current_task()
                    if self._turn.stop_reason != "cancelled" or (
                        task is not None and task.cancelling()
                    ):
                        raise
                    await self._connection.cancel(session_id=self._session_id)
                finally:
                    self._prompt_task = None
                self._is_new = False
            except BaseException as error:
                self.trace.root_reply = self._turn.reply.strip()
                await run_shielded(self._stop(graceful=False))
                if not isinstance(error, Exception):
                    raise
                detail = (
                    error.data.get("details")
                    if isinstance(error, RequestError) and isinstance(error.data, dict)
                    else None
                ) or str(error)
                if stderr := self._stderr_tail.decode(errors="replace").strip():
                    detail = f"{detail}\n\nACP process stderr:\n{stderr}"
                raise RuntimeError(detail) from error
            turn = self._turn.model_copy(deep=True)
            self.trace.root_reply = turn.reply.strip()
            harness = cast(ACPHarness, self.harness)
            harness._consume_protocol_metadata(self.trace, turn.response_metadata)
            harness.acp_turn_result(self.trace, turn)
            if self.trace.stop_condition is None and not any(
                call.node is not None for call in self.trace.calls[calls_before:]
            ):
                detail = turn.reply.strip()[-500:] or "<no output>"
                raise RuntimeError(
                    "ACP agent completed without committing a model turn: " + detail
                )
            return ProgramResult(exit_code=0, stdout=turn.reply, stderr="")

    async def _stop(self, *, graceful: bool) -> dict[str, Any]:
        process, self._process = self._process, None
        connection, self._connection = self._connection, None
        gate = self._gate
        stderr_task, self._stderr_task = self._stderr_task, None
        session_id, self._session_id = self._session_id, None
        response_metadata: dict[str, Any] = {}
        try:
            if connection is not None:
                capabilities = (
                    self._capabilities and self._capabilities.session_capabilities
                )
                if (
                    graceful
                    and session_id
                    and capabilities
                    and capabilities.close is not None
                ):
                    with contextlib.suppress(Exception):
                        response = await asyncio.wait_for(
                            connection.close_session(session_id=session_id), timeout=10
                        )
                        if response is not None:
                            response_metadata = dict(response.field_meta or {})
                with contextlib.suppress(Exception):
                    # SDK shutdown flushes queued writes; a stalled runtime must not
                    # prevent process termination.
                    await asyncio.wait_for(connection.close(), timeout=5)
            if process is not None:
                for stop in (process.terminate, process.kill):
                    with contextlib.suppress(Exception):
                        await stop()
                    try:
                        await asyncio.wait_for(process.wait(), timeout=5)
                        break
                    except TimeoutError:
                        continue
        finally:
            if stderr_task is not None:
                if not stderr_task.done():
                    stderr_task.cancel()
                with contextlib.suppress(BaseException):
                    await stderr_task
            if gate is not None:
                self._gate = None
                await gate.aclose()
        return response_metadata

    async def close(self) -> None:
        if self._closed:
            return
        # Publish closure before taking the turn lock, so queued turns cannot restart it.
        await super().close()

        async def close_process() -> None:
            async with self._lock:
                response_metadata = await self._stop(graceful=True)
                if response_metadata:
                    harness = cast(ACPHarness, self.harness)
                    harness._consume_protocol_metadata(self.trace, response_metadata)
                    harness.acp_close_result(self.trace, response_metadata)

        await run_shielded(close_process())
