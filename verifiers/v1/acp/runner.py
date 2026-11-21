"""Run harness segments through an ACP agent."""

import asyncio
import json
import os
import sys
from contextlib import AsyncExitStack, suppress
from dataclasses import dataclass
from typing import Any

import httpx
from acp import (
    PROTOCOL_VERSION,
    Client,
    image_block,
    spawn_agent_process,
    text_block,
)
from acp.schema import (
    AgentMessageChunk,
    AllowedOutcome,
    ClientCapabilities,
    DeniedOutcome,
    HttpHeader,
    HttpMcpServer,
    PermissionOption,
    RequestPermissionResponse,
    TextContentBlock,
)


@dataclass(frozen=True)
class ACPTurn:
    reply: str
    stop_reason: str | None
    response_metadata: dict[str, Any]
    update_metadata: list[dict[str, Any]]


class ToolGate:
    """The rollout's `/tool` gate, asked before every tool call the agent wants to run."""

    def __init__(self, url: str, secret: str) -> None:
        self.url = url
        self.client = httpx.AsyncClient(
            headers={"Authorization": f"Bearer {secret}"},
            # The host may execute a remote tool before answering. Its tool and
            # rollout deadlines own that work; timing out here could let A
            # continue while the denied-looking command still runs in B.
            timeout=httpx.Timeout(None, connect=5),
        )

    async def decision(self, tool_call_id: str, arguments: Any) -> str:
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
            response = await self.client.post(
                self.url, json={"tool_call_id": tool_call_id, "arguments": arguments}
            )
            response.raise_for_status()
            return response.json()["action"]
        except Exception as error:  # noqa: BLE001 - an unreachable gate lets nothing run
            print(f"tool gate denied {tool_call_id}: {error}", file=sys.stderr)
            return "deny"


class VerifiersACPClient(Client):
    def __init__(self) -> None:
        self.gate: ToolGate | None = None
        self.prompt_task: asyncio.Task | None = None
        self.visible_reply = ""
        self.message_id: str | None = None
        self.stop_reason: str | None = None
        self.response_metadata: dict[str, Any] = {}
        self.update_metadata: list[dict[str, Any]] = []
        self.tool_inputs: dict[tuple[str, str], Any] = {}

    def reset(self) -> None:
        self.visible_reply = ""
        self.message_id = None
        self.stop_reason = None
        self.response_metadata = {}
        self.update_metadata = []
        self.tool_inputs = {}

    def turn_result(self) -> ACPTurn:
        return ACPTurn(
            reply=self.visible_reply,
            stop_reason=self.stop_reason,
            response_metadata=self.response_metadata,
            update_metadata=self.update_metadata,
        )

    async def session_update(self, session_id: str, update: Any, **kwargs: Any) -> None:
        tool_call_id = getattr(update, "tool_call_id", None)
        raw_input = getattr(update, "raw_input", None)
        if tool_call_id is not None and raw_input is not None:
            self.tool_inputs[session_id, tool_call_id] = raw_input
        elif tool_call_id is not None and getattr(update, "status", None) in {
            "pending",
            "in_progress",
        }:
            # Kimi requests permission before its final rawInput update. Its
            # streaming updates replace one text block with cumulative JSON.
            content = getattr(update, "content", None) or []
            if len(content) == 1 and isinstance(
                block := getattr(content[0], "content", None), TextContentBlock
            ):
                with suppress(json.JSONDecodeError):
                    arguments = json.loads(block.text)
                    if isinstance(arguments, dict):
                        self.tool_inputs.setdefault(
                            (session_id, tool_call_id), arguments
                        )
        metadata = dict(kwargs)
        if isinstance(field_meta := getattr(update, "field_meta", None), dict):
            metadata.update(field_meta)
        if metadata:
            self.update_metadata.append(metadata)
        if isinstance(update, AgentMessageChunk) and isinstance(
            update.content, TextContentBlock
        ):
            message_id = getattr(update, "message_id", None)
            if message_id is not None and message_id != self.message_id:
                self.visible_reply = ""
                self.message_id = message_id
            self.visible_reply += update.content.text

    async def request_permission(
        self,
        session_id: str,
        tool_call: Any,
        options: list[PermissionOption],
        **kwargs: Any,
    ) -> RequestPermissionResponse:
        """Ask the rollout before execution: deny rejects one call; stop cancels the turn."""
        kinds = ("allow_once", "allow_always")
        if self.gate is not None:
            # Kimi and DeepSeek send argument-bearing tool updates first, then
            # an ID-only permission request. Preserve an explicitly supplied
            # request input; only fill the omitted field from this session.
            arguments = tool_call.raw_input
            if arguments is None:
                arguments = self.tool_inputs.get((session_id, tool_call.tool_call_id))
            decision = await self.gate.decision(tool_call.tool_call_id, arguments)
            if decision == "stop":
                self.stop_reason = "cancelled"
                if self.prompt_task is not None:
                    self.prompt_task.cancel()
                return RequestPermissionResponse(
                    outcome=DeniedOutcome(outcome="cancelled")
                )
            if decision != "allow":
                kinds = ("reject_once", "reject_always")
        option = next(
            (item for kind in kinds for item in options if item.kind == kind), None
        )
        outcome = (
            AllowedOutcome(outcome="selected", option_id=option.option_id)
            if option
            else DeniedOutcome(outcome="cancelled")
        )
        return RequestPermissionResponse(outcome=outcome)


def user_content_blocks(contents: list, supports_images: bool) -> list:
    """Render one user turn's ordered VF contents as ACP prompt blocks."""
    blocks = []
    for index, content in enumerate(contents):
        if index:
            blocks.append(text_block("\n\n"))
        content = content or ""
        parts = (
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


def mcp_servers(config: dict) -> list[HttpMcpServer]:
    return [
        HttpMcpServer(
            type="http",
            name=name,
            url=url,
            headers=[
                HttpHeader(name=key, value=value)
                for key, value in config.get("mcp_headers", {}).get(name, {}).items()
            ],
        )
        for name, url in config["mcp_urls"].items()
    ]


async def prompt(
    client: VerifiersACPClient,
    connection: Any,
    capabilities: Any,
    session_id: str,
    config: dict,
    *,
    is_new: bool,
) -> ACPTurn:
    client.reset()
    prompt_capabilities = capabilities and capabilities.prompt_capabilities
    supports_images = bool(prompt_capabilities and prompt_capabilities.image)
    blocks = []
    if is_new and config["system_prompt"]:
        blocks.append(text_block(f"(system)\n{config['system_prompt']}\n\n[user]\n"))
    blocks.extend(user_content_blocks(config["user_contents"], supports_images))
    if not blocks:
        raise ValueError("ACP prompt has no content")
    try:
        client.prompt_task = asyncio.create_task(
            connection.prompt(session_id=session_id, prompt=blocks)
        )
        response = await client.prompt_task
        client.stop_reason = response.stop_reason
        client.response_metadata = dict(response.field_meta or {})
    except asyncio.CancelledError:
        if client.stop_reason != "cancelled":
            raise
        await connection.cancel(session_id=session_id)
    finally:
        client.prompt_task = None
    return client.turn_result()


class ACPSession:
    """One live ACP process, connection, and session shared by several turns."""

    def __init__(self, *, env: dict[str, str] | None = None) -> None:
        self.client = VerifiersACPClient()
        self._env = env
        self._reset()

    def _reset(self) -> None:
        self.stack = AsyncExitStack()
        self.connection: Any = None
        self.capabilities: Any = None
        self.supports_steering = False
        self.session_id: str | None = None
        self.is_new = True

    async def start(self, config: dict) -> None:
        command = config["command"]
        proc_env = {**os.environ, **(self._env or {})}
        cwd = config.get("cwd") or os.getcwd()
        try:
            agent_process = await self.stack.enter_async_context(
                spawn_agent_process(
                    self.client,
                    command[0],
                    *command[1:],
                    cwd=cwd,
                    env=proc_env,
                    transport_kwargs={"stderr": None},
                )
            )
            self.connection = agent_process[0]
            initialized = await self.connection.initialize(
                protocol_version=PROTOCOL_VERSION,
                client_capabilities=ClientCapabilities.model_validate(
                    config["client_capabilities"]
                ),
            )
            required = config.get("required_agent_metadata")
            if required and required not in (initialized.field_meta or {}):
                raise ValueError(f"ACP agent does not advertise {required}")
            self.capabilities = initialized.agent_capabilities
            self.supports_steering = (initialized.field_meta or {}).get(
                "steering", {}
            ).get("supported") is True
            session = await self.connection.new_session(
                cwd=config.get("cwd") or os.getcwd(),
                mcp_servers=mcp_servers(config),
                **config["session_meta"],
            )
            if title := config.get("session_title"):
                await self.connection.ext_method(
                    "session/rename", {"sessionId": session.session_id, "title": title}
                )
            if ready_file := config.get("ready_file"):
                deadline = asyncio.get_running_loop().time() + 30
                while not os.path.isfile(ready_file):
                    if asyncio.get_running_loop().time() >= deadline:
                        raise RuntimeError(
                            "Native tool gate did not activate before the first prompt"
                        )
                    await asyncio.sleep(0.05)
        except BaseException:
            with suppress(BaseException):
                await self.stack.aclose()
            self._reset()
            raise
        self.session_id = session.session_id
        self.is_new = True

    async def run(self, config: dict) -> ACPTurn:
        gate = config.get("tool_interception")
        if gate and self.client.gate is None:
            self.client.gate = ToolGate(gate["url"], gate["secret"])
        if self.connection is None:
            await self.start(config)
        assert self.session_id is not None
        result = await prompt(
            self.client,
            self.connection,
            self.capabilities,
            self.session_id,
            config,
            is_new=self.is_new,
        )
        self.is_new = False
        return result

    async def steer(self, message: str, message_id: str | None = None) -> dict:
        if self.connection is None or self.session_id is None:
            return {"outcome": "promptRequired", "reason": "noRunningTurn"}
        if not self.supports_steering:
            raise NotImplementedError("ACP agent does not advertise steering support")
        params = {
            "sessionId": self.session_id,
            "prompt": [{"type": "text", "text": message}],
            "_meta": {"steering": {"idleBehavior": "promptRequired"}},
        }
        if message_id is not None:
            params["messageId"] = message_id
        return await self.connection.ext_method("session/steering", params)

    async def close(self, *, graceful: bool = True) -> dict[str, Any]:
        response_metadata: dict[str, Any] = {}
        try:
            if graceful and self.connection is not None and self.session_id is not None:
                session_capabilities = (
                    self.capabilities and self.capabilities.session_capabilities
                )
                if session_capabilities and session_capabilities.close is not None:
                    with suppress(Exception):
                        response = await self.connection.close_session(
                            session_id=self.session_id
                        )
                        response_metadata = dict(response.field_meta or {})
        finally:
            try:
                await self.stack.aclose()
            finally:
                if self.client.gate is not None:
                    await self.client.gate.client.aclose()
                    self.client.gate = None
                self._reset()
        return response_metadata
