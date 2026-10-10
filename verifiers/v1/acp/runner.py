# /// script
# requires-python = ">=3.10,<3.15"
# dependencies = ["agent-client-protocol==0.12.1", "httpx"]
# ///
"""Run harness segments through an ACP agent."""

import asyncio
import json
import os
import random
import signal
import sys
import time
import traceback
from contextlib import AsyncExitStack, suppress
from dataclasses import asdict, dataclass
from typing import Any

import httpx
from acp import (
    PROTOCOL_VERSION,
    Client,
    RequestError,
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

MAX_PACKET_BYTES = 128 * 1024 * 1024
# The rollout stamps its answers; an unstamped one came from a tunnel or proxy in between.
STAMP_HEADER = "x-verifiers-interception"
RETRY_SECONDS = 300.0


@dataclass(frozen=True)
class ACPTurn:
    reply: str
    stop_reason: str | None
    response_metadata: dict[str, Any]
    update_metadata: list[dict[str, Any]]


class ToolGate:
    """The rollout's `/tool` gate, asked before every tool call the agent wants to run."""

    def __init__(self, url: str, secret: str, failed: str | None = None) -> None:
        self.url = url
        self.failed = failed
        """A file the agent's own gate hook writes when it couldn't ask the gate."""
        self.error: str | None = None
        self.client = httpx.AsyncClient(
            headers={"Authorization": f"Bearer {secret}"},
            timeout=httpx.Timeout(120, connect=5),
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
            decision = await self.ask(
                {"tool_call_id": tool_call_id, "arguments": arguments}
            )
            return decision["action"]
        except Exception as error:  # noqa: BLE001 - an unreachable gate lets nothing run
            # A denial would reach the model as the policy's verdict: fail the turn.
            print(f"tool gate failed for {tool_call_id}: {error}", file=sys.stderr)
            self.error = self.error or f"tool gate failed for {tool_call_id}: {error}"
            return "stop"

    async def ask(self, payload: dict) -> dict:
        """Retry what a tunnel or proxy dropped or answered, marked so the rollout
        answers a repeat with its first verdict."""
        deadline = time.monotonic() + RETRY_SECONDS
        delay, retry = 0.5, 0
        while True:
            try:
                response = await self.client.post(
                    self.url,
                    json=payload,
                    headers={"x-stainless-retry-count": str(retry)},
                )
                status = response.status_code
                if response.headers.get(STAMP_HEADER) or not (
                    status in (404, 408, 429) or status >= 500
                ):
                    response.raise_for_status()
                    return response.json()
                failure = f"HTTP {status}"
            except httpx.TransportError as error:
                failure = repr(error)
            if time.monotonic() + delay > deadline:
                raise RuntimeError(f"unreachable for {RETRY_SECONDS:.0f}s: {failure}")
            await asyncio.sleep(delay * random.uniform(0.5, 1.5))
            delay, retry = min(delay * 2, 10.0), retry + 1

    def failure(self) -> str | None:
        if self.error is None and self.failed and os.path.exists(self.failed):
            with open(self.failed, errors="replace") as file:
                self.error = file.read(2000) or "tool gate hook failed"
        return self.error


class VerifiersACPClient(Client):
    def __init__(self) -> None:
        self.gate: ToolGate | None = None
        self.prompt_task: asyncio.Task | None = None
        self.visible_reply = ""
        self.message_id: str | None = None
        self.stop_reason: str | None = None
        self.response_metadata: dict[str, Any] = {}
        self.update_metadata: list[dict[str, Any]] = []

    def reset(self) -> None:
        self.visible_reply = ""
        self.message_id = None
        self.stop_reason = None
        self.response_metadata = {}
        self.update_metadata = []

    def turn_result(self) -> ACPTurn:
        return ACPTurn(
            reply=self.visible_reply,
            stop_reason=self.stop_reason,
            response_metadata=self.response_metadata,
            update_metadata=self.update_metadata,
        )

    async def session_update(self, session_id: str, update: Any, **kwargs: Any) -> None:
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
            decision = await self.gate.decision(
                tool_call.tool_call_id, tool_call.raw_input
            )
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

    def __init__(self) -> None:
        self.client = VerifiersACPClient()
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
        try:
            agent_process = await self.stack.enter_async_context(
                spawn_agent_process(
                    self.client,
                    command[0],
                    *command[1:],
                    env=os.environ.copy(),
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
            self.capabilities = initialized.agent_capabilities
            self.supports_steering = (initialized.field_meta or {}).get(
                "steering", {}
            ).get("supported") is True
            session = await self.connection.new_session(
                cwd=os.getcwd(),
                mcp_servers=mcp_servers(config),
                **config["session_meta"],
            )
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
            self.client.gate = ToolGate(gate["url"], gate["secret"], gate.get("failed"))
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
        if self.client.gate is not None and (error := self.client.gate.failure()):
            raise RuntimeError(error)
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

    async def close(self) -> dict[str, Any]:
        response_metadata: dict[str, Any] = {}
        try:
            if self.connection is not None and self.session_id is not None:
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


async def read_packet(stream: asyncio.StreamReader) -> dict | None:
    try:
        header = await stream.readexactly(8)
    except asyncio.IncompleteReadError as error:
        if not error.partial:
            return None
        raise EOFError("ACP session packet ended early") from error
    size = int.from_bytes(header, "big")
    if size > MAX_PACKET_BYTES:
        raise ValueError(f"ACP session packet is too large: {size} bytes")
    try:
        return json.loads((await stream.readexactly(size)).decode())
    except asyncio.IncompleteReadError as error:
        raise EOFError("ACP session packet ended early") from error


def write_packet(stream: Any, value: dict) -> None:
    data = json.dumps(value, ensure_ascii=False).encode()
    if len(data) > MAX_PACKET_BYTES:
        raise ValueError(f"ACP session packet is too large: {len(data)} bytes")
    stream.write(len(data).to_bytes(8, "big"))
    stream.write(data)
    stream.flush()


async def serve_stream() -> None:
    session = ACPSession()
    reader = asyncio.StreamReader()
    protocol = asyncio.StreamReaderProtocol(reader)
    await asyncio.get_running_loop().connect_read_pipe(
        lambda: protocol, sys.stdin.buffer
    )
    tasks: set[asyncio.Task] = set()
    prompt_lock = asyncio.Lock()

    async def handle(request: dict) -> None:
        operation = request.get("operation")
        try:
            if operation == "prompt":
                async with prompt_lock:
                    result = asdict(await session.run(request["config"]))
            elif operation == "steer":
                result = await session.steer(
                    request["message"], request.get("message_id")
                )
            elif operation == "shutdown":
                result = {"response_metadata": await session.close()}
            else:
                raise ValueError(f"unknown ACP session operation: {operation!r}")
            response = {"ok": True, "result": result}
        except Exception as error:  # noqa: BLE001 - serialize protocol failures
            traceback.print_exc()
            response = {"ok": False, "error": f"{type(error).__name__}: {error}"}
            if isinstance(error, RequestError) and isinstance(error.data, dict):
                response["error"] = error.data.get("details") or str(error)
                response["error_data"] = error.data
            if operation == "prompt":
                response["result"] = asdict(session.client.turn_result())
        response["id"] = request.get("id")
        write_packet(sys.stdout.buffer, response)

    try:
        while request := await read_packet(reader):
            if request.get("operation") == "shutdown":
                if tasks:
                    await asyncio.gather(*tasks)
                await handle(request)
                break
            task = asyncio.create_task(handle(request))
            tasks.add(task)
            task.add_done_callback(tasks.discard)
    finally:
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        await session.close()


async def main() -> None:
    task = asyncio.current_task()
    loop = asyncio.get_running_loop()
    if task is not None:
        for sig in (signal.SIGTERM, signal.SIGINT):
            with suppress(NotImplementedError):
                loop.add_signal_handler(sig, task.cancel)
    with suppress(asyncio.CancelledError):
        await serve_stream()


if __name__ == "__main__":
    asyncio.run(main())
