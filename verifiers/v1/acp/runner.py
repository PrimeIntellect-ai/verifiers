# /// script
# requires-python = ">=3.10,<3.15"
# dependencies = ["agent-client-protocol==0.12.1", "httpx"]
# ///
"""Run harness segments through an ACP agent."""

import asyncio
import json
import os
import signal
import sys
import traceback
from contextlib import AsyncExitStack, asynccontextmanager, suppress
from dataclasses import asdict, dataclass
from typing import Any

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
    HttpMcpServer,
    PermissionOption,
    RequestPermissionResponse,
    TextContentBlock,
    ToolCall,
    ToolCallUpdate,
)

# {tool_interception}

MAX_PACKET_BYTES = 128 * 1024 * 1024
TOOL_INTERCEPTION_METHOD = "verifiers/tool_interception"


@dataclass(frozen=True)
class ACPTurn:
    reply: str
    stop_reason: str | None
    response_metadata: dict[str, Any]
    update_metadata: list[dict[str, Any]]


class LiveACPClient(Client):
    """ACP client with a private awaited extension for native tool policy."""

    def __init__(self) -> None:
        self.visible_reply = ""
        self.message_id: str | None = None
        self.stop_reason: str | None = None
        self.response_metadata: dict[str, Any] = {}
        self.update_metadata: list[dict[str, Any]] = []
        self.tool_calls: dict[str, str] = {}
        self.tool_interceptor: ToolInterceptionClient | None = None  # noqa: F821
        self.prompt_error: Exception | None = None

    def reset(self) -> None:
        self.visible_reply = ""
        self.message_id = None
        self.stop_reason = None
        self.response_metadata = {}
        self.update_metadata = []
        self.tool_calls = {}
        self.prompt_error = None

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
        if isinstance(update, ToolCall):
            self.tool_calls[update.tool_call_id] = update.status or "pending"
        elif isinstance(update, ToolCallUpdate):
            if update.status:
                self.tool_calls[update.tool_call_id] = update.status
        elif isinstance(update, AgentMessageChunk) and isinstance(
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
        option = next(
            (item for item in options if item.kind in ("allow_once", "allow_always")),
            None,
        )
        outcome = (
            AllowedOutcome(outcome="selected", option_id=option.option_id)
            if option
            else DeniedOutcome(outcome="cancelled")
        )
        return RequestPermissionResponse(outcome=outcome)

    async def ext_method(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        if method != TOOL_INTERCEPTION_METHOD:
            raise RequestError.method_not_found(method)
        try:
            if self.tool_interceptor is None:
                raise RuntimeError("Tool interception is not configured")
            return await asyncio.to_thread(self.tool_interceptor.request, params)
        except Exception as error:
            self.prompt_error = RuntimeError(
                f"Tool interception is unavailable: {error}"
            )
            raise RequestError.internal_error({"details": str(error)}) from error


@asynccontextmanager
async def running_tool_proxy(command: list[str], client: LiveACPClient):
    process = await asyncio.create_subprocess_exec(
        *command,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=None,
        env=os.environ.copy(),
    )
    assert process.stdin is not None
    assert process.stdout is not None
    relay: asyncio.Task[None] | None = None
    try:
        line = await asyncio.wait_for(process.stdout.readline(), timeout=15)
        url = line.decode().strip()
        if not url.startswith("ws://127.0.0.1:"):
            raise RuntimeError(
                f"Tool interception proxy returned an invalid endpoint: {url!r}"
            )

        async def relay_requests() -> None:
            async for line in process.stdout:
                request = json.loads(line)
                request_id = request.get("id")
                try:
                    decision = await client.ext_method(
                        TOOL_INTERCEPTION_METHOD, request["body"]
                    )
                    response = {"id": request_id, "decision": decision}
                except Exception as error:  # noqa: BLE001 - proxy needs the failure text
                    response = {"id": request_id, "error": str(error)}
                process.stdin.write((json.dumps(response) + "\n").encode())
                await process.stdin.drain()

        relay = asyncio.create_task(relay_requests())
        yield url
    finally:
        if relay is not None:
            relay.cancel()
            with suppress(asyncio.CancelledError):
                await relay
        if process.returncode is None:
            process.terminate()
            try:
                await asyncio.wait_for(process.wait(), timeout=5)
            except TimeoutError:
                process.kill()
                await process.wait()


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
        HttpMcpServer(type="http", name=name, url=url, headers=[])
        for name, url in config["mcp_urls"].items()
    ]


async def prompt(
    client: LiveACPClient,
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
        response = await connection.prompt(session_id=session_id, prompt=blocks)
        client.stop_reason = response.stop_reason
        client.response_metadata = dict(response.field_meta or {})
    except RequestError as error:
        if client.prompt_error is not None:
            raise client.prompt_error from error
        detail = error.data.get("details") if isinstance(error.data, dict) else None
        raise RuntimeError(detail or str(error)) from error
    if client.prompt_error is not None:
        raise client.prompt_error

    tool_statuses = list(client.tool_calls.values())
    tools_finished = all(status in ("completed", "failed") for status in tool_statuses)
    if config.get("require_terminal_tool_status", False) and not tools_finished:
        raise RuntimeError(
            f"ACP agent ended with unfinished tool calls: {tool_statuses}"
        )
    completed_tool_turn = (
        response.stop_reason == "end_turn" and bool(tool_statuses) and tools_finished
    )
    if not client.visible_reply.strip() and not completed_tool_turn:
        raise RuntimeError(
            "ACP agent produced no visible reply "
            f"(stop_reason={response.stop_reason}, tool_statuses={tool_statuses})"
        )
    return client.turn_result()


class ACPSession:
    """One live ACP process, connection, and session shared by several turns."""

    def __init__(self) -> None:
        self.client = LiveACPClient()
        self._reset()

    def _reset(self) -> None:
        self.stack = AsyncExitStack()
        self.connection: Any = None
        self.capabilities: Any = None
        self.session_id: str | None = None
        self.is_new = True

    async def start(self, config: dict) -> None:
        command = config["command"]
        interception = config.get("toolInterception")
        self.client.tool_interceptor = None
        try:
            agent_env = os.environ.copy()
            if interception is not None:
                self.client.tool_interceptor = ToolInterceptionClient(  # noqa: F821
                    interception["url"], interception["secret"]
                )
            if proxy_command := config.get("toolInterceptionProxy"):
                if self.client.tool_interceptor is None:
                    raise RuntimeError("Tool interception is not configured")
                agent_env[
                    "VF_CODE_MODE_HOST_URL"
                ] = await self.stack.enter_async_context(
                    running_tool_proxy(proxy_command, self.client)
                )
            agent_process = await self.stack.enter_async_context(
                spawn_agent_process(
                    self.client,
                    command[0],
                    *command[1:],
                    env=agent_env,
                    transport_kwargs={"stderr": None},
                )
            )
            self.connection = agent_process[0]
            client_capabilities = ClientCapabilities()
            initialized = await self.connection.initialize(
                protocol_version=PROTOCOL_VERSION,
                client_capabilities=client_capabilities,
            )
            self.capabilities = initialized.agent_capabilities
            session = await self.connection.new_session(
                cwd=os.getcwd(),
                mcp_servers=mcp_servers(config),
                **config["session_meta"],
            )
        except BaseException:
            with suppress(BaseException):
                await self.stack.aclose()
            if self.client.tool_interceptor is not None:
                self.client.tool_interceptor.close()
                self.client.tool_interceptor = None
            self._reset()
            raise
        self.session_id = session.session_id
        self.is_new = True

    async def run(self, config: dict) -> ACPTurn:
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
                if self.client.tool_interceptor is not None:
                    self.client.tool_interceptor.close()
                    self.client.tool_interceptor = None
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
    try:
        while request := await read_packet(reader):
            stop = False
            try:
                operation = request.get("operation")
                if operation == "prompt":
                    response = {
                        "ok": True,
                        "result": asdict(await session.run(request["config"])),
                    }
                elif operation == "shutdown":
                    stop = True
                    response = {
                        "ok": True,
                        "result": {"response_metadata": await session.close()},
                    }
                else:
                    raise ValueError(f"unknown ACP session operation: {operation!r}")
            except Exception as error:  # noqa: BLE001 - serialize protocol failures
                traceback.print_exc()
                response = {
                    "ok": False,
                    "error": f"{type(error).__name__}: {error}",
                }
                if operation == "prompt":
                    response["result"] = asdict(session.client.turn_result())
            write_packet(sys.stdout.buffer, response)
            if stop:
                break
    finally:
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
