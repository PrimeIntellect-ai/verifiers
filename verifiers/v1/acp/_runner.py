# /// script
# requires-python = ">=3.10,<3.15"
# dependencies = ["agent-client-protocol==0.11.0"]
# ///
"""Run harness segments through an ACP agent."""

import asyncio
import json
import os
import sys
import traceback
from contextlib import AsyncExitStack, suppress
from pathlib import Path
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
)

MAX_PACKET_BYTES = 128 * 1024 * 1024
PROBE_UNAVAILABLE_EXIT_CODE = 75


class VerifiersClient(Client):
    def __init__(self) -> None:
        self.visible_reply = ""
        self.message_id: str | None = None

    def reset(self) -> None:
        self.visible_reply = ""
        self.message_id = None

    async def session_update(self, session_id: str, update: Any, **kwargs: Any) -> None:
        if not isinstance(update, AgentMessageChunk) or not isinstance(
            update.content, TextContentBlock
        ):
            return
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


def content_blocks(messages: list[dict], supports_images: bool) -> list:
    blocks = []
    transcript = len(messages) != 1 or messages[0].get("role") != "user"
    for message in messages:
        if transcript:
            separator = "\n\n" if blocks else ""
            blocks.append(
                text_block(f"{separator}[{message.get('role', 'message')}]\n")
            )
        content = message.get("content") or ""
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
        metadata = {
            key: value
            for key, value in message.items()
            if key not in ("role", "content") and value
        }
        if metadata:
            blocks.append(text_block("\n" + json.dumps(metadata, ensure_ascii=False)))
    return blocks


def mcp_servers(config: dict) -> list[HttpMcpServer]:
    return [
        HttpMcpServer(type="http", name=name, url=url, headers=[])
        for name, url in config["mcp_urls"].items()
    ]


def segment_messages(config: dict, is_new: bool) -> list[dict]:
    messages = config["messages"]
    if not is_new:
        last_assistant = max(
            (
                index
                for index, message in enumerate(messages)
                if message.get("role") == "assistant"
            ),
            default=-1,
        )
        messages = messages[last_assistant + 1 :]
    if is_new and config["system_prompt"]:
        messages = [
            {"role": "system", "content": config["system_prompt"]},
            *messages,
        ]
    return messages


async def prompt(
    client: VerifiersClient,
    connection: Any,
    capabilities: Any,
    session_id: str,
    config: dict,
    *,
    is_new: bool,
) -> str:
    prompt_capabilities = capabilities and capabilities.prompt_capabilities
    supports_images = bool(prompt_capabilities and prompt_capabilities.image)
    blocks = content_blocks(segment_messages(config, is_new), supports_images)
    if not blocks:
        raise ValueError("ACP prompt has no content")
    client.reset()
    try:
        response = await connection.prompt(session_id=session_id, prompt=blocks)
    except RequestError as error:
        detail = error.data.get("details") if isinstance(error.data, dict) else None
        raise RuntimeError(detail or str(error)) from error
    if not client.visible_reply.strip():
        raise RuntimeError(
            "ACP agent produced no visible reply "
            f"(stop_reason={response.stop_reason!r})"
        )
    return client.visible_reply


async def run_once(config: dict) -> str:
    client = VerifiersClient()
    command = config["command"]
    async with spawn_agent_process(
        client,
        command[0],
        *command[1:],
        env=os.environ.copy(),
        transport_kwargs={"stderr": None},
    ) as (connection, _process):
        initialized = await connection.initialize(
            protocol_version=PROTOCOL_VERSION,
            client_capabilities=ClientCapabilities(),
        )
        capabilities = initialized.agent_capabilities
        session_path = Path(config["session_path"]) if config["session_path"] else None
        is_new = session_path is None or not session_path.exists()
        servers = mcp_servers(config)
        if is_new:
            session = await connection.new_session(cwd=os.getcwd(), mcp_servers=servers)
            session_id = session.session_id
        else:
            session_id = session_path.read_text().strip()
            session_capabilities = capabilities and capabilities.session_capabilities
            if session_capabilities and session_capabilities.resume is not None:
                await connection.resume_session(
                    cwd=os.getcwd(), session_id=session_id, mcp_servers=servers
                )
            elif capabilities and capabilities.load_session:
                await connection.load_session(
                    cwd=os.getcwd(), session_id=session_id, mcp_servers=servers
                )
            else:
                raise RuntimeError("ACP agent does not support resuming sessions")

        reply = await prompt(
            client,
            connection,
            capabilities,
            session_id,
            config,
            is_new=is_new,
        )
        if session_path and is_new:
            session_path.parent.mkdir(parents=True, exist_ok=True)
            session_path.write_text(session_id)
        return reply


class PersistentSession:
    """One live ACP process, connection, and session shared by several segments."""

    def __init__(self) -> None:
        self.stack = AsyncExitStack()
        self.client = VerifiersClient()
        self.connection: Any = None
        self.capabilities: Any = None
        self.session_id: str | None = None
        self.command: list[str] | None = None
        self.server_urls: dict[str, str] | None = None
        self.system_prompt: str | None = None
        self.is_new = True

    async def start(self, config: dict) -> None:
        command = config["command"]
        try:
            self.connection, _process = await self.stack.enter_async_context(
                spawn_agent_process(
                    self.client,
                    command[0],
                    *command[1:],
                    env=os.environ.copy(),
                    transport_kwargs={"stderr": None},
                )
            )
            initialized = await self.connection.initialize(
                protocol_version=PROTOCOL_VERSION,
                client_capabilities=ClientCapabilities(),
            )
            self.capabilities = initialized.agent_capabilities
            session = await self.connection.new_session(
                cwd=os.getcwd(), mcp_servers=mcp_servers(config)
            )
        except BaseException:
            try:
                await self.stack.aclose()
            except BaseException:
                pass
            self.stack = AsyncExitStack()
            self.connection = None
            self.capabilities = None
            self.session_id = None
            self.command = None
            self.server_urls = None
            self.system_prompt = None
            self.is_new = True
            raise
        self.session_id = session.session_id
        self.command = command
        self.server_urls = config["mcp_urls"]
        self.system_prompt = config["system_prompt"]

    async def run(self, config: dict) -> str:
        if self.connection is None:
            await self.start(config)
        elif (
            config["command"] != self.command
            or config["mcp_urls"] != self.server_urls
            or config["system_prompt"] != self.system_prompt
        ):
            raise RuntimeError("ACP sidecar session configuration changed")
        assert self.session_id is not None
        reply = await prompt(
            self.client,
            self.connection,
            self.capabilities,
            self.session_id,
            config,
            is_new=self.is_new,
        )
        self.is_new = False
        return reply

    async def close(self) -> None:
        if self.connection is not None and self.session_id is not None:
            session_capabilities = (
                self.capabilities and self.capabilities.session_capabilities
            )
            if session_capabilities and session_capabilities.close is not None:
                with suppress(Exception):
                    await self.connection.close_session(session_id=self.session_id)
        await self.stack.aclose()
        self.connection = None


async def read_packet(reader: asyncio.StreamReader) -> dict:
    size = int.from_bytes(await reader.readexactly(8), "big")
    if size > MAX_PACKET_BYTES:
        raise ValueError(f"ACP sidecar packet is too large: {size} bytes")
    return json.loads((await reader.readexactly(size)).decode())


async def write_packet(writer: asyncio.StreamWriter, value: dict) -> None:
    data = json.dumps(value, ensure_ascii=False).encode()
    writer.write(len(data).to_bytes(8, "big"))
    writer.write(data)
    await writer.drain()


async def serve_sidecar(socket_path: str) -> None:
    path = Path(socket_path)
    path.unlink(missing_ok=True)
    session = PersistentSession()
    lock = asyncio.Lock()
    shutdown = asyncio.Event()

    async def handle(
        reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        try:
            request = await read_packet(reader)
            operation = request.get("operation")
            if operation == "ping":
                response = {"ok": True}
            elif operation == "shutdown":
                try:
                    await session.close()
                    response = {"ok": True}
                finally:
                    shutdown.set()
            else:
                async with lock:
                    if operation == "prompt":
                        reply = await session.run(request["config"])
                        response = {"ok": True, "reply": reply}
                    else:
                        raise ValueError(
                            f"unknown ACP sidecar operation: {operation!r}"
                        )
        except Exception as error:
            traceback.print_exc()
            response = {
                "ok": False,
                "error": f"{type(error).__name__}: {error}",
            }
        try:
            await write_packet(writer, response)
        finally:
            writer.close()
            await writer.wait_closed()

    server = await asyncio.start_unix_server(handle, path=socket_path)
    os.chmod(path, 0o600)
    try:
        async with server:
            await shutdown.wait()
    finally:
        server.close()
        await server.wait_closed()
        await session.close()
        path.unlink(missing_ok=True)


async def connect(
    socket_path: str,
    wait_seconds: float = 60,
) -> tuple[asyncio.StreamReader, asyncio.StreamWriter]:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + wait_seconds
    while True:
        try:
            return await asyncio.open_unix_connection(socket_path)
        except (FileNotFoundError, ConnectionRefusedError):
            if loop.time() >= deadline:
                raise RuntimeError("timed out waiting for ACP sidecar")
            await asyncio.sleep(0.1)


async def request_sidecar(
    socket_path: str,
    request: dict,
    wait_seconds: float = 60,
    response_seconds: float | None = None,
) -> dict:
    reader, writer = await connect(socket_path, wait_seconds)
    try:
        await write_packet(writer, request)
        response = (
            await read_packet(reader)
            if response_seconds is None
            else await asyncio.wait_for(read_packet(reader), response_seconds)
        )
    finally:
        writer.close()
        await writer.wait_closed()
    if not response.get("ok"):
        raise RuntimeError(response.get("error") or "ACP sidecar request failed")
    return response


def read_config(path_value: str) -> dict:
    path = Path(path_value)
    config = json.loads(path.read_text())
    path.unlink()
    return config


async def main() -> None:
    operation = sys.argv[1]
    if operation == "once":
        sys.stdout.write(await run_once(read_config(sys.argv[2])))
    elif operation == "serve":
        await serve_sidecar(sys.argv[2])
    elif operation == "request":
        response = await request_sidecar(
            sys.argv[3],
            {"operation": "prompt", "config": read_config(sys.argv[2])},
        )
        sys.stdout.write(response["reply"])
    elif operation == "shutdown":
        await request_sidecar(
            sys.argv[2],
            {"operation": "shutdown"},
            wait_seconds=2,
            response_seconds=5,
        )
    elif operation == "probe":
        wait_seconds = float(sys.argv[3]) if len(sys.argv) > 3 else 0
        try:
            await asyncio.wait_for(
                request_sidecar(
                    sys.argv[2],
                    {"operation": "ping"},
                    wait_seconds=wait_seconds,
                ),
                timeout=max(2, wait_seconds + 1),
            )
        except RuntimeError as error:
            if str(error) == "timed out waiting for ACP sidecar":
                raise SystemExit(PROBE_UNAVAILABLE_EXIT_CODE) from None
            raise
    else:
        raise ValueError(f"unknown ACP runner operation: {operation!r}")


if __name__ == "__main__":
    asyncio.run(main())
