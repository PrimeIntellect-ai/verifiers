# /// script
# requires-python = ">=3.10,<3.15"
# dependencies = ["agent-client-protocol==0.11.0"]
# ///
"""Run harness segments through an ACP agent."""

import asyncio
import json
import os
import signal
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
    ToolCall,
    ToolCallUpdate,
)

MAX_PACKET_BYTES = 128 * 1024 * 1024
LATE_UPDATE_GRACE_SECONDS = 1.0


class VerifiersACPClient(Client):
    def __init__(self) -> None:
        self.visible_reply = ""
        self.message_id: str | None = None
        self.tool_calls: dict[str, str] = {}
        self.output_changed = asyncio.Condition()

    def reset(self) -> None:
        self.visible_reply = ""
        self.message_id = None
        self.tool_calls = {}

    async def session_update(self, session_id: str, update: Any, **kwargs: Any) -> None:
        async with self.output_changed:
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
            else:
                return
            self.output_changed.notify_all()

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


def segment_messages(
    config: dict, *, is_new: bool, replayed_transcript: bool
) -> list[dict]:
    messages = config["messages"]
    if replayed_transcript:
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
    client: VerifiersACPClient,
    connection: Any,
    capabilities: Any,
    session_id: str,
    config: dict,
    *,
    is_new: bool,
    replayed_transcript: bool,
) -> str:
    prompt_capabilities = capabilities and capabilities.prompt_capabilities
    supports_images = bool(prompt_capabilities and prompt_capabilities.image)
    blocks = content_blocks(
        segment_messages(
            config,
            is_new=is_new,
            replayed_transcript=replayed_transcript,
        ),
        supports_images,
    )
    if not blocks:
        raise ValueError("ACP prompt has no content")
    client.reset()
    try:
        response = await connection.prompt(session_id=session_id, prompt=blocks)
    except RequestError as error:
        detail = error.data.get("details") if isinstance(error.data, dict) else None
        raise RuntimeError(detail or str(error)) from error

    # ACP 0.11 dispatches notifications in background tasks but resolves a request
    # response directly in its receive loop. An agent that sends its final
    # session/update immediately before session/prompt returns can therefore wake
    # this coroutine before the update handler has run. Wait specifically for text:
    # a completed tool update may also arrive first and must not hide a later reply.
    def has_visible_reply() -> bool:
        return bool(client.visible_reply.strip())

    if not has_visible_reply():
        async with client.output_changed:
            try:
                await asyncio.wait_for(
                    client.output_changed.wait_for(has_visible_reply),
                    timeout=LATE_UPDATE_GRACE_SECONDS,
                )
            except asyncio.TimeoutError:  # noqa: UP041 - Python 3.10 compatibility
                pass

    tool_statuses = list(client.tool_calls.values())
    completed_tool_turn = (
        config.get("allow_empty_tool_reply", False)
        and response.stop_reason == "end_turn"
        and bool(tool_statuses)
        and all(status in ("completed", "failed") for status in tool_statuses)
    )
    if not has_visible_reply() and not completed_tool_turn:
        raise RuntimeError(
            "ACP agent produced no visible reply "
            f"(stop_reason={response.stop_reason}, tool_statuses={tool_statuses})"
        )
    return client.visible_reply


async def run_once(config: dict) -> str:
    client = VerifiersACPClient()
    command = config["command"]
    async with spawn_agent_process(
        client,
        command[0],
        *command[1:],
        env=os.environ.copy(),
        transport_kwargs={"stderr": None},
    ) as agent_process:
        connection = agent_process[0]
        initialized = await connection.initialize(
            protocol_version=PROTOCOL_VERSION,
            client_capabilities=ClientCapabilities(),
        )
        capabilities = initialized.agent_capabilities
        session_path = Path(config["session_path"]) if config["session_path"] else None
        session_meta = config["session_meta"]
        is_new = session_path is None or not session_path.exists()
        servers = mcp_servers(config)
        if is_new:
            session = await connection.new_session(
                cwd=os.getcwd(), mcp_servers=servers, **session_meta
            )
            session_id = session.session_id
        else:
            session_id = session_path.read_text().strip()
            session_capabilities = capabilities and capabilities.session_capabilities
            if session_capabilities and session_capabilities.resume is not None:
                await connection.resume_session(
                    cwd=os.getcwd(),
                    session_id=session_id,
                    mcp_servers=servers,
                    **session_meta,
                )
            elif capabilities and capabilities.load_session:
                await connection.load_session(
                    cwd=os.getcwd(),
                    session_id=session_id,
                    mcp_servers=servers,
                    **session_meta,
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
            replayed_transcript=not is_new,
        )
        if session_path and is_new:
            session_path.parent.mkdir(parents=True, exist_ok=True)
            session_path.write_text(session_id)
        return reply


class LiveACPSession:
    """One live ACP process, connection, and session shared by several turns."""

    def __init__(self) -> None:
        self.client = VerifiersACPClient()
        self._reset()

    def _reset(self) -> None:
        self.stack = AsyncExitStack()
        self.connection: Any = None
        self.capabilities: Any = None
        self.session_id: str | None = None
        self.command: list[str] | None = None
        self.server_urls: dict[str, str] | None = None
        self.system_prompt: str | None = None
        self.session_meta: dict | None = None
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
                client_capabilities=ClientCapabilities(),
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
            self._reset()
            raise
        self.session_id = session.session_id
        self.command = command
        self.server_urls = config["mcp_urls"]
        self.system_prompt = config["system_prompt"]
        self.session_meta = config["session_meta"]
        self.is_new = True

    async def run(self, config: dict) -> str:
        if self.connection is None:
            await self.start(config)
        elif (
            config["command"] != self.command
            or config["mcp_urls"] != self.server_urls
            or config["system_prompt"] != self.system_prompt
            or config["session_meta"] != self.session_meta
        ):
            raise RuntimeError("ACP session configuration changed")
        assert self.session_id is not None
        reply = await prompt(
            self.client,
            self.connection,
            self.capabilities,
            self.session_id,
            config,
            is_new=self.is_new,
            # HarnessSession.turn() already passes only this interaction's new
            # messages; unlike run_once resume, this is not a replayed transcript.
            replayed_transcript=False,
        )
        self.is_new = False
        return reply

    async def close(self) -> None:
        try:
            if self.connection is not None and self.session_id is not None:
                session_capabilities = (
                    self.capabilities and self.capabilities.session_capabilities
                )
                if session_capabilities and session_capabilities.close is not None:
                    with suppress(Exception):
                        await self.connection.close_session(session_id=self.session_id)
            await self.stack.aclose()
        finally:
            self._reset()


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
    session = LiveACPSession()
    closed = False
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
                        "reply": await session.run(request["config"]),
                    }
                elif operation == "shutdown":
                    await session.close()
                    closed = True
                    stop = True
                    response = {"ok": True}
                else:
                    raise ValueError(f"unknown ACP session operation: {operation!r}")
            except Exception as error:  # noqa: BLE001 - serialize protocol failures
                traceback.print_exc()
                response = {
                    "ok": False,
                    "error": f"{type(error).__name__}: {error}",
                }
            write_packet(sys.stdout.buffer, response)
            if stop:
                break
    finally:
        if not closed:
            await session.close()


def read_config(path_value: str) -> dict:
    path = Path(path_value)
    config = json.loads(path.read_text())
    path.unlink()
    return config


async def main() -> None:
    operation = sys.argv[1]
    if operation == "once":
        sys.stdout.write(await run_once(read_config(sys.argv[2])))
    elif operation == "stream":
        task = asyncio.current_task()
        loop = asyncio.get_running_loop()
        if task is not None:
            for sig in (signal.SIGTERM, signal.SIGINT):
                with suppress(NotImplementedError):
                    loop.add_signal_handler(sig, task.cancel)
        with suppress(asyncio.CancelledError):
            await serve_stream()
    else:
        raise ValueError(f"unknown ACP runner operation: {operation!r}")


if __name__ == "__main__":
    asyncio.run(main())
