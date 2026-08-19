# /// script
# requires-python = ">=3.10,<3.15"
# dependencies = ["agent-client-protocol==0.12.1"]
# ///
"""Run harness segments through an ACP agent."""

import asyncio
import json
import os
import signal
import sys
import traceback
from contextlib import AsyncExitStack, suppress
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
KEEPALIVE_INTERVAL_SECONDS = 15.0
EMPTY_REPLY_ATTEMPTS = 3
EMPTY_REPLY_REPAIR_PROMPT = (
    "Your previous turn ended without a visible reply. Continue the pending task "
    "and return a visible final response."
)
# Bytes written while the host is detached from the process stream are dropped,
# so a reattached host may be mid-frame. Sync responses carry this marker (plus
# the request nonce) so the host can re-align the length-prefixed framing.
RESYNC_MAGIC = b"\xff\x00vf-acp-resync\x00\xff"


class VerifiersACPClient(Client):
    def __init__(self) -> None:
        self.visible_reply = ""
        self.message_id: str | None = None
        self.tool_calls: dict[str, str] = {}

    def reset(self) -> None:
        self.visible_reply = ""
        self.message_id = None
        self.tool_calls = {}

    async def session_update(self, session_id: str, update: Any, **kwargs: Any) -> None:
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
    client: VerifiersACPClient,
    connection: Any,
    capabilities: Any,
    session_id: str,
    config: dict,
    *,
    is_new: bool,
) -> str:
    prompt_capabilities = capabilities and capabilities.prompt_capabilities
    supports_images = bool(prompt_capabilities and prompt_capabilities.image)
    blocks = []
    if is_new and config["system_prompt"]:
        blocks.append(text_block(f"(system)\n{config['system_prompt']}\n\n[user]\n"))
    blocks.extend(user_content_blocks(config["user_contents"], supports_images))
    if not blocks:
        raise ValueError("ACP prompt has no content")

    for attempt in range(1, EMPTY_REPLY_ATTEMPTS + 1):
        client.reset()
        try:
            response = await connection.prompt(session_id=session_id, prompt=blocks)
        except RequestError as error:
            detail = error.data.get("details") if isinstance(error.data, dict) else None
            raise RuntimeError(detail or str(error)) from error

        tool_statuses = list(client.tool_calls.values())
        completed_tool_turn = (
            config.get("allow_empty_tool_reply", False)
            and response.stop_reason == "end_turn"
            and bool(tool_statuses)
            and all(status in ("completed", "failed") for status in tool_statuses)
        )
        if client.visible_reply.strip() or completed_tool_turn:
            return client.visible_reply

        retryable_empty_reply = (
            response.stop_reason == "end_turn" and not tool_statuses
        )
        if not retryable_empty_reply or attempt == EMPTY_REPLY_ATTEMPTS:
            raise RuntimeError(
                "ACP agent produced no visible reply "
                f"after {attempt} attempt(s) "
                f"(stop_reason={response.stop_reason}, tool_statuses={tool_statuses})"
            )
        blocks = [text_block(EMPTY_REPLY_REPAIR_PROMPT)]

    raise AssertionError("empty reply retry loop did not return or raise")


class ACPSession:
    """One live ACP process, connection, and session shared by several turns."""

    def __init__(self) -> None:
        self.client = VerifiersACPClient()
        self._reset()

    def _reset(self) -> None:
        self.stack = AsyncExitStack()
        self.connection: Any = None
        self.capabilities: Any = None
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
        self.is_new = True

    async def run(self, config: dict) -> str:
        if self.connection is None:
            await self.start(config)
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
    stream.write(len(data).to_bytes(8, "big") + data)
    stream.flush()


async def emit_keepalives(stream: Any) -> None:
    while True:
        await asyncio.sleep(KEEPALIVE_INTERVAL_SECONDS)
        write_packet(stream, {"type": "keepalive"})


async def serve_stream() -> None:
    session = ACPSession()
    reader = asyncio.StreamReader()
    protocol = asyncio.StreamReaderProtocol(reader)
    await asyncio.get_running_loop().connect_read_pipe(
        lambda: protocol, sys.stdin.buffer
    )
    keepalive = asyncio.create_task(emit_keepalives(sys.stdout.buffer))
    last_seq = 0
    last_response: dict | None = None
    try:
        while request := await read_packet(reader):
            stop = False
            operation = request.get("operation")
            seq = request.get("seq")
            if operation == "sync":
                # The host reattached to our process stream after a transport
                # failure: it may be mid-frame and does not know whether the
                # last response reached it. Emit a raw marker so it can
                # re-align framing, then report what we last answered.
                sync = {
                    "ok": True,
                    "sync": True,
                    "last_seq": last_seq,
                    "last_response": last_response,
                }
                data = json.dumps(sync, ensure_ascii=False).encode()
                nonce = str(request.get("nonce", "")).encode()
                sys.stdout.buffer.write(
                    RESYNC_MAGIC
                    + nonce
                    + RESYNC_MAGIC
                    + len(data).to_bytes(8, "big")
                    + data
                )
                sys.stdout.buffer.flush()
                continue
            if isinstance(seq, int) and seq <= last_seq:
                continue  # duplicate of an already-answered request
            try:
                if operation == "prompt":
                    response = {
                        "ok": True,
                        "reply": await session.run(request["config"]),
                    }
                elif operation == "shutdown":
                    await session.close()
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
            if isinstance(seq, int):
                last_seq = seq
                response = {**response, "seq": seq}
                last_response = response
            write_packet(sys.stdout.buffer, response)
            if stop:
                break
    finally:
        keepalive.cancel()
        with suppress(asyncio.CancelledError):
            await keepalive
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
