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
    SessionInfoUpdate,
    TextContentBlock,
    ToolCall,
    ToolCallUpdate,
)

MAX_PACKET_BYTES = 128 * 1024 * 1024
LATE_REPLY_GRACE_SECONDS = 1.0


class VerifiersACPClient(Client):
    """ACP output collector with an opt-in correlated lifecycle consumer."""

    _NON_ANSWER_META = frozenset(("compaction", "refinement", "subagents"))

    def __init__(self) -> None:
        self.visible_reply = ""
        self.message_id: str | None = None
        self.tool_calls: dict[str, str] = {}
        self.output_changed = asyncio.Condition()
        self.lifecycle_namespace: str | None = None
        self.stop_reason: str | None = None
        self.prompt_turn_id = 0
        self.response_boundary: dict[str, Any] | None = None
        self.terminal_quiescence: dict[str, Any] | None = None
        self.lifecycle_error: str | None = None
        self._last_event_sequence = 0

    def reset(self, lifecycle_namespace: str | None = None) -> None:
        self.visible_reply = ""
        self.message_id = None
        self.tool_calls = {}
        self.lifecycle_namespace = lifecycle_namespace
        self.stop_reason = None
        self.response_boundary = None
        self.terminal_quiescence = None
        self.lifecycle_error = None
        if lifecycle_namespace is not None:
            self.prompt_turn_id += 1

    def _lifecycle_meta(self, update: Any) -> dict[str, Any] | None:
        if self.lifecycle_namespace is None:
            return None
        field_meta = getattr(update, "field_meta", None)
        if not isinstance(field_meta, dict):
            return None
        if self.lifecycle_namespace not in field_meta:
            return None
        event = field_meta[self.lifecycle_namespace]
        if not isinstance(event, dict):
            self.lifecycle_error = "Prime Agent lifecycle metadata must be an object"
            return None
        return event

    def _consume_lifecycle(self, event: dict[str, Any] | None) -> None:
        if event is None:
            return
        sequence = event.get("eventSequence")
        if type(sequence) is not int or sequence <= self._last_event_sequence:
            self.lifecycle_error = "Prime Agent lifecycle eventSequence is invalid"
            return
        self._last_event_sequence = sequence
        turn_id = event.get("promptTurnId")
        if type(turn_id) is not int:
            self.lifecycle_error = "Prime Agent lifecycle promptTurnId is invalid"
            return
        if turn_id != self.prompt_turn_id:
            return
        phase = event.get("phase")
        if phase == "responseBoundary":
            outcome = event.get("outcome")
            terminal_expected = event.get("terminalQuiescenceExpected")
            if (
                outcome not in ("result", "error")
                or type(terminal_expected) is not bool
                or (outcome == "result" and not terminal_expected)
            ):
                self.lifecycle_error = "Prime Agent responseBoundary is malformed"
            elif self.response_boundary is not None:
                self.lifecycle_error = "Prime Agent emitted duplicate responseBoundary"
            else:
                self.response_boundary = event
            return
        if phase != "terminalQuiescence":
            return
        quiescence = event.get("quiescence")
        outstanding = (
            quiescence.get("outstandingSubagents")
            if isinstance(quiescence, dict)
            else None
        )
        remaining = (
            quiescence.get("remainingAutonomousContinuations")
            if isinstance(quiescence, dict)
            else None
        )
        if (
            self.response_boundary is None
            or event.get("outcome") != self.response_boundary.get("outcome")
            or type(outstanding) is not int
            or outstanding != 0
            or type(remaining) is not int
            or remaining < 0
        ):
            self.lifecycle_error = "Prime Agent terminalQuiescence is malformed"
            return
        if self.terminal_quiescence is not None:
            self.lifecycle_error = "Prime Agent emitted duplicate terminalQuiescence"
            return
        self.terminal_quiescence = event

    def _is_current_turn_event(self, event: dict[str, Any] | None) -> bool:
        if self.lifecycle_namespace is None:
            return True
        return bool(
            event
            and type(event.get("promptTurnId")) is int
            and event.get("promptTurnId") == self.prompt_turn_id
            and event.get("phase") == "event"
        )

    def _is_answer_chunk(self, event: dict[str, Any] | None) -> bool:
        return self._is_current_turn_event(event) and not (
            event and self._NON_ANSWER_META.intersection(event)
        )

    async def session_update(self, session_id: str, update: Any, **kwargs: Any) -> None:
        async with self.output_changed:
            event = self._lifecycle_meta(update)
            self._consume_lifecycle(event)
            if isinstance(update, ToolCall):
                if self._is_current_turn_event(event):
                    self.tool_calls[update.tool_call_id] = update.status or "pending"
            elif isinstance(update, ToolCallUpdate):
                if update.status and self._is_current_turn_event(event):
                    self.tool_calls[update.tool_call_id] = update.status
            elif (
                isinstance(update, AgentMessageChunk)
                and isinstance(update.content, TextContentBlock)
                and self._is_answer_chunk(event)
            ):
                message_id = getattr(update, "message_id", None)
                if message_id is not None and message_id != self.message_id:
                    self.visible_reply = ""
                    self.message_id = message_id
                self.visible_reply += update.content.text
            elif not isinstance(update, SessionInfoUpdate):
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
) -> dict[str, Any]:
    prompt_capabilities = capabilities and capabilities.prompt_capabilities
    supports_images = bool(prompt_capabilities and prompt_capabilities.image)
    blocks = []
    if is_new and config["system_prompt"]:
        blocks.append(text_block(f"(system)\n{config['system_prompt']}\n\n[user]\n"))
    blocks.extend(user_content_blocks(config["user_contents"], supports_images))
    if not blocks:
        raise ValueError("ACP prompt has no content")
    client.reset(config.get("lifecycle_meta_namespace"))
    prompt_error: RequestError | None = None
    try:
        response = await connection.prompt(session_id=session_id, prompt=blocks)
        client.stop_reason = response.stop_reason
    except RequestError as error:
        prompt_error = error

    # ACP 0.11 dispatches notifications in background tasks but resolves a request
    # response directly in its receive loop. An agent that sends its final
    # session/update immediately before session/prompt returns can therefore wake
    # this coroutine before the update handler has run. Wait specifically for text:
    # a completed tool update may also arrive first and must not hide a later reply.
    def has_visible_reply() -> bool:
        return bool(client.visible_reply.strip())

    if prompt_error is None and not has_visible_reply():
        async with client.output_changed:
            try:
                await asyncio.wait_for(
                    client.output_changed.wait_for(has_visible_reply),
                    timeout=LATE_REPLY_GRACE_SECONDS,
                )
            except asyncio.TimeoutError:  # noqa: UP041 - Python 3.10 compatibility
                pass

    if (
        client.lifecycle_namespace is not None
        and prompt_error is not None
        and client.response_boundary is None
        and client.lifecycle_error is None
    ):
        # Prime drains its boundary notification before returning a request error,
        # while ACP 0.11 dispatches that notification in a background task.
        async with client.output_changed:
            try:
                await asyncio.wait_for(
                    client.output_changed.wait_for(
                        lambda: (
                            client.response_boundary is not None
                            or client.lifecycle_error is not None
                        )
                    ),
                    timeout=LATE_REPLY_GRACE_SECONDS,
                )
            except asyncio.TimeoutError:  # noqa: UP041 - Python 3.10 compatibility
                pass

    terminal_expected = prompt_error is None or bool(
        client.response_boundary
        and client.response_boundary.get("terminalQuiescenceExpected") is True
    )
    if (
        client.lifecycle_namespace is not None
        and terminal_expected
        and client.terminal_quiescence is None
        and client.lifecycle_error is None
    ):
        # Do not impose a short protocol grace here: descendants can settle long after
        # the prompt response. The owning rollout/action timeout remains the hard bound.
        async with client.output_changed:
            await client.output_changed.wait_for(
                lambda: (
                    client.terminal_quiescence is not None
                    or client.lifecycle_error is not None
                )
            )
    if client.lifecycle_error is not None:
        raise RuntimeError(client.lifecycle_error)
    if (
        client.lifecycle_namespace is not None
        and terminal_expected
        and client.terminal_quiescence is None
    ):
        raise RuntimeError(
            "Prime Agent prompt returned without correlated terminalQuiescence "
            f"(stop_reason={client.stop_reason})"
        )
    if prompt_error is not None:
        data = getattr(prompt_error, "data", None)
        detail = data.get("details") if isinstance(data, dict) else None
        raise RuntimeError(detail or str(prompt_error)) from prompt_error
    if (
        client.terminal_quiescence is not None
        and client.terminal_quiescence["outcome"] == "error"
    ):
        raise RuntimeError("Prime Agent reported a terminal lifecycle error")

    tool_statuses = list(client.tool_calls.values())
    completed_tool_turn = (
        config.get("allow_empty_tool_reply", False)
        and client.stop_reason == "end_turn"
        and bool(tool_statuses)
        and all(status in ("completed", "failed") for status in tool_statuses)
    )
    if not has_visible_reply() and not completed_tool_turn:
        raise RuntimeError(
            "ACP agent produced no visible reply "
            f"(stop_reason={client.stop_reason}, tool_statuses={tool_statuses})"
        )
    return {
        "reply": client.visible_reply,
        "stop_reason": client.stop_reason,
        "response_boundary": client.response_boundary,
        "lifecycle": client.terminal_quiescence,
    }


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

    async def run(self, config: dict) -> dict[str, Any]:
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
                        **await session.run(request["config"]),
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
                    "stop_reason": session.client.stop_reason,
                    "response_boundary": session.client.response_boundary,
                    "lifecycle": session.client.terminal_quiescence,
                }
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
