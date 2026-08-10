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
    SessionInfoUpdate,
    TextContentBlock,
    ToolCall,
    ToolCallUpdate,
)

MAX_PACKET_BYTES = 128 * 1024 * 1024
LATE_METADATA_SETTLE_SECONDS = 0.05
LATE_UPDATE_GRACE_SECONDS = 1.0


class VerifiersACPClient(Client):
    """ACP client with a fail-closed, producer-versioned metadata contract.

    ACP 0.11 does not correlate extension updates to requests. Timing is therefore
    only liveness: metadata is attached only when the producer gives this client
    a complete, ordered envelope for the currently open prompt turn.
    """

    def __init__(self) -> None:
        self.visible_reply = ""
        self.message_id: str | None = None
        self.tool_calls: dict[str, str] = {}
        self.acp_meta: dict[str, list[Any]] = {}
        self.turn_acp_meta: dict[str, list[Any]] = {}
        self.unattributed_acp_meta: dict[str, list[Any]] = {}
        self._metadata_lifecycle_open = False
        self._metadata_lifecycle_started = False
        self._ambiguous_meta = False
        self._prompt_turn_id = 0
        self._expected_prompt_turn_id: int | None = None
        self._last_event_sequence = 0
        self._boundary_outcome: str | None = None
        self._terminal_outcome: str | None = None
        self._correlation_error: str | None = None
        self.output_changed = asyncio.Condition()

    def reset(self) -> None:
        self.visible_reply = ""
        self.message_id = None
        self.tool_calls = {}

    def begin_prompt_metadata(self, *, expected: bool) -> None:
        """Open one exact producer envelope; never infer ownership from time."""
        if self._ambiguous_meta:
            namespaces = ", ".join(sorted(self.unattributed_acp_meta))
            raise RuntimeError(
                "ACP metadata arrived after its prompt lifecycle closed "
                f"(namespaces: {namespaces or 'unknown'}); refusing to attach it "
                "to a later turn"
            )
        self.turn_acp_meta = {}
        self._metadata_lifecycle_open = expected
        self._metadata_lifecycle_started = True
        self._prompt_turn_id += 1
        self._expected_prompt_turn_id = self._prompt_turn_id if expected else None
        self._last_event_sequence = 0
        self._boundary_outcome = None
        self._terminal_outcome = None
        self._correlation_error = None

    def close_prompt_metadata(self) -> None:
        self._metadata_lifecycle_open = False

    def _reject_metadata(self, namespace: str, event: Any, reason: str) -> None:
        self.unattributed_acp_meta.setdefault(namespace, []).append(event)
        self._ambiguous_meta = True
        self._correlation_error = reason

    def _accept_metadata(self, namespace: str, event: Any) -> None:
        """Validate one envelope event before it can enter trace-visible metadata."""
        if not isinstance(event, dict):
            self._reject_metadata(
                namespace, event, "ACP metadata event must be an object"
            )
            return
        turn_id = event.get("promptTurnId")
        sequence = event.get("eventSequence")
        phase = event.get("phase")
        outcome = event.get("outcome")
        if type(turn_id) is not int or turn_id <= 0:
            self._reject_metadata(
                namespace, event, "ACP metadata has invalid promptTurnId"
            )
        elif turn_id != self._expected_prompt_turn_id:
            self._reject_metadata(
                namespace, event, "ACP metadata belongs to a foreign prompt turn"
            )
        elif type(sequence) is not int or sequence <= 0:
            self._reject_metadata(
                namespace, event, "ACP metadata has invalid eventSequence"
            )
        elif sequence <= self._last_event_sequence:
            self._reject_metadata(
                namespace, event, "ACP metadata eventSequence regressed or duplicated"
            )
        elif phase not in ("responseBoundary", "terminalQuiescence"):
            self._reject_metadata(
                namespace, event, "ACP metadata has an unsupported phase"
            )
        elif outcome not in ("result", "error"):
            self._reject_metadata(
                namespace, event, "ACP metadata has an unsupported outcome"
            )
        elif phase == "responseBoundary":
            if self._boundary_outcome is not None:
                self._reject_metadata(
                    namespace, event, "ACP metadata has duplicate responseBoundary"
                )
            else:
                self._last_event_sequence = sequence
                self._boundary_outcome = outcome
                self.turn_acp_meta.setdefault(namespace, []).append(event)
        elif self._boundary_outcome is None:
            self._reject_metadata(
                namespace, event, "ACP terminalQuiescence preceded responseBoundary"
            )
        elif self._terminal_outcome is not None:
            self._reject_metadata(
                namespace, event, "ACP metadata has duplicate terminalQuiescence"
            )
        elif outcome != self._boundary_outcome:
            self._reject_metadata(
                namespace, event, "ACP terminal outcome disagrees with responseBoundary"
            )
        else:
            quiescence = event.get("terminalQuiescence")
            if not isinstance(quiescence, dict) or (
                type(quiescence.get("outstandingSubagents")) is not int
                or quiescence["outstandingSubagents"] != 0
                or type(quiescence.get("remainingAutonomousContinuations")) is not int
                or quiescence["remainingAutonomousContinuations"] != 0
            ):
                self._reject_metadata(
                    namespace,
                    event,
                    "ACP terminalQuiescence must contain explicit zero counters",
                )
            else:
                self._last_event_sequence = sequence
                self._terminal_outcome = outcome
                self.turn_acp_meta.setdefault(namespace, []).append(event)

    def require_terminal_metadata(self, *, expected: bool) -> None:
        """Reject partial envelopes; `end_turn` and elapsed time are not evidence."""
        if not expected:
            return
        if self._correlation_error is not None:
            raise RuntimeError(self._correlation_error)
        if self._boundary_outcome is None or self._terminal_outcome is None:
            raise RuntimeError(
                "ACP metadata lacks a correlated result/error, responseBoundary, and terminalQuiescence"
            )
        if self._terminal_outcome == "error":
            raise RuntimeError("ACP producer reported a correlated error")

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

    async def session_update(self, session_id: str, update: Any, **kwargs: Any) -> None:
        async with self.output_changed:
            if isinstance(update, ToolCall):
                self.tool_calls[update.tool_call_id] = update.status or "pending"
            elif isinstance(update, ToolCallUpdate):
                if update.status:
                    self.tool_calls[update.tool_call_id] = update.status
            elif isinstance(update, SessionInfoUpdate):
                for namespace, event in (update.field_meta or {}).items():
                    if self._metadata_lifecycle_open:
                        self._accept_metadata(namespace, event)
                    else:
                        self.unattributed_acp_meta.setdefault(namespace, []).append(
                            event
                        )
                        if self._metadata_lifecycle_started:
                            self._ambiguous_meta = True
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


def _meta_event_count(client: VerifiersACPClient) -> int:
    return sum(len(events) for events in client.turn_acp_meta.values())


async def wait_for_late_metadata(client: VerifiersACPClient, *, expected: bool) -> None:
    """Settle metadata only for a harness that explicitly negotiated it.

    Ordinary ACP harnesses must retain their historical zero-metadata fast path:
    they neither open a metadata lifecycle nor pay this grace period. A capable
    producer still gets one bounded grace window for its final ordered updates.
    """
    if not expected:
        return

    async def settle() -> None:
        while True:
            before = _meta_event_count(client)
            # Nothing has arrived yet: hold the full grace window for the first
            # event. Once metadata exists, only wait the short settle interval for
            # a straggler, so a metadata-bearing turn pays no fixed delay.
            timeout = (
                LATE_METADATA_SETTLE_SECONDS if before else LATE_UPDATE_GRACE_SECONDS
            )
            async with client.output_changed:
                try:
                    await asyncio.wait_for(
                        client.output_changed.wait_for(
                            lambda seen=before: _meta_event_count(client) != seen
                        ),
                        timeout=timeout,
                    )
                except asyncio.TimeoutError:  # noqa: UP041 - Python 3.10 compatibility
                    return

    # Overall ceiling, so a chatty stream cannot extend the turn indefinitely.
    try:
        await asyncio.wait_for(settle(), timeout=LATE_UPDATE_GRACE_SECONDS)
    except asyncio.TimeoutError:  # noqa: UP041 - Python 3.10 compatibility
        pass


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
            {
                "role": "system",
                "content": config["system_prompt"],
            },
            *messages,
        ]
    return messages


def write_meta(path: str | None, client: VerifiersACPClient) -> None:
    if path is not None:
        Path(path).write_text(json.dumps(client.acp_meta, ensure_ascii=False))


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
    blocks = content_blocks(segment_messages(config, is_new), supports_images)
    if not blocks:
        raise ValueError("ACP prompt has no content")
    client.reset()
    metadata_expected = bool(config.get("metadata_expected", False))
    # Initialization, session creation, and resume can emit SessionInfoUpdates.
    # The harness must explicitly opt in before a prompt lifecycle accepts them:
    # no-metadata ACP agents retain their exact historical prompt fast path.
    client.begin_prompt_metadata(expected=metadata_expected)
    try:
        try:
            await connection.prompt(session_id=session_id, prompt=blocks)
        except RequestError as error:
            detail = error.data.get("details") if isinstance(error.data, dict) else None
            raise RuntimeError(detail or str(error)) from error

        # ACP 0.11 dispatches notifications in background tasks but resolves a
        # request response directly in its receive loop. An agent that sends its
        # final session/update immediately before session/prompt returns can
        # therefore wake this coroutine before the update handler has run.
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

        # This wait supplies liveness only. Ownership is established solely by
        # the producer envelope validated below, never by a deadline or stop reason.
        await wait_for_late_metadata(client, expected=metadata_expected)
        client.require_terminal_metadata(expected=metadata_expected)

        if not has_visible_reply():
            raise RuntimeError("ACP agent produced no visible reply")
        return client.visible_reply
    finally:
        # ACP 0.11 has no event-to-prompt correlation id. Closing before the
        # caller exposes this response ensures a late update cannot leak into a
        # subsequent turn; it is quarantined and fails that later turn instead.
        client.close_prompt_metadata()


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

        try:
            reply = await prompt(
                client,
                connection,
                capabilities,
                session_id,
                config,
                is_new=is_new,
            )
        finally:
            write_meta(config.get("meta_path"), client)
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
                    if session.client.turn_acp_meta:
                        response["meta"] = session.client.turn_acp_meta
                    session.client.turn_acp_meta = {}
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
                # A validated terminal envelope belongs to this request even when
                # its outcome is error; preserve it for the persistent-stream caller.
                if session.client.turn_acp_meta:
                    response["meta"] = session.client.turn_acp_meta
                session.client.turn_acp_meta = {}
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
