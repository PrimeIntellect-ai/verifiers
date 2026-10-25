# /// script
# requires-python = ">=3.10,<3.15"
# dependencies = ["agent-client-protocol==0.12.1", "httpx"]
# ///
"""Run harness segments through an ACP agent."""

import asyncio
import json
import os
import re
import signal
import sys
import traceback
from collections.abc import AsyncIterator
from contextlib import AsyncExitStack, suppress
from dataclasses import asdict, dataclass
from typing import Any
from urllib.parse import urlsplit

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

# A Prime tunnel's gateway answers this page itself while the tunnel has no frpc
# registered: for the moment after its frpc's connection drops, until frpc logs in again.
TUNNEL_NOT_FOUND = b"Tunnel Not Found"
# How long the relay keeps resending a request the gateway turned away.
RELAY_RETRY_SECONDS = 120.0
_HOP_HEADERS = {
    "connection",
    "content-length",
    "host",
    "keep-alive",
    "proxy-connection",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
}


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
            response = await self.client.post(
                self.url, json={"tool_call_id": tool_call_id, "arguments": arguments}
            )
            response.raise_for_status()
            return response.json()["action"]
        except Exception as error:  # noqa: BLE001 - an unreachable gate lets nothing run
            print(f"tool gate denied {tool_call_id}: {error}", file=sys.stderr)
            return "deny"


class ProviderRelay:
    """Front the rollout's model endpoint inside the box, so a request the tunnel gateway
    turned away is sent again instead of failing the agent.

    The endpoint is a Prime tunnel to the host's interception server. When the tunnel's
    frpc connection drops, the gateway unregisters the tunnel and, until frpc logs in
    again (under a second), answers every new request with its own `404 - Tunnel Not
    Found` page. Such a request never reached the interception server, so it is safe to
    resend; the agent can't tell, and OpenCode treats a provider 404 as fatal. Every other
    answer, the interception server's own errors included, streams back unchanged.

    One request per connection: each response is chunked and closes the connection."""

    def __init__(self, upstream: str) -> None:
        parts = urlsplit(upstream)
        self.origin = f"{parts.scheme}://{parts.netloc}"
        self.client = httpx.AsyncClient(
            timeout=httpx.Timeout(None, connect=30),
            limits=httpx.Limits(max_connections=None, max_keepalive_connections=32),
        )
        self.server: asyncio.Server | None = None
        self.local_origin = ""

    async def start(self) -> None:
        self.server = await asyncio.start_server(
            self._serve, "127.0.0.1", 0, limit=1024 * 1024
        )
        port = self.server.sockets[0].getsockname()[1]
        self.local_origin = f"http://127.0.0.1:{port}"

    def route(self, env: dict[str, str]) -> dict[str, str]:
        """`env` with the endpoint's origin pointed at the relay."""
        origin = re.compile(re.escape(self.origin) + r"(?![\w.:-])")
        return {key: origin.sub(self.local_origin, value) for key, value in env.items()}

    async def close(self) -> None:
        if self.server is not None:
            self.server.close()
        await self.client.aclose()

    async def _serve(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        try:
            request = await _read_request(reader)
            if request is not None:
                await self._forward(*request, writer)
        except (
            ConnectionError,
            asyncio.IncompleteReadError,
            asyncio.LimitOverrunError,
            httpx.HTTPError,
        ):
            pass  # either side hung up mid-exchange: the agent sees the cut, as unrelayed
        except Exception:  # noqa: BLE001 - one bad request must not stop the relay
            traceback.print_exc()
        finally:
            writer.close()
            with suppress(Exception):
                await writer.wait_closed()

    async def _send(
        self, method: str, target: str, headers: list[tuple[str, str]], body: bytes
    ) -> tuple[httpx.Response, bytes | None]:
        """The upstream response, streaming; or a 404, closed, with its raw body."""
        loop = asyncio.get_running_loop()
        deadline = loop.time() + RELAY_RETRY_SECONDS
        delay = 0.25
        while True:
            request = self.client.build_request(
                method, self.origin + target, headers=headers, content=body
            )
            try:
                response = await self.client.send(request, stream=True)
            except httpx.ConnectError as error:  # never sent
                if loop.time() + delay > deadline:
                    raise
                reason = f"{type(error).__name__}: {error}"
            else:
                if response.status_code != 404:
                    return response, None
                try:
                    raw = b"".join([chunk async for chunk in response.aiter_raw()])
                finally:
                    await response.aclose()
                page = httpx.Response(404, headers=response.headers, content=raw)
                if (
                    TUNNEL_NOT_FOUND not in page.content
                    or loop.time() + delay > deadline
                ):
                    return response, raw
                reason = "the tunnel gateway answered 404 Tunnel Not Found"
            print(
                f"provider relay: {reason}; resending {method} {target} in {delay:.2f}s",
                file=sys.stderr,
            )
            await asyncio.sleep(delay)
            delay = min(delay * 2, 8.0)

    async def _forward(
        self,
        method: str,
        target: str,
        headers: list[tuple[str, str]],
        body: bytes,
        writer: asyncio.StreamWriter,
    ) -> None:
        try:
            response, raw = await self._send(method, target, headers, body)
        except httpx.HTTPError as error:
            message = f"provider relay: {type(error).__name__}: {error}".encode()
            head = (
                "HTTP/1.1 502 Bad Gateway\r\nContent-Type: text/plain\r\n"
                f"Content-Length: {len(message)}\r\nConnection: close\r\n\r\n"
            )
            writer.write(head.encode() + message)
            await writer.drain()
            return
        try:
            head = [f"HTTP/1.1 {response.status_code} {response.reason_phrase}"]
            head += [
                f"{key}: {value}"
                for key, value in response.headers.multi_items()
                if key.lower() not in _HOP_HEADERS
            ]
            has_body = method != "HEAD" and response.status_code not in (204, 304)
            if has_body:
                head.append("Transfer-Encoding: chunked")
            head.append("Connection: close")
            writer.write(("\r\n".join(head) + "\r\n\r\n").encode("latin-1"))
            if has_body:
                chunks = response.aiter_raw() if raw is None else _once(raw)
                async for chunk in chunks:
                    if chunk:
                        writer.write(b"%x\r\n%s\r\n" % (len(chunk), chunk))
                        await writer.drain()
                writer.write(b"0\r\n\r\n")
            await writer.drain()
        finally:
            await response.aclose()


async def _once(data: bytes) -> AsyncIterator[bytes]:
    yield data


async def _read_request(
    reader: asyncio.StreamReader,
) -> tuple[str, str, list[tuple[str, str]], bytes] | None:
    """One HTTP/1.1 request: method, target, end-to-end headers and the whole body."""
    try:
        head = await reader.readuntil(b"\r\n\r\n")
    except asyncio.IncompleteReadError as error:
        if not error.partial:
            return None
        raise
    request_line, *lines = head.decode("latin-1").split("\r\n")
    method, target, _ = request_line.split(" ", 2)
    headers = []
    length = 0
    chunked = False
    for line in lines:
        if not line:
            continue
        key, _, value = line.partition(":")
        key, value = key.strip(), value.strip()
        lower = key.lower()
        if lower == "content-length":
            length = int(value)
        elif lower == "transfer-encoding":
            chunked = "chunked" in value.lower()
        if lower not in _HOP_HEADERS and lower != "expect":
            headers.append((key, value))
    if not chunked:
        return method, target, headers, await reader.readexactly(length)
    body = bytearray()
    while size := int((await reader.readuntil(b"\r\n")).split(b";")[0], 16):
        body += (await reader.readexactly(size + 2))[:-2]
    while await reader.readuntil(b"\r\n") != b"\r\n":
        pass  # trailers
    return method, target, headers, bytes(body)


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
        self.relay: ProviderRelay | None = None
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
        env = os.environ.copy()
        if upstream := config.get("provider_url"):
            if self.relay is None:
                relay = ProviderRelay(upstream)
                await relay.start()
                self.relay = relay
            env = self.relay.route(env)
        try:
            agent_process = await self.stack.enter_async_context(
                spawn_agent_process(
                    self.client,
                    command[0],
                    *command[1:],
                    env=env,
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
                if self.relay is not None:
                    await self.relay.close()
                    self.relay = None
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
