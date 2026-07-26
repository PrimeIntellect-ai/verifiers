"""Streamable-HTTP MCP client embedded into the shared null/bash program."""

import asyncio
import json
from contextlib import asynccontextmanager

import httpx
from anyio import BrokenResourceError, ClosedResourceError, EndOfStream
from mcp.shared.exceptions import McpError
from mcp.types import CONNECTION_CLOSED
from tenacity import (
    AsyncRetrying,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential_jitter,
)

MCP_CALL_ATTEMPTS = 6
MCP_CANCELLED_CLOSE_GRACE = 1.0  # then re-cancel a hung stateful-session DELETE
MCP_ERROR_PREFIX = "VF_MCP_ERROR="
MCP_TIMEOUT = httpx.Timeout(600.0, connect=5.0)  # the OpenAI SDK client defaults


class MCPTransportFailure(Exception):
    """A lost MCP transport, with enough policy data for the parent harness to classify it."""

    def __init__(
        self,
        server: str,
        operation: str,
        initialized: bool,
        replay_safe: bool,
        cause: BaseException,
    ) -> None:
        leaf = cause
        while nested := getattr(leaf, "exceptions", ()):
            leaf = next(
                (e for e in nested if not isinstance(e, asyncio.CancelledError)),
                nested[0],
            )
        status = getattr(getattr(leaf, "response", None), "status_code", None)
        transient = (
            isinstance(
                leaf,
                (
                    asyncio.CancelledError,
                    BrokenResourceError,
                    ClosedResourceError,
                    EndOfStream,
                    httpx.TransportError,
                    OSError,
                    TimeoutError,
                ),
            )
            or (
                isinstance(leaf, McpError)
                and leaf.error.code == CONNECTION_CLOSED
                and leaf.error.message == "Connection closed"
            )
            or status in (408, 429)
            or (status is not None and status >= 500)
        )
        can_replay = not initialized or replay_safe
        self.session_retry_safe = transient and can_replay
        detail = {
            "cause_type": type(leaf).__name__,
            "delivery": "response_unknown" if initialized else "operation_not_started",
            "operation": operation,
            "phase": operation if initialized else "initialize",
            "replay_safe": can_replay,
            "server": server,
            "session_retry_safe": self.session_retry_safe,
            "status_code": status,
            "transient": transient,
        }
        super().__init__(MCP_ERROR_PREFIX + json.dumps(detail, separators=(",", ":")))


async def _mcp_transport(
    spec: dict, ready: asyncio.Future, stop: asyncio.Event
) -> BaseException | None:
    """Own the SDK transport task group separately so its failure cannot cancel the session owner."""
    from mcp.client.streamable_http import (
        create_mcp_http_client,
        streamable_http_client,
    )

    try:
        async with (
            create_mcp_http_client(
                headers=spec.get("headers") or None, timeout=MCP_TIMEOUT
            ) as http_client,
            streamable_http_client(spec["url"], http_client=http_client) as streams,
        ):
            if not ready.done():
                ready.set_result(streams)
            await stop.wait()
    except BaseException as error:  # noqa: BLE001 - preserve transport cancellation
        if not ready.done():
            ready.set_exception(error)
        return error
    return None


@asynccontextmanager
async def mcp_session(server: str, spec: dict, operation: str, replay_safe: bool):
    """Open one session while keeping the transport's AnyIO cancel scope in its owner task."""
    from mcp import ClientSession

    ready = asyncio.get_running_loop().create_future()
    stop = asyncio.Event()
    transport = asyncio.create_task(
        _mcp_transport(spec, ready, stop), name=f"mcp-transport-{server}"
    )
    failure = None
    initialized = False
    operation_completed = False
    try:
        read, write, *_ = await ready
        async with ClientSession(read, write) as session:
            await session.initialize()
            initialized = True
            yield session
            operation_completed = True
    except BaseException as error:  # noqa: BLE001 - preserve caller cancellation
        task = asyncio.current_task()
        if not operation_completed or (task is not None and task.cancelling()):
            failure = error
    finally:
        stop.set()
        task = asyncio.current_task()
        aborting = (
            isinstance(failure, asyncio.CancelledError)
            and task is not None
            and task.cancelling()
        )
        if aborting:
            transport.cancel()
        while not transport.done():
            cancel_count = task.cancelling() if task is not None else 0
            try:
                if aborting:
                    await asyncio.wait_for(
                        asyncio.shield(transport), MCP_CANCELLED_CLOSE_GRACE
                    )
                else:
                    await asyncio.shield(transport)
            except TimeoutError:
                transport.cancel()
            except asyncio.CancelledError as error:
                task = asyncio.current_task()
                if task is not None and task.cancelling() > cancel_count:
                    failure = error
                    aborting = True
                    transport.cancel()
        try:
            transport_failure = transport.result()
        except asyncio.CancelledError as error:
            transport_failure = error
        failure_leaf = failure
        while nested := getattr(failure_leaf, "exceptions", ()):
            failure_leaf = next(
                (e for e in nested if not isinstance(e, asyncio.CancelledError)),
                nested[0],
            )
        connection_closed = (
            isinstance(failure_leaf, McpError)
            and failure_leaf.error.code == CONNECTION_CLOSED
            and failure_leaf.error.message == "Connection closed"
        )
        if transport_failure is None and (
            connection_closed
            or isinstance(
                failure_leaf,
                (
                    BrokenResourceError,
                    ClosedResourceError,
                    EndOfStream,
                    httpx.TransportError,
                    OSError,
                    TimeoutError,
                ),
            )
        ):
            transport_failure = failure

    if failure is None:
        return
    task = asyncio.current_task()
    if isinstance(failure, asyncio.CancelledError) and (
        task is None or task.cancelling() or transport_failure is None
    ):
        raise failure
    if transport_failure is None:
        raise failure
    raise MCPTransportFailure(
        server, operation, initialized, replay_safe, transport_failure
    ) from transport_failure


async def with_retry(call):
    """Retry only transient failures before a replay-safe operation has completed."""
    async for attempt in AsyncRetrying(
        stop=stop_after_attempt(MCP_CALL_ATTEMPTS),
        wait=wait_exponential_jitter(initial=0.5, max=30),
        retry=retry_if_exception(
            lambda error: (
                isinstance(error, MCPTransportFailure) and error.session_retry_safe
            )
        ),
        reraise=True,
    ):
        with attempt:
            return await call()


async def connect_mcp(
    config: dict, reserved: set[str] | None = None
) -> tuple[list[dict], dict, dict]:
    """Enumerate MCP tools and build advertised-name dispatch metadata."""
    reserved = reserved or set()
    tool_schemas: list[dict] = []
    dispatch: dict[str, tuple] = {}
    servers: dict[str, dict] = {}
    for name, spec in config.get("mcpServers", {}).items():
        servers[name] = spec

        async def list_tools(name: str = name, spec: dict = spec):
            async with mcp_session(
                name, spec, "list_tools", replay_safe=True
            ) as session:
                return (await session.list_tools()).tools

        for tool in await with_retry(list_tools):
            full = f"{name}_{tool.name}" if name else tool.name
            if full in reserved or full in dispatch:
                raise ValueError(
                    f"duplicate tool name {full!r}; keep MCP tool names qualified"
                )
            tool_schemas.append(
                {
                    "type": "function",
                    "function": {
                        "name": full,
                        "description": tool.description or "",
                        "parameters": tool.inputSchema,
                    },
                }
            )
            dispatch[full] = (name, tool.name)
    return tool_schemas, dispatch, servers


def mcp_content_to_chat_content(blocks) -> str | list[dict]:
    parts = []
    for block in blocks:
        if block.type == "text":
            parts.append({"type": "text", "text": block.text})
        elif block.type == "image":
            url = f"data:{block.mimeType};base64,{block.data}"
            parts.append({"type": "image_url", "image_url": {"url": url}})
        else:
            parts.append({"type": "text", "text": str(block)})
    if not parts:
        return str(blocks)
    if all(part["type"] == "text" for part in parts):
        return "\n".join(part["text"] for part in parts)
    return parts


async def call_mcp(
    servers: dict, dispatch: dict, name: str, arguments: dict
) -> str | list[dict]:
    """Call once after initialization; a lost response is never replayed in this session."""
    server, raw = dispatch[name]

    async def call():
        async with mcp_session(
            server, servers[server], "call_tool", replay_safe=False
        ) as session:
            return await session.call_tool(raw, arguments)

    result = await with_retry(call)
    return mcp_content_to_chat_content(result.content)
