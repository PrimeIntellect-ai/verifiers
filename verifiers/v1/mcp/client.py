import asyncio
import json
from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from contextlib import asynccontextmanager
from typing import Any, TypeVar, cast

import httpx
from anyio import BrokenResourceError, ClosedResourceError, EndOfStream
from mcp import ClientSession
from mcp.shared.exceptions import McpError
from mcp.types import CONNECTION_CLOSED
from tenacity import (
    AsyncRetrying,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential_jitter,
)

MCP_CALL_ATTEMPTS = 6
MCP_CANCELLED_CLOSE_GRACE = 1.0
MCP_ERROR_PREFIX = "VF_MCP_ERROR="
MCP_TIMEOUT = 600.0

T = TypeVar("T")


def _cancel_count(task: asyncio.Task[Any] | None) -> int:
    """Return pending cancellations while remaining importable on Python 3.10."""
    return task.cancelling() if task is not None and hasattr(task, "cancelling") else 0


class MCPTransportFailure(Exception):
    """A lost MCP transport with enough context to decide whether replay is safe."""

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
                (
                    error
                    for error in nested
                    if not isinstance(error, asyncio.CancelledError)
                ),
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
    spec: dict[str, Any], ready: asyncio.Future[Any], stop: asyncio.Event
) -> BaseException | None:
    """Own the SDK transport task group so its failure cannot cancel the session owner."""
    from mcp.client.streamable_http import (
        create_mcp_http_client,
        streamable_http_client,
    )

    timeout = httpx.Timeout(
        spec.get("timeout", MCP_TIMEOUT),
        connect=spec.get("connect_timeout", 5.0),
    )
    try:
        async with (
            create_mcp_http_client(
                headers=spec.get("headers") or None,
                timeout=timeout,
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
async def mcp_session(
    spec: dict[str, Any],
    *,
    server: str = "",
    operation: str = "operation",
    replay_safe: bool = False,
) -> AsyncIterator[ClientSession]:
    """Open one streamable-HTTP session with transport ownership in a dedicated task.

    Initialization is replay-safe. Once the caller's operation starts, transport loss
    is replay-safe only when the caller explicitly says the operation is repeatable.
    Teardown noise cannot replace an operation that already returned or raised.
    """
    ready = asyncio.get_running_loop().create_future()
    stop = asyncio.Event()
    transport = asyncio.create_task(
        _mcp_transport(spec, ready, stop), name=f"mcp-transport-{server}"
    )
    failure: BaseException | None = None
    initialized = False
    operation_completed = False
    outer_cancel: asyncio.CancelledError | None = None
    try:
        read, write, *_ = await ready
        async with ClientSession(read, write) as session:
            try:
                await session.initialize()
                initialized = True
                yield session
            except asyncio.CancelledError as error:
                outer_cancel = error
                raise
            operation_completed = True
    except BaseException as error:  # noqa: BLE001 - preserve caller cancellation
        task = asyncio.current_task()
        if outer_cancel is None and isinstance(error, asyncio.CancelledError):
            outer_cancel = error
        if _cancel_count(task):
            failure = outer_cancel or asyncio.CancelledError()
        elif not operation_completed:
            failure = error
    finally:
        stop.set()
        task = asyncio.current_task()
        aborting = (
            isinstance(failure, asyncio.CancelledError) and _cancel_count(task) > 0
        )
        if aborting:
            transport.cancel()
        while not transport.done():
            cancel_count = _cancel_count(task)
            try:
                if aborting:
                    await asyncio.wait_for(
                        asyncio.shield(transport), MCP_CANCELLED_CLOSE_GRACE
                    )
                else:
                    await asyncio.shield(transport)
            except asyncio.TimeoutError:  # noqa: UP041 - distinct on Python 3.10
                transport.cancel()
            except asyncio.CancelledError as error:
                task = asyncio.current_task()
                if _cancel_count(task) > cancel_count:
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
                (
                    error
                    for error in nested
                    if not isinstance(error, asyncio.CancelledError)
                ),
                nested[0],
            )
        session_lost = (
            isinstance(failure_leaf, McpError)
            and failure_leaf.error.code == CONNECTION_CLOSED
            and failure_leaf.error.message == "Connection closed"
        ) or isinstance(
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
        if session_lost:
            transport_failure = failure

    if failure is None:
        return
    task = asyncio.current_task()
    if isinstance(failure, asyncio.CancelledError) and (
        task is None or _cancel_count(task) or transport_failure is None
    ):
        raise failure
    if transport_failure is not failure:
        raise failure
    raise MCPTransportFailure(
        server, operation, initialized, replay_safe, transport_failure
    ) from transport_failure


async def with_retry(call: Callable[[], Awaitable[T]]) -> T:
    """Retry transient failures only while replaying the operation is safe."""
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
    raise RuntimeError("retrying stopped without returning or raising")


async def connect_mcp(
    config: dict[str, Any], reserved: set[str] | None = None
) -> tuple[
    list[dict[str, Any]],
    dict[str, tuple[str, str, bool]],
    dict[str, dict[str, Any]],
]:
    """Enumerate configured MCP tools and build chat schemas and dispatch metadata."""
    tool_schemas: list[dict[str, Any]] = []
    dispatch: dict[str, tuple[str, str, bool]] = {}
    servers: dict[str, dict[str, Any]] = {}
    reserved = reserved or set()
    for name, spec in config.get("mcpServers", {}).items():
        servers[name] = spec

        async def list_tools(name: str = name, spec: dict[str, Any] = spec):
            async with mcp_session(
                spec,
                server=name,
                operation="list_tools",
                replay_safe=True,
            ) as session:
                return (await session.list_tools()).tools

        for tool in await with_retry(list_tools):
            full = f"{name}_{tool.name}" if name else tool.name
            if full in reserved or full in dispatch:
                raise ValueError(
                    f"duplicate tool name {full!r}; keep MCP tool names qualified"
                )
            annotations = tool.annotations
            replay_safe = bool(
                annotations
                and (
                    annotations.readOnlyHint is True
                    or annotations.idempotentHint is True
                )
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
            dispatch[full] = (name, tool.name, replay_safe)
    return tool_schemas, dispatch, servers


def mcp_content_to_chat_content(
    blocks: Sequence[Any],
) -> str | list[dict[str, Any]]:
    """Convert MCP content blocks to OpenAI chat tool-result content."""
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
        return "\n".join(cast(str, part["text"]) for part in parts)
    return parts


async def call_mcp(
    servers: dict[str, dict[str, Any]],
    dispatch: dict[str, tuple[str, str, bool]],
    name: str,
    arguments: dict[str, Any],
) -> str | list[dict[str, Any]]:
    """Call a tool, replaying a lost response only when its annotations permit it."""
    server, raw, replay_safe = dispatch[name]

    async def call():
        async with mcp_session(
            servers[server],
            server=server,
            operation="call_tool",
            replay_safe=replay_safe,
        ) as session:
            return await session.call_tool(raw, arguments)

    result = await with_retry(call)
    return mcp_content_to_chat_content(result.content)
