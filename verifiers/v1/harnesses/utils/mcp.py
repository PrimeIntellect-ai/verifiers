import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from contextlib import AsyncExitStack, asynccontextmanager, suppress
from typing import Any, TypeVar, cast

import httpx2
from anyio import CancelScope
from mcp import Client
from mcp.client.streamable_http import create_mcp_http_client, streamable_http_client

MCP_CALL_ATTEMPTS = 6
MCP_TIMEOUT = 600.0

T = TypeVar("T")


@asynccontextmanager
async def mcp_client(spec: dict[str, Any]) -> AsyncIterator[Client]:
    """Open one fresh MCP client in the caller's task.

    The client negotiates the newest protocol and falls back for older servers.
    Teardown failures after the body completes are suppressed so closing noise cannot
    fail or replay a call whose result is already available.
    """
    stack = AsyncExitStack()
    try:
        http_client = await stack.enter_async_context(
            create_mcp_http_client(
                headers=spec.get("headers") or None,
                timeout=httpx2.Timeout(
                    spec.get("timeout", MCP_TIMEOUT),
                    connect=spec.get("connect_timeout", 5.0),
                ),
            )
        )
        transport = streamable_http_client(spec["url"], http_client=http_client)
        yield await stack.enter_async_context(Client(transport))
    finally:
        with suppress(Exception):
            await stack.aclose()


class MCPConnection:
    """One rollout-owned client, with operations serialized across reconnects.

    The SDK's AnyIO contexts enter and exit in the owner task, even when discovery
    runs under wait_for or a tool call is cancelled by another task.
    """

    def __init__(self, spec: dict[str, Any]):
        # Bundled programs without MCP should not import the retry machinery.
        from tenacity import AsyncRetrying, stop_after_attempt, wait_exponential_jitter

        self.spec = spec
        self.task: asyncio.Task[None] | None = None
        self.ready: asyncio.Future[Client] | None = None
        self.cancel_scope: CancelScope | None = None
        self.lock = asyncio.Lock()
        self.run = AsyncRetrying(
            stop=stop_after_attempt(MCP_CALL_ATTEMPTS),
            wait=wait_exponential_jitter(initial=0.5, max=30),
            reraise=True,
        ).wraps(self._run)

    async def serve(
        self, ready: asyncio.Future["Client"], cancel_scope: CancelScope
    ) -> None:
        stack = AsyncExitStack()
        try:
            stack.enter_context(cancel_scope)
            client = await stack.enter_async_context(mcp_client(self.spec))
            ready.set_result(client)
            await asyncio.Future()
        except asyncio.CancelledError:
            pass
        except Exception as error:  # noqa: BLE001 - deliver owner failures to the caller
            if not ready.done():
                ready.set_exception(error)
        finally:
            if not ready.done():
                ready.cancel()
            # Owner teardown must not replace the caller's error with cancellation.
            with suppress(asyncio.CancelledError):
                await stack.aclose()

    async def _run(self, operation: Callable[["Client"], Awaitable[T]]) -> T:
        await self.lock.acquire()
        try:
            if self.task is None or self.task.done():
                self.ready = asyncio.get_running_loop().create_future()
                self.cancel_scope = CancelScope()
                self.task = asyncio.create_task(
                    self.serve(self.ready, self.cancel_scope)
                )
            assert self.ready is not None
            client = await self.ready
            return await operation(client)
        except BaseException as error:
            await self.aclose(abort=isinstance(error, asyncio.CancelledError))
            raise
        finally:
            self.lock.release()

    async def aclose(self, *, abort: bool = False) -> None:
        if self.task is None:
            return
        task, self.task = self.task, None
        cancel_scope = self.cancel_scope
        assert cancel_scope is not None
        caller = asyncio.current_task()
        assert caller is not None
        # Stack cleanup can follow cancellation while awaiting a model or local tool.
        if abort or caller.cancelling():
            cancel_scope.cancel()
        else:
            task.cancel()
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            cancel_scope.cancel()
            await asyncio.shield(task)
            raise


async def connect_mcp(
    config: dict[str, Any], stack: AsyncExitStack, reserved: set[str] | None = None
) -> tuple[
    list[dict[str, Any]],
    dict[str, tuple[str, str]],
    dict[str, MCPConnection],
]:
    """Enumerate MCP tools and return their schemas, dispatch map, and servers."""
    tool_schemas: list[dict[str, Any]] = []
    dispatch: dict[str, tuple[str, str]] = {}
    servers: dict[str, MCPConnection] = {}
    reserved = reserved or set()
    for name, spec in config.get("mcpServers", {}).items():
        server = servers[name] = MCPConnection(spec)
        stack.push_async_callback(server.aclose)
        result = await server.run(lambda client: client.list_tools())
        for tool in result.tools:
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
                        "parameters": tool.input_schema,
                    },
                }
            )
            dispatch[full] = (name, tool.name)
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
            url = f"data:{block.mime_type};base64,{block.data}"
            parts.append({"type": "image_url", "image_url": {"url": url}})
        else:
            parts.append({"type": "text", "text": str(block)})
    if not parts:
        return str(blocks)
    if all(part["type"] == "text" for part in parts):
        return "\n".join(cast(str, part["text"]) for part in parts)
    return parts


async def call_mcp(
    servers: dict[str, MCPConnection],
    dispatch: dict[str, tuple[str, str]],
    name: str,
    arguments: dict[str, Any],
) -> str | list[dict[str, Any]]:
    """Reuse the rollout's client, reconnecting before retrying a failed call."""
    server_name, raw = dispatch[name]

    result = await servers[server_name].run(
        lambda client: client.call_tool(raw, arguments)
    )
    return mcp_content_to_chat_content(result.content)
