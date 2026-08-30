from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from contextlib import AsyncExitStack, asynccontextmanager, suppress
from typing import Any, TypeVar, cast

import httpx
from mcp import ClientSession
from mcp.client.streamable_http import create_mcp_http_client, streamable_http_client
from tenacity import AsyncRetrying, stop_after_attempt, wait_exponential_jitter

MCP_CALL_ATTEMPTS = 6
MCP_TIMEOUT = 600.0

T = TypeVar("T")


@asynccontextmanager
async def mcp_session(spec: dict[str, Any]) -> AsyncIterator[ClientSession]:
    """Open one fresh streamable-HTTP session in the caller's task.

    Teardown failures after the body completes are suppressed so closing noise cannot
    fail or replay a call whose result is already available.
    """
    stack = AsyncExitStack()
    try:
        http_client = await stack.enter_async_context(
            create_mcp_http_client(
                headers=spec.get("headers") or None,
                timeout=httpx.Timeout(
                    spec.get("timeout", MCP_TIMEOUT),
                    connect=spec.get("connect_timeout", 5.0),
                ),
            )
        )
        read, write, *_ = await stack.enter_async_context(
            streamable_http_client(spec["url"], http_client=http_client)
        )
        session = await stack.enter_async_context(ClientSession(read, write))
        await session.initialize()
        yield session
    finally:
        with suppress(Exception):
            await stack.aclose()


async def with_retry(call: Callable[[], Awaitable[T]]) -> T:
    """Run one session-scoped operation with the existing at-least-once retries."""
    async for attempt in AsyncRetrying(
        stop=stop_after_attempt(MCP_CALL_ATTEMPTS),
        wait=wait_exponential_jitter(initial=0.5, max=30),
        reraise=True,
    ):
        with attempt:
            return await call()
    raise RuntimeError("retrying stopped without returning or raising")


async def connect_mcp(
    config: dict[str, Any], reserved: set[str] | None = None
) -> tuple[
    list[dict[str, Any]],
    dict[str, tuple[str, str]],
    dict[str, dict[str, Any]],
]:
    """Enumerate MCP tools and return their schemas, dispatch map, and servers."""
    tool_schemas: list[dict[str, Any]] = []
    dispatch: dict[str, tuple[str, str]] = {}
    servers: dict[str, dict[str, Any]] = {}
    reserved = reserved or set()
    for name, spec in config.get("mcpServers", {}).items():
        servers[name] = spec

        async def list_tools(spec: dict[str, Any] = spec):
            async with mcp_session(spec) as session:
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
    dispatch: dict[str, tuple[str, str]],
    name: str,
    arguments: dict[str, Any],
) -> str | list[dict[str, Any]]:
    """Call one MCP tool on a fresh session per retry attempt."""
    server_name, raw = dispatch[name]

    async def call():
        async with mcp_session(servers[server_name]) as session:
            return await session.call_tool(raw, arguments)

    result = await with_retry(call)
    return mcp_content_to_chat_content(result.content)
