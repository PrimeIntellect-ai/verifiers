"""Shared Chat Completions and MCP loop embedded in direct harness programs."""

import asyncio
import json
from contextlib import AsyncExitStack, asynccontextmanager, suppress
from pathlib import Path

import httpx
from openai import AsyncOpenAI
from tenacity import AsyncRetrying, stop_after_attempt, wait_exponential_jitter

# {tool_interception}

MCP_CALL_ATTEMPTS = 6
MCP_TIMEOUT = 600.0


def add_chat_arguments(parser) -> None:
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--api-key", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--system-prompt", default="")
    parser.add_argument("--prompt", default="")
    parser.add_argument("--initial-messages-file", default="")
    parser.add_argument("--mcp-config", default="")
    parser.add_argument("--tool-interception-url", default="")
    parser.add_argument("--tool-interception-secret-bytes", type=int, default=0)


@asynccontextmanager
async def mcp_session(spec: dict):
    """Open one operation-scoped streamable HTTP session."""
    from mcp import ClientSession
    from mcp.client.streamable_http import (
        create_mcp_http_client,
        streamable_http_client,
    )

    stack = AsyncExitStack()
    try:
        http_client = await stack.enter_async_context(
            create_mcp_http_client(
                headers=spec.get("headers") or None,
                timeout=httpx.Timeout(spec.get("timeout", MCP_TIMEOUT), connect=5.0),
            )
        )
        read, write, *_ = await stack.enter_async_context(
            streamable_http_client(spec["url"], http_client=http_client)
        )
        session = await stack.enter_async_context(ClientSession(read, write))
        await session.initialize()
        yield session
    finally:
        # Closing noise must not fail or replay an already answered tool call.
        with suppress(Exception):
            await stack.aclose()


async def with_mcp_retry(call):
    """Retry transient session failures; MCP has at-least-once delivery."""
    async for attempt in AsyncRetrying(
        stop=stop_after_attempt(MCP_CALL_ATTEMPTS),
        wait=wait_exponential_jitter(initial=0.5, max=30),
        reraise=True,
    ):
        with attempt:
            return await call()


async def connect_mcp(config: dict, reserved: set[str]) -> tuple[list, dict, dict]:
    """Enumerate tools and build their advertised-name dispatch table."""
    schemas = []
    dispatch = {}
    servers = {}
    for server_name, spec in config.get("mcpServers", {}).items():
        servers[server_name] = spec

        async def list_tools(spec: dict = spec):
            async with mcp_session(spec) as session:
                return (await session.list_tools()).tools

        for tool in await with_mcp_retry(list_tools):
            name = f"{server_name}_{tool.name}" if server_name else tool.name
            if name in reserved or name in dispatch:
                raise ValueError(
                    f"duplicate tool name {name!r}; keep MCP tool names qualified"
                )
            schemas.append(
                {
                    "type": "function",
                    "function": {
                        "name": name,
                        "description": tool.description or "",
                        "parameters": tool.inputSchema,
                    },
                }
            )
            dispatch[name] = (server_name, tool.name)
    return schemas, dispatch, servers


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


async def call_mcp(servers: dict, dispatch: dict, name: str, arguments: dict):
    server_name, raw_name = dispatch[name]

    async def call():
        async with mcp_session(servers[server_name]) as session:
            return await session.call_tool(raw_name, arguments)

    result = await with_mcp_retry(call)
    return mcp_content_to_chat_content(result.content)


async def run_chat(
    args,
    local_tools: list[dict],
    execute_local=None,
    *,
    harness: str,
    model_timeout=None,
    mcp_connect_timeout: float | None = None,
) -> None:
    """Run the common model/tool loop around harness-specific local tools."""
    secret = read_tool_secret(  # noqa: F821 - injected runtime client
        args.tool_interception_secret_bytes, harness
    )
    interceptor = (
        ToolInterceptionClient(args.tool_interception_url, secret)  # noqa: F821
        if args.tool_interception_url
        else None
    )
    initial = []
    if args.initial_messages_file:
        path = Path(args.initial_messages_file)
        initial = json.loads(path.read_bytes())
        path.unlink()
    client_options = {"timeout": model_timeout} if model_timeout is not None else {}
    client = AsyncOpenAI(
        base_url=args.base_url,
        api_key=args.api_key,
        **client_options,
    )
    config = json.loads(args.mcp_config or "{}")
    reserved = {tool["function"]["name"] for tool in local_tools}
    if config.get("mcpServers"):
        if mcp_connect_timeout is None:
            mcp_tools, dispatch, servers = await connect_mcp(config, reserved)
        else:
            async with asyncio.timeout(mcp_connect_timeout):
                mcp_tools, dispatch, servers = await connect_mcp(config, reserved)
    else:
        mcp_tools, dispatch, servers = [], {}, {}
    tools = [*local_tools, *mcp_tools]
    messages = (
        [{"role": "system", "content": args.system_prompt}]
        if args.system_prompt
        else []
    )
    if initial:
        messages.extend(initial)
    elif args.prompt:
        messages.append({"role": "user", "content": args.prompt})

    try:
        while True:
            completion = await client.chat.completions.create(
                model=args.model,
                messages=messages,
                tools=tools or None,
            )
            message = completion.choices[0].message
            messages.append(message.model_dump(exclude_none=True))
            if not message.tool_calls:
                return
            for tool_call in message.tool_calls:
                name = tool_call.function.name
                tool_message = {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": "",
                    "name": name,
                }
                if interceptor is not None:
                    decision = await asyncio.to_thread(
                        interceptor.call, "before", tool_message
                    )
                    if decision["action"] == "rewrite":
                        messages.append(decision["message"])
                        continue
                try:
                    arguments = json.loads(tool_call.function.arguments or "{}")
                except json.JSONDecodeError as error:
                    content = (
                        "error: invalid JSON in tool arguments "
                        f"({error}); resend the call with valid JSON"
                    )
                else:
                    if not isinstance(arguments, dict):
                        content = (
                            "error: tool arguments must be a JSON object, got "
                            f"{type(arguments).__name__}; resend as an object"
                        )
                    elif name in dispatch:
                        content = await call_mcp(servers, dispatch, name, arguments)
                    elif execute_local is not None:
                        content = await execute_local(name, arguments)
                    else:
                        content = f"error: unknown tool {name!r}"
                tool_message["content"] = content
                if interceptor is not None:
                    decision = await asyncio.to_thread(
                        interceptor.call, "after", tool_message
                    )
                    if decision["action"] == "rewrite":
                        tool_message = decision["message"]
                messages.append(tool_message)
    finally:
        if interceptor is not None:
            interceptor.close()
