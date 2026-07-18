# /// script
# requires-python = ">=3.11"
# dependencies = ["openai", "mcp"]
# ///
"""The interception endpoint and secret arrive through argv rather than the environment."""

import argparse
import asyncio
import json
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
from openai import AsyncOpenAI

MCP_CALL_ATTEMPTS = 6
MCP_CALL_BACKOFF = 0.2  # seconds, exponential up to the cap
MCP_CALL_MAX_BACKOFF = 2.0
MCP_TIMEOUT = httpx.Timeout(60.0, read=300.0)


async def chat(
    client: AsyncOpenAI, model: str, messages: list[dict], tools: list[dict]
):
    completion = await client.chat.completions.create(
        model=model, messages=messages, tools=tools or None
    )
    return completion.choices[0].message


@asynccontextmanager
async def mcp_session(spec: dict):
    """Open a fresh streamable-HTTP session to one MCP server, entered and exited within the
    caller's task so AnyIO cancellation scopes stay correctly nested."""
    from mcp import ClientSession
    from mcp.client.streamable_http import (
        create_mcp_http_client,
        streamable_http_client,
    )

    async with (
        create_mcp_http_client(
            headers=spec.get("headers") or None, timeout=MCP_TIMEOUT
        ) as http_client,
        streamable_http_client(spec["url"], http_client=http_client) as (
            read,
            write,
            *_,
        ),
        ClientSession(read, write) as session,
    ):
        await session.initialize()
        yield session


async def connect_mcp(config: dict) -> tuple[list[dict], dict, dict]:
    """Enumerate each configured MCP server's tools (a streamable-HTTP `url`); return (tool schemas,
    dispatch mapping advertised name -> (server name, raw tool name), servers mapping name -> spec).
    No session is held — a stateless-HTTP server is reconnected per call. Tools are advertised as
    `<server>_<tool>`; a server named `""` (TOOL_PREFIX = None) advertises its tools bare, so names
    must be unique across the rollout's servers."""
    tool_schemas: list[dict] = []
    dispatch: dict[str, tuple] = {}
    servers: dict[str, dict] = {}
    for name, spec in config.get("mcpServers", {}).items():
        servers[name] = spec
        async with mcp_session(spec) as session:
            for tool in (await session.list_tools()).tools:
                full = f"{name}_{tool.name}" if name else tool.name
                if full in dispatch:
                    raise ValueError(
                        f"duplicate tool name {full!r} across servers; keep qualified names"
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
    """Call a tool on a fresh session, retrying a dropped connection so connection churn under high
    concurrency doesn't crash the harness. Retrying any Exception is sound: a tool that fails for
    real returns an error in its content (not an exception), and cancellation is a BaseException."""
    server_name, raw = dispatch[name]
    spec = servers[server_name]
    for attempt in range(MCP_CALL_ATTEMPTS):
        try:
            async with mcp_session(spec) as session:
                result = await session.call_tool(raw, arguments)
                return mcp_content_to_chat_content(result.content)
        except Exception:
            if attempt + 1 == MCP_CALL_ATTEMPTS:
                raise
            await asyncio.sleep(
                min(MCP_CALL_BACKOFF * 2**attempt, MCP_CALL_MAX_BACKOFF)
            )
    raise RuntimeError("unreachable")  # loop either returns or raises


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--api-key", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--system-prompt", default="")
    parser.add_argument("--prompt", default="")
    parser.add_argument("--initial-messages-file", default="")
    parser.add_argument("--mcp-config", default="")
    return parser.parse_args()


async def main() -> None:
    args = parse_args()
    initial = []
    if args.initial_messages_file:
        path = Path(args.initial_messages_file)
        payload = path.read_bytes()
        path.unlink()
        initial = json.loads(payload)
    client = AsyncOpenAI(base_url=args.base_url, api_key=args.api_key)
    config = json.loads(args.mcp_config or "{}")
    if config.get("mcpServers"):
        # Bound only tool enumeration; each session is opened and closed within this task.
        async with asyncio.timeout(60):
            tools, dispatch, servers = await connect_mcp(config)
    else:
        tools, dispatch, servers = [], {}, {}
    messages = (
        [{"role": "system", "content": args.system_prompt}]
        if args.system_prompt
        else []
    )
    if initial:
        messages.extend(initial)
    elif args.prompt:
        messages.append({"role": "user", "content": args.prompt})
    while True:
        message = await chat(client, args.model, messages, tools)
        messages.append(message.model_dump(exclude_none=True))
        if not message.tool_calls:
            break
        for call in message.tool_calls:
            name = call.function.name
            try:
                tool_args = json.loads(call.function.arguments or "{}")
            except json.JSONDecodeError as e:
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.id,
                        "content": f"error: invalid JSON in tool arguments ({e}); resend the call with valid JSON",
                    }
                )
                continue
            # Valid JSON can still be a non-object (`[]`, `42`, `null`); the MCP dispatch
            # assumes a dict, so reject anything else as a tool error rather than crashing.
            if not isinstance(tool_args, dict):
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.id,
                        "content": f"error: tool arguments must be a JSON object, got {type(tool_args).__name__}; resend as an object",
                    }
                )
                continue
            if name in dispatch:
                content = await call_mcp(servers, dispatch, name, tool_args)
            else:
                content = f"error: unknown tool {name!r}"
            messages.append(
                {"role": "tool", "tool_call_id": call.id, "content": content}
            )


if __name__ == "__main__":
    asyncio.run(main())
