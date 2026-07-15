# /// script
# requires-python = ">=3.11"
# dependencies = ["openai", "mcp"]
# ///
"""The interception endpoint and secret arrive through argv rather than the environment."""

import argparse
import asyncio
import json
from contextlib import AsyncExitStack
from pathlib import Path

from openai import AsyncOpenAI


async def chat(
    client: AsyncOpenAI, model: str, messages: list[dict], tools: list[dict]
):
    completion = await client.chat.completions.create(
        model=model, messages=messages, tools=tools or None
    )
    return completion.choices[0].message


async def connect_mcp(stack: AsyncExitStack, config: dict) -> tuple[list[dict], dict]:
    """Connect to each configured MCP server (a streamable-HTTP `url`); return
    (tool schemas, dispatch mapping `<server>_<tool>` -> (session, raw tool name))."""
    from mcp import ClientSession
    from mcp.client.streamable_http import (
        create_mcp_http_client,
        streamable_http_client,
    )

    tool_schemas: list[dict] = []
    dispatch: dict[str, tuple] = {}
    for name, spec in config.get("mcpServers", {}).items():
        http_client = await stack.enter_async_context(
            create_mcp_http_client(headers=spec.get("headers") or None)
        )
        read, write, *_ = await stack.enter_async_context(
            streamable_http_client(spec["url"], http_client=http_client)
        )
        session = await stack.enter_async_context(ClientSession(read, write))
        await session.initialize()
        for tool in (await session.list_tools()).tools:
            full = f"{name}_{tool.name}"
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
            dispatch[full] = (session, tool.name)
    return tool_schemas, dispatch


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


async def call_mcp(dispatch: dict, name: str, arguments: dict) -> str | list[dict]:
    session, raw = dispatch[name]
    result = await session.call_tool(raw, arguments)
    return mcp_content_to_chat_content(result.content)


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
    # Retries are safe: the interception server records each committed turn and replays a
    # retried (byte-identical) request from that record instead of re-sampling, so a retry after
    # a committed turn returns the recorded completion rather than forking the trace into a
    # dead-end branch. Retries ride out transient transport/provider blips; the generous timeout
    # lets a slow turn finish instead of being cut short and re-sent.
    client = AsyncOpenAI(base_url=args.base_url, api_key=args.api_key, timeout=1800.0)
    config = json.loads(args.mcp_config or "{}")
    async with AsyncExitStack() as stack:
        # asyncio.timeout, not wait_for: the MCP/httpx cancel scopes entered onto
        # `stack` must be exited by this task, not a wait_for-spawned child task.
        if config.get("mcpServers"):
            async with asyncio.timeout(60):
                tools, dispatch = await connect_mcp(stack, config)
        else:
            tools, dispatch = [], {}
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
                    content = await call_mcp(dispatch, name, tool_args)
                else:
                    content = f"error: unknown tool {name!r}"
                messages.append(
                    {"role": "tool", "tool_call_id": call.id, "content": content}
                )


if __name__ == "__main__":
    asyncio.run(main())
