# /// script
# requires-python = ">=3.10"
# dependencies = ["openai", "mcp"]
# ///
"""The default harness's program: a chat loop with the taskset's MCP tools (and none of its own).

A growing-message-list chat loop. When the harness sets MCP_CONFIG (a standard `mcpServers` URL
map) it connects to those servers over streamable HTTP, exposes their tools to the model as
`<server>_<tool>`, and routes those calls to the server. The loop runs until the model answers
without a tool call (immediately, when no tools are offered) — unless the task has a user simulator
(`--user-url`, its own MCP server), in which case a no-tool-call turn is a turn boundary: the
program calls the simulator's `respond` tool, appends the user message(s) it returns, and
re-prompts, until the simulator returns no further turns. A no-prompt task is opened the same way
(`respond("")` before the first model call). Carrying every user turn in the program's own
conversation keeps the recorded message graph linear and the user turns regular user messages.

It runs as a uv script (deps: openai, mcp), so the chat + tool plumbing — including the user
simulator, which is just another MCP server — is just the SDKs; the harness bootstraps `uv` in the
runtime. The interception endpoint, per-rollout secret, and model arrive as argv (not env), so
nothing the program spawns inherits them.
"""

import argparse
import asyncio
import json
import os
from contextlib import AsyncExitStack

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


async def connect_user_sim(stack: AsyncExitStack, url: str):
    """Connect to the task's user simulator — its own MCP server (a `vf.User`), never shown to the
    model — and return an async `respond(text)` that calls its `respond` tool, returning the next
    user message(s) as OpenAI wire dicts (an empty list = the simulator is done)."""
    from mcp import ClientSession
    from mcp.client.streamable_http import (
        create_mcp_http_client,
        streamable_http_client,
    )

    http_client = await stack.enter_async_context(create_mcp_http_client())
    read, write, *_ = await stack.enter_async_context(
        streamable_http_client(url, http_client=http_client)
    )
    session = await stack.enter_async_context(ClientSession(read, write))
    await session.initialize()

    async def respond(text: str) -> list[dict]:
        result = await session.call_tool("respond", {"message": text})
        parts = [b.text for b in result.content if getattr(b, "type", None) == "text"]
        return json.loads("\n".join(parts))["messages"]

    return respond


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--api-key", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--system-prompt", default="")
    parser.add_argument("--prompt", default="")
    parser.add_argument("--mcp-config", default="")
    parser.add_argument("--user-url", default="")
    return parser.parse_args()


async def main() -> None:
    args = parse_args()
    client = AsyncOpenAI(base_url=args.base_url, api_key=args.api_key)
    config = json.loads(args.mcp_config or "{}")
    user_url = args.user_url or None
    async with AsyncExitStack() as stack:
        tools, dispatch = (
            await connect_mcp(stack, config) if config.get("mcpServers") else ([], {})
        )
        user_respond = await connect_user_sim(stack, user_url) if user_url else None
        messages = (
            [{"role": "system", "content": args.system_prompt}]
            if args.system_prompt
            else []
        )
        # A Messages prompt (e.g. an image-bearing prompt) arrives pre-built as OpenAI wire dicts
        # via INITIAL_MESSAGES (kept in env: it can be large multimodal content that overflows
        # argv, and it's prompt content, not a credential); otherwise --prompt is the opening
        # message. Both empty means the task has no prompt — the user simulator opens it.
        initial = json.loads(os.environ.get("INITIAL_MESSAGES", "[]"))
        if initial:
            messages.extend(initial)
        elif args.prompt:
            messages.append({"role": "user", "content": args.prompt})
        elif user_respond:
            opening = await user_respond("")
            if not opening:
                return
            messages.extend(opening)
        while True:
            message = await chat(client, args.model, messages, tools)
            messages.append(message.model_dump(exclude_none=True))
            if message.tool_calls:
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
                    if name in dispatch:
                        content = await call_mcp(dispatch, name, tool_args)
                    else:
                        content = f"error: unknown tool {name!r}"
                    messages.append(
                        {"role": "tool", "tool_call_id": call.id, "content": content}
                    )
                continue
            # No tool call: a final answer for the current user turn. With a user simulator, ask it
            # for the next user turn and re-prompt; an empty reply means it's done. Without one, the
            # conversation is over.
            if user_respond:
                reply = await user_respond(message.content or "")
                if not reply:
                    break
                messages.extend(reply)
                continue
            break


if __name__ == "__main__":
    asyncio.run(main())
