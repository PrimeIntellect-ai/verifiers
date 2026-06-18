# /// script
# requires-python = ">=3.10"
# dependencies = ["openai", "mcp"]
# ///
"""The default harness's program: a chat loop with the taskset's MCP tools (and none of its own).

A growing-message-list chat loop. When the harness sets MCP_CONFIG (a standard `mcpServers` URL
map) it connects to those servers over streamable HTTP, exposes their tools to the model as
`<server>_<tool>`, and routes those calls to the server. The loop runs until the model answers
without a tool call (immediately, when no tools are offered).

It runs as a uv script (deps: openai, mcp), so the chat + tool plumbing is just the
SDKs — the harness bootstraps `uv` in the runtime. Model calls go to the interception
server (OPENAI_BASE_URL/API_KEY).
"""

import asyncio
import json
import os
import sys
from contextlib import AsyncExitStack

from openai import AsyncOpenAI, omit

client = AsyncOpenAI()


async def chat(messages: list[dict], tools: list[dict]):
    completion = await client.chat.completions.create(
        model=os.environ["OPENAI_MODEL"], messages=messages, tools=tools or omit
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


async def main() -> None:
    config = json.loads(os.environ.get("MCP_CONFIG", "{}"))
    use_responses = "--responses" in sys.argv[2:]
    async with AsyncExitStack() as stack:
        tools, dispatch = (
            await connect_mcp(stack, config) if config.get("mcpServers") else ([], {})
        )
        if use_responses:
            tools = [{"type": "function", **tool["function"]} for tool in tools]
        system_prompt = os.environ.get("APPEND_SYSTEM_PROMPT", "")
        messages = (
            [{"role": "system", "content": system_prompt}] if system_prompt else []
        )
        # A Messages prompt (e.g. an image-bearing prompt) arrives pre-built as OpenAI
        # wire dicts; otherwise the single argv string is the first user message. An empty argv
        # means the task has no prompt — the framework's user simulator seeds the opening turn,
        # so send no user message and let the interception server inject it.
        initial = json.loads(os.environ.get("INITIAL_MESSAGES", "[]"))
        if initial:
            messages.extend(initial)
        elif sys.argv[1]:
            messages.append({"role": "user", "content": sys.argv[1]})
        while True:
            if use_responses:
                response = await client.responses.create(
                    model=os.environ["OPENAI_MODEL"],
                    input=messages,
                    tools=tools or omit,
                )
                messages.extend(
                    item.model_dump(mode="json", exclude_unset=True)
                    for item in response.output
                )
                calls = [
                    item for item in response.output if item.type == "function_call"
                ]
            else:
                message = await chat(messages, tools)
                messages.append(message.model_dump(exclude_none=True))
                calls = message.tool_calls or []
            if not calls:
                break
            for call in calls:
                function = call if use_responses else call.function
                if function.name in dispatch:
                    content = await call_mcp(
                        dispatch,
                        function.name,
                        json.loads(function.arguments or "{}"),
                    )
                else:
                    content = f"error: unknown tool {function.name!r}"
                if use_responses and isinstance(content, list):
                    for part in content:
                        part["type"] = f"input_{part['type'].removesuffix('_url')}"
                        if part["type"] == "input_image":
                            part["image_url"] = part["image_url"]["url"]
                messages.append(
                    {
                        "type": "function_call_output",
                        "call_id": call.call_id,
                        "output": content,
                    }
                    if use_responses
                    else {"role": "tool", "tool_call_id": call.id, "content": content}
                )


if __name__ == "__main__":
    asyncio.run(main())
