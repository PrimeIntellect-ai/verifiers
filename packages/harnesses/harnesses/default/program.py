# /// script
# requires-python = ">=3.10"
# dependencies = ["openai", "mcp"]
# ///
"""The default harness's program: a chat loop with an optional bash tool (+ optional MCP tools).

A growing-message-list chat loop. It offers a local `bash` tool when ENABLE_BASH is set
(the default); when the harness sets MCP_CONFIG (a standard `mcpServers` URL map) it also
connects to those servers over streamable HTTP, exposes their tools to the model as
`<server>_<tool>`, and routes those calls to the server. The loop runs until the model
answers without a tool call (immediately, when no tools are offered).

It runs as a uv script (deps: openai, mcp), so the chat + tool plumbing is just the
SDKs — the harness bootstraps `uv` in the runtime. Model calls go to the interception
server (OPENAI_BASE_URL/API_KEY); the bash tool runs locally in the runtime.
"""

import asyncio
import json
import os
import subprocess
import sys
from contextlib import AsyncExitStack

from openai import AsyncOpenAI

BASH_TOOL = {
    "type": "function",
    "function": {
        "name": "bash",
        "description": "Run a bash command and return its combined stdout and stderr.",
        "parameters": {
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "The bash command to run."}
            },
            "required": ["command"],
        },
    },
}

# base_url + api_key come from OPENAI_BASE_URL / OPENAI_API_KEY.
client = AsyncOpenAI()


def run_bash(command: str) -> str:
    try:
        result = subprocess.run(
            ["bash", "-c", command], capture_output=True, text=True, timeout=60 * 60
        )
        return result.stdout + result.stderr
    except Exception as e:
        return f"error: {e}"


async def chat(messages: list[dict], tools: list[dict]):
    completion = await client.chat.completions.create(
        model=os.environ["OPENAI_MODEL"], messages=messages, tools=tools or None
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


async def call_mcp(dispatch: dict, name: str, arguments: dict) -> str:
    session, raw = dispatch[name]
    result = await session.call_tool(raw, arguments)
    texts = [b.text for b in result.content if getattr(b, "type", None) == "text"]
    return "\n".join(texts) if texts else str(result.content)


async def main() -> None:
    config = json.loads(os.environ.get("MCP_CONFIG", "{}"))
    async with AsyncExitStack() as stack:
        mcp_tools, dispatch = (
            await connect_mcp(stack, config) if config.get("mcpServers") else ([], {})
        )
        enable_bash = os.environ.get("ENABLE_BASH", "0") == "1"
        tools = ([BASH_TOOL] if enable_bash else []) + mcp_tools
        system_prompt = os.environ.get("APPEND_SYSTEM_PROMPT", "")
        messages = (
            [{"role": "system", "content": system_prompt}] if system_prompt else []
        )
        # A Messages instruction (e.g. an image-bearing prompt) arrives pre-built as OpenAI
        # wire dicts; otherwise the single argv string is the first user message.
        initial = json.loads(os.environ.get("INITIAL_MESSAGES", "[]"))
        if initial:
            messages.extend(initial)
        else:
            messages.append({"role": "user", "content": sys.argv[1]})
        while True:
            message = await chat(messages, tools)
            messages.append(message.model_dump(exclude_none=True))
            if not message.tool_calls:
                break
            for call in message.tool_calls:
                name = call.function.name
                args = json.loads(call.function.arguments or "{}")
                if name in dispatch:
                    content = await call_mcp(dispatch, name, args)
                elif name == "bash":
                    content = await asyncio.to_thread(run_bash, args.get("command", ""))
                else:
                    content = f"error: unknown tool {name!r}"
                messages.append(
                    {"role": "tool", "tool_call_id": call.id, "content": content}
                )


if __name__ == "__main__":
    asyncio.run(main())
