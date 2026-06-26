# /// script
# requires-python = ">=3.10"
# dependencies = ["openai", "mcp", "httpx"]
# ///
"""The bash harness's program: a chat loop with a local `bash` tool (+ optional MCP tools).

A growing-message-list chat loop. It always offers a local `bash` tool that runs shell commands
in the runtime; when the harness sets MCP_CONFIG (a standard `mcpServers` URL map) it also
connects to those servers over streamable HTTP, exposes their tools to the model as
`<server>_<tool>`, and routes those calls to the server. The loop runs until the model answers
without a tool call — unless the task has a user simulator (`--user-url`), in which case a
no-tool-call turn is a turn boundary: the program POSTs the assistant's text to `/user`, appends
the simulated user reply, and re-prompts, until the simulator signals `done` (a no-prompt task is
opened the same way). Carrying every turn in the program's own conversation keeps the recorded
message graph linear.

It runs as a uv script (deps: openai, mcp, httpx), so the chat + tool plumbing is just the SDKs —
the harness bootstraps `uv` in the runtime. The interception endpoint, per-rollout secret, and
model arrive as argv (not env), so the bash tool's local subprocesses never inherit them.
"""

import argparse
import asyncio
import json
import os
import subprocess
from contextlib import AsyncExitStack

import httpx
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


def run_bash(command: str) -> str:
    try:
        result = subprocess.run(
            ["bash", "-c", command], capture_output=True, text=True, timeout=3600
        )
        return result.stdout + result.stderr
    except Exception as e:
        return f"error: {e}"


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


async def next_user_turn(
    http: httpx.AsyncClient, url: str, secret: str, message: str
) -> tuple[list[dict], bool]:
    """Drive the task's user simulator: POST the model's last text to `/user` and return its next
    user message(s) (already OpenAI wire dicts) and whether the trajectory should now end."""
    resp = await http.post(
        url, json={"message": message}, headers={"Authorization": f"Bearer {secret}"}
    )
    resp.raise_for_status()
    data = resp.json()
    return data["messages"], data["done"]


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
        mcp_tools, dispatch = (
            await connect_mcp(stack, config) if config.get("mcpServers") else ([], {})
        )
        tools = [BASH_TOOL] + mcp_tools
        http = (
            await stack.enter_async_context(httpx.AsyncClient(timeout=120))
            if user_url
            else None
        )
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
        elif user_url:
            opening, done = await next_user_turn(http, user_url, args.api_key, "")
            messages.extend(opening)
            if done:
                return
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
                    elif name == "bash":
                        content = await asyncio.to_thread(
                            run_bash, tool_args.get("command", "")
                        )
                    else:
                        content = f"error: unknown tool {name!r}"
                    messages.append(
                        {"role": "tool", "tool_call_id": call.id, "content": content}
                    )
                continue
            # No tool call: a final answer for the current user turn. With a user simulator, fetch
            # the next user turn and re-prompt; otherwise the conversation is over.
            if user_url:
                reply, done = await next_user_turn(
                    http, user_url, args.api_key, message.content or ""
                )
                messages.extend(reply)
                if done:
                    break
                continue
            break


if __name__ == "__main__":
    asyncio.run(main())
