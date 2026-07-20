# /// script
# requires-python = ">=3.10"
# dependencies = ["openai", "mcp"]
# ///
"""The compacting harness's program: a context-REWRITE loop (so every turn branches).

Unlike the bash harness (which appends to a growing message list), this sends a FRESH
`[system, user]` every turn — the task on the first turn, then the model's own
carried-over `notes` (the task is never shown again, so the notes are the durable memory).
Because the prompt is rewritten rather than extended, every turn is its own branch (see
verifiers.v1.branching). This mirrors context-tools' `context_rewrite=True`.

To make progress with tools (the harness sets MCP_CONFIG, a standard `mcpServers` URL
map), the rewrite also carries the LAST tool call's output into the next turn's prompt —
kept for exactly one turn, so the model can actually read a result and act on it before
it's gone. Anything it needs longer it copies into <notes>; the raw output is dropped on
the next rewrite. So tools work, and the trajectory still branches every turn.

It runs as a uv script (deps: openai, mcp), so the chat + tool plumbing is just the
SDKs — the harness bootstraps `uv` in the runtime. Model calls go to the interception
server (OPENAI_BASE_URL/API_KEY); MCP servers are reached over streamable HTTP.
"""

import asyncio
import json
import os
import re
import sys
from contextlib import AsyncExitStack

from openai import AsyncOpenAI

SYSTEM = (
    "You solve a task across several turns; your NOTES are your only lasting memory. The "
    "first turn shows the task; after that you see only your notes — plus, the turn right "
    "after you call a tool, that tool's output (shown ONCE, so copy what you need into your "
    "notes before it's gone). Each turn, reply with brief reasoning, then your COMPLETE "
    "updated notes in <notes>...</notes>, and either call a tool to gather more or give the "
    "final answer in <answer>...</answer>."
)

# base_url + api_key come from OPENAI_BASE_URL / OPENAI_API_KEY.
client = AsyncOpenAI()


async def chat(messages: list[dict], tools: list[dict]):
    completion = await client.chat.completions.create(
        model=os.environ["OPENAI_MODEL"], messages=messages, tools=tools or None
    )
    return completion.choices[0].message


def extract(tag: str, text: str) -> str | None:
    match = re.search(rf"<{tag}>(.*?)</{tag}>", text, re.DOTALL)
    return match.group(1).strip() if match else None


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
    task = sys.stdin.read()
    config = json.loads(os.environ.get("MCP_CONFIG", "{}"))
    notes: str | None = None  # the durable memory carried across turns
    tool_output: str | None = None  # the last tool result, kept for exactly one turn
    async with AsyncExitStack() as stack:
        tools, dispatch = (
            await connect_mcp(stack, config) if config.get("mcpServers") else ([], {})
        )
        while True:  # each turn is a fresh prompt — a new branch
            # The rewrite: the task on the first turn, then only the carried-over notes;
            # plus the last tool output, kept one turn so the model can actually use it.
            parts = [f"Task: {task}" if notes is None else f"Notes:\n{notes}"]
            if tool_output is not None:
                parts.append(f"Latest tool output:\n{tool_output}")
            messages = [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": "\n\n".join(parts)},
            ]
            message = await chat(messages, tools)
            notes = extract("notes", message.content or "") or notes
            answer = extract("answer", message.content or "")
            if answer is not None:
                print(answer)
                return
            # Carry only the latest tool output to the next turn (dropped after that).
            tool_output = None
            if message.tool_calls:
                results = []
                for call in message.tool_calls:
                    args = json.loads(call.function.arguments or "{}")
                    result = (
                        await call_mcp(dispatch, call.function.name, args)
                        if call.function.name in dispatch
                        else f"error: unknown tool {call.function.name!r}"
                    )
                    results.append(f"{call.function.name}:\n{result}")
                tool_output = "\n\n".join(results)


if __name__ == "__main__":
    asyncio.run(main())
