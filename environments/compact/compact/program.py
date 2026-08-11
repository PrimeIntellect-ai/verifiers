# /// script
# requires-python = ">=3.10"
# dependencies = ["openai", "mcp>=1.24.0,<2"]
# ///
"""The compacting harness's program: a context-REWRITE loop (so every turn branches).

Unlike the bash harness (which appends to a growing message list), this sends a FRESH
`[system, user]` every turn — the task on the first turn, then the model's own
carried-over `notes` (the task is never shown again, so the notes are the durable memory).
Because the prompt is rewritten rather than extended, every turn is its own branch (see
verifiers.v1.branching). This mirrors context-tools' `context_rewrite=True`.

Tools (the harness sets MCP_CONFIG, a standard `mcpServers` URL map) follow a strict
one-call-per-turn protocol: the model may make at most ONE tool call, sees its result
in-context, and must then reply without tools — reasoning plus updated <notes>. Only the
notes flow into the next turn's prompt; the tool result never does, so anything worth
keeping must be written into the notes. A disallowed tool call — more than one in a
reply, or any call after the result — ends the rollout immediately with no answer, so
the violation itself becomes training signal.

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
    "first turn shows the task; after that you see only your notes. Each turn you may "
    "call at most ONE tool; you will see its result immediately, and must then reply "
    "WITHOUT calling tools — brief reasoning, then your COMPLETE updated notes in "
    "<notes>...</notes>. Only the notes carry to the next turn (the tool result does "
    "not), so copy what you need into them. Any disallowed tool call ends the run. When "
    "you know the final answer, give it in <answer>...</answer>."
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
    task = sys.argv[1]
    config = json.loads(os.environ.get("MCP_CONFIG", "{}"))
    notes: str | None = None  # the durable memory carried across turns
    async with AsyncExitStack() as stack:
        tools, dispatch = (
            await connect_mcp(stack, config) if config.get("mcpServers") else ([], {})
        )
        while True:  # each turn is a fresh prompt — a new branch
            # The rewrite: the task on the first turn, then only the carried-over notes.
            prompt = f"Task: {task}" if notes is None else f"Notes:\n{notes}"
            messages = [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": prompt},
            ]
            message = await chat(messages, tools)
            notes = extract("notes", message.content or "") or notes
            if message.tool_calls:
                if len(message.tool_calls) > 1:
                    return  # more than one call in a reply ends the rollout
                call = message.tool_calls[0]
                args = json.loads(call.function.arguments or "{}")
                result = (
                    await call_mcp(dispatch, call.function.name, args)
                    if call.function.name in dispatch
                    else f"error: unknown tool {call.function.name!r}"
                )
                messages.append(message)
                messages.append(
                    {"role": "tool", "tool_call_id": call.id, "content": result}
                )
                # The summary reply: tools stay advertised so a violation is a parsed
                # tool call we can detect, not junk text in the notes.
                message = await chat(messages, tools)
                if message.tool_calls:
                    return  # a call after the result ends the rollout
                notes = extract("notes", message.content or "") or notes
            answer = extract("answer", message.content or "")
            if answer is not None:
                print(answer)
                return


if __name__ == "__main__":
    asyncio.run(main())
