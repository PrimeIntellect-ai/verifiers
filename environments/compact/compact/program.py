# /// script
# requires-python = ">=3.10"
# dependencies = ["openai", "mcp>=1.24.0,<2"]
# ///
"""The compacting harness's program: a context-REWRITE loop (so every turn branches).

Unlike the bash harness (which appends to a growing message list), this sends a FRESH
`[system, user]` every turn — the task on the first turn, then the model's own
carried-over notes (the task is never shown again, so the notes are the durable memory).
Because the prompt is rewritten rather than extended, every turn is its own branch (see
verifiers.v1.branching). This mirrors context-tools' `context_rewrite=True`.

Notes are saved through a harness-provided `summarize` tool — no tag parsing. Each turn
the model either calls ONE task tool (MCP, from the harness's MCP_CONFIG), sees its
result in-context, and must then save updated notes via `summarize`; or it calls
`summarize` directly. A plain-text reply (no tool call) finishes the run and is printed
as the answer. A disallowed call — more than one tool in a reply, or a task tool after a
result — ends the rollout immediately with no answer, so the violation itself becomes
training signal.

It runs as a uv script (deps: openai, mcp), so the chat + tool plumbing is just the
SDKs — the harness bootstraps `uv` in the runtime. Model calls go to the interception
server (OPENAI_BASE_URL/API_KEY); MCP servers are reached over streamable HTTP.
"""

import asyncio
import json
import os
import sys
from contextlib import AsyncExitStack

from openai import AsyncOpenAI

SYSTEM = (
    "You work in turns, and your context is wiped between turns: the next turn shows "
    "only this system message plus the notes you last saved with the `summarize` tool. "
    "The task is shown on the first turn only; tool results and your own replies do NOT "
    "carry over — your saved notes are your entire memory, so write them complete and "
    "self-contained. Each turn, either call ONE task tool (you will see its result "
    "immediately, then you must save updated notes with `summarize`), or call "
    "`summarize` directly. Calling more than one task tool in a turn ends the run as a "
    "failure."
)

SUMMARIZE = {
    "type": "function",
    "function": {
        "name": "summarize",
        "description": (
            "Save your complete, self-contained notes — the only thing you will see "
            "next turn."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "notes": {
                    "type": "string",
                    "description": "The full notes to carry into the next turn.",
                }
            },
            "required": ["notes"],
        },
    },
}

# base_url + api_key come from OPENAI_BASE_URL / OPENAI_API_KEY.
client = AsyncOpenAI()


async def chat(
    messages: list[dict], tools: list[dict], tool_choice: dict | None = None
):
    completion = await client.chat.completions.create(
        model=os.environ["OPENAI_MODEL"],
        messages=messages,
        tools=tools or None,
        tool_choice=tool_choice or "auto" if tools else None,
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
    task = sys.argv[1]
    config = json.loads(os.environ.get("MCP_CONFIG", "{}"))
    notes: str | None = None  # the durable memory carried across turns
    async with AsyncExitStack() as stack:
        tools, dispatch = (
            await connect_mcp(stack, config) if config.get("mcpServers") else ([], {})
        )
        toolset = [*tools, SUMMARIZE]
        while True:  # each turn is a fresh prompt — a new branch
            # The rewrite: the task on the first turn, then only the carried-over notes.
            prompt = f"Task: {task}" if notes is None else f"Notes:\n{notes}"
            messages = [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": prompt},
            ]
            message = await chat(messages, toolset)
            if not message.tool_calls:
                print(message.content or "")  # plain text finishes the run
                return
            if len(message.tool_calls) > 1:
                return  # more than one call in a reply ends the rollout
            call = message.tool_calls[0]
            args = json.loads(call.function.arguments or "{}")
            if call.function.name == "summarize":
                notes = args.get("notes") or notes
                continue
            result = (
                await call_mcp(dispatch, call.function.name, args)
                if call.function.name in dispatch
                else f"error: unknown tool {call.function.name!r}"
            )
            messages.append(message)
            messages.append(
                {"role": "tool", "tool_call_id": call.id, "content": result}
            )
            # After a tool result, `summarize` is forced via tool_choice — saving
            # notes is the only action left in the turn.
            message = await chat(
                messages,
                [SUMMARIZE],
                {"type": "function", "function": {"name": "summarize"}},
            )
            if not message.tool_calls:
                print(message.content or "")  # finishing right after a result is fine
                return
            call = message.tool_calls[0]
            if len(message.tool_calls) > 1 or call.function.name != "summarize":
                return  # a non-`summarize` call after the result ends the rollout
            args = json.loads(call.function.arguments or "{}")
            notes = args.get("notes") or notes


if __name__ == "__main__":
    asyncio.run(main())
