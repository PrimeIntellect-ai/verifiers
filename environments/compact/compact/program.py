# /// script
# requires-python = ">=3.10"
# dependencies = ["openai", "mcp==2.0.0", "httpx2", "tenacity"]
# ///
"""The compacting harness's program: a context-rewrite loop (every turn branches).

Each turn sends a fresh `[system, user]` — the task on the first turn, then only the
notes the model saved via the `summarize` tool. Per turn: at most one task tool call,
its result shown in-context, then a forced `summarize`; a plain-text reply finishes the
run and is printed as the answer; a disallowed call ends the rollout with no answer
(training signal). Model calls go to the interception server (OPENAI_BASE_URL/API_KEY);
MCP clients use the task's declared transport.
"""

import asyncio
import json
import os
import sys
from contextlib import AsyncExitStack
from typing import TYPE_CHECKING

from openai import AsyncOpenAI

if TYPE_CHECKING:
    from verifiers.v1.harnesses.utils.mcp import call_mcp, connect_mcp  # noqa: TC004

SYSTEM = (
    "You work in turns, and your context is wiped between turns: the next turn shows "
    "only this system message plus the notes you last saved with the `summarize` tool. "
    "The task is shown on the first turn only; tool results and your own replies do NOT "
    "carry over — your saved notes are your entire memory, so write them complete and "
    "self-contained. Each turn, either call ONE task tool (you will see its result "
    "immediately, then you must save updated notes with `summarize`), call `summarize` "
    "directly, or reply with plain text and no tool call to finish the run and deliver "
    "your final answer. Calling more than one task tool in a turn ends the run as a "
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


async def main() -> None:
    task = sys.argv[1]
    config = json.loads(os.environ.get("MCP_CONFIG", "{}"))
    notes: str | None = None  # the durable memory carried across turns
    async with AsyncExitStack() as stack:
        tools, dispatch, servers = await connect_mcp(config, stack, {"summarize"})
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
                await call_mcp(servers, dispatch, call.function.name, args)
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
