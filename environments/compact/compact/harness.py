"""The compacting harness: an in-process context-rewrite loop (every turn branches).

Each turn sends a fresh `[system, user]` — the task on the first turn, then only the
notes the model saved via the `summarize` tool. Runs in-process on the host event loop,
spawning no container or sandbox.
"""

import json
from contextlib import AsyncExitStack
from typing import Any

from openai import APIError, AsyncOpenAI

from verifiers.v1.clients import ModelContext
from verifiers.v1.configs.harness import HarnessConfig
from verifiers.v1.harness import Harness
from verifiers.v1.harnesses.utils.mcp import call_mcp, connect_mcp
from verifiers.v1.runtimes import ProgramResult, Runtime
from verifiers.v1.task import TaskData
from verifiers.v1.trace import Trace

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

SUMMARIZE: dict[str, Any] = {
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


class CompactingHarnessConfig(HarnessConfig):
    """A context-rewrite harness: rebuilds its prompt from carried-over notes each
    compaction instead of appending, so the trajectory branches at every compaction."""


class CompactingHarness(Harness[CompactingHarnessConfig]):
    SUPPORTS_MCP = True
    EXECUTES_CODE = False
    NEEDS_CONTAINER = False
    RUNS_ON_HOST = True

    async def launch(
        self,
        ctx: ModelContext,
        trace: Trace,
        runtime: Runtime,
        endpoint: str,
        secret: str,
        mcp_urls: dict[str, str],
        data: TaskData,
    ) -> ProgramResult:
        _, prompt = self.resolve_text_prompt(data)
        if prompt is None:
            raise ValueError("Compacting harness requires a string task prompt")
        mcp_cfg = (
            {
                "mcpServers": {
                    name: {"url": url, "timeout": self.config.tool_timeout}
                    for name, url in mcp_urls.items()
                }
            }
            if mcp_urls
            else {}
        )
        notes: str | None = None
        async with (
            AsyncOpenAI(base_url=endpoint, api_key=secret, max_retries=0) as client,
            AsyncExitStack() as stack,
        ):
            tools, dispatch, servers = (
                await connect_mcp(mcp_cfg, stack, {"summarize"})
                if mcp_cfg.get("mcpServers")
                else ([], {}, {})
            )
            toolset = [*tools, SUMMARIZE]
            while True:
                turn_prompt = f"Task: {prompt}" if notes is None else f"Notes:\n{notes}"
                messages: list[dict[str, Any]] = [
                    {"role": "system", "content": SYSTEM},
                    {"role": "user", "content": turn_prompt},
                ]
                try:
                    completion = await client.chat.completions.create(
                        model=ctx.model, messages=messages, tools=toolset
                    )
                except APIError as e:
                    return ProgramResult(exit_code=1, stdout="", stderr=str(e))
                message = completion.choices[0].message
                if not message.tool_calls:
                    return ProgramResult(
                        exit_code=0, stdout=message.content or "", stderr=""
                    )
                if len(message.tool_calls) > 1:
                    return ProgramResult(exit_code=0, stdout="", stderr="")
                call = message.tool_calls[0]
                args = json.loads(call.function.arguments or "{}")
                if call.function.name == "summarize":
                    notes = args.get("notes") or notes
                    continue
                if call.function.name in dispatch and isinstance(args, dict):
                    raw = await call_mcp(servers, dispatch, call.function.name, args)
                    result = raw if isinstance(raw, str) else json.dumps(raw)
                else:
                    result = f"error: unknown tool {call.function.name!r}"
                messages.append(message.model_dump(exclude_none=True))
                messages.append(
                    {"role": "tool", "tool_call_id": call.id, "content": result}
                )
                try:
                    completion = await client.chat.completions.create(
                        model=ctx.model,
                        messages=messages,
                        tools=[SUMMARIZE],
                        tool_choice={
                            "type": "function",
                            "function": {"name": "summarize"},
                        },
                    )
                except APIError as e:
                    return ProgramResult(exit_code=1, stdout="", stderr=str(e))
                message = completion.choices[0].message
                if not message.tool_calls:
                    return ProgramResult(
                        exit_code=0, stdout=message.content or "", stderr=""
                    )
                call = message.tool_calls[0]
                if len(message.tool_calls) > 1 or call.function.name != "summarize":
                    return ProgramResult(exit_code=0, stdout="", stderr="")
                args = json.loads(call.function.arguments or "{}")
                notes = args.get("notes") or notes
