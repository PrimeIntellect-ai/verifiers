"""A judge-based check for `@vf.intercept` handlers: an LLM judge allows or blocks each
tool call; the handler decides what a rejection means."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from verifiers.v1.judge import Judge, JudgeConfig, JudgeResponse, judge_verdict
from verifiers.v1.types import ID

if TYPE_CHECKING:
    from verifiers.v1.intercept import InterceptExchange


class ToolCallJudgeConfig(JudgeConfig):
    id: ID = "tool_call"
    """Pinned to the built-in, so a code-level default entry needs no explicit id."""
    prompt_file: Path | None = Path(__file__).resolve().parent / "gate.txt"
    """The bundled allow/block policy prompt; point at another file to reword the policy
    (the template fills `{question}`, `{tool}`, and `{arguments}`)."""


class ToolCallJudge(Judge[bool, ToolCallJudgeConfig]):
    """Decides ALLOWED/BLOCKED for one tool call; `parse` maps the verdict to a bool
    (True = allowed). A verdict matching neither raises, like the scoring judges — a
    judge failure must not silently block (or allow) the call."""

    def parse(self, response: JudgeResponse[bool]) -> bool:
        return judge_verdict(response.text, ("ALLOWED", "BLOCKED")) == "ALLOWED"


async def judge_tools(exchange: InterceptExchange, judge: Judge | None = None) -> bool:
    """True when every tool call in the response is allowed by `judge` (default
    `ToolCallJudge()`); False on the first rejected call. Request side or no tool calls
    means there is nothing to judge: True. Pass any `Judge` subclass instance to gate on
    a custom prompt/policy."""
    if exchange.direction != "response":
        return True
    message = exchange.message
    calls = (message.tool_calls if message else None) or []
    if not calls:
        return True
    gate = judge or ToolCallJudge()
    task = getattr(getattr(exchange.trace, "task", None), "data", None)
    question = getattr(task, "prompt_text", "") or ""
    for call in calls:
        result = await gate.evaluate(
            trace=exchange.trace,
            question=question,
            tool=call.name,
            arguments=call.arguments,
        )
        if not result.parsed:
            return False
    return True


__all__ = ["ToolCallJudge", "ToolCallJudgeConfig", "judge_tools"]
