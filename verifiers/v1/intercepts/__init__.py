"""Ready-made checks for `@vf.intercept` handlers, plus fuzzy tool-name matching.

The companions to `verifiers.v1.intercept`: each takes the `exchange` and returns a
verdict — `match_tool_calls` matches tool calls by pattern (`match_tool`),
`judge_tools` vets each tool call with an LLM judge (`ToolCallJudge`), and
`strip_provider_tools` strips provider-side tools in place. The handler decides what a
verdict means:

    @vf.intercept
    async def no_git(self, exchange):
        if match_tool_calls(exchange, "git"):
            return vf.UserMessage(content="git is not allowed")
"""

from verifiers.v1.intercepts.gate import ToolCallJudge, ToolCallJudgeConfig, judge_tools
from verifiers.v1.intercepts.match import TOOL_SYNONYMS, match_tool, normalize_tool_name
from verifiers.v1.intercepts.tools import match_tool_calls, strip_provider_tools

__all__ = [
    "TOOL_SYNONYMS",
    "normalize_tool_name",
    "match_tool",
    "match_tool_calls",
    "strip_provider_tools",
    "ToolCallJudge",
    "ToolCallJudgeConfig",
    "judge_tools",
]
