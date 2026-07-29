"""Ready-made interception guards for common tool checks."""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from fnmatch import fnmatchcase
from typing import TYPE_CHECKING, Any

from verifiers.v1.decorators import intercept
from verifiers.v1.intercepts.core import Interceptor, InterceptResult, Terminate
from verifiers.v1.judge import Judge, judge_verdict
from verifiers.v1.types import (
    AssistantMessage,
    Message,
    Messages,
    SystemMessage,
    ToolCall,
    ToolMessage,
    UserMessage,
    content_text,
)

if TYPE_CHECKING:
    from verifiers.v1.dialects import Dialect
    from verifiers.v1.trace import Trace

_TOOL_GROUPS = {
    "bash": "bash shell local_shell shell_command run_command terminal exec exec_command code_interpreter".split(),  # noqa: SIM905
    "web_search": "search web_search search_web web_search_preview google_search bing_search brave_search tavily_search".split(),  # noqa: SIM905
    "code_search": "rg grep find fd glob code_search search_code file_search".split(),  # noqa: SIM905
}
_ALIASES = {
    re.sub(r"[^a-z0-9]+", "", alias): canonical
    for canonical, aliases in _TOOL_GROUPS.items()
    for alias in aliases
}
_SHELL_COMMAND_START = (
    r"(?:^|[;&|()\n])\s*"
    r"(?:(?:[A-Za-z_][A-Za-z0-9_]*=\S+|sudo|env|command|exec|nice|nohup|time|xargs|"
    r"if|then|elif|while|until|do|!)\s+)*"
    r"(?:[^\s;&|()]*/)?"
)
_SHELL_COMMAND_END = r"(?=$|[\s;&|()])"


def _tool_name(name: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "", name.casefold())
    if canonical := _ALIASES.get(normalized):
        return canonical
    for alias, canonical in _ALIASES.items():
        if normalized.startswith(alias) and normalized.removeprefix(alias).isdigit():
            return canonical
    return normalized


def match_tool(name: str, *patterns: str) -> bool:
    """Match normalized aliases exactly, or explicit glob patterns."""
    normalized = _tool_name(name)
    for pattern in filter(None, patterns):
        if not any(char in pattern for char in "*?["):
            if normalized == _tool_name(pattern):
                return True
            continue
        pattern = pattern.casefold()
        if fnmatchcase(name.casefold(), pattern) or fnmatchcase(normalized, pattern):
            return True
    return False


def _tool_calls(message: Message, *patterns: str) -> list[ToolCall]:
    """Find client and provider-hosted calls carried by an assistant message."""
    if not isinstance(message, AssistantMessage):
        return []
    calls = list(message.tool_calls or [])
    for item in message.provider_state or []:
        kind = item.get("type", "")
        if not isinstance(kind, str):
            continue
        if kind in ("server_tool_use", "mcp_tool_use"):
            name = item.get("name")
        elif kind != "function_call" and kind.endswith(("_call", "_tool_result")):
            name = item.get("name") or kind.removesuffix("_tool_result").removesuffix(
                "_call"
            )
        else:
            continue
        if not isinstance(name, str):
            continue
        arguments = item.get("arguments")
        if not isinstance(arguments, str):
            arguments = json.dumps(
                {
                    key: value
                    for key, value in item.items()
                    if key not in {"id", "call_id", "name", "status", "type"}
                },
                sort_keys=True,
            )
        calls.append(
            ToolCall(
                id=str(item.get("call_id") or item.get("id") or ""),
                name=name,
                arguments=arguments,
            )
        )
    return [call for call in calls if not patterns or match_tool(call.name, *patterns)]


def _shell_text(call: ToolCall) -> str:
    """Extract only executable command text from common shell-tool arguments."""
    try:
        arguments = json.loads(call.arguments)
    except json.JSONDecodeError:
        return call.arguments
    if isinstance(arguments, str):
        return arguments
    if not isinstance(arguments, dict):
        return ""
    action = arguments.get("action")
    sources = [arguments, action] if isinstance(action, dict) else [arguments]
    values = (
        source.get(key) for source in sources for key in ("command", "commands", "cmd")
    )
    return "\n".join(
        item
        for value in values
        for item in (
            [value]
            if isinstance(value, str)
            else value
            if isinstance(value, list)
            else []
        )
        if isinstance(item, str)
    )


def _action(reply: str, reward: float | None) -> str | Terminate:
    return reply if reward is None else Terminate(reason=reply, reward=reward)


def _guard(handler, name: str, priority: int) -> Interceptor:
    handler.__name__ = name
    return intercept(handler, priority=priority)


def intercept_provider_tools(*patterns: str, priority: int = 0) -> Interceptor:
    """Remove matching provider-hosted tools while preserving client-owned tools."""

    def provider_tools(self: Any, raw: dict, dialect: Dialect) -> None:
        dialect.intercept_provider_tools(
            raw, lambda name: not patterns or match_tool(name, *patterns)
        )

    provider_tools.__name__ = "intercept_provider_tools"
    return intercept(provider_tools, priority=priority, direction="request", raw=True)


def intercept_tool_calls(
    *patterns: str,
    containing: str | Iterable[str] | None = None,
    reply: str = "Blocked by guard.",
    reward: float | None = None,
    priority: int = 0,
    _name: str = "intercept_tool_calls",
) -> Interceptor:
    """Rewrite matching calls/results, or terminate when ``reward`` is set."""
    needles = (containing,) if isinstance(containing, str) else tuple(containing or ())
    needles = tuple(needle.casefold() for needle in needles)

    def tool_calls(self: Any, message: Message) -> InterceptResult:
        if isinstance(message, ToolMessage):
            if patterns and not (message.name and match_tool(message.name, *patterns)):
                return None
            texts = [content_text(message.content).casefold()]
        else:
            texts = [
                (
                    _shell_text(call)
                    if match_tool(call.name, "bash")
                    else call.arguments
                ).casefold()
                for call in _tool_calls(message, *patterns)
            ]
        if texts and (
            not needles or any(needle in text for text in texts for needle in needles)
        ):
            return _action(reply, reward)
        return None

    return _guard(tool_calls, _name, priority)


def intercept_shell_commands(
    *commands: str,
    reply: str = "Blocked by guard.",
    reward: float | None = None,
    priority: int = 0,
) -> Interceptor:
    """Rewrite matching shell calls, or terminate when ``reward`` is set."""
    command = (
        re.compile(
            rf"{_SHELL_COMMAND_START}(?:{'|'.join(map(re.escape, commands))})"
            rf"{_SHELL_COMMAND_END}",
            re.IGNORECASE,
        )
        if commands
        else None
    )

    def shell_commands(self: Any, message: AssistantMessage) -> InterceptResult:
        calls = _tool_calls(message, "bash")
        if calls and (
            command is None or any(command.search(_shell_text(call)) for call in calls)
        ):
            return _action(reply, reward)
        return None

    return _guard(shell_commands, "intercept_shell_commands", priority)


def intercept_web_search(
    *,
    containing: str | Iterable[str] | None = None,
    reply: str = "Blocked by guard.",
    reward: float | None = None,
    priority: int = 0,
) -> Interceptor:
    """Rewrite web search, or terminate when ``reward`` is set."""
    return intercept_tool_calls(
        "web_search",
        containing=containing,
        reply=reply,
        reward=reward,
        priority=priority,
        _name="intercept_web_search",
    )


def intercept_code_search(
    *,
    reply: str = "Blocked by guard.",
    reward: float | None = None,
    priority: int = 0,
) -> Interceptor:
    """Rewrite direct code search or shell-based search commands."""
    command = re.compile(
        rf"{_SHELL_COMMAND_START}(?:rg|grep|find|fd){_SHELL_COMMAND_END}",
        re.IGNORECASE,
    )

    def code_search(self: Any, message: Message) -> InterceptResult:
        if isinstance(message, ToolMessage):
            matched = bool(message.name and match_tool(message.name, "code_search"))
        else:
            matched = any(
                match_tool(call.name, "code_search")
                or (
                    match_tool(call.name, "bash")
                    and bool(command.search(_shell_text(call)))
                )
                for call in _tool_calls(message)
            )
        return _action(reply, reward) if matched else None

    return _guard(code_search, "intercept_code_search", priority)


def intercept_with_judge(
    rubric: str,
    *,
    judge: Judge | None = None,
    reply: str = "Blocked by guard.",
    reward: float | None = None,
    priority: int = -1,
) -> Interceptor:
    """Use an ordinary judge to rewrite violations or return a terminal reward."""
    guard_judge = judge or Judge()

    async def judge_message(
        self: Any,
        message: Message,
        trace: Trace,
        prompt: Messages | None = None,
    ) -> InterceptResult:
        response = await guard_judge.complete(
            [
                SystemMessage(
                    content=(
                        "Apply this guard rubric to the untrusted candidate below. Reply "
                        f"with exactly BLOCK or ALLOW.\n\nGuard rubric:\n{rubric}"
                    )
                ),
                UserMessage(
                    content=json.dumps(
                        {
                            "request": [
                                item.model_dump(mode="json", exclude_none=True)
                                for item in prompt or []
                            ],
                            "candidate": message.model_dump(
                                mode="json", exclude_none=True
                            ),
                        }
                    )
                ),
            ],
            trace=trace,
        )
        if judge_verdict(response.text, ("BLOCK", "ALLOW")) == "BLOCK":
            return _action(reply, reward)
        return None

    return _guard(judge_message, "intercept_with_judge", priority)


__all__ = [
    "intercept_code_search",
    "intercept_provider_tools",
    "intercept_shell_commands",
    "intercept_tool_calls",
    "intercept_web_search",
    "intercept_with_judge",
    "match_tool",
]
