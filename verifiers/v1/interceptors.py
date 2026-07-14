"""Reusable message interceptors for common tool policies."""

import json
import re
from collections.abc import Awaitable, Callable, Iterable
from fnmatch import fnmatchcase

from verifiers.v1.decorators import intercept
from verifiers.v1.types import AssistantMessage, Message, ToolCall, ToolMessage

Interceptor = Callable[..., Awaitable[str | None]]
_SHELL_TOOLS = (
    "bash",
    "*shell",
    "shell_command",
    "exec",
    "exec_command",
    "execute_command",
    "run_command",
    "terminal",
    "*code_execution",
    "*code_interpreter*",
)
_CODE_SEARCH_TOOLS = (
    "grep",
    "glob",
    "*code*search*",
    "*search*code*",
    "*file*search*",
)
_CODE_SEARCH_COMMANDS = ("rg", "grep", "find", "fd")


def _matches(name: str, patterns: tuple[str, ...]) -> bool:
    return not patterns or any(
        fnmatchcase(name.casefold(), pattern) for pattern in patterns
    )


def find_tool_calls(message: Message, *patterns: str) -> list[ToolCall]:
    """Return ordinary and provider-native tool calls, optionally filtered by glob patterns.

    Provider-hosted calls such as Responses web search and Anthropic server tools are returned
    after they have run upstream; ordinary function/MCP calls are visible before the harness runs
    them. Matching is case-insensitive.
    """
    if not isinstance(message, AssistantMessage):
        return []
    calls = list(message.tool_calls or [])
    for item in message.provider_state or []:
        kind = item.get("type", "")
        if kind == "server_tool_use":
            name = item.get("name")
        elif kind != "function_call" and kind.endswith("_call"):
            name = item.get("name") or kind.removesuffix("_call")
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
    normalized = tuple(pattern.casefold() for pattern in patterns)
    return [call for call in calls if _matches(call.name, normalized)]


def block_tool_calls(
    *patterns: str,
    containing: str | Iterable[str] | None = None,
    reply: str = "Blocked by policy.",
    priority: int = 0,
) -> Interceptor:
    """Build an interceptor that blocks matching client or provider-native tools.

    Patterns are case-insensitive globs. When `containing` is set, a matching call is blocked only
    if the complete assistant response or tool result contains one of those strings.
    """
    names = tuple(pattern.casefold() for pattern in patterns)
    needles = (containing,) if isinstance(containing, str) else tuple(containing or ())
    needles = tuple(needle.casefold() for needle in needles)

    async def blocker(self, message: Message) -> str | None:
        if isinstance(message, ToolMessage):
            matched = not names or (
                message.name is not None and _matches(message.name, names)
            )
        else:
            matched = bool(find_tool_calls(message, *names))
        if not matched:
            return None
        haystack = message.model_dump_json().casefold()
        if needles and not any(needle in haystack for needle in needles):
            return None
        return reply

    return intercept(blocker, priority=priority)


def block_shell_commands(
    *commands: str,
    reply: str = "Blocked by policy.",
    priority: int = 0,
) -> Interceptor:
    """Build an interceptor that blocks shell calls containing the given command names.

    This catches common Bash, shell, terminal, Codex exec, and provider code-execution shapes.
    It is a convenience policy, not a sandbox boundary: equivalent or obfuscated commands may
    still require a custom classifier or runtime isolation.
    """
    command_patterns = tuple(
        re.compile(rf"(?<![\w.-]){re.escape(command)}(?![\w.-])", re.IGNORECASE)
        for command in commands
    )

    async def blocker(self, message: AssistantMessage) -> str | None:
        calls = find_tool_calls(message, *_SHELL_TOOLS)
        if calls and (
            not command_patterns
            or any(
                pattern.search(call.arguments)
                for call in calls
                for pattern in command_patterns
            )
        ):
            return reply
        return None

    return intercept(blocker, priority=priority)


def block_web_search(
    *,
    containing: str | Iterable[str] | None = None,
    reply: str = "Blocked by policy.",
    priority: int = 0,
) -> Interceptor:
    """Block client or provider-hosted web search, optionally by response content."""
    return block_tool_calls(
        "*web*search*", containing=containing, reply=reply, priority=priority
    )


def block_code_search(
    *, reply: str = "Blocked by policy.", priority: int = 0
) -> Interceptor:
    """Block common code-search tools and rg, grep, find, or fd shell calls."""
    tool_blocker = block_tool_calls(*_CODE_SEARCH_TOOLS, reply=reply)
    shell_blocker = block_shell_commands(*_CODE_SEARCH_COMMANDS, reply=reply)

    async def blocker(self, message: Message) -> str | None:
        return await tool_blocker(self, message) or await shell_blocker(self, message)

    return intercept(blocker, priority=priority)


__all__ = [
    "block_code_search",
    "block_shell_commands",
    "block_tool_calls",
    "block_web_search",
    "find_tool_calls",
]
