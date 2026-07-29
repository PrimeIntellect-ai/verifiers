"""Ready-made interception guards for common tool checks."""

from __future__ import annotations

import json
import re
import shlex
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
    "bash": "bash shell local_shell shell_command run_command terminal exec exec_command code_interpreter code_execution".split(),  # noqa: SIM905
    "web_search": "search web_search search_web web_search_preview google_search bing_search brave_search tavily_search".split(),  # noqa: SIM905
    "code_search": "rg grep find fd glob code_search search_code file_search".split(),  # noqa: SIM905
}
_ALIASES = {
    re.sub(r"[^a-z0-9]+", "", alias): canonical
    for canonical, aliases in _TOOL_GROUPS.items()
    for alias in aliases
}
_SHELL_WRAPPER_OPTIONS = {
    "sudo": {"-u", "--user", "-g", "--group", "-h", "--host", "-p", "--prompt"},
    "env": {"-u", "--unset", "-C", "--chdir", "-S", "--split-string"},
    "nice": {"-n", "--adjustment"},
    "time": {"-f", "--format", "-o", "--output"},
    "xargs": {
        "-a",
        "--arg-file",
        "-E",
        "--eof",
        "-I",
        "--replace",
        "-J",
        "-L",
        "--max-lines",
        "-n",
        "--max-args",
        "-P",
        "--max-procs",
        "-R",
        "-S",
        "-s",
        "--max-chars",
        "-d",
        "--delimiter",
    },
    "command": set(),
    "exec": set(),
    "nohup": set(),
    "if": set(),
    "then": set(),
    "elif": set(),
    "else": set(),
    "while": set(),
    "until": set(),
    "do": set(),
    "!": set(),
}
_SHELL_INTERPRETERS = {"sh", "bash", "dash", "ksh", "zsh"}
_SHELL_INTERPRETER_OPTIONS = {"-o", "+o", "-O", "+O", "--rcfile", "--init-file"}


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


def _shell_text(
    call: ToolCall, *, commands_only: bool = False, raw_fallback: bool = False
) -> str:
    """Extract shell text, optionally reduced to normalized invocation lines."""
    try:
        arguments = json.loads(call.arguments)
    except json.JSONDecodeError:
        return call.arguments
    if isinstance(arguments, str):
        text = arguments
    elif isinstance(arguments, dict):
        action = arguments.get("action")
        sources = [arguments, action] if isinstance(action, dict) else [arguments]
        has_command = any(
            key in source
            for source in sources
            for key in ("command", "commands", "cmd")
        )
        values = (
            source.get(key)
            for source in sources
            for key in ("command", "commands", "cmd")
        )
        text = "\n".join(
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
        if not has_command and raw_fallback:
            return call.arguments
    else:
        text = ""
    if not commands_only or not text:
        return text

    # Tokenize command fields only, then unwrap launchers and nested shell payloads.
    pending = [text]
    invocations = []
    while pending:
        lexer = shlex.shlex(
            pending.pop().replace("\\\n", "").replace("\n", ";"),
            posix=True,
            punctuation_chars="{};&|()<>",
        )
        lexer.whitespace_split = True
        lexer.commenters = ""
        try:
            tokens = list(lexer)
        except ValueError:
            continue
        segments = [[]]
        for token in tokens:
            if token and set(token) <= set("{};&|()"):
                segments.append([])
            else:
                segments[-1].append(token)

        for segment in segments:
            index = 0
            while index < len(segment):
                token = segment[index]
                redirect = token and set(token) <= set("<>&")
                descriptor = (
                    token.isdigit()
                    and index + 1 < len(segment)
                    and set(segment[index + 1]) <= set("<>&")
                )
                if redirect:
                    index += 2
                    continue
                if descriptor:
                    index += 3
                    continue
                if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*=.*", token):
                    index += 1
                    continue

                name = token.rsplit("/", 1)[-1]
                line = " ".join(segment[index:])
                invocations.append(line)
                if name != token:
                    invocations.append(" ".join([name, *segment[index + 1 :]]))
                index += 1

                if name in _SHELL_INTERPRETERS:
                    while index < len(segment) and segment[index].startswith(
                        ("-", "+")
                    ):
                        option = segment[index]
                        option_name, separator, option_value = option.partition("=")
                        command_option = option_name == "--command" or (
                            option_name.startswith("-")
                            and not option.startswith("--")
                            and "c" in option_name[1:]
                        )
                        if command_option:
                            if separator:
                                pending.append(option_value)
                            elif index + 1 < len(segment):
                                pending.append(segment[index + 1])
                            break
                        index += (
                            2
                            if option_name in _SHELL_INTERPRETER_OPTIONS
                            and not separator
                            else 1
                        )
                    break

                if name not in _SHELL_WRAPPER_OPTIONS:
                    break
                option_arguments = _SHELL_WRAPPER_OPTIONS[name]
                while index < len(segment) and segment[index].startswith("-"):
                    option = segment[index]
                    option_name, separator, option_value = option.partition("=")
                    if name == "env" and option_name in ("-S", "--split-string"):
                        if separator:
                            pending.append(option_value)
                        elif index + 1 < len(segment):
                            pending.append(segment[index + 1])
                    index += (
                        2 if option_name in option_arguments and not separator else 1
                    )

    return "\n".join(invocations)


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
                    _shell_text(call, raw_fallback=True)
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
            rf"^(?:{'|'.join(map(re.escape, commands))})(?=$|\s)",
            re.IGNORECASE | re.MULTILINE,
        )
        if commands
        else None
    )

    def shell_commands(self: Any, message: AssistantMessage) -> InterceptResult:
        calls = _tool_calls(message, "bash")
        if calls and (
            command is None
            or any(
                command.search(_shell_text(call, commands_only=True)) for call in calls
            )
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
    command = re.compile(r"^(?:rg|grep|find|fd)(?=$|\s)", re.IGNORECASE | re.MULTILINE)

    def code_search(self: Any, message: Message) -> InterceptResult:
        if isinstance(message, ToolMessage):
            matched = bool(message.name and match_tool(message.name, "code_search"))
        else:
            matched = any(
                match_tool(call.name, "code_search")
                or (
                    match_tool(call.name, "bash")
                    and bool(command.search(_shell_text(call, commands_only=True)))
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
