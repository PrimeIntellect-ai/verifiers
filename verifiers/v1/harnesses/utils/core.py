"""Chat loop, local tools, and interception hook for bundled chat programs."""

import argparse
import asyncio
import json
import re
import shlex
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

import httpx
from openai import APIStatusError, AsyncOpenAI

if TYPE_CHECKING:
    # The harness bundles this module into the generated script before execution.
    from verifiers.v1.harnesses.utils.compaction import (  # noqa: TC004
        CompactionFailed,
        Compactor,
        bound_tool_message,
        compactable,
        discover_threshold,
        estimated_tokens,
        is_context_overflow,
    )
    from verifiers.v1.harnesses.utils.mcp import call_mcp, connect_mcp  # noqa: TC004

SERPER_URL = "https://google.serper.dev/search"

BASH_TOOL = {
    "type": "function",
    "function": {
        "name": "bash",
        "description": "Run a bash command and return its combined stdout and stderr.",
        "parameters": {
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "The bash command to run."}
            },
            "required": ["command"],
        },
    },
}

EDIT_TOOL = {
    "type": "function",
    "function": {
        "name": "edit",
        "description": (
            "Replace a unique string in a file. old_str must appear exactly once in the file."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "File path (relative to cwd or absolute).",
                },
                "old_str": {
                    "type": "string",
                    "description": "Exact string to find (must appear exactly once).",
                },
                "new_str": {"type": "string", "description": "Replacement string."},
            },
            "required": ["path", "old_str", "new_str"],
        },
    },
}

SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "search",
        "description": (
            "Run a web search via Serper (Google) and return the top organic results as title, "
            "URL, and snippet. Issue focused queries and call it several times to cover different "
            "angles; use the bash tool (e.g. curl) to read a result page in full."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The search query."},
                "num_results": {
                    "type": "integer",
                    "description": "Number of results to return (default 5).",
                },
            },
            "required": ["query"],
        },
    },
}


def format_results(results, query: str) -> str:
    """Format Serper organic results as title/URL/snippet blocks."""
    sections = []
    for i, result in enumerate(results, 1):
        title = (result.get("title") or "").strip() or "Untitled"
        lines = [f"Result {i}: {title}"]
        link = (result.get("link") or "").strip()
        if link:
            lines.append(f"URL: {link}")
        snippet = (result.get("snippet") or "").strip()
        if snippet:
            lines.append(f"  - {snippet}")
        sections.append("\n".join(lines))
    if not sections:
        return f"No results returned for query: {query}"
    return "\n\n---\n\n".join(sections)


def run_search(query: str, api_key: str, num_results: int = 5) -> str:
    """Serper Google web search -> formatted organic results.

    The key arrives as an argument (handed in by the harness over argv, like the interception
    secret) instead of from `$SERPER_API_KEY`, so the agent's `bash` subprocesses never inherit it.
    The whole call is wrapped so a bad query or malformed payload becomes a tool error rather than
    raising out of the chat loop and killing the rollout."""
    if not api_key:
        return "Error: no Serper API key (SERPER_API_KEY was not set in the eval environment)"
    # num_results comes straight from model tool JSON, so it may be a non-int (e.g. "ten"); coerce
    # defensively — `organic[:num_results]` would otherwise raise on a bad slice.
    try:
        num_results = max(1, int(num_results))
    except (TypeError, ValueError):
        num_results = 5
    try:
        response = httpx.post(
            SERPER_URL,
            json={"q": query},
            headers={"X-API-KEY": api_key, "Content-Type": "application/json"},
            timeout=45,
        )
        response.raise_for_status()
        organic = response.json().get("organic") or []
        return format_results(organic[:num_results], query)
    except Exception as e:  # noqa: BLE001 - tool failures are returned to the model
        return f"search failed ({e}). Try again or rephrase the query."


# Ports nano-rlm's git history guard: ordinary git commands stay available, but `git log`
# invocations that read beyond current-branch history (`--all`, `--branches`, ...) are
# refused. The refusal string is reused verbatim so agent-visible logs stay consistent
# across the harnesses.
GIT_REFUSAL = (
    "Git history option '{cmd}' is not allowed. Use current-branch history only."
)
_BASH_SEPARATORS = re.compile(r"&&|\|\||;|\|")
_RESTRICTED_LOG_OPTIONS = {
    "--all",
    "-all",
    "--alternate-refs",
    "--reflog",
    "--walk-reflogs",
    "-g",
}
_RESTRICTED_LOG_OPTION_PREFIXES = ("--branches", "--glob", "--remotes", "--tags")
_GIT_GLOBAL_OPTIONS_WITH_VALUE = {
    "-C",
    "-c",
    "--config-env",
    "--exec-path",
    "--git-dir",
    "--namespace",
    "--work-tree",
}


def _split_segment(segment: str) -> list:
    try:
        return shlex.split(segment)
    except ValueError:
        return segment.strip().split()


def _is_git_binary(token: str) -> bool:
    return token == "git" or token.rsplit("/", 1)[-1] == "git"


def _skip_git_global_options(argv: list, index: int) -> int:
    while index < len(argv):
        token = argv[index]
        if token == "--":
            return index + 1
        if not token.startswith("-"):
            return index
        option = token.split("=", 1)[0]
        if option in _GIT_GLOBAL_OPTIONS_WITH_VALUE and "=" not in token:
            index += 2
        else:
            index += 1
    return index


def _is_restricted_log_option(token: str) -> bool:
    if token in _RESTRICTED_LOG_OPTIONS:
        return True
    return any(
        token == option or token.startswith(f"{option}=")
        for option in _RESTRICTED_LOG_OPTION_PREFIXES
    )


def find_blocked_git_log_option(argv: list) -> "str | None":
    if not argv or not _is_git_binary(argv[0]):
        return None
    subcommand_index = _skip_git_global_options(argv, 1)
    if subcommand_index >= len(argv) or argv[subcommand_index] != "log":
        return None
    for token in argv[subcommand_index + 1 :]:
        if token == "--":
            return None
        if _is_restricted_log_option(token):
            return token
    return None


def find_blocked_command(command: str) -> "str | None":
    for segment in _BASH_SEPARATORS.split(command):
        blocked = find_blocked_git_log_option(_split_segment(segment))
        if blocked is not None:
            return blocked
    return None


def run_bash(command: str, allow_git: bool = True) -> str:
    if not allow_git:
        blocked = find_blocked_command(command)
        if blocked is not None:
            return GIT_REFUSAL.format(cmd=blocked)
    try:
        result = subprocess.run(
            ["bash", "-c", command],
            capture_output=True,
            text=True,
            timeout=3600,
            check=False,
        )
        return result.stdout + result.stderr
    except Exception as e:  # noqa: BLE001 - tool failures are returned to the model
        return f"error: {e}"


def run_edit(path: str, old_str: str, new_str: str) -> str:
    if not isinstance(path, str) or not path:
        return "error: 'path' is required"
    if not isinstance(old_str, str) or not isinstance(new_str, str):
        return "error: 'old_str' and 'new_str' must be strings"
    if not old_str:
        # '' matches everywhere (''.count('') == 1 on an empty file), so it would insert
        # rather than replace — reject it to keep the "exactly once" contract honest.
        return "error: 'old_str' must be a non-empty string"
    filepath = Path(path)
    if not filepath.is_absolute():
        filepath = Path.cwd() / filepath
    if not filepath.exists():
        return f"error: {path} not found"
    # Reading/writing can fail on a directory, permissions, or non-text content; return the
    # error as a tool result instead of letting it abort the chat loop.
    try:
        content = filepath.read_text()
    except Exception as e:  # noqa: BLE001 - tool failures are returned to the model
        return f"error: could not read {path}: {e}"
    count = content.count(old_str)
    if count != 1:
        return f"error: old_str must appear exactly once in {path} (found {count})"
    try:
        filepath.write_text(content.replace(old_str, new_str, 1))
    except Exception as e:  # noqa: BLE001 - tool failures are returned to the model
        return f"error: could not write {path}: {e}"
    return f"Edited {path}"


async def chat(
    client: AsyncOpenAI,
    model: str,
    messages: list[dict],
    tools: list[dict],
    *,
    tool_choice: str | None = None,
):
    kwargs = {"model": model, "messages": messages, "tools": tools or None}
    if tools and tool_choice is not None:
        kwargs["tool_choice"] = tool_choice
    return await client.chat.completions.create(**kwargs)


async def run_tool_hook(
    client: httpx.AsyncClient,
    url: str,
    api_key: str,
    phase: str,
    message: dict,
) -> dict:
    response = await client.post(
        url,
        headers={"Authorization": f"Bearer {api_key}"},
        json={"phase": phase, "message": message},
    )
    response.raise_for_status()
    decision = response.json()
    if decision["action"] == "stop":
        raise RuntimeError(decision["reason"])
    return decision


async def run_chat_loop(
    args: argparse.Namespace,
    compactor: "Compactor",
    messages: list[dict],
    dispatch: dict,
    servers: dict,
    tool_client: httpx.AsyncClient | None,
) -> None:
    """Run the tool-calling conversation until a text-only reply or context exhaustion."""
    while True:
        try:
            completion, messages = await compactor.complete(messages)
        except CompactionFailed:
            # The context is exhausted and could not be summarized: end the run
            # cleanly with what the conversation holds - still a trainable sample.
            return
        except APIStatusError as error:
            # Null cannot compact, so context exhaustion ends it with the transcript so far.
            if args.bash or not is_context_overflow(error):
                raise
            return
        message = completion.choices[0].message
        messages.append(message.model_dump(exclude_none=True))
        if not message.tool_calls:
            return
        tool_result_tokens = 0
        for call in message.tool_calls:
            name = call.function.name
            tool_message = {
                "role": "tool",
                "tool_call_id": call.id,
                "content": "",
                "name": name,
            }
            if args.tool_interception_url:
                assert tool_client is not None
                decision = await run_tool_hook(
                    tool_client,
                    args.tool_interception_url,
                    args.api_key,
                    "before",
                    tool_message,
                )
                if decision["action"] == "rewrite":
                    rewritten = bound_tool_message(decision["message"])
                    messages.append(rewritten)
                    tool_result_tokens += estimated_tokens(
                        str(rewritten.get("content", ""))
                    )
                    continue
            try:
                tool_args = json.loads(call.function.arguments or "{}")
            except json.JSONDecodeError as e:
                content = f"error: invalid JSON in tool arguments ({e}); resend the call with valid JSON"
            else:
                # Valid JSON can still be a non-object (`[]`, `42`, `null`).
                if not isinstance(tool_args, dict):
                    content = f"error: tool arguments must be a JSON object, got {type(tool_args).__name__}; resend as an object"
                elif name in dispatch:
                    content = await call_mcp(servers, dispatch, name, tool_args)
                elif name == "bash" and args.bash:
                    content = await asyncio.to_thread(
                        run_bash, tool_args.get("command", ""), args.allow_git
                    )
                elif name == "edit" and args.edit:
                    content = await asyncio.to_thread(
                        run_edit,
                        tool_args.get("path"),
                        tool_args.get("old_str"),
                        tool_args.get("new_str"),
                    )
                elif name == "search" and args.search:
                    content = await asyncio.to_thread(
                        run_search,
                        tool_args.get("query", ""),
                        args.serper_key,
                        tool_args.get("num_results", 5),
                    )
                else:
                    content = f"error: unknown tool {name!r}"
            tool_message["content"] = content
            tool_message = bound_tool_message(tool_message)
            if args.tool_interception_url:
                assert tool_client is not None
                decision = await run_tool_hook(
                    tool_client,
                    args.tool_interception_url,
                    args.api_key,
                    "after",
                    tool_message,
                )
                if decision["action"] == "rewrite":
                    tool_message = bound_tool_message(decision["message"])
            messages.append(tool_message)
            tool_result_tokens += estimated_tokens(str(tool_message["content"]))
        if compactor.reached(completion, tool_result_tokens) and compactable(messages):
            try:
                messages = await compactor.compact(messages)
            except CompactionFailed:
                return


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--api-key", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--system-prompt", default="")
    parser.add_argument("--prompt", default="")
    parser.add_argument("--initial-messages-file", default="")
    parser.add_argument("--mcp-config", default="")
    parser.add_argument("--tool-interception-url", default="")
    parser.add_argument("--bash", action="store_true")
    parser.add_argument("--allow-git", action="store_true")
    parser.add_argument("--compaction", action="store_true")
    parser.add_argument("--summarize-at-tokens", type=int)
    parser.add_argument("--edit", action="store_true")
    parser.add_argument("--search", action="store_true")
    parser.add_argument("--serper-key", default="")
    return parser.parse_args()


async def main() -> None:
    args = parse_args()
    initial = []
    if args.initial_messages_file:
        path = Path(args.initial_messages_file)
        payload = path.read_bytes()
        path.unlink()
        initial = json.loads(payload)
    client = AsyncOpenAI(
        base_url=args.base_url,
        api_key=args.api_key,
        timeout=httpx.Timeout(600.0 if args.bash else None, connect=5.0),
    )
    tool_client = (
        httpx.AsyncClient(timeout=httpx.Timeout(None, connect=5.0))
        if args.tool_interception_url
        else None
    )
    config = json.loads(args.mcp_config or "{}")
    tools = [BASH_TOOL] if args.bash else []
    reserved = {"bash"} if args.bash else set()
    if args.edit:
        tools.append(EDIT_TOOL)
        reserved.add("edit")
    if args.search:
        tools.append(SEARCH_TOOL)
        reserved.add("search")
    if config.get("mcpServers"):
        mcp_tools, dispatch, servers = await asyncio.wait_for(
            connect_mcp(config, reserved), timeout=None if args.bash else 60
        )
    else:
        mcp_tools, dispatch, servers = [], {}, {}
    tools += mcp_tools
    messages = (
        [{"role": "system", "content": args.system_prompt}]
        if args.system_prompt
        else []
    )
    if initial:
        messages.extend(initial)
    elif args.prompt:
        messages.append({"role": "user", "content": args.prompt})
    compactor = Compactor(
        client,
        args.model,
        tools,
        args.compaction,
        args.summarize_at_tokens,
    )
    if compactor.enabled and compactor.threshold is None:
        compactor.threshold = await discover_threshold(client, args.model)
    # The initial conversation is the floor for checkpoint fallbacks: a first-turn
    # checkpoint must never retry from an empty base.
    compactor.note_good(messages)
    await run_chat_loop(args, compactor, messages, dispatch, servers, tool_client)
    if tool_client is not None:
        await tool_client.aclose()


# Inert on package import; the entry point once this module ends the bundled script.
if __name__ == "__main__":
    asyncio.run(main())
