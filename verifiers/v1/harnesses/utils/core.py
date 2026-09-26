"""Chat loop, local tools, and interception hook for bundled chat programs."""

import argparse
import asyncio
import json
import logging
import subprocess
from contextlib import AsyncExitStack
from pathlib import Path
from typing import TYPE_CHECKING

import httpx
from openai import APIConnectionError, APIStatusError, AsyncOpenAI, omit
from openai.lib.streaming.chat import AsyncChatCompletionStream
from tenacity import (
    AsyncRetrying,
    before_sleep_log,
    retry_if_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)

if TYPE_CHECKING:
    # The harness bundles this module into the generated script before execution.
    from verifiers.v1.harnesses.utils.compaction import (  # noqa: TC004
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


def run_bash(command: str) -> str:
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


def _accumulate_streamed_message(accumulated: dict, delta: dict) -> None:
    """Preserve repeated role and reasoning-detail metadata the SDK concatenates."""
    if role := delta.get("role"):
        accumulated["role"] = role

    delta_details = delta.get("reasoning_details") or []
    if not delta_details:
        return
    reasoning_details = accumulated.setdefault("reasoning_details", [])
    for detail in delta_details:
        previous = reasoning_details[-1] if reasoning_details else {}
        detail_type = detail.get("type")
        content_field = {
            "reasoning.summary": "summary",
            "reasoning.text": "text",
        }.get(detail_type)
        if (
            content_field
            and detail_type == previous.get("type")
            and all(
                previous.get(field_name) is None
                or detail.get(field_name) is None
                or previous[field_name] == detail[field_name]
                for field_name in ("id", "index", "format")
            )
        ):
            previous[content_field] = (previous.get(content_field) or "") + (
                detail.get(content_field) or ""
            )
            for field_name in ("id", "index", "signature", "format"):
                if (
                    previous.get(field_name) is None
                    and detail.get(field_name) is not None
                ):
                    previous[field_name] = detail[field_name]
        else:
            reasoning_details.append(dict(detail))


async def chat(
    client: AsyncOpenAI,
    model: str,
    messages: list[dict],
    tools: list[dict],
    *,
    tool_choice: str | None = None,
):
    kwargs = {"model": model, "messages": messages}
    if tools:
        kwargs["tools"] = tools
    if tools and tool_choice is not None:
        kwargs["tool_choice"] = tool_choice
    async for attempt in AsyncRetrying(
        retry=retry_if_exception_type((APIConnectionError, httpx.TransportError)),
        stop=stop_after_attempt(client.max_retries + 1),
        wait=wait_random_exponential(multiplier=0.5, max=8.0),
        before_sleep=before_sleep_log(logging.getLogger(__name__), logging.WARNING),
        reraise=True,
    ):
        # Reuse the interception server's body-digest replay guard on stream retries.
        retry_count = attempt.retry_state.attempt_number - 1
        headers = {"x-stainless-retry-count": str(retry_count)} if retry_count else omit
        raw_stream = await client.chat.completions.create(
            **kwargs,
            stream=True,
            stream_options={"include_usage": True},
            extra_headers=headers,
        )
        # The SDK retries request setup; only stream consumption is retried here.
        with attempt:
            return await _read_chat_completion(raw_stream)


async def _read_chat_completion(raw_stream):
    # Accumulate native deltas without auto-parsing tool arguments or treating
    # finish_reason="length" as an exception: compaction owns that decision.
    async with AsyncChatCompletionStream(
        raw_stream=raw_stream, response_format=omit, input_tools=[]
    ) as response:
        completion = None
        message_overrides: dict[int, dict] = {}
        async for event in response:
            if event.type == "chunk":
                completion = event.snapshot
                for choice in event.chunk.choices:
                    delta = choice.delta.model_dump(exclude_none=True)
                    if delta.get("role") or delta.get("reasoning_details"):
                        _accumulate_streamed_message(
                            message_overrides.setdefault(choice.index, {}), delta
                        )
        if (
            completion is None
            or not completion.choices
            or any(choice.finish_reason is None for choice in completion.choices)
        ):
            raise APIConnectionError(
                message="Model stream ended before a completion finished",
                request=raw_stream.response.request,
            )
        for choice in completion.choices:
            overrides = message_overrides.setdefault(choice.index, {})
            overrides.setdefault("role", "assistant")
            for field_name, value in overrides.items():
                setattr(choice.message, field_name, value)
        return completion


async def gate_tool_call(
    client: httpx.AsyncClient, url: str, api_key: str, call
) -> dict:
    """Ask the rollout's gate before running `call`. A denial carries the result to record
    in place of executing; a stopped rollout ends the program."""
    try:
        arguments = json.loads(call.function.arguments or "{}")
    except json.JSONDecodeError:
        arguments = call.function.arguments
    response = await client.post(
        url,
        headers={"Authorization": f"Bearer {api_key}"},
        json={
            "tool_call_id": call.id,
            "name": call.function.name,
            "arguments": arguments,
        },
    )
    response.raise_for_status()
    decision = response.json()
    if decision["action"] == "stop":
        raise RuntimeError(decision["reason"])
    return decision


def initial_messages(args: argparse.Namespace) -> list[dict]:
    """Consume a resume transcript once, or start with the supplied prompt."""
    initial = []
    if args.initial_messages_file:
        path = Path(args.initial_messages_file)
        payload = path.read_bytes()
        path.unlink()
        initial = json.loads(payload)
    messages = (
        [{"role": "system", "content": args.system_prompt}]
        if args.system_prompt
        else []
    )
    if initial:
        messages.extend(initial)
    elif args.prompt:
        messages.append({"role": "user", "content": args.prompt})
    return messages


async def run_chat_loop(
    args: argparse.Namespace,
    client: AsyncOpenAI,
    tools: list[dict],
    messages: list[dict],
    local_tools: dict,
    dispatch: dict,
    servers: dict,
    *,
    compactor: "Compactor | None" = None,
    tool_client: httpx.AsyncClient | None = None,
) -> None:
    """Run a conversation, with bounded streaming turns when a compactor is supplied.

    Other programs use ordinary SDK completions and retain tool results verbatim.
    """
    while True:
        try:
            if compactor is None:
                completion = await client.chat.completions.create(
                    model=args.model, messages=messages, tools=tools or None
                )
            else:
                completion, messages = await compactor.complete(messages)
        except APIStatusError as error:
            # Bounded chat programs treat exhaustion without an active compaction
            # threshold as a budget stop. Ordinary SDK clients propagate the error.
            if (
                compactor is None
                or (compactor.enabled and compactor.threshold is not None)
                or not is_context_overflow(error)
            ):
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
            }
            if compactor is not None:
                tool_message["name"] = name
            if tool_client is not None:
                decision = await gate_tool_call(
                    tool_client, args.tool_interception_url, args.api_key, call
                )
                if decision["action"] == "deny":
                    denied = decision["message"]
                    if compactor is not None:
                        denied = bound_tool_message(denied)
                        tool_result_tokens += estimated_tokens(
                            str(denied.get("content", ""))
                        )
                    messages.append(denied)
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
                elif name in local_tools:
                    content = await asyncio.to_thread(local_tools[name], tool_args)
                else:
                    content = f"error: unknown tool {name!r}"
            tool_message["content"] = content
            # Results are rewritten at the model boundary, not here.
            if compactor is not None:
                tool_message = bound_tool_message(tool_message)
                tool_result_tokens += estimated_tokens(str(tool_message["content"]))
            messages.append(tool_message)
        if (
            compactor is not None
            and compactor.reached(completion, tool_result_tokens)
            and compactable(messages)
        ):
            messages = await compactor.compact(messages)


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
    parser.add_argument("--compaction", action="store_true")
    parser.add_argument("--summarize-at-tokens", type=int)
    parser.add_argument("--edit", action="store_true")
    parser.add_argument("--search", action="store_true")
    parser.add_argument("--serper-key", default="")
    return parser.parse_args()


async def main() -> None:
    args = parse_args()
    messages = initial_messages(args)
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
    local_tools = (
        {"bash": lambda values: run_bash(values.get("command", ""))}
        if args.bash
        else {}
    )
    reserved = {"bash"} if args.bash else set()
    if args.edit:
        tools.append(EDIT_TOOL)
        reserved.add("edit")
        local_tools["edit"] = lambda values: run_edit(
            values.get("path"), values.get("old_str"), values.get("new_str")
        )
    if args.search:
        tools.append(SEARCH_TOOL)
        reserved.add("search")
        local_tools["search"] = lambda values: run_search(
            values.get("query", ""), args.serper_key, values.get("num_results", 5)
        )
    async with AsyncExitStack() as mcp_stack:
        await mcp_stack.enter_async_context(client)
        if tool_client is not None:
            await mcp_stack.enter_async_context(tool_client)
        if config.get("mcpServers"):
            mcp_tools, dispatch, servers = await asyncio.wait_for(
                connect_mcp(config, mcp_stack, reserved),
                timeout=None if args.bash else 60,
            )
        else:
            mcp_tools, dispatch, servers = [], {}, {}
        tools += mcp_tools
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
        await run_chat_loop(
            args,
            client,
            tools,
            messages,
            local_tools,
            dispatch,
            servers,
            compactor=compactor,
            tool_client=tool_client,
        )
