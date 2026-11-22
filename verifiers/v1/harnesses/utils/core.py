"""Chat loop, local tools, and interception hook for bundled chat programs."""

import argparse
import asyncio
import fcntl
import json
import logging
import os
import select
import selectors
import struct
import subprocess
import termios
import threading
import time
import traceback
from contextlib import AsyncExitStack, suppress
from pathlib import Path
from typing import TYPE_CHECKING

import certifi
import httpx
from openai import (
    APIConnectionError,
    APIStatusError,
    AsyncOpenAI,
    DefaultAsyncHttpxClient,
    omit,
)
from openai.lib.streaming.chat import AsyncChatCompletionStream
from pydantic import AnyHttpUrl, ValidationError
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
        TOOL_OUTPUT_MAX_BYTES,
        Compactor,
        bound_tool_message,
        compactable,
        discover_threshold,
        estimated_tokens,
        is_context_overflow,
    )
    from verifiers.v1.harnesses.utils.mcp import call_mcp, connect_mcp  # noqa: TC004
    from verifiers.v1.utils.network import network_rule_matches  # noqa: TC004

# Mirror of verifiers.v1.errors.MODEL_TRANSPORT_ERROR_EXIT_CODE;
# this bundled script cannot import verifiers.
MODEL_TRANSPORT_ERROR_EXIT_CODE = 97

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


def run_search(
    query: str,
    api_key: str,
    num_results: int = 5,
    network_block: list[str] | None = None,
) -> str:
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
        if network_block:
            filtered = []
            for result in organic:
                try:
                    url = AnyHttpUrl(result.get("link", ""))
                except ValidationError:
                    continue
                if not any(
                    network_rule_matches(
                        rule,
                        url.scheme,
                        url.host.strip("[]"),
                        url.port,
                        subdomains=True,
                    )
                    for rule in network_block
                ):
                    filtered.append(result)
            organic = filtered
        return format_results(organic[:num_results], query)
    except Exception as e:  # noqa: BLE001 - tool failures are returned to the model
        return f"search failed ({e}). Try again or rephrase the query."


class _BashOutput:
    def __init__(self) -> None:
        self.head = b""
        self.tail = b""
        self.size = 0

    def add(self, chunk: bytes) -> None:
        self.head += chunk[: max(0, TOOL_OUTPUT_MAX_BYTES - len(self.head))]
        self.tail = (self.tail + chunk)[-TOOL_OUTPUT_MAX_BYTES:]
        self.size += len(chunk)


BASH_TIMEOUT = 3600
"""Seconds a command may run before the call fails and bash is killed."""

EXIT_GRACE = 0.5
"""Seconds to keep reading after bash exits, for processes it does not wait for, such as
process substitutions, to finish writing and close the pipes."""

BACKGROUND_NOTE = "[a process started by this command is still running and holds its output; later output is not shown]"
"""Ends a result whose output a process the command left running still holds open."""


def _pending(fd: int) -> int:
    """Bytes waiting in a pipe right now."""
    return struct.unpack("i", fcntl.ioctl(fd, termios.FIONREAD, b"\0" * 4))[0]


def _at_end(fd: int) -> bool:
    """Whether every process has closed a pipe whose waiting bytes were just read. Bytes
    that arrived since are a background job's, written after the grace, and dropped."""
    poller = select.poll()  # not select.select, which rejects fds >= FD_SETSIZE (1024)
    poller.register(fd, select.POLLIN)
    if not poller.poll(0):
        return False
    return not os.read(fd, 65536)


def _discard_until_eof(fd: int) -> None:
    """Read a pipe a background job still holds to its end, so the job's writes neither
    block on a full pipe nor fail on a closed one once the call has returned."""
    with open(fd, "rb", buffering=0) as pipe:
        while pipe.read(65536):
            pass


def _drain_in_background(fd: int) -> None:
    """Hand a pipe still held open to a daemon thread. Never raises: if that fails, the
    pipe is closed with the call, as on a timeout."""
    try:
        copy = os.dup(fd)
    except OSError:
        return
    try:
        threading.Thread(target=_discard_until_eof, args=(copy,), daemon=True).start()
    except Exception:  # noqa: BLE001 - e.g. no more threads; the result still stands
        with suppress(OSError):
            os.close(copy)


def run_bash(command: str) -> str:
    held_open = False
    try:
        with subprocess.Popen(
            ["bash", "-c", command],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        ) as process:
            stdout, stderr = _BashOutput(), _BashOutput()
            deadline = time.monotonic() + BASH_TIMEOUT
            exited = None
            try:
                with selectors.DefaultSelector() as selector:
                    assert process.stdout is not None and process.stderr is not None
                    selector.register(process.stdout, selectors.EVENT_READ, stdout)
                    selector.register(process.stderr, selectors.EVENT_READ, stderr)
                    while selector.get_map():
                        now = time.monotonic()
                        if exited is None and process.poll() is not None:
                            exited = now
                        if exited is None:
                            if now >= deadline:
                                raise subprocess.TimeoutExpired(
                                    process.args, BASH_TIMEOUT
                                )
                            wait = min(deadline - now, 0.1)
                        elif now - exited < EXIT_GRACE:
                            # Bash has exited, but a process it did not wait for, such
                            # as a process substitution, may still be writing.
                            wait = exited + EXIT_GRACE - now
                        else:
                            # A pipe still open is held by a job left running. Read what
                            # is waiting and hand the rest to a thread off the call.
                            for key in list(selector.get_map().values()):
                                pending = _pending(key.fd)
                                while pending > 0:
                                    chunk = os.read(key.fd, min(pending, 65536))
                                    if not chunk:
                                        break
                                    pending -= len(chunk)
                                    key.data.add(chunk)
                                selector.unregister(key.fileobj)
                                if not _at_end(key.fd):
                                    held_open = True
                                    _drain_in_background(key.fd)
                            break
                        for key, _ in selector.select(wait):
                            chunk = os.read(key.fd, 8192)
                            if chunk:
                                key.data.add(chunk)
                            else:
                                selector.unregister(key.fileobj)
                    process.wait(timeout=max(0, deadline - time.monotonic()))
            finally:
                # Do not communicate() here: descendants may still hold the pipes. On a
                # timeout the pipes are closed, not drained, so a process still writing
                # to them dies on its next write.
                if process.returncode is None:
                    process.kill()
                    process.wait()
        size = stdout.size + stderr.size
        # The note fits in the output budget, so the result is not truncated again.
        budget = TOOL_OUTPUT_MAX_BYTES - (
            len(BACKGROUND_NOTE.encode()) + 1 if held_open else 0
        )
        if size <= budget:
            # A job still writing may have been cut off mid-character at the hand-off.
            text = "".join(
                output.head.decode(errors="replace")
                .replace("\r\n", "\n")
                .replace("\r", "\n")
                for output in (stdout, stderr)
            )
        else:
            marker = f"\n[... bash output truncated; {size} bytes total ...]\n"
            keep = (budget - len(marker.encode())) // 2
            # Preserve stdout-then-stderr ordering, including when just one is large.
            head = (stdout.head + stderr.head)[:keep].decode(errors="ignore")
            tail = (stdout.tail + stderr.tail)[-keep:].decode(errors="ignore")
            text = head + marker + tail
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        if held_open:
            text += ("\n" if text and not text.endswith("\n") else "") + BACKGROUND_NOTE
        return text
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


_STREAMED_MESSAGE_FIELDS = (
    "role",
    "reasoning",
    "reasoning_content",
    "reasoning_details",
)


def _accumulate_streamed_message(accumulated: dict, delta: dict) -> None:
    """Accumulate message fields whose stream semantics differ from the SDK defaults."""
    if role := delta.get("role"):
        accumulated["role"] = role

    for field_name in ("reasoning", "reasoning_content"):
        if value := delta.get(field_name):
            accumulated[field_name] = accumulated.get(field_name, "") + value

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
    try:
        async for attempt in AsyncRetrying(
            retry=retry_if_exception_type((APIConnectionError, httpx.TransportError)),
            stop=stop_after_attempt(client.max_retries + 1),
            wait=wait_random_exponential(multiplier=0.5, max=8.0),
            before_sleep=before_sleep_log(logging.getLogger(__name__), logging.WARNING),
            reraise=True,
        ):
            # Reuse the interception server's body-digest replay guard on stream retries.
            retry_count = attempt.retry_state.attempt_number - 1
            headers = (
                {"x-stainless-retry-count": str(retry_count)} if retry_count else omit
            )
            raw_stream = await client.chat.completions.create(
                **kwargs,
                stream=True,
                stream_options={"include_usage": True},
                extra_headers=headers,
            )
            # The SDK retries request setup; only stream consumption is retried here.
            with attempt:
                return await _read_chat_completion(raw_stream)
    except (APIConnectionError, httpx.TransportError) as error:
        # Preserve the original transport error in stderr for the host's diagnostic.
        traceback.print_exc()
        raise SystemExit(MODEL_TRANSPORT_ERROR_EXIT_CODE) from error


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
                    if any(
                        delta.get(field_name) for field_name in _STREAMED_MESSAGE_FIELDS
                    ):
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
        except APIStatusError as error:
            # Without compaction (off, or on with no discoverable window to compact
            # against), context exhaustion is a budget limit, not a crash: end the run
            # with the transcript so far. When the compactor could act, it already
            # tried, so an overflow reaching here is a real failure.
            compacting = compactor.enabled and compactor.threshold is not None
            if compacting or not is_context_overflow(error):
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
                decision = await gate_tool_call(
                    tool_client, args.tool_interception_url, args.api_key, call
                )
                if decision["action"] == "deny":
                    denied = bound_tool_message(decision["message"])
                    messages.append(denied)
                    tool_result_tokens += estimated_tokens(
                        str(denied.get("content", ""))
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
                        run_bash, tool_args.get("command", "")
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
                        args.search_block,
                    )
                else:
                    content = f"error: unknown tool {name!r}"
            tool_message["content"] = content
            # Results are rewritten at the model boundary, not here.
            tool_message = bound_tool_message(tool_message)
            messages.append(tool_message)
            tool_result_tokens += estimated_tokens(str(tool_message["content"]))
        if compactor.reached(completion, tool_result_tokens) and compactable(messages):
            messages = await compactor.compact(messages)


def read_staged(path: str) -> bytes:
    """Read and delete a file the harness staged for this launch; empty without one."""
    if not path:
        return b""
    file = Path(path)
    data = file.read_bytes()
    file.unlink()
    return data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--api-key", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--system-prompt-file", default="")
    parser.add_argument("--initial-messages-file", default="")
    parser.add_argument("--mcp-config", default="")
    parser.add_argument("--tool-interception-url", default="")
    parser.add_argument("--bash", action="store_true")
    parser.add_argument("--compaction", action="store_true")
    parser.add_argument("--summarize-at-tokens", type=int)
    parser.add_argument("--edit", action="store_true")
    parser.add_argument("--search", action="store_true")
    parser.add_argument("--search-block", type=json.loads, default=[])
    parser.add_argument("--serper-key", default="")
    return parser.parse_args()


async def main() -> None:
    args = parse_args()
    system_prompt = read_staged(args.system_prompt_file).decode()
    initial = json.loads(read_staged(args.initial_messages_file) or b"[]")
    # Minimal task images may lack a system CA bundle.
    client = AsyncOpenAI(
        base_url=args.base_url,
        api_key=args.api_key,
        timeout=httpx.Timeout(600.0 if args.bash else None, connect=5.0),
        http_client=DefaultAsyncHttpxClient(verify=certifi.where()),
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
    async with AsyncExitStack() as mcp_stack:
        if config.get("mcpServers"):
            mcp_tools, dispatch, servers = await asyncio.wait_for(
                connect_mcp(config, mcp_stack, reserved),
                timeout=None if args.bash else 60,
            )
        else:
            mcp_tools, dispatch, servers = [], {}, {}
        tools += mcp_tools
        messages = (
            [{"role": "system", "content": system_prompt}] if system_prompt else []
        )
        messages.extend(initial)
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
