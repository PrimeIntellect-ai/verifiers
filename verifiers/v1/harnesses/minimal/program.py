# /// script
# requires-python = ">=3.10"
# dependencies = ["openai", "mcp==2.0.0", "httpx", "httpx2", "tenacity"]
# ///
"""Shared Null/Bash chat program; secrets use argv so tools do not inherit them."""

import argparse
import asyncio
import json
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

import httpx
from openai import APIError, APIStatusError, AsyncOpenAI

if TYPE_CHECKING:
    # The harness bundles this module into the generated script before execution.
    from verifiers.v1.mcp.client import call_mcp, connect_mcp  # noqa: TC004

SERPER_URL = "https://google.serper.dev/search"

CONTEXT_OVERFLOW_MARKERS = (
    "context_length_exceeded",
    "exceeds the context window",
    "reduce the length of the messages",
    "maximum context length",
    "prompt is too long",
    "request_too_large",
    "request entity too large",
    "exceeds the maximum number of tokens",
    "maximum prompt length is",
)


def is_context_overflow(error: APIStatusError) -> bool:
    details = f"{error} {error.body or ''}".casefold()
    return error.status_code in (400, 413) and any(
        marker in details for marker in CONTEXT_OVERFLOW_MARKERS
    )


RESERVE_TOKENS = 16_384
"""Compact when this many tokens remain below the model context window."""

COMPACTION_ATTEMPTS = 3
"""Checkpoint attempts before compaction fails: a rejected request falls back to the
last good snapshot; an empty or tool-calling reply is resampled."""

TOOL_OUTPUT_MAX_BYTES = 20_000
"""Middle-out truncation budget for one tool result before it enters the conversation."""

CHECKPOINT_COMPACTION_PROMPT = """You are performing a CONTEXT CHECKPOINT COMPACTION. Create a handoff summary for another LLM that will resume the task.

Include:
- Current progress and key decisions made
- Important context, constraints, or user preferences
- What remains to be done (clear next steps)
- Any critical data, examples, or references needed to continue

Be concise, structured, and focused on helping the next LLM seamlessly continue the work.

Reply with the summary as plain text. Do not call any tools - summarize from the conversation as it stands."""

POST_COMPACTION_FRAMING = """Another language model started to solve this problem and produced \
a summary of its thinking process. You also have access to the state of the tools that \
were used by that language model. Use this to build on the work \
that has already been done and avoid duplicating work. Here is \
the summary produced by the other language model, use the \
information in this summary to assist with your own analysis:"""

CONTEXT_OVERFLOW_MARKERS = (
    # OpenAI error code "context_length_exceeded"; OpenRouter relays the raw body.
    "context_length_exceeded",
    # OpenAI Responses/Completions: "Your input exceeds the context window of this model".
    "exceeds the context window",
    # OpenAI chat: "Input tokens exceed the configured limit of N tokens. Please reduce
    # the length of the messages."; Groq words it the same way.
    "reduce the length of the messages",
    # vLLM: "This model's maximum context length is N tokens"; the renderers pre-flight:
    # "Prompt length (N) exceeds maximum context length (M)"; Mistral uses the same words.
    "maximum context length",
    # Anthropic: "prompt is too long: N tokens > M maximum".
    "prompt is too long",
    # Anthropic byte-size overflow: HTTP 413 {"type": "request_too_large"}.
    "request_too_large",
    # HTTP proxies reject an oversized body with 413 "Request Entity Too Large".
    "request entity too large",
    # Google: "The input token count (N) exceeds the maximum number of tokens allowed (M)".
    "exceeds the maximum number of tokens",
    # xAI: "This model's maximum prompt length is N but the request contains M tokens".
    "maximum prompt length is",
)

CONTEXT_WINDOW_FIELDS = (
    "max_model_len",
    "context_length",
    "context_window",
    "max_context_length",
)


def default_threshold(context_window: int) -> int:
    """Leave a fixed reserve below the window; small windows keep at least half."""
    return max(context_window - RESERVE_TOKENS, context_window // 2)


async def discover_threshold(client: AsyncOpenAI, model: str) -> int | None:
    """The compaction threshold, when the provider's model card advertises a context window.

    `models.list()` keeps provider extensions in each card's `model_extra`; a raw
    `cast_to` parse breaks on one Python version or another."""
    try:
        page = await client.models.list()
    except APIError:
        return None
    for card in page.data:
        if card.id != model:
            continue
        extra = card.model_extra or {}
        for field in CONTEXT_WINDOW_FIELDS:
            value = extra.get(field)
            if isinstance(value, int) and not isinstance(value, bool) and value > 0:
                return default_threshold(value)
        break
    return None


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


class CompactionFailed(Exception):
    """Every checkpoint attempt failed - the caller ends the run cleanly instead."""


def compactable(messages: list[dict]) -> bool:
    """Whether compaction can reclaim anything - some history beyond the task exists."""
    first_user = next(
        (i for i, m in enumerate(messages) if m.get("role") == "user"), None
    )
    return any(
        m.get("role") != "system" and i != first_user for i, m in enumerate(messages)
    )


def bound_tool_message(message: dict) -> dict:
    """Bound a tool message before it enters the conversation - rewrites included."""
    content = message.get("content")
    if isinstance(content, str):
        return {**message, "content": truncate_tool_output(content)}
    # Multimodal results come back as content-part lists; only plain text is truncated.
    return message


def truncate_tool_output(text: str) -> str:
    """Keep the head and tail of an oversized tool result and say what was cut."""
    data = text.encode("utf-8")
    if len(data) <= TOOL_OUTPUT_MAX_BYTES:
        return text
    keep = TOOL_OUTPUT_MAX_BYTES // 2
    head = data[:keep].decode("utf-8", errors="ignore")
    tail = data[-keep:].decode("utf-8", errors="ignore")
    return (
        f"Warning: truncated output (original token count: {estimated_tokens(text)})\n"
        f"Total output lines: {text.count(chr(10)) + 1}\n\n"
        f"{head}\n[... {len(data) - 2 * keep} bytes truncated ...]\n{tail}"
    )


def estimated_tokens(chars: str) -> int:
    """Rough token count at four characters per token."""
    return (len(chars) + 3) // 4


def context_tokens(completion) -> int:
    usage = completion.usage
    if usage is None:
        return 0
    return (usage.prompt_tokens or 0) + (usage.completion_tokens or 0)


class Compactor:
    """Compact once and retry once when a model turn exhausts its context."""

    def __init__(self, client, model, tools, enabled, threshold):
        self.client = client
        self.model = model
        self.tools = tools
        self.enabled = enabled
        self.threshold = threshold
        self.compacted = False
        self.last_good = 0
        """Message count of the newest state that passed a threshold check - by
        definition a state with a full reserve of room, so a checkpoint over it fits."""

    def reached(self, completion, extra_tokens: int = 0) -> bool:
        return (
            self.enabled
            and self.threshold is not None
            and context_tokens(completion) + extra_tokens >= self.threshold
        )

    def note_good(self, messages: list[dict]) -> None:
        self.last_good = len(messages)

    async def complete(self, messages: list[dict]):
        try:
            completion = await chat(self.client, self.model, messages, self.tools)
        except APIStatusError as error:
            if (
                not self.enabled
                or self.threshold is None
                or not is_context_overflow(error)
            ):
                raise
            if not compactable(messages):
                if self.compacted:
                    # The conversation is already a compaction floor and still
                    # overflows - out of moves, end cleanly.
                    raise CompactionFailed(
                        "the compacted conversation still overflows"
                    ) from error
                raise
        else:
            choice = completion.choices[0]
            if not self.reached(completion):
                # Usage-verified: this exact prompt was accepted with a full
                # reserve of room, so it is a safe checkpoint fallback.
                self.note_good(messages)
                return completion, messages
            if choice.finish_reason != "length" or not compactable(messages):
                return completion, messages

        messages = await self.compact(messages)
        try:
            completion = await chat(self.client, self.model, messages, self.tools)
        except APIStatusError as error:
            # The rebuilt conversation is sized to fit, so this is out of moves.
            if is_context_overflow(error):
                raise CompactionFailed(
                    "the rebuilt conversation still overflows"
                ) from error
            raise
        return completion, messages

    async def compact(self, messages: list[dict]) -> list[dict]:
        # A rejected checkpoint falls back to the last good snapshot (which has a
        # full reserve of room, so it fits); an empty or tool-calling reply is
        # resampled. Reasoning is never part of the summary.
        system = [message for message in messages if message.get("role") == "system"]
        base = messages
        for _ in range(COMPACTION_ATTEMPTS):
            checkpoint = [
                *base,
                {"role": "user", "content": CHECKPOINT_COMPACTION_PROMPT},
            ]
            try:
                completion = await chat(
                    self.client,
                    self.model,
                    checkpoint,
                    self.tools,
                    tool_choice="none",
                )
            except APIStatusError as error:
                if not is_context_overflow(error):
                    raise
                base = messages[: self.last_good]
                continue
            message = completion.choices[0].message
            # Reasoning never enters the summary: only the reply's final text
            # counts, so a reply that lives entirely in the reasoning channel
            # is resampled like an empty one.
            text = (message.content or "").strip()
            if not message.tool_calls and text:
                framed = POST_COMPACTION_FRAMING + "\n\n" + text
                rebuilt = [*system, {"role": "user", "content": framed}]
                self.note_good(rebuilt)
                self.compacted = True
                return rebuilt
        raise CompactionFailed(
            f"no usable summary after {COMPACTION_ATTEMPTS} attempts"
        )


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
    while True:
        try:
            completion, messages = await compactor.complete(messages)
        except CompactionFailed:
            # The context is exhausted and could not be summarized: end the run
            # cleanly with what the conversation holds - still a trainable sample.
            break
        except APIStatusError as error:
            # Null cannot compact, so context exhaustion ends it with the transcript so far.
            if args.bash or not is_context_overflow(error):
                raise
            return
        message = completion.choices[0].message
        messages.append(message.model_dump(exclude_none=True))
        if not message.tool_calls:
            break
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
                break
    if tool_client is not None:
        await tool_client.aclose()


if __name__ == "__main__":
    asyncio.run(main())
