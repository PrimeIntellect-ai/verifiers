"""Tool-output bounding and context compaction for bundled chat programs."""

from typing import TYPE_CHECKING

from openai import APIError, APIStatusError, AsyncOpenAI

if TYPE_CHECKING:
    # The harness bundles this module into the generated script before execution.
    from verifiers.v1.harnesses.utils.core import chat  # noqa: TC004

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


class CompactionFailed(Exception):
    """Every checkpoint attempt failed - the caller ends the run cleanly instead."""


def is_context_overflow(error: APIStatusError) -> bool:
    details = f"{error} {error.body or ''}"
    # An overflow is deterministic: a 400, or a 413 for a byte-size cap.
    return error.status_code in (400, 413) and any(
        marker in details.casefold() for marker in CONTEXT_OVERFLOW_MARKERS
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
