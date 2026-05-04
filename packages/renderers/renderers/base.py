from __future__ import annotations

import logging
import queue
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Literal, Protocol, TypedDict, runtime_checkable

logger = logging.getLogger("renderers.base")


# ---------------------------------------------------------------------------
# Message types — strong typing for the conversation data model
# ---------------------------------------------------------------------------


class TextPart(TypedDict):
    """A chunk of text content in a message."""

    type: Literal["text"]
    text: str


class ThinkingPart(TypedDict):
    """Model's internal reasoning (chain-of-thought) as a content part."""

    type: Literal["thinking"]
    thinking: str


ContentPart = TextPart | ThinkingPart

# Content is either a plain string or a list of structured parts.
Content = str | list[ContentPart]


class ToolCallFunction(TypedDict):
    """Function body within a tool call."""

    name: str
    arguments: dict[str, Any] | str


class ToolCall(TypedDict, total=False):
    """Structured tool invocation following OpenAI function-calling format."""

    type: str  # "function"
    id: str
    function: ToolCallFunction


class ToolSpec(TypedDict):
    """Tool specification (OpenAI function-calling format)."""

    name: str
    description: str
    parameters: dict[str, Any]


class Message(TypedDict, total=False):
    """A single turn in a multi-turn conversation.

    Required keys: role, content.
    Optional keys mirror the OpenAI chat format for tool calling.
    """

    role: str
    content: Content
    tool_calls: list[ToolCall]
    tool_call_id: str
    name: str
    reasoning_content: str


# ---------------------------------------------------------------------------
# Renderer data types
# ---------------------------------------------------------------------------


@dataclass
class RenderedTokens:
    """Result of rendering messages to tokens.

    Each token carries an index into the original message list so callers can
    build per-token loss masks without re-rendering.  Tokens from structural
    scaffolding (generation prompt, im_start/im_end wrapping) carry index -1.
    """

    token_ids: list[int] = field(default_factory=list)
    message_indices: list[int] = field(default_factory=list)


@dataclass
class ParsedResponse:
    """Result of parsing completion tokens back into a structured message."""

    content: str
    reasoning_content: str | None = None
    tool_calls: list[dict[str, Any]] | None = None


@dataclass
class RenderedConversation:
    """Exact token state for a rendered conversation."""

    prompt_ids: list[int]
    completion_ids: list[int] = field(default_factory=list)
    completion_logprobs: list[float] = field(default_factory=list)
    messages: list[Message] = field(default_factory=list)
    parsed_completion: ParsedResponse | None = None

    @property
    def token_ids(self) -> list[int]:
        return self.prompt_ids + self.completion_ids

    def with_completion(
        self,
        completion_ids: list[int],
        *,
        completion_logprobs: list[float] | None = None,
        parsed_completion: ParsedResponse | None = None,
    ) -> "RenderedConversation":
        return RenderedConversation(
            prompt_ids=list(self.prompt_ids),
            completion_ids=list(completion_ids),
            completion_logprobs=list(completion_logprobs or []),
            messages=list(self.messages),
            parsed_completion=parsed_completion,
        )


@runtime_checkable
class Renderer(Protocol):
    """Owns message ↔ token conversion for a specific model family."""

    def render(
        self,
        messages: list[Message],
        *,
        tools: list[ToolSpec] | None = None,
        add_generation_prompt: bool = False,
    ) -> RenderedTokens:
        """Render messages to token IDs with per-token message attribution."""
        ...

    def render_ids(
        self,
        messages: list[Message],
        *,
        tools: list[ToolSpec] | None = None,
        add_generation_prompt: bool = False,
    ) -> list[int]:
        """Render messages to token IDs (without attribution metadata)."""
        ...

    def parse_response(self, token_ids: list[int]) -> ParsedResponse:
        """Parse completion tokens back into a structured message."""
        ...

    def get_stop_token_ids(self) -> list[int]:
        """Return token IDs that signal generation should stop."""
        ...

    def bridge_to_next_turn(
        self,
        previous_prompt_ids: list[int],
        previous_completion_ids: list[int],
        new_messages: list[Message],
        *,
        tools: list[ToolSpec] | None = None,
    ) -> list[int] | None:
        """Extend ``prev_prompt_ids + prev_completion_ids`` with the tokens
        the next turn adds, without re-rendering the sampled tokens.

        Contract: if the return value ``B`` is not None, then
        ``B[: len(prev_prompt) + len(prev_completion)] == prev_prompt + prev_completion``
        and ``B`` ends at the position where the next assistant turn begins
        generating (i.e. equivalent to rendering the full message list so far
        with ``add_generation_prompt=True`` — except prev sampled tokens are
        kept verbatim rather than re-rendered).

        Return ``None`` whenever the renderer can't prove that contract
        holds — the caller falls back to a full re-render. In particular,
        bridges refuse assistant messages in ``new_messages`` (those would
        re-tokenize model-sampled content). Hand-coded renderers know their
        canonical close and synthesise it on truncated priors;
        DefaultRenderer always returns ``None`` because the template's
        close is unknown.
        """
        ...


class RendererPool:
    """Thread-safe pool of Renderer instances for parallel pretokenization.

    Each Renderer wraps its own tokenizer copy, avoiding contention.

    Construction parallelism matters: ``AutoTokenizer.from_pretrained`` takes
    hundreds of ms per call (JSON parse + Rust tokenizer build + HF cache
    lookup), so populating a 32-slot pool serially costs ~10-15s on startup
    and shows up directly as a step-0 stall. We fan the factory out across a
    short-lived thread pool; since HF fast tokenizers release the GIL during
    the Rust build phase, this parallelizes well.
    """

    def __init__(self, factory: Callable[[], Renderer], size: int):
        from concurrent.futures import ThreadPoolExecutor

        self._factory = factory
        self._pool: queue.Queue[Renderer] = queue.Queue(maxsize=size)
        # Cap workers so we don't spawn an oversized thread pool just to init
        # a small pool; clamp to 8 because past that the GIL-bound Python
        # portion of from_pretrained stops scaling.
        workers = min(size, 8)
        with ThreadPoolExecutor(max_workers=workers) as executor:
            for renderer in executor.map(lambda _: factory(), range(size)):
                self._pool.put(renderer)

    @contextmanager
    def checkout(self):
        renderer = self._pool.get()
        try:
            yield renderer
        finally:
            self._pool.put(renderer)

    @property
    def size(self) -> int:
        return self._pool.maxsize


RENDERER_REGISTRY: dict[str, type] = {}

# Exact canonical HF model names → renderer. We do NOT use prefix
# matching because models with the same architecture may ship different
# chat templates (base vs instruct, tuned vs pretrained) — matching on
# prefix silently routes them to a renderer that doesn't produce
# template-parity output. Fine-tunes and renamed checkpoints MUST pass
# ``renderer=<name>`` explicitly; the auto path falls back to
# ``DefaultRenderer`` (which uses ``apply_chat_template`` verbatim) and
# logs a loud INFO line with the chosen fallback.
MODEL_RENDERER_MAP: dict[str, str] = {
    # Qwen3 — base and Instruct variants share the same chat template.
    "Qwen/Qwen3-0.6B": "qwen3",
    "Qwen/Qwen3-1.7B": "qwen3",
    "Qwen/Qwen3-4B": "qwen3",
    "Qwen/Qwen3-4B-Instruct-2507": "qwen3",
    "Qwen/Qwen3-4B-Thinking-2507": "qwen3",
    "Qwen/Qwen3-8B": "qwen3",
    "Qwen/Qwen3-14B": "qwen3",
    "Qwen/Qwen3-32B": "qwen3",
    "Qwen/Qwen3-30B-A3B": "qwen3",
    "Qwen/Qwen3-235B-A22B": "qwen3",
    # Qwen3.5.
    "Qwen/Qwen3.5-9B": "qwen3.5",
    "Qwen/Qwen3.5-35B-A3B": "qwen3.5",
    "Qwen/Qwen3.5-122B-A10B": "qwen3.5",
    # Qwen3.6.
    "Qwen/Qwen3.6-35B-A3B": "qwen3.6",
    # Qwen3-VL.
    "Qwen/Qwen3-VL-4B-Instruct": "qwen3_vl",
    "Qwen/Qwen3-VL-8B-Instruct": "qwen3_vl",
    "Qwen/Qwen3-VL-30B-A3B-Instruct": "qwen3_vl",
    # GLM-5 family (GLM-4.7 reuses the GLM-5 template).
    "zai-org/GLM-5": "glm5",
    "zai-org/GLM-4.7-Flash": "glm5",
    "zai-org/GLM-5.1": "glm5.1",
    # GLM-4.5.
    "THUDM/GLM-4.5-Air": "glm4.5",
    "zai-org/GLM-4.5-Air": "glm4.5",
    # MiniMax.
    "MiniMaxAI/MiniMax-M2": "minimax-m2",
    "MiniMaxAI/MiniMax-M2.5": "minimax-m2",
    # DeepSeek V3.
    "deepseek-ai/DeepSeek-V3": "deepseek_v3",
    "deepseek-ai/DeepSeek-V3-Base": "deepseek_v3",
    # Kimi K2 (K2.5 and K2.6 share the K2.5 template, distinct from K2).
    "moonshotai/Kimi-K2-Instruct": "kimi_k2",
    "moonshotai/Kimi-K2.5": "kimi_k25",
    "moonshotai/Kimi-K2.6": "kimi_k25",
    # Nemotron 3.
    "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16": "nemotron3",
    "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16": "nemotron3",
    # GPT-OSS.
    "openai/gpt-oss-20b": "gpt_oss",
    "openai/gpt-oss-120b": "gpt_oss",
}


def _populate_registry():
    if RENDERER_REGISTRY:
        return
    from renderers.default import DefaultRenderer
    from renderers.deepseek_v3 import DeepSeekV3Renderer
    from renderers.glm5 import GLM5Renderer, GLM51Renderer
    from renderers.glm45 import GLM45Renderer
    from renderers.gpt_oss import GptOssRenderer
    from renderers.kimi_k2 import KimiK2Renderer
    from renderers.kimi_k25 import KimiK25Renderer
    from renderers.minimax_m2 import MiniMaxM2Renderer
    from renderers.nemotron3 import Nemotron3Renderer
    from renderers.qwen3 import Qwen3KeepThinkingRenderer, Qwen3Renderer
    from renderers.qwen3_vl import Qwen3VLRenderer
    from renderers.qwen35 import Qwen35Renderer
    from renderers.qwen36 import Qwen36Renderer

    RENDERER_REGISTRY.update(
        {
            "default": DefaultRenderer,
            "qwen3": Qwen3Renderer,
            "qwen3-keep-thinking": Qwen3KeepThinkingRenderer,
            "qwen3_vl": Qwen3VLRenderer,
            "qwen3.5": Qwen35Renderer,
            "qwen3.6": Qwen36Renderer,
            "glm5": GLM5Renderer,
            "glm5.1": GLM51Renderer,
            "glm4.5": GLM45Renderer,
            "minimax-m2": MiniMaxM2Renderer,
            "deepseek_v3": DeepSeekV3Renderer,
            "kimi_k2": KimiK2Renderer,
            "kimi_k25": KimiK25Renderer,
            "nemotron3": Nemotron3Renderer,
            "gpt_oss": GptOssRenderer,
        }
    )


def create_renderer_pool(
    tokenizer_name_or_path: str,
    renderer: str = "auto",
    size: int = 16,
    *,
    tool_parser: str | None = None,
    reasoning_parser: str | None = None,
) -> RendererPool:
    """Create a RendererPool with *size* independent tokenizer copies.

    Each slot loads its own tokenizer so threads never share mutable state.
    HuggingFace fast tokenizers release the GIL during Rust encoding, so
    threads achieve real parallelism.

    ``tool_parser`` and ``reasoning_parser`` are forwarded to
    ``create_renderer`` when the pool falls back to ``DefaultRenderer``.
    """

    def factory() -> Renderer:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name_or_path, trust_remote_code=True
        )
        return create_renderer(
            tokenizer,
            renderer=renderer,
            tool_parser=tool_parser,
            reasoning_parser=reasoning_parser,
        )

    return RendererPool(factory, size=size)


def create_renderer(
    tokenizer,
    renderer: str = "auto",
    *,
    tool_parser: str | None = None,
    reasoning_parser: str | None = None,
) -> Renderer:
    """Create a Renderer by name, or auto-detect from the tokenizer's model name.

    Args:
        tokenizer: HuggingFace tokenizer instance.
        renderer: Renderer name ('qwen3', 'qwen3_vl', 'qwen3.5', 'glm5', 'glm4.5',
                  'minimax-m2', 'deepseek_v3', 'kimi_k2', 'kimi_k25', 'nemotron3',
                  'gpt_oss', 'default') or 'auto' to detect from model name.
        tool_parser: Name of a tool parser registered in ``renderers.parsers``.
                  Only consumed by DefaultRenderer. Model-specific renderers
                  have their own parsing wired in.
        reasoning_parser: Name of a reasoning parser registered in
                  ``renderers.parsers``. Only consumed by DefaultRenderer.
    """
    _populate_registry()

    default_kwargs: dict = {}
    if tool_parser is not None:
        default_kwargs["tool_parser"] = tool_parser
    if reasoning_parser is not None:
        default_kwargs["reasoning_parser"] = reasoning_parser

    if renderer != "auto":
        cls = RENDERER_REGISTRY.get(renderer)
        if cls is None:
            raise ValueError(
                f"Unknown renderer {renderer!r}. Available: {', '.join(sorted(RENDERER_REGISTRY))}"
            )
        if renderer == "default":
            return cls(tokenizer, **default_kwargs)
        if default_kwargs:
            logger.info(
                "tool_parser / reasoning_parser are only consumed by "
                "DefaultRenderer; ignoring for renderer=%r which has "
                "built-in behavior.",
                renderer,
            )
        return cls(tokenizer)

    # Auto-detect from model name via exact match on the canonical HF id.
    # Fine-tunes and renamed checkpoints miss on purpose — their chat
    # template may differ from the original even when the architecture
    # matches, so silently mapping them would produce template-parity
    # bugs. Set ``renderer=<name>`` explicitly for those.
    model_name = getattr(tokenizer, "name_or_path", "")
    renderer_name = MODEL_RENDERER_MAP.get(model_name)
    if renderer_name is not None:
        return RENDERER_REGISTRY[renderer_name](tokenizer)

    # No match — fall back to default (apply_chat_template). For fine-tunes
    # with customized chat templates this is the *correct* choice, so we don't
    # warn. Note the pick at INFO and advertise the parser knobs.
    logger.info(
        "No model-specific renderer matched %r. Using DefaultRenderer "
        "(apply_chat_template). Pass tool_parser=<name> or "
        "reasoning_parser=<name> to enable structured output parsing.",
        model_name or "<unnamed tokenizer>",
    )
    return RENDERER_REGISTRY["default"](tokenizer, **default_kwargs)


# ---------------------------------------------------------------------------
# Standalone helpers that work with any Renderer implementation
# ---------------------------------------------------------------------------


def build_training_sample(
    renderer: Renderer,
    messages: list[Message],
    *,
    role_to_mask: Callable[[Message], bool],
    tools: list[ToolSpec] | None = None,
) -> tuple[list[int], list[bool]]:
    """Build (token_ids, loss_mask) for supervised training.

    Single render() call + message_indices → per-token mask.
    Replaces build_incremental_token_mask (O(N) renders → O(1)).
    """
    rendered = renderer.render(messages, tools=tools)
    loss_mask: list[bool] = []
    for msg_idx in rendered.message_indices:
        if msg_idx < 0:
            loss_mask.append(False)
        else:
            loss_mask.append(role_to_mask(messages[msg_idx]))
    return rendered.token_ids, loss_mask


def _common_prefix_len(a: list[int], b: list[int]) -> int:
    max_len = min(len(a), len(b))
    for idx in range(max_len):
        if a[idx] != b[idx]:
            return idx
    return max_len


def trim_to_turn_close(
    previous_prompt_ids: list[int],
    previous_completion_ids: list[int],
    close_token_ids: set[int],
    *,
    synthesize_close: int | None = None,
) -> list[int] | None:
    """Return the longest prefix of ``prev_prompt + prev_completion`` that
    ends at a turn-close token, or ``None`` if none exists and
    ``synthesize_close`` is not provided.

    Scans only within ``prev_completion_ids`` — a close token in
    ``prev_prompt_ids`` is structural template scaffolding, not a turn
    boundary the current step's completion produced.

    When ``prev_completion_ids`` has no close token, the prior turn was
    truncated at max_tokens. The caller opts in to synthesising the
    canonical close by passing ``synthesize_close`` (its token id).
    Otherwise the caller falls back to a fresh re-render.

    Hand-coded renderers pass this helper a set they know describes their
    turn boundaries. DefaultRenderer can't know its template's close, so
    it doesn't call this — it returns ``None`` from ``bridge_to_next_turn``
    unconditionally.
    """
    previous_ids = list(previous_prompt_ids) + list(previous_completion_ids)
    for idx in range(len(previous_ids) - 1, len(previous_prompt_ids) - 1, -1):
        if previous_ids[idx] in close_token_ids:
            return previous_ids[: idx + 1]
    if synthesize_close is None:
        return None
    previous_ids.append(synthesize_close)
    return previous_ids


def reject_assistant_in_extension(new_messages: list[Message]) -> bool:
    """Return True if any message in ``new_messages`` is an assistant turn.

    Bridges refuse to re-tokenize assistant content because it would
    replace model-sampled tokens with canonical template text — violating
    the contract that sampled tokens land in training exactly as emitted.
    """
    return any(m.get("role") == "assistant" for m in new_messages)


def build_trajectory_step(
    renderer: Renderer,
    prompt_messages: list[Message],
    completion_messages: list[Message],
    *,
    tools: list[ToolSpec] | None = None,
) -> dict[str, Any]:
    """Build prompt_ids / completion_ids / masks for a trajectory step.

    Uses common_prefix_len to find the split point because generation prompts
    may diverge from the full sequence at token boundaries (e.g., ``\\n`` vs
    ``\\n\\n`` when thinking content is empty in Qwen3.5).
    """
    has_completion = len(completion_messages) > 0
    prompt_ids = renderer.render_ids(
        prompt_messages, tools=tools, add_generation_prompt=has_completion
    )
    full_ids = renderer.render_ids(prompt_messages + completion_messages, tools=tools)

    split_idx = _common_prefix_len(prompt_ids, full_ids)
    completion_ids = full_ids[split_idx:]

    return {
        "prompt_ids": full_ids[:split_idx],
        "prompt_mask": [False] * split_idx,
        "completion_ids": completion_ids,
        "completion_mask": [True] * len(completion_ids),
        "completion_logprobs": [0.0] * len(completion_ids),
        "routed_experts": None,
    }
