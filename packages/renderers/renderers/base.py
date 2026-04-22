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


class ImagePart(TypedDict):
    """A chunk of image content in a message (URL, data-URI, or raw bytes)."""

    type: Literal["image"]
    image: str  # URL or data URI


ContentPart = TextPart | ImagePart | ThinkingPart

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

    # Opt-in flag that ``build_incremental_prompt_ids`` reads when the prior
    # turn's completion was truncated (no stop token in completion_ids). When
    # True, the bridge appends ``get_stop_token_ids()[0]`` as a synthetic close
    # so the resulting prompt extends the prior step's tokens exactly. Default
    # False because this is only sound for renderers whose canonical close
    # token we *know* — which is true for all hand-coded renderers but not for
    # DefaultRenderer (which wraps arbitrary HF chat templates).
    synthesize_close_on_truncation: bool = False

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

# Maps model name prefixes to renderer names. Checked in order;
# longer prefixes first so "Qwen/Qwen3.5" matches before "Qwen/Qwen3".
MODEL_RENDERER_MAP: dict[str, str] = {
    "Qwen/Qwen3.5": "qwen3.5",
    "Qwen/Qwen3-VL": "qwen3_vl",
    "Qwen/Qwen3": "qwen3",
    "zai-org/GLM-5": "glm5",
    "zai-org/GLM-4.7": "glm5",
    "THUDM/GLM-4.5": "glm4.5",
    "MiniMaxAI/MiniMax-M2": "minimax-m2",
    "deepseek-ai/DeepSeek": "deepseek_v3",
    "moonshotai/Kimi-K2.5": "kimi_k25",
    "moonshotai/Kimi-K2": "kimi_k2",
    "nvidia/Llama-3": "nemotron3",
    "nvidia/Nemotron": "nemotron3",
}


def _populate_registry():
    if RENDERER_REGISTRY:
        return
    from renderers.default import DefaultRenderer
    from renderers.deepseek_v3 import DeepSeekV3Renderer
    from renderers.glm5 import GLM5Renderer
    from renderers.glm45 import GLM45Renderer
    from renderers.gpt_oss import GptOssRenderer
    from renderers.kimi_k2 import KimiK2Renderer
    from renderers.kimi_k25 import KimiK25Renderer
    from renderers.minimax_m2 import MiniMaxM2Renderer
    from renderers.nemotron3 import Nemotron3Renderer
    from renderers.qwen3 import Qwen3Renderer
    from renderers.qwen3_vl import Qwen3VLRenderer
    from renderers.qwen35 import Qwen35Renderer

    RENDERER_REGISTRY.update(
        {
            "default": DefaultRenderer,
            "qwen3": Qwen3Renderer,
            "qwen3_vl": Qwen3VLRenderer,
            "qwen3.5": Qwen35Renderer,
            "glm5": GLM5Renderer,
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
    synthesize_close_on_truncation: bool = False,
) -> RendererPool:
    """Create a RendererPool with *size* independent tokenizer copies.

    Each slot loads its own tokenizer so threads never share mutable state.
    HuggingFace fast tokenizers release the GIL during Rust encoding, so
    threads achieve real parallelism.

    ``tool_parser``, ``reasoning_parser``, and ``synthesize_close_on_truncation``
    are forwarded to ``create_renderer`` when the pool falls back to
    ``DefaultRenderer``.
    """

    def factory(
        _name=tokenizer_name_or_path,
        _renderer=renderer,
        _tool_parser=tool_parser,
        _reasoning_parser=reasoning_parser,
        _synth_close=synthesize_close_on_truncation,
    ) -> Renderer:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(_name, trust_remote_code=True)
        return create_renderer(
            tokenizer,
            renderer=_renderer,
            tool_parser=_tool_parser,
            reasoning_parser=_reasoning_parser,
            synthesize_close_on_truncation=_synth_close,
        )

    return RendererPool(factory, size=size)


def create_renderer(
    tokenizer,
    renderer: str = "auto",
    *,
    tool_parser: str | None = None,
    reasoning_parser: str | None = None,
    synthesize_close_on_truncation: bool = False,
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
        synthesize_close_on_truncation: When True, DefaultRenderer bridges over
                  vLLM-truncated turns by appending the tokenizer's EOS token
                  in place of the missing end-of-turn marker. See the package
                  README for when it's safe to enable. Only consumed by
                  DefaultRenderer; hand-coded renderers set this themselves.
    """
    _populate_registry()

    default_kwargs: dict = {}
    if tool_parser is not None:
        default_kwargs["tool_parser"] = tool_parser
    if reasoning_parser is not None:
        default_kwargs["reasoning_parser"] = reasoning_parser
    if synthesize_close_on_truncation:
        default_kwargs["synthesize_close_on_truncation"] = True

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
                "tool_parser / reasoning_parser / synthesize_close_on_truncation "
                "are only consumed by DefaultRenderer; ignoring for renderer=%r "
                "which has built-in behavior.",
                renderer,
            )
        return cls(tokenizer)

    # Auto-detect from model name
    model_name = getattr(tokenizer, "name_or_path", "")
    for prefix, renderer_name in MODEL_RENDERER_MAP.items():
        if model_name.startswith(prefix):
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


def build_supervised_sample(
    renderer: Renderer,
    messages: list[Message],
    *,
    role_to_mask: Callable[[Message], bool],
    tools: list[ToolSpec] | None = None,
    collapse_consecutive_tool_messages: bool = False,
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


def build_incremental_prompt_ids(
    renderer: Renderer,
    previous_prompt_ids: list[int],
    previous_completion_ids: list[int],
    new_messages: list[Message],
    *,
    tools: list[ToolSpec] | None = None,
) -> list[int] | None:
    """Append new environment messages to exact previous tokens.

    This mirrors the old token route's bridge trick: render a dummy assistant
    message followed by the new environment messages, then keep only the suffix
    after the dummy assistant boundary.  The sampled assistant tokens themselves
    are never re-rendered.
    """
    if not previous_prompt_ids or not previous_completion_ids or not new_messages:
        return None

    previous_ids = list(previous_prompt_ids) + list(previous_completion_ids)
    try:
        stop_token_ids_list = list(renderer.get_stop_token_ids())
    except Exception:
        stop_token_ids_list = []
    stop_token_ids = set(stop_token_ids_list)

    boundary_idx: int | None = None
    if stop_token_ids:
        for idx in range(len(previous_ids) - 1, len(previous_prompt_ids) - 1, -1):
            if previous_ids[idx] in stop_token_ids:
                boundary_idx = idx
                break

    if boundary_idx is None:
        # No stop token in previous_completion_ids — vLLM truncated the prior
        # turn at max_tokens.
        #
        # Only renderers that explicitly opt in via
        # ``synthesize_close_on_truncation = True`` will bridge over this case.
        # The opt-in is gated per-renderer because synthesizing a close token
        # that the model didn't emit is only sound when we *know* the template's
        # expected close (which we do for hand-coded renderers: Qwen3, GLM,
        # DeepSeekV3, etc. — their get_stop_token_ids()[0] is the canonical
        # end-of-turn marker by construction).
        #
        # DefaultRenderer leaves this False because it wraps an unknown HF chat
        # template; ``tokenizer.eos_token_id`` is usually the right close for
        # chatml-family fine-tunes but not universally. Returning None here
        # matches main's TITO behavior on truncation — the caller falls back to
        # a full re-render, which extends whenever BPE round-trip is stable.
        # Fragmentation rate lands at ~main's 22%, not the 100% you'd get
        # otherwise.
        #
        # The synthetic close is KL-safe for opt-in renderers: it lands in the
        # next step's prompt_ids AFTER step_N's completion_ids, so when
        # interleave_rollout merges the two steps into one sample, it becomes a
        # mask=False "context" token. The trainer's KL sum weights it by the
        # mask, so the token's logprob never enters the loss.
        synth_ok = getattr(renderer, "synthesize_close_on_truncation", False)
        if not synth_ok or not stop_token_ids_list:
            return None
        previous_ids = previous_ids + [stop_token_ids_list[0]]
        boundary_idx = len(previous_ids) - 1

    previous_ids = previous_ids[: boundary_idx + 1]
    boundary_token_id = previous_ids[-1]

    dummy_assistant: Message = {"role": "assistant", "content": "x"}

    try:
        bridge_full_ids = renderer.render_ids(
            [dummy_assistant, *new_messages],
            tools=tools,
            add_generation_prompt=True,
        )
        bridge_base_ids = renderer.render_ids(
            [dummy_assistant],
            tools=tools,
            add_generation_prompt=False,
        )
    except Exception:
        return None

    if bridge_full_ids[: len(bridge_base_ids)] != bridge_base_ids:
        return None

    gap: int | None = None
    for idx in range(len(bridge_base_ids) - 1, -1, -1):
        if bridge_base_ids[idx] == boundary_token_id:
            gap = len(bridge_base_ids) - idx - 1
            break
    if gap is None:
        return None

    bridge_ids = list(bridge_full_ids[len(bridge_base_ids) - gap :])
    if bridge_ids and bridge_ids[0] == boundary_token_id:
        bridge_ids = bridge_ids[1:]
    return previous_ids + bridge_ids


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
