from __future__ import annotations

import queue
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol, runtime_checkable


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


@runtime_checkable
class Renderer(Protocol):
    """Owns message ↔ token conversion for a specific model family."""

    def render(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        add_generation_prompt: bool = False,
    ) -> RenderedTokens:
        """Render messages to token IDs with per-token message attribution."""
        ...

    def render_ids(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
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
    """

    def __init__(self, factory: Callable[[], Renderer], size: int):
        self._factory = factory
        self._pool: queue.Queue[Renderer] = queue.Queue(maxsize=size)
        for _ in range(size):
            self._pool.put(factory())

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

# Maps model name prefixes to renderer names. Checked in order.
MODEL_RENDERER_MAP: dict[str, str] = {
    "Qwen/Qwen3.5": "qwen3.5",
    "Qwen/Qwen3": "qwen3",
    "PrimeIntellect/Qwen3": "qwen3",
    "zai-org/GLM-5": "glm5",
    "zai-org/GLM-4.7": "glm5",
    "THUDM/GLM-4.5": "glm4.5",
    "MiniMaxAI/MiniMax-M2": "minimax-m2",
}


def _populate_registry():
    if RENDERER_REGISTRY:
        return
    from renderers.default import DefaultRenderer
    from renderers.glm45 import GLM45Renderer
    from renderers.glm5 import GLM5Renderer
    from renderers.minimax_m2 import MiniMaxM2Renderer
    from renderers.qwen3 import Qwen3Renderer
    from renderers.qwen35 import Qwen35Renderer

    RENDERER_REGISTRY.update(
        {
            "default": DefaultRenderer,
            "qwen3": Qwen3Renderer,
            "qwen3.5": Qwen35Renderer,
            "glm5": GLM5Renderer,
            "glm4.5": GLM45Renderer,
            "minimax-m2": MiniMaxM2Renderer,
        }
    )


def create_renderer(tokenizer, renderer: str = "auto") -> Renderer:
    """Create a Renderer by name, or auto-detect from the tokenizer's model name.

    Args:
        tokenizer: HuggingFace tokenizer instance.
        renderer: Renderer name ('qwen3', 'qwen3.5', 'glm5', 'glm4.5', 'minimax-m2',
                  'intellect', 'default') or 'auto' to detect from model name.
    """
    _populate_registry()

    if renderer != "auto":
        cls = RENDERER_REGISTRY.get(renderer)
        if cls is None:
            raise ValueError(
                f"Unknown renderer {renderer!r}. Available: {', '.join(sorted(RENDERER_REGISTRY))}"
            )
        return cls(tokenizer)

    # Auto-detect from model name
    model_name = getattr(tokenizer, "name_or_path", "")
    for prefix, renderer_name in MODEL_RENDERER_MAP.items():
        if model_name.startswith(prefix):
            return RENDERER_REGISTRY[renderer_name](tokenizer)

    # No match — fall back to default (apply_chat_template)
    return RENDERER_REGISTRY["default"](tokenizer)


# ---------------------------------------------------------------------------
# Standalone helpers that work with any Renderer implementation
# ---------------------------------------------------------------------------


def build_supervised_sample(
    renderer: Renderer,
    messages: list[dict[str, Any]],
    *,
    role_to_mask: Callable[[dict[str, Any]], bool],
    tools: list[dict[str, Any]] | None = None,
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


def build_trajectory_step(
    renderer: Renderer,
    prompt_messages: list[dict[str, Any]],
    completion_messages: list[dict[str, Any]],
    *,
    tools: list[dict[str, Any]] | None = None,
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
