"""Qwen3.6 Renderer — mirrors the Qwen3.6 Jinja chat template.

Delta vs Qwen3.5 (template lines 100 and 122):

1. Optional ``preserve_thinking`` flag retains historical ``<think>`` blocks
   for assistant turns before the last user query. The template default
   (flag unset) matches Qwen3.5's "thinking only after last query" behaviour,
   so the renderer keeps that as the default.
2. Tool-call argument serialization changed to ``tojson`` for every non-string
   value. Bools now render as ``true``/``false`` (not ``True``/``False``) and
   ``None`` as ``null`` (not ``None``), fixing the single-turn extension-break
   mode where a boolean parameter's case drifted across a re-render.

Everything else — tool system prompt, tool-call XML structure, thinking
markers, bridge logic, parser — is identical to Qwen3.5.
"""

from __future__ import annotations

import json
from typing import Any

from renderers.qwen35 import Qwen35Renderer


class Qwen36Renderer(Qwen35Renderer):
    """Deterministic message → token renderer for Qwen3.6 models."""

    def __init__(
        self,
        tokenizer,
        *,
        enable_thinking: bool = True,
        preserve_thinking: bool = False,
    ):
        super().__init__(tokenizer, enable_thinking=enable_thinking)
        self._preserve_thinking = preserve_thinking

    def _should_render_thinking(self, msg_idx: int, last_query_index: int) -> bool:
        if self._preserve_thinking:
            return True
        return msg_idx > last_query_index

    @staticmethod
    def _render_arg_value(arg_value: Any) -> str:
        if isinstance(arg_value, str):
            return arg_value
        return json.dumps(arg_value, ensure_ascii=False)
