"""Default Renderer — falls back to tokenizer.apply_chat_template() for unsupported models.

This is the escape hatch: works with any model that has a Jinja chat template,
but doesn't provide message_indices (so build_supervised_sample uses incremental
rendering) and parse_response is basic text extraction.
"""

from __future__ import annotations

import json
import re
from typing import Any

from transformers.tokenization_utils import PreTrainedTokenizer

from renderers.base import ParsedResponse, RenderedTokens


class DefaultRenderer:
    """Fallback renderer using tokenizer.apply_chat_template().

    Works with any model but is slower (Jinja) and doesn't track per-token
    message attribution. Use a model-specific renderer when available.
    """

    def __init__(self, tokenizer: PreTrainedTokenizer, **chat_template_kwargs):
        self._tokenizer = tokenizer
        self._chat_template_kwargs = chat_template_kwargs

    def render(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        add_generation_prompt: bool = False,
    ) -> RenderedTokens:
        # Incremental rendering to get per-token message attribution
        token_ids: list[int] = []
        message_indices: list[int] = []
        prev_len = 0

        for idx, message in enumerate(messages):
            cur_ids = self._apply(messages[: idx + 1], tools=tools)
            new_tokens = cur_ids[prev_len:]
            token_ids = cur_ids
            message_indices.extend([idx] * len(new_tokens))
            prev_len = len(cur_ids)

        if add_generation_prompt:
            full_ids = self._apply(messages, tools=tools, add_generation_prompt=True)
            gen_tokens = full_ids[prev_len:]
            token_ids = full_ids
            message_indices.extend([-1] * len(gen_tokens))

        return RenderedTokens(token_ids=token_ids, message_indices=message_indices)

    def _apply(self, messages, *, tools=None, add_generation_prompt=False) -> list[int]:
        kwargs = dict(self._chat_template_kwargs)
        kwargs["add_generation_prompt"] = add_generation_prompt
        kwargs["tokenize"] = True
        if tools is not None:
            kwargs["tools"] = tools
        kwargs["return_dict"] = False
        result = self._tokenizer.apply_chat_template(messages, **kwargs)
        return list(result)

    def render_ids(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        add_generation_prompt: bool = False,
    ) -> list[int]:
        return self._apply(messages, tools=tools, add_generation_prompt=add_generation_prompt)

    def parse_response(self, token_ids: list[int]) -> ParsedResponse:
        text = self._tokenizer.decode(token_ids, skip_special_tokens=True)
        # Basic thinking extraction
        reasoning_content = None
        if "</think>" in text:
            before, after = text.split("</think>", 1)
            if "<think>" in before:
                reasoning_content = before.split("<think>")[-1].strip()
            else:
                reasoning_content = before.strip()
            text = after.strip()

        return ParsedResponse(
            content=text.strip(),
            reasoning_content=reasoning_content if reasoning_content else None,
            tool_calls=None,
        )

    def get_stop_token_ids(self) -> list[int]:
        stop_ids = []
        if self._tokenizer.eos_token_id is not None:
            stop_ids.append(self._tokenizer.eos_token_id)
        return stop_ids
