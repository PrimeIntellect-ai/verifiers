"""Default Renderer — falls back to tokenizer.apply_chat_template() for unsupported models.

This is the escape hatch: works with any model that has a Jinja chat template,
but doesn't provide message_indices (so build_supervised_sample uses incremental
rendering) and parse_response is basic text extraction unless tool/reasoning
parsers are plugged in.
"""

from __future__ import annotations

from transformers.tokenization_utils import PreTrainedTokenizer

from renderers.base import (
    Message,
    ParsedResponse,
    RenderedTokens,
    ToolSpec,
    build_incremental_prompt_ids,
)
from renderers.parsers import (
    ReasoningParser,
    ToolParser,
    get_reasoning_parser,
    get_tool_parser,
)


class DefaultRenderer:
    """Fallback renderer using tokenizer.apply_chat_template().

    Works with any model. Pass ``tool_parser`` and/or ``reasoning_parser``
    (by name, resolved against the registries in ``renderers.parsers``) to
    enable structured output extraction.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        *,
        tool_parser: str | ToolParser | None = None,
        reasoning_parser: str | ReasoningParser | None = None,
        synthesize_close_on_truncation: bool = False,
        **chat_template_kwargs,
    ):
        self._tokenizer = tokenizer
        self._chat_template_kwargs = chat_template_kwargs
        self._tool_parser = _resolve_parser(tool_parser, tokenizer, get_tool_parser)
        self._reasoning_parser = _resolve_parser(
            reasoning_parser, tokenizer, get_reasoning_parser
        )
        self.synthesize_close_on_truncation = synthesize_close_on_truncation

    @property
    def supports_tools(self) -> bool:
        return self._tool_parser is not None

    def render(
        self,
        messages: list[Message],
        *,
        tools: list[ToolSpec] | None = None,
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
        messages: list[Message],
        *,
        tools: list[ToolSpec] | None = None,
        add_generation_prompt: bool = False,
    ) -> list[int]:
        return self._apply(
            messages, tools=tools, add_generation_prompt=add_generation_prompt
        )

    def parse_response(self, token_ids: list[int]) -> ParsedResponse:
        # 1. Extract tool calls while we still have token ids (most formats
        #    use special-token delimiters, so id-level matching is reliable).
        if self._tool_parser is not None:
            content_ids, tool_calls = self._tool_parser.extract(list(token_ids))
        else:
            content_ids = list(token_ids)
            tool_calls = None

        # 2. Decode (keep special tokens so a downstream reasoning parser can
        #    still see things like <think>/</think> when they're tokens).
        text = self._tokenizer.decode(content_ids, skip_special_tokens=False)

        # 3. Extract reasoning from the decoded text. Falls back to a built-in
        #    <think>...</think> sniff so unconfigured users get the same behavior
        #    as before.
        if self._reasoning_parser is not None:
            reasoning_content, text = self._reasoning_parser.extract(text)
        else:
            reasoning_content = None
            if "</think>" in text:
                before, after = text.split("</think>", 1)
                if "<think>" in before:
                    reasoning_content = before.split("<think>", 1)[-1].strip()
                else:
                    reasoning_content = before.strip()
                text = after.strip()

        # Strip any remaining special tokens from the final content (we kept
        # them around for the reasoning parser above).
        text = _strip_special_tokens(self._tokenizer, text)

        return ParsedResponse(
            content=text.strip(),
            reasoning_content=reasoning_content if reasoning_content else None,
            tool_calls=tool_calls,
        )

    def get_stop_token_ids(self) -> list[int]:
        stop_ids = []
        if self._tokenizer.eos_token_id is not None:
            stop_ids.append(self._tokenizer.eos_token_id)
        return stop_ids

    def bridge_to_next_turn(
        self,
        previous_prompt_ids: list[int],
        previous_completion_ids: list[int],
        new_messages: list[Message],
        *,
        tools: list[ToolSpec] | None = None,
    ) -> list[int] | None:
        """Return prompt_ids for the next turn that extend prev_prompt + prev_completion.

        DefaultRenderer doesn't know its template, so it defers to the generic
        ``build_incremental_prompt_ids`` algorithm (which walks the template
        output via the dummy-assistant trick). Hand-coded renderers should
        override this with template-specific logic.
        """
        return build_incremental_prompt_ids(
            self,
            previous_prompt_ids,
            previous_completion_ids,
            new_messages,
            tools=tools,
        )


def _resolve_parser(value, tokenizer, factory):
    if value is None:
        return None
    if isinstance(value, str):
        return factory(value, tokenizer)
    return value


def _strip_special_tokens(tokenizer, text: str) -> str:
    """Remove any special-token substrings that slipped into decoded text."""
    specials = getattr(tokenizer, "all_special_tokens", None) or []
    for token in specials:
        if token and token in text:
            text = text.replace(token, "")
    return text
