"""Qwen3-VL renderer (text-only).

Mirrors the Qwen3-VL chat template for text-only conversations. Image and
video content parts are not supported — the renderer pipeline does not
carry image payloads through to vLLM. Pass multimodal inputs through MITO
(server-side templating) instead.
"""

from __future__ import annotations

import json
from typing import Any

from transformers.tokenization_utils import PreTrainedTokenizer

from renderers.base import (
    Message,
    ParsedResponse,
    RenderedTokens,
    ToolSpec,
    reject_assistant_in_extension,
    trim_to_turn_close,
)
from renderers.parsing import parse_qwen3

_TOOLS_HEADER = (
    "# Tools\n\n"
    "You may call one or more functions to assist with the user query.\n\n"
    "You are provided with function signatures within <tools></tools> XML tags:\n"
    "<tools>"
)

_TOOLS_FOOTER = (
    "\n</tools>\n\n"
    "For each function call, return a json object with function name and arguments "
    "within <tool_call></tool_call> XML tags:\n"
    "<tool_call>\n"
    '{"name": <function-name>, "arguments": <args-json-object>}\n'
    "</tool_call>"
)


class Qwen3VLRenderer:
    """Deterministic message to token renderer for Qwen3-VL models (text-only)."""

    def __init__(self, tokenizer: PreTrainedTokenizer):
        self._tokenizer = tokenizer

        self._im_start = self._token_id("<|im_start|>")
        self._im_end = self._token_id("<|im_end|>")
        self._endoftext = self._token_id("<|endoftext|>")
        self._tool_call = self._token_id("<tool_call>")
        self._tool_call_end = self._token_id("</tool_call>")
        self._tool_response = self._token_id("<tool_response>")
        self._tool_response_end = self._token_id("</tool_response>")

    def _token_id(self, token: str) -> int:
        tid = self._tokenizer.convert_tokens_to_ids(token)
        assert isinstance(tid, int) and tid != self._tokenizer.unk_token_id, (
            f"Special token {token!r} not found in tokenizer vocabulary"
        )
        return tid

    def _encode(self, text: str) -> list[int]:
        if not text:
            return []
        return self._tokenizer.encode(text, add_special_tokens=False)

    @staticmethod
    def _render_text_content(content: Any) -> str:
        if content is None:
            return ""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict) and "text" in item:
                    parts.append(item["text"])
                else:
                    raise ValueError(f"Unexpected content item: {item}")
            return "".join(parts)
        raise TypeError(f"Unexpected content type: {type(content)}")

    def render(
        self,
        messages: list[Message],
        *,
        tools: list[ToolSpec] | None = None,
        add_generation_prompt: bool = False,
    ) -> RenderedTokens:
        if not messages:
            raise ValueError("No messages provided.")

        tokens: list[int] = []
        indices: list[int] = []

        def emit_ids(ids: list[int], msg_idx: int) -> None:
            tokens.extend(ids)
            indices.extend([msg_idx] * len(ids))

        def emit_special(token_id: int, msg_idx: int) -> None:
            tokens.append(token_id)
            indices.append(msg_idx)

        def emit_text(text: str, msg_idx: int) -> None:
            emit_ids(self._encode(text), msg_idx)

        first_is_system = messages[0].get("role") == "system"

        if tools:
            sys_idx = 0 if first_is_system else -1
            emit_special(self._im_start, sys_idx)
            tool_text = "system\n"
            if first_is_system:
                sys_content = self._render_text_content(messages[0].get("content"))
                tool_text += sys_content + "\n\n"
            tool_text += _TOOLS_HEADER
            for tool in tools:
                tool_text += "\n" + json.dumps(tool, ensure_ascii=False)
            tool_text += _TOOLS_FOOTER
            emit_text(tool_text, sys_idx)
            emit_special(self._im_end, sys_idx)
            emit_text("\n", sys_idx)
        elif first_is_system:
            emit_special(self._im_start, 0)
            sys_content = self._render_text_content(messages[0].get("content"))
            emit_text("system\n" + sys_content, 0)
            emit_special(self._im_end, 0)
            emit_text("\n", 0)

        for i, msg in enumerate(messages):
            role = msg["role"]

            if role == "system":
                continue

            content = self._render_text_content(msg.get("content"))

            if role == "user":
                emit_special(self._im_start, i)
                emit_text("user\n" + content, i)
                emit_special(self._im_end, i)
                emit_text("\n", i)

            elif role == "assistant":
                self._render_assistant(
                    msg, i, emit_special=emit_special, emit_text=emit_text
                )

            elif role == "tool":
                self._render_tool(
                    messages,
                    i,
                    content,
                    emit_special=emit_special,
                    emit_text=emit_text,
                )

            else:
                raise ValueError(f"Unexpected message role: {role}")

        if add_generation_prompt:
            emit_special(self._im_start, -1)
            emit_text("assistant\n", -1)

        return RenderedTokens(token_ids=tokens, message_indices=indices)

    def render_ids(
        self,
        messages: list[Message],
        *,
        tools: list[ToolSpec] | None = None,
        add_generation_prompt: bool = False,
    ) -> list[int]:
        return self.render(
            messages, tools=tools, add_generation_prompt=add_generation_prompt
        ).token_ids

    def parse_response(self, token_ids: list[int]) -> ParsedResponse:
        return parse_qwen3(
            self._tokenizer,
            token_ids,
            stop_ids={self._im_end, self._endoftext},
            tool_call_id=self._tool_call,
            tool_call_end_id=self._tool_call_end,
        )

    def get_stop_token_ids(self) -> list[int]:
        return [self._im_end, self._endoftext]

    def bridge_to_next_turn(
        self,
        previous_prompt_ids: list[int],
        previous_completion_ids: list[int],
        new_messages: list[Message],
        *,
        tools: list[ToolSpec] | None = None,
    ) -> list[int] | None:
        if (
            not previous_prompt_ids
            or not new_messages
            or reject_assistant_in_extension(new_messages)
        ):
            return None

        previous_ids = trim_to_turn_close(
            previous_prompt_ids,
            previous_completion_ids,
            {self._im_end, self._endoftext},
            synthesize_close=self._im_end,
        )
        if previous_ids is None:
            return None

        ext: list[int] = []

        def emit_special(token_id: int, _msg_idx: int = -1) -> None:
            ext.append(token_id)

        def emit_text(text: str, _msg_idx: int = -1) -> None:
            ext.extend(self._encode(text))

        emit_text("\n", -1)

        for i, msg in enumerate(new_messages):
            role = msg.get("role")
            content = self._render_text_content(msg.get("content"))
            if role == "user":
                emit_special(self._im_start, i)
                emit_text("user\n" + content, i)
                emit_special(self._im_end, i)
                emit_text("\n", i)
            elif role == "system":
                emit_special(self._im_start, i)
                emit_text("system\n" + content, i)
                emit_special(self._im_end, i)
                emit_text("\n", i)
            elif role == "tool":
                self._render_tool(
                    new_messages,
                    i,
                    content,
                    emit_special=emit_special,
                    emit_text=emit_text,
                )
            else:
                return None

        emit_special(self._im_start, -1)
        emit_text("assistant\n", -1)

        return previous_ids + ext

    def _render_assistant(
        self,
        msg: Message,
        msg_idx: int,
        *,
        emit_special,
        emit_text,
    ) -> None:
        content = self._render_text_content(msg.get("content"))
        original_content = msg.get("content")
        tool_calls = msg.get("tool_calls") or []

        emit_special(self._im_start, msg_idx)

        prefix = "assistant\n" + content
        if not tool_calls:
            emit_text(prefix, msg_idx)
        else:
            for tc_idx, tc in enumerate(tool_calls):
                if tc_idx == 0:
                    separator = "\n" if original_content else ""
                    emit_text(prefix + separator, msg_idx)
                else:
                    emit_text("\n", msg_idx)

                func = tc.get("function") or tc
                name = func.get("name", "")
                arguments = func.get("arguments", {})
                args_str = (
                    arguments
                    if isinstance(arguments, str)
                    else json.dumps(arguments, ensure_ascii=False)
                )

                emit_special(self._tool_call, msg_idx)
                emit_text(
                    '\n{"name": "' + name + '", "arguments": ' + args_str + "}\n",
                    msg_idx,
                )
                emit_special(self._tool_call_end, msg_idx)

        emit_special(self._im_end, msg_idx)
        emit_text("\n", msg_idx)

    def _render_tool(
        self,
        messages: list[Message],
        msg_idx: int,
        content: str,
        *,
        emit_special,
        emit_text,
    ) -> None:
        prev_is_tool = msg_idx > 0 and messages[msg_idx - 1]["role"] == "tool"
        next_is_tool = (
            msg_idx + 1 < len(messages) and messages[msg_idx + 1]["role"] == "tool"
        )

        if not prev_is_tool:
            emit_special(self._im_start, msg_idx)
            emit_text("user", msg_idx)

        emit_text("\n", msg_idx)
        emit_special(self._tool_response, msg_idx)
        emit_text("\n" + content + "\n", msg_idx)
        emit_special(self._tool_response_end, msg_idx)

        if not next_is_tool:
            emit_special(self._im_end, msg_idx)
            emit_text("\n", msg_idx)
