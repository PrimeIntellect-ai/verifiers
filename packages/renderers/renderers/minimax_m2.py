"""MiniMax M2.5 Renderer — hard-coded Python mirroring the MiniMax M2.5 Jinja chat template.

Unique characteristics:
- Token format: ]~!b[ (BOS), ]~b] (role prefix), [e~[ (EOS)
- Role "assistant" rendered as "ai"
- Default system message injected if none provided
- Tool calls use <minimax:tool_call>/<invoke>/<parameter> XML format
- Tool responses wrapped in <response> tags (regular text, not special tokens)
- Thinking only for assistant messages after last user turn
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
from renderers.parsing import parse_minimax

_DEFAULT_SYSTEM = (
    "You are a helpful assistant. Your name is MiniMax-M2.5 and is built by MiniMax."
)

_TOOLS_HEADER = (
    "\n\n# Tools\n"
    "You may call one or more tools to assist with the user query.\n"
    "Here are the tools available in JSONSchema format:\n"
    "\n<tools>\n"
)

_TOOLS_FOOTER_PREFIX = "</tools>\n\n"

_TOOLS_INSTRUCTIONS = (
    "When making tool calls, use XML format to invoke tools and pass parameters:\n"
    "\n<minimax:tool_call>\n"
    '<invoke name="tool-name-1">\n'
    '<parameter name="param-key-1">param-value-1</parameter>\n'
    '<parameter name="param-key-2">param-value-2</parameter>\n'
    "...\n"
    "</invoke>\n"
    "</minimax:tool_call>"
)


class MiniMaxM2Renderer:
    """Deterministic message → token renderer for MiniMax M2 / M2.5 models."""

    synthesize_close_on_truncation = True

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        *,
        default_system: str = _DEFAULT_SYSTEM,
    ):
        self._tokenizer = tokenizer
        self._default_system = default_system

        self._bos = self._token_id("]~!b[")
        self._role = self._token_id("]~b]")
        self._eos = self._token_id("[e~[")
        self._think = self._token_id("<think>")
        self._think_end = self._token_id("</think>")
        self._tool_call_tok = self._token_id("<minimax:tool_call>")
        self._tool_call_end_tok = self._token_id("</minimax:tool_call>")

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
    def _visible_text(content: Any) -> str:
        if content is None:
            return "None"
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict) and item.get("type") == "text":
                    parts.append(item.get("text", ""))
            return "".join(parts)
        return str(content)

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

        def emit_special(token_id: int, msg_idx: int) -> None:
            tokens.append(token_id)
            indices.append(msg_idx)

        def emit_text(text: str, msg_idx: int) -> None:
            ids = self._encode(text)
            tokens.extend(ids)
            indices.extend([msg_idx] * len(ids))

        # ── Extract system message ──────────────────────────────────
        first_is_system = messages[0].get("role") == "system"
        sys_idx = 0 if first_is_system else -1
        conversation: list[Message] = messages[1:] if first_is_system else messages

        # ── System block (always present) ───────────────────────────
        emit_special(self._bos, sys_idx)
        emit_special(self._role, sys_idx)

        sys_content = (
            self._visible_text(messages[0].get("content")) if first_is_system else ""
        )
        system_text = "system\n" + (sys_content or self._default_system)

        if tools:
            system_text += _TOOLS_HEADER
            for tool in tools:
                func = tool.get("function", tool)
                system_text += (
                    "<tool>" + json.dumps(func, ensure_ascii=False) + "</tool>\n"
                )
            system_text += _TOOLS_FOOTER_PREFIX
            system_text += _TOOLS_INSTRUCTIONS

        emit_text(system_text, sys_idx)
        emit_special(self._eos, sys_idx)
        emit_text("\n", sys_idx)

        # ── Compute last_user_index (relative to conversation) ──────
        last_ui = -1
        for ci, msg in enumerate(conversation):
            if msg.get("role") == "user":
                last_ui = ci

        # ── Iterate conversation messages ───────────────────────────
        for ci, msg in enumerate(conversation):
            role = msg["role"]
            # Map back to original message index for attribution
            orig_idx = ci + (1 if first_is_system else 0)

            if role == "user":
                emit_special(self._role, orig_idx)
                emit_text("user\n" + self._visible_text(msg.get("content")), orig_idx)
                emit_special(self._eos, orig_idx)
                emit_text("\n", orig_idx)

            elif role == "assistant":
                self._render_assistant(
                    msg,
                    orig_idx,
                    ci,
                    last_ui,
                    emit_special=emit_special,
                    emit_text=emit_text,
                )

            elif role == "tool":
                self._render_tool(
                    conversation,
                    ci,
                    orig_idx,
                    msg,
                    emit_special=emit_special,
                    emit_text=emit_text,
                )

        # ── Generation prompt ───────────────────────────────────────
        if add_generation_prompt:
            emit_special(self._role, -1)
            emit_text("ai\n", -1)
            emit_special(self._think, -1)
            emit_text("\n", -1)

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
        return parse_minimax(
            self._tokenizer,
            token_ids,
            stop_ids={self._eos},
            think_id=self._think,
            think_end_id=self._think_end,
            tool_call_id=self._tool_call_tok,
            tool_call_end_id=self._tool_call_end_tok,
        )

    def get_stop_token_ids(self) -> list[int]:
        return [self._eos]

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
            {self._eos},
            synthesize_close=(
                self._eos if self.synthesize_close_on_truncation else None
            ),
        )
        if previous_ids is None:
            return None

        ext: list[int] = []

        def emit_special(token_id: int, _msg_idx: int = -1) -> None:
            ext.append(token_id)

        def emit_text(text: str, _msg_idx: int = -1) -> None:
            ext.extend(self._encode(text))

        # Trailing ``\n`` after the ``[e~[`` turn close — see ``render()``.
        emit_text("\n", -1)

        for i, msg in enumerate(new_messages):
            role = msg.get("role")
            content = self._visible_text(msg.get("content"))
            if role == "user":
                emit_special(self._role, i)
                emit_text("user\n" + content, i)
                emit_special(self._eos, i)
                emit_text("\n", i)
            elif role == "system":
                emit_special(self._role, i)
                emit_text("system\n" + content, i)
                emit_special(self._eos, i)
                emit_text("\n", i)
            elif role == "tool":
                self._render_tool(
                    new_messages,
                    i,
                    i,
                    msg,
                    emit_special=emit_special,
                    emit_text=emit_text,
                )
            else:
                return None

        # Generation prompt.
        emit_special(self._role, -1)
        emit_text("ai\n", -1)
        emit_special(self._think, -1)
        emit_text("\n", -1)

        return previous_ids + ext

    def _render_assistant(
        self, msg, orig_idx, conv_idx, last_user_index, *, emit_special, emit_text
    ):
        content = self._visible_text(msg.get("content"))

        reasoning_content = ""
        if isinstance(msg.get("reasoning_content"), str):
            reasoning_content = msg["reasoning_content"]
        elif "</think>" in content:
            before, after = content.split("</think>", 1)
            if "<think>" in before:
                reasoning_content = before.split("<think>")[-1].strip("\n")
            else:
                reasoning_content = before.strip("\n")
            content = after.strip("\n")

        emit_special(self._role, orig_idx)

        # Build the full text between ]~b]ai and [e~[ with special tokens
        # interspersed. Keep text segments contiguous to preserve BPE merges.
        tool_calls = msg.get("tool_calls") or []

        if reasoning_content and conv_idx > last_user_index:
            emit_text("ai\n", orig_idx)
            emit_special(self._think, orig_idx)
            emit_text("\n" + reasoning_content + "\n", orig_idx)
            emit_special(self._think_end, orig_idx)
            # \n\n + content must be contiguous for BPE
            after_think = "\n\n" + content if content else "\n\n"
        else:
            after_think = "ai\n" + content if content else "ai\n"

        if tool_calls:
            # \n before <minimax:tool_call> must be contiguous with preceding text
            emit_text(after_think + "\n", orig_idx)
            emit_special(self._tool_call_tok, orig_idx)

            invoke_block = "\n"
            for tc in tool_calls:
                func = tc.get("function") or tc
                name = func.get("name", "")
                arguments = func.get("arguments", {})

                invoke_block += '<invoke name="' + name + '">\n'
                # OpenAI canonical form: arguments is a JSON string. Parse it so the
                # per-argument rendering below still works.
                if isinstance(arguments, str):
                    try:
                        arguments = json.loads(arguments)
                    except json.JSONDecodeError:
                        arguments = {}
                if isinstance(arguments, dict):
                    for arg_name, arg_value in arguments.items():
                        val_str = (
                            arg_value
                            if isinstance(arg_value, str)
                            else json.dumps(arg_value, ensure_ascii=False)
                        )
                        invoke_block += (
                            '<parameter name="'
                            + arg_name
                            + '">'
                            + val_str
                            + "</parameter>\n"
                        )
                invoke_block += "</invoke>\n"

            emit_text(invoke_block, orig_idx)
            emit_special(self._tool_call_end_tok, orig_idx)
        else:
            emit_text(after_think, orig_idx)

        emit_special(self._eos, orig_idx)
        emit_text("\n", orig_idx)

    def _render_tool(
        self,
        conversation: list[Message],
        conv_idx: int,
        orig_idx: int,
        msg: Message,
        *,
        emit_special,
        emit_text,
    ) -> None:
        prev_is_tool = conv_idx > 0 and conversation[conv_idx - 1]["role"] == "tool"
        next_is_tool = (
            conv_idx + 1 < len(conversation)
            and conversation[conv_idx + 1]["role"] == "tool"
        )

        if not prev_is_tool:
            emit_special(self._role, orig_idx)
            emit_text("tool", orig_idx)

        content = self._visible_text(msg.get("content"))
        # Leading ``\n`` before ``<response>`` only on the first of a
        # consecutive run — subsequent ones piggyback on the trailing ``\n``
        # emitted below, so BPE can merge ``</response>\n<response>``
        # through a single emit_text call instead of splitting the merge.
        prefix = "" if prev_is_tool else "\n"
        suffix = "\n" if next_is_tool else ""
        emit_text(prefix + "<response>" + content + "</response>" + suffix, orig_idx)

        if not next_is_tool:
            emit_special(self._eos, orig_idx)
            emit_text("\n", orig_idx)
