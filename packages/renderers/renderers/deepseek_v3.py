"""DeepSeek V3 Renderer — hard-coded Python mirroring the DeepSeek V3 Jinja chat template.

Special tokens use fullwidth Unicode vertical bar (｜ = U+FF5C) and underscores
rendered as ▁ (U+2581), e.g. <｜begin▁of▁sentence｜>.

Format:
    <｜begin▁of▁sentence｜>{system}<｜User｜>{user}<｜Assistant｜>{assistant}<｜end▁of▁sentence｜>

Thinking uses plain text tags <think>...</think> (NOT special tokens).
When enable_thinking=True the generation prompt prefills <think>\\n to trigger reasoning.
"""

from __future__ import annotations

import json

from transformers.tokenization_utils import PreTrainedTokenizer

from renderers.base import (
    Message,
    ParsedResponse,
    RenderedTokens,
    ToolSpec,
    reject_assistant_in_extension,
    trim_to_turn_close,
)
from renderers.parsing import parse_deepseek_v3

# Fullwidth vertical bar used in DeepSeek special token names.
_SEP = "\uff5c"  # ｜  (U+FF5C)
# Fullwidth underscore substitute used in DeepSeek special token names.
_US = "\u2581"  # ▁  (U+2581)


def _ds_token(name: str) -> str:
    """Build a DeepSeek special-token string: <｜{name}｜>."""
    return f"<{_SEP}{name}{_SEP}>"


class DeepSeekV3Renderer:
    """Deterministic message → token renderer for DeepSeek V3 models."""

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        *,
        enable_thinking: bool = True,
    ):
        self._tokenizer = tokenizer
        self._enable_thinking = enable_thinking

        # ── BOS / EOS ────────────────────────────────────────────────
        self._bos = self._get_special_token(f"begin{_US}of{_US}sentence")
        self._eos = self._get_special_token(f"end{_US}of{_US}sentence")

        # ── Role tokens ───────────────────────────────────────────────
        self._user_token = self._get_special_token("User")
        self._assistant_token = self._get_special_token("Assistant")

        # ── Tool call section tokens ──────────────────────────────────
        self._tool_calls_begin = self._get_special_token(f"tool{_US}calls{_US}begin")
        self._tool_calls_end = self._get_special_token(f"tool{_US}calls{_US}end")
        self._tool_call_begin = self._get_special_token(f"tool{_US}call{_US}begin")
        self._tool_call_end = self._get_special_token(f"tool{_US}call{_US}end")
        self._tool_sep = self._get_special_token(f"tool{_US}sep")

        # ── Tool output section tokens ────────────────────────────────
        self._tool_outputs_begin = self._get_special_token(
            f"tool{_US}outputs{_US}begin"
        )
        self._tool_outputs_end = self._get_special_token(f"tool{_US}outputs{_US}end")
        self._tool_output_begin = self._get_special_token(f"tool{_US}output{_US}begin")
        self._tool_output_end = self._get_special_token(f"tool{_US}output{_US}end")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_special_token(self, name: str) -> int:
        """Encode <｜{name}｜> and assert it maps to exactly one token."""
        token_str = _ds_token(name)
        ids = self._tokenizer.encode(token_str, add_special_tokens=False)
        assert len(ids) == 1, f"Expected single token for {token_str!r}, got {ids}"
        return ids[0]

    def _encode(self, text: str) -> list[int]:
        if not text:
            return []
        return self._tokenizer.encode(text, add_special_tokens=False)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

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

        # ── 1. BOS token ─────────────────────────────────────────────
        emit_special(self._bos, -1)

        # ── 2. Collect system messages at the start ───────────────────
        # All leading system messages are concatenated with "\n\n" and emitted
        # before the first non-system message (no role token), matching the HF
        # chat template behaviour.
        sys_parts: list[str] = []
        first_non_sys = 0
        for msg in messages:
            if msg["role"] == "system":
                content = msg.get("content") or ""
                if isinstance(content, list):
                    content = "".join(
                        p.get("text", "") for p in content if isinstance(p, dict)
                    )
                sys_parts.append(str(content))
                first_non_sys += 1
            else:
                break

        if sys_parts:
            # Attribute the concatenated system text to the first system message (index 0).
            emit_text("\n\n".join(sys_parts), 0)

        # ── 3. Render non-system messages ─────────────────────────────
        num_messages = len(messages)
        for i in range(first_non_sys, num_messages):
            msg = messages[i]
            role = msg["role"]

            if role == "system":
                # System messages after the initial block — treat as user turns.
                content = msg.get("content") or ""
                if isinstance(content, list):
                    content = "".join(
                        p.get("text", "") for p in content if isinstance(p, dict)
                    )
                emit_special(self._user_token, i)
                emit_text(str(content), i)

            elif role == "user":
                content = msg.get("content") or ""
                if isinstance(content, list):
                    content = "".join(
                        p.get("text", "")
                        if isinstance(p, dict) and p.get("type") == "text"
                        else p.get("text", "")
                        if isinstance(p, dict)
                        else ""
                        for p in content
                    )
                emit_special(self._user_token, i)
                emit_text(str(content), i)

            elif role == "assistant":
                self._render_assistant(
                    msg,
                    i,
                    messages,
                    emit_special=emit_special,
                    emit_text=emit_text,
                )

            elif role == "tool":
                self._render_tool(
                    messages,
                    i,
                    emit_special=emit_special,
                    emit_text=emit_text,
                )

        # ── 4. Generation prompt ──────────────────────────────────────
        if add_generation_prompt:
            # Don't add <｜Assistant｜> after tool outputs — content flows directly.
            last_role = messages[-1]["role"] if messages else None
            if last_role != "tool":
                emit_special(self._assistant_token, -1)
            if self._enable_thinking:
                emit_text("<think>\n", -1)

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
        return parse_deepseek_v3(
            self._tokenizer,
            token_ids,
            stop_ids={self._eos},
            tool_calls_begin_id=self._tool_calls_begin,
            tool_calls_end_id=self._tool_calls_end,
            tool_call_begin_id=self._tool_call_begin,
            tool_call_end_id=self._tool_call_end,
            tool_sep_id=self._tool_sep,
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
            synthesize_close=self._eos,
        )
        if previous_ids is None:
            return None

        ext: list[int] = []

        def emit_special(token_id: int, _msg_idx: int = -1) -> None:
            ext.append(token_id)

        def emit_text(text: str, _msg_idx: int = -1) -> None:
            ext.extend(self._encode(text))

        for i, msg in enumerate(new_messages):
            role = msg.get("role")
            content = msg.get("content") or ""
            if isinstance(content, list):
                content = "".join(
                    p.get("text", "") for p in content if isinstance(p, dict)
                )
            content = str(content)

            if role == "user":
                emit_special(self._user_token, i)
                emit_text(content, i)
            elif role == "system":
                # Post-initial system messages render as user turns.
                emit_special(self._user_token, i)
                emit_text(content, i)
            elif role == "tool":
                prev_is_tool = i > 0 and new_messages[i - 1].get("role") == "tool"
                next_is_tool = (
                    i + 1 < len(new_messages)
                    and new_messages[i + 1].get("role") == "tool"
                )
                if not prev_is_tool:
                    emit_special(self._tool_outputs_begin, i)
                emit_special(self._tool_output_begin, i)
                emit_text(content, i)
                emit_special(self._tool_output_end, i)
                if not next_is_tool:
                    emit_special(self._tool_outputs_end, i)
            else:
                return None

        # Generation prompt — skip ``<｜Assistant｜>`` when the prior new
        # message was a tool response (matches render()'s behaviour: tool
        # output flows directly into assistant content).
        last_role = new_messages[-1].get("role") if new_messages else None
        if last_role != "tool":
            emit_special(self._assistant_token, -1)
        if self._enable_thinking:
            emit_text("<think>\n", -1)

        return previous_ids + ext

    # ------------------------------------------------------------------
    # Assistant rendering
    # ------------------------------------------------------------------

    def _render_assistant(
        self,
        msg: Message,
        msg_idx: int,
        messages: list[Message],
        *,
        emit_special,
        emit_text,
    ) -> None:
        # Determine whether this message follows a tool output sequence.
        # The HF template emits <｜tool▁outputs▁end｜> before the assistant content
        # without a new <｜Assistant｜> token in that case.
        prev_is_tool = msg_idx > 0 and messages[msg_idx - 1]["role"] == "tool"

        content = msg.get("content") or ""
        # Support structured content (ThinkingPart / TextPart list).
        if isinstance(content, list):
            parts_text: list[str] = []
            for p in content:
                if not isinstance(p, dict):
                    continue
                if p.get("type") == "thinking":
                    thinking = p.get("thinking", "")
                    parts_text.append(f"<think>{thinking}</think>")
                elif p.get("type") == "text":
                    parts_text.append(p.get("text", ""))
            content = "".join(parts_text)
        # Also accept reasoning_content stored separately (OpenAI-style).
        elif isinstance(msg.get("reasoning_content"), str) and msg["reasoning_content"]:
            reasoning = msg["reasoning_content"]
            content = f"<think>{reasoning}</think>{content}"

        tool_calls = msg.get("tool_calls") or []

        if not prev_is_tool:
            emit_special(self._assistant_token, msg_idx)

        if not tool_calls:
            emit_text(content, msg_idx)
        else:
            # Emit any pre-tool-call content first.
            emit_text(content, msg_idx)

            # Tool call section.
            emit_special(self._tool_calls_begin, msg_idx)
            for tc in tool_calls:
                func = tc.get("function") or tc
                name = func.get("name", "")
                arguments = func.get("arguments", {})
                args_str = (
                    json.dumps(arguments, ensure_ascii=False)
                    if not isinstance(arguments, str)
                    else arguments
                )
                # Format: <｜tool▁call▁begin｜>function<｜tool▁sep｜>{name}\n```json\n{args}\n```<｜tool▁call▁end｜>
                # tool_sep is a special token; type ("function") and name+args are plain text.
                emit_special(self._tool_call_begin, msg_idx)
                emit_text("function", msg_idx)
                emit_special(self._tool_sep, msg_idx)
                emit_text(f"{name}\n```json\n{args_str}\n```", msg_idx)
                emit_special(self._tool_call_end, msg_idx)
            emit_special(self._tool_calls_end, msg_idx)

        emit_special(self._eos, msg_idx)

    # ------------------------------------------------------------------
    # Tool (tool-response) rendering
    # ------------------------------------------------------------------

    def _render_tool(
        self,
        messages: list[Message],
        msg_idx: int,
        *,
        emit_special,
        emit_text,
    ) -> None:
        prev_is_tool = msg_idx > 0 and messages[msg_idx - 1]["role"] == "tool"
        next_is_tool = (
            msg_idx + 1 < len(messages) and messages[msg_idx + 1]["role"] == "tool"
        )

        content = messages[msg_idx].get("content") or ""
        if isinstance(content, list):
            content = "".join(p.get("text", "") for p in content if isinstance(p, dict))

        if not prev_is_tool:
            emit_special(self._tool_outputs_begin, msg_idx)

        emit_special(self._tool_output_begin, msg_idx)
        emit_text(str(content), msg_idx)
        emit_special(self._tool_output_end, msg_idx)

        if not next_is_tool:
            emit_special(self._tool_outputs_end, msg_idx)
