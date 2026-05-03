"""Qwen3.5 Renderer — hard-coded Python that mirrors the Qwen3.5 Jinja chat template.

Produces token-for-token identical output to tokenizer.apply_chat_template() while
also tracking which message produced each token (for per-token loss masks).
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
from renderers.parsing import parse_qwen35

# ---------------------------------------------------------------------------
# Tool system prompt constants (must match the Jinja template exactly)
# ---------------------------------------------------------------------------

_TOOLS_HEADER = "# Tools\n\nYou have access to the following functions:\n\n<tools>"

_TOOLS_FOOTER = "\n</tools>"

_TOOLS_INSTRUCTIONS = (
    "\n\nIf you choose to call a function ONLY reply in the following format with NO suffix:"
    "\n\n<tool_call>\n<function=example_function_name>\n<parameter=example_parameter_1>\nvalue_1"
    "\n</parameter>\n<parameter=example_parameter_2>\nThis is the value for the second parameter"
    "\nthat can span\nmultiple lines\n</parameter>\n</function>\n</tool_call>"
    "\n\n<IMPORTANT>\nReminder:"
    "\n- Function calls MUST follow the specified format:"
    " an inner <function=...></function> block must be nested within"
    " <tool_call></tool_call> XML tags"
    "\n- Required parameters MUST be specified"
    "\n- You may provide optional reasoning for your function call"
    " in natural language BEFORE the function call, but NOT after"
    "\n- If there is no function call available, answer the question like normal"
    " with your current knowledge and do not tell the user about function calls"
    "\n</IMPORTANT>"
)


class Qwen35Renderer:
    """Deterministic message → token renderer for Qwen3.5 models."""

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        *,
        enable_thinking: bool = True,
        keep_thinking: bool = False,
    ):
        self._tokenizer = tokenizer
        self._enable_thinking = enable_thinking
        self._keep_thinking = keep_thinking

        # Look up special token IDs from the tokenizer (not hardcoded)
        self._im_start = self._token_id("<|im_start|>")
        self._im_end = self._token_id("<|im_end|>")
        self._endoftext = self._token_id("<|endoftext|>")
        self._think = self._token_id("<think>")
        self._think_end = self._token_id("</think>")
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

    # ------------------------------------------------------------------
    # Content rendering (mirrors the render_content Jinja macro)
    # ------------------------------------------------------------------

    @staticmethod
    def _render_content(content: Any) -> str:
        """Render message content to a text string (before tokenization).

        Handles string, list of text parts, and None.
        """
        if content is None:
            return ""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict):
                    if "text" in item:
                        parts.append(item["text"])
                    else:
                        raise ValueError(f"Unexpected content item: {item}")
            return "".join(parts)
        raise TypeError(f"Unexpected content type: {type(content)}")

    # ------------------------------------------------------------------
    # last_query_index computation
    # ------------------------------------------------------------------

    @staticmethod
    def _last_query_index(messages: list[Message]) -> int:
        """Find the index of the last 'real' user query (not a tool_response wrapper).

        Returns ``len(messages)`` — an out-of-range sentinel — when no such
        query exists. Callers compare ``msg_idx > last_query_index`` to
        decide whether an assistant turn sits after the last user query
        (and so keeps its thinking block). The sentinel makes that check
        uniformly ``False``, which is the only reasonable default for
        assistant-only inputs (e.g. the bridge's dummy-assistant render).
        """
        for i in range(len(messages) - 1, -1, -1):
            msg = messages[i]
            if msg.get("role") != "user":
                continue
            content = Qwen35Renderer._render_content(msg.get("content")).strip()
            if not (
                content.startswith("<tool_response>")
                and content.endswith("</tool_response>")
            ):
                return i
        return len(messages)

    # ------------------------------------------------------------------
    # Core render method
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

        # ── 1. System message + optional tools ──────────────────────
        first_is_system = messages[0].get("role") == "system"

        if tools:
            # System message index for attribution
            sys_idx = 0 if first_is_system else -1

            emit_special(self._im_start, sys_idx)
            emit_text("system\n", sys_idx)

            # Tools header + JSON definitions
            tool_text = _TOOLS_HEADER
            for tool in tools:
                tool_text += "\n" + json.dumps(tool, ensure_ascii=False)
            tool_text += _TOOLS_FOOTER
            tool_text += _TOOLS_INSTRUCTIONS

            # Append user's system content if present
            if first_is_system:
                sys_content = self._render_content(messages[0].get("content")).strip()
                if sys_content:
                    tool_text += "\n\n" + sys_content

            emit_text(tool_text, sys_idx)
            emit_special(self._im_end, sys_idx)
            emit_text("\n", sys_idx)
        elif first_is_system:
            sys_content = self._render_content(messages[0].get("content")).strip()
            emit_special(self._im_start, 0)
            emit_text("system\n" + sys_content, 0)
            emit_special(self._im_end, 0)
            emit_text("\n", 0)

        # ── 2. Compute last_query_index ─────────────────────────────
        last_qi = self._last_query_index(messages)

        # ── 3. Iterate messages ─────────────────────────────────────
        for i, msg in enumerate(messages):
            role = msg["role"]
            content = self._render_content(msg.get("content")).strip()

            if role == "system":
                if i != 0:
                    raise ValueError("System message must be at the beginning.")
                continue  # Already handled above

            elif role == "user":
                emit_special(self._im_start, i)
                emit_text("user\n" + content, i)
                emit_special(self._im_end, i)
                emit_text("\n", i)

            elif role == "assistant":
                self._render_assistant(
                    msg,
                    i,
                    content,
                    last_qi,
                    emit_special=emit_special,
                    emit_text=emit_text,
                    emit_ids=emit_ids,
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

        # ── 4. Generation prompt ────────────────────────────────────
        if add_generation_prompt:
            emit_special(self._im_start, -1)
            emit_text("assistant\n", -1)
            if self._enable_thinking:
                emit_special(self._think, -1)
                emit_text("\n", -1)
            else:
                emit_special(self._think, -1)
                emit_text("\n\n", -1)
                emit_special(self._think_end, -1)
                emit_text("\n\n", -1)

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
        return parse_qwen35(
            self._tokenizer,
            token_ids,
            stop_ids={self._im_end, self._endoftext},
            think_id=self._think,
            think_end_id=self._think_end,
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

        def emit_ids(ids: list[int], _msg_idx: int = -1) -> None:
            ext.extend(ids)

        # Trailing ``\n`` after ``<|im_end|>`` — ``render()`` emits it as
        # part of the prior turn, but vLLM stops on ``<|im_end|>`` so the
        # ``\n`` never makes it into prev_completion.
        emit_text("\n", -1)

        for i, msg in enumerate(new_messages):
            role = msg.get("role")
            content = self._render_content(msg.get("content")).strip()
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

        # Generation prompt — matches the gen-prompt branch of ``render()``.
        emit_special(self._im_start, -1)
        emit_text("assistant\n", -1)
        if self._enable_thinking:
            emit_special(self._think, -1)
            emit_text("\n", -1)
        else:
            emit_special(self._think, -1)
            emit_text("\n\n", -1)
            emit_special(self._think_end, -1)
            emit_text("\n\n", -1)

        return previous_ids + ext

    # ------------------------------------------------------------------
    # Assistant message rendering
    # ------------------------------------------------------------------

    def _should_render_thinking(self, msg_idx: int, last_query_index: int) -> bool:
        """Whether to emit a ``<think>`` block for the assistant message at ``msg_idx``.

        Qwen3.5 only emits thinking for assistant turns that sit after the
        last real user query. ``keep_thinking=True`` overrides this so
        prior-turn thinking is retained across user turns.
        """
        if self._keep_thinking:
            return True
        return msg_idx > last_query_index

    @staticmethod
    def _render_arg_value(arg_value: Any) -> str:
        """Serialise a single tool-call argument value to text.

        Mirrors the Jinja args-value branch: dicts/lists → compact JSON;
        everything else (including bool, int, None) → ``str(...)``. That
        matches Qwen3.5's ``tojson`` for mapping/sequence / ``string`` for
        the rest. Qwen3.6 overrides this to push non-strings through JSON
        so bools round-trip as ``true``/``false`` instead of ``True``/``False``.
        """
        if isinstance(arg_value, (dict, list)):
            return json.dumps(arg_value, ensure_ascii=False)
        return str(arg_value)

    def _render_assistant(
        self,
        msg: Message,
        msg_idx: int,
        content: str,
        last_query_index: int,
        *,
        emit_special,
        emit_text,
        emit_ids,
    ) -> None:
        # Extract reasoning_content
        reasoning_content = ""
        if isinstance(msg.get("reasoning_content"), str):
            reasoning_content = msg["reasoning_content"]
        elif "</think>" in content:
            # Split on </think> to separate reasoning from content
            before_think_end, after_think_end = content.split("</think>", 1)
            # Extract text after <think> (if present)
            if "<think>" in before_think_end:
                reasoning_content = before_think_end.split("<think>")[-1].lstrip("\n")
            else:
                reasoning_content = before_think_end.lstrip("\n")
            reasoning_content = reasoning_content.rstrip("\n")
            content = after_think_end.lstrip("\n")

        reasoning_content = reasoning_content.strip()

        emit_special(self._im_start, msg_idx)

        if self._should_render_thinking(msg_idx, last_query_index):
            # Include thinking block
            emit_text("assistant\n", msg_idx)
            emit_special(self._think, msg_idx)
            emit_text("\n" + reasoning_content + "\n", msg_idx)
            emit_special(self._think_end, msg_idx)
            emit_text("\n\n" + content, msg_idx)
        else:
            emit_text("assistant\n" + content, msg_idx)

        # Tool calls
        tool_calls = msg.get("tool_calls") or []
        if tool_calls:
            for tc_idx, tc in enumerate(tool_calls):
                func = tc.get("function") or tc
                name = func.get("name", "")
                arguments = func.get("arguments", {})

                # Separator before <tool_call>
                if tc_idx == 0:
                    if content.strip():
                        emit_text("\n\n", msg_idx)
                    # else: no separator
                else:
                    emit_text("\n", msg_idx)

                emit_special(self._tool_call, msg_idx)
                emit_text("\n<function=" + name + ">\n", msg_idx)

                # Render arguments
                # OpenAI canonical form: arguments is a JSON string. Parse it so the
                # per-argument rendering below still works.
                if isinstance(arguments, str):
                    try:
                        arguments = json.loads(arguments)
                    except json.JSONDecodeError:
                        arguments = {}
                if isinstance(arguments, dict):
                    for arg_name, arg_value in arguments.items():
                        value_str = self._render_arg_value(arg_value)
                        emit_text(
                            "<parameter="
                            + arg_name
                            + ">\n"
                            + value_str
                            + "\n</parameter>\n",
                            msg_idx,
                        )

                emit_text("</function>\n", msg_idx)
                emit_special(self._tool_call_end, msg_idx)

        emit_special(self._im_end, msg_idx)
        emit_text("\n", msg_idx)

    # ------------------------------------------------------------------
    # Tool message rendering
    # ------------------------------------------------------------------

    def _render_tool(
        self,
        messages: list[Message],
        msg_idx: int,
        content: str,
        *,
        emit_special,
        emit_text,
    ) -> None:
        # Consecutive tool messages are grouped under a single <|im_start|>user block
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
