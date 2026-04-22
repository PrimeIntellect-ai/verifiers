"""Nemotron 3 Renderer — hard-coded Python that mirrors the Nemotron 3 chat template.

Nemotron 3 uses the same <|im_start|>/<|im_end|> format as Qwen3.5 but differs in:

1. Tool declarations: XML format inside <tools>...</tools> (not JSON-per-line).
2. System message ordering: system prompt goes BEFORE tools block.
3. Thinking block scope: <think></think> is prepended to ALL assistant messages
   that lack thinking content (not just those after the last user query).
4. Think separator: single \\n after </think> (not \\n\\n like Qwen3.5).
5. Empty system message: always prepends an empty system message if none exists.
6. Disable-thinking generation suffix: <think></think> with no trailing newlines.
7. Tool response format: trailing newline after </tool_response>.
"""

from __future__ import annotations

import json
from typing import Any

from transformers.tokenization_utils import PreTrainedTokenizer

from renderers.base import Message, ParsedResponse, RenderedTokens, ToolSpec
from renderers.parsing import parse_qwen35

# ---------------------------------------------------------------------------
# Tool system prompt constants
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


def _render_extra_keys(obj: dict[str, Any], handled_keys: set[str]) -> list[str]:
    """Render extra dict keys as XML, mirroring the HF template's render_extra_keys macro.

    Dicts and lists are JSON-encoded; scalars are string-coerced.
    """
    lines: list[str] = []
    for key, value in obj.items():
        if key in handled_keys:
            continue
        if isinstance(value, (dict, list)):
            lines.append(f"<{key}>{json.dumps(value)}</{key}>")
        else:
            lines.append(f"<{key}>{value!s}</{key}>")
    return lines


class Nemotron3Renderer:
    """Deterministic message → token renderer for Nemotron 3 models."""

    synthesize_close_on_truncation = True

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        *,
        enable_thinking: bool = True,
    ):
        self._tokenizer = tokenizer
        self._enable_thinking = enable_thinking

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
    # Content rendering
    # ------------------------------------------------------------------

    @staticmethod
    def _render_content(content: Any) -> str:
        """Render message content to a text string (before tokenization)."""
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
                    if (
                        item.get("type") == "image"
                        or "image" in item
                        or "image_url" in item
                    ):
                        parts.append("<|vision_start|><|image_pad|><|vision_end|>")
                    elif item.get("type") == "video" or "video" in item:
                        parts.append("<|vision_start|><|video_pad|><|vision_end|>")
                    elif "text" in item:
                        parts.append(item["text"])
                    else:
                        raise ValueError(f"Unexpected content item: {item}")
            return "".join(parts)
        raise TypeError(f"Unexpected content type: {type(content)}")

    # ------------------------------------------------------------------
    # Tool declaration formatting (XML, Nemotron 3 style)
    # ------------------------------------------------------------------

    @staticmethod
    def _format_tool_declaration(tool: ToolSpec) -> str:
        """Format a single tool declaration in Nemotron 3 XML format."""
        lines = [
            "<function>",
            f"<name>{tool['name']}</name>",
        ]
        description = tool.get("description", "").strip()
        if description:
            lines.append(f"<description>{description}</description>")
        lines.append("<parameters>")
        params = tool.get("parameters") or {}
        if isinstance(params, dict) and "properties" in params:
            for param_name, param_fields in params["properties"].items():
                lines.append("<parameter>")
                lines.append(f"<name>{param_name}</name>")
                if "type" in param_fields:
                    lines.append(f"<type>{param_fields['type']!s}</type>")
                if "description" in param_fields:
                    lines.append(
                        f"<description>{param_fields['description'].strip()}</description>"
                    )
                if "enum" in param_fields:
                    lines.append(f"<enum>{json.dumps(param_fields['enum'])}</enum>")
                lines.extend(
                    _render_extra_keys(
                        param_fields, {"name", "type", "description", "enum"}
                    )
                )
                lines.append("</parameter>")
        if isinstance(params, dict):
            lines.extend(_render_extra_keys(params, {"type", "properties", "required"}))
        if isinstance(params, dict) and "required" in params:
            lines.append(f"<required>{json.dumps(params['required'])}</required>")
        lines.append("</parameters>")
        lines.extend(
            _render_extra_keys(tool, {"type", "name", "description", "parameters"})
        )
        lines.append("</function>")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Message normalization
    # ------------------------------------------------------------------

    def _normalize_messages(self, messages: list[Message]) -> list[Message]:
        """Prepend empty system message if none exists.

        Nemotron 3's HF template always outputs a system message block even
        when none is provided.
        """
        if not messages or messages[0].get("role") != "system":
            return [{"role": "system", "content": ""}] + list(messages)
        return list(messages)

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

        # Always ensure an empty system message is present.
        messages = self._normalize_messages(messages)

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
            # Nemotron 3: system prompt BEFORE tools block
            sys_idx = 0 if first_is_system else -1

            emit_special(self._im_start, sys_idx)
            emit_text("system\n", sys_idx)

            # Build system content: user's system text first, then tools
            if first_is_system:
                sys_content = self._render_content(messages[0].get("content")).strip()
            else:
                sys_content = ""

            tool_declarations = "\n".join(
                self._format_tool_declaration(t) for t in tools
            )
            tools_block = (
                _TOOLS_HEADER
                + "\n"
                + tool_declarations
                + _TOOLS_FOOTER
                + _TOOLS_INSTRUCTIONS
            )

            if sys_content:
                full_sys = sys_content + "\n\n" + tools_block
            else:
                full_sys = tools_block

            emit_text(full_sys, sys_idx)
            emit_special(self._im_end, sys_idx)
            emit_text("\n", sys_idx)

        elif first_is_system:
            sys_content = self._render_content(messages[0].get("content")).strip()
            emit_special(self._im_start, 0)
            emit_text("system\n" + sys_content, 0)
            emit_special(self._im_end, 0)
            emit_text("\n", 0)

        # ── 2. Iterate messages ─────────────────────────────────────
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

        # ── 3. Generation prompt ────────────────────────────────────
        if add_generation_prompt:
            emit_special(self._im_start, -1)
            emit_text("assistant\n", -1)
            if self._enable_thinking:
                emit_special(self._think, -1)
                emit_text("\n", -1)
            else:
                # Disable-thinking suffix: <think></think> with no trailing newlines
                emit_special(self._think, -1)
                emit_special(self._think_end, -1)

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

    # ------------------------------------------------------------------
    # Assistant message rendering
    # ------------------------------------------------------------------

    def _render_assistant(
        self,
        msg: Message,
        msg_idx: int,
        content: str,
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
            before_think_end, after_think_end = content.split("</think>", 1)
            if "<think>" in before_think_end:
                reasoning_content = before_think_end.split("<think>")[-1].lstrip("\n")
            else:
                reasoning_content = before_think_end.lstrip("\n")
            reasoning_content = reasoning_content.rstrip("\n")
            content = after_think_end.lstrip("\n")

        reasoning_content = reasoning_content.strip()

        emit_special(self._im_start, msg_idx)
        emit_text("assistant\n", msg_idx)

        # Nemotron 3: <think></think> is prepended to ALL assistant messages
        # that lack thinking content (not just those after the last user query).
        if reasoning_content:
            # Has thinking: emit full think block with single \n separator
            emit_special(self._think, msg_idx)
            emit_text("\n" + reasoning_content + "\n", msg_idx)
            emit_special(self._think_end, msg_idx)
            # Single \n separator (not \n\n like Qwen3.5)
            emit_text("\n" + content, msg_idx)
        else:
            # No thinking: prepend empty <think></think>
            emit_special(self._think, msg_idx)
            emit_special(self._think_end, msg_idx)
            emit_text(content, msg_idx)

        # Tool calls
        tool_calls = msg.get("tool_calls") or []
        if tool_calls:
            for tc_idx, tc in enumerate(tool_calls):
                func = tc.get("function") or tc
                name = func.get("name", "")
                arguments = func.get("arguments", {})

                # Single \n separator before <tool_call>
                if tc_idx == 0:
                    if content.strip():
                        emit_text("\n", msg_idx)
                    # else: no separator
                else:
                    emit_text("\n", msg_idx)

                emit_special(self._tool_call, msg_idx)
                emit_text("\n<function=" + name + ">\n", msg_idx)

                # Render arguments
                if isinstance(arguments, dict):
                    for arg_name, arg_value in arguments.items():
                        if isinstance(arg_value, (dict, list)):
                            value_str = json.dumps(arg_value, ensure_ascii=False)
                        else:
                            value_str = str(arg_value)
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
                # Trailing \n after </tool_call> (Nemotron 3 specific)
                emit_text("\n", msg_idx)

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
        # Nemotron 3: trailing \n after </tool_response>
        emit_text("\n", msg_idx)

        if not next_is_tool:
            emit_special(self._im_end, msg_idx)
            emit_text("\n", msg_idx)
