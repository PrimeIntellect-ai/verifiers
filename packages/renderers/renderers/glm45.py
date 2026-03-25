"""GLM-4.5 Air Renderer — hard-coded Python mirroring the GLM-4.5 Jinja chat template.

Key differences from GLM-5:
- \\n after every role marker (<|user|>\\n, <|assistant|>\\n)
- <think></think>\\n separator (vs bare </think> in GLM-5)
- Tool calls have \\n between arg tags
- Thinking disabled via /nothink appended to user content
- Gen prompt (thinking=True): just <|assistant|> (no <think>)
"""

from __future__ import annotations

import json
import re
from typing import Any

from transformers.tokenization_utils import PreTrainedTokenizer

from renderers.base import ParsedResponse, RenderedTokens

_TOOLS_HEADER = (
    "\n# Tools\n\n"
    "You may call one or more functions to assist with the user query.\n\n"
    "You are provided with function signatures within <tools></tools> XML tags:\n"
    "<tools>\n"
)

_TOOLS_FOOTER = (
    "</tools>\n\n"
    "For each function call, output the function name and arguments "
    "within the following XML format:\n"
    "<tool_call>{function-name}\n"
    "<arg_key>{arg-key-1}</arg_key>\n"
    "<arg_value>{arg-value-1}</arg_value>\n"
    "<arg_key>{arg-key-2}</arg_key>\n"
    "<arg_value>{arg-value-2}</arg_value>\n"
    "...\n"
    "</tool_call>"
)


class GLM45Renderer:
    """Deterministic message → token renderer for GLM-4.5 Air models."""

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        *,
        enable_thinking: bool = True,
    ):
        self._tokenizer = tokenizer
        self._enable_thinking = enable_thinking

        self._gmask = self._token_id("[gMASK]")
        self._sop = self._token_id("<sop>")
        self._system = self._token_id("<|system|>")
        self._user = self._token_id("<|user|>")
        self._assistant = self._token_id("<|assistant|>")
        self._observation = self._token_id("<|observation|>")
        self._endoftext = self._token_id("<|endoftext|>")
        self._think = self._token_id("<think>")
        self._think_end = self._token_id("</think>")
        self._tool_call_tok = self._token_id("<tool_call>")
        self._tool_call_end_tok = self._token_id("</tool_call>")
        self._arg_key = self._token_id("<arg_key>")
        self._arg_key_end = self._token_id("</arg_key>")
        self._arg_value = self._token_id("<arg_value>")
        self._arg_value_end = self._token_id("</arg_value>")

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

    @staticmethod
    def _last_user_index(messages: list[dict[str, Any]]) -> int:
        for i in range(len(messages) - 1, -1, -1):
            if messages[i].get("role") == "user":
                return i
        return -1

    def render(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
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

        # ── Prefix ──────────────────────────────────────────────────
        emit_special(self._gmask, -1)
        emit_special(self._sop, -1)

        # ── Tools in system prompt ──────────────────────────────────
        if tools:
            emit_special(self._system, -1)
            tool_text = _TOOLS_HEADER
            for tool in tools:
                tool_text += json.dumps(tool, ensure_ascii=False) + "\n"
            tool_text += _TOOLS_FOOTER
            emit_text(tool_text, -1)

        # ── Compute last_user_index ─────────────────────────────────
        last_ui = self._last_user_index(messages)

        # ── Iterate messages ────────────────────────────────────────
        for i, msg in enumerate(messages):
            role = msg.get("role")
            content = self._visible_text(msg.get("content"))

            if role == "system":
                emit_special(self._system, i)
                emit_text("\n" + content, i)

            elif role == "user":
                emit_special(self._user, i)
                user_text = "\n" + content
                if not self._enable_thinking and not content.endswith("/nothink"):
                    user_text += "/nothink"
                emit_text(user_text, i)

            elif role == "assistant":
                self._render_assistant(msg, i, content, last_ui, emit_special=emit_special, emit_text=emit_text)

            elif role == "tool":
                self._render_tool(messages, i, content, emit_special=emit_special, emit_text=emit_text)

        # ── Generation prompt ───────────────────────────────────────
        if add_generation_prompt:
            emit_special(self._assistant, -1)
            if not self._enable_thinking:
                emit_text("\n", -1)
                emit_special(self._think, -1)
                emit_special(self._think_end, -1)

        return RenderedTokens(token_ids=tokens, message_indices=indices)

    def render_ids(self, messages, *, tools=None, add_generation_prompt=False):
        return self.render(messages, tools=tools, add_generation_prompt=add_generation_prompt).token_ids

    def parse_response(self, token_ids: list[int]) -> ParsedResponse:
        text = self._tokenizer.decode(token_ids, skip_special_tokens=False)
        for marker in ["<|endoftext|>", "<|user|>", "<|observation|>"]:
            text = text.split(marker)[0]

        reasoning_content = None
        if "</think>" in text:
            before, after = text.split("</think>", 1)
            if "<think>" in before:
                before = before.split("<think>")[-1]
            reasoning = before.strip()
            if reasoning:
                reasoning_content = reasoning
            text = after

        tool_calls = None
        if "<tool_call>" in text:
            tool_calls = []
            parts = text.split("<tool_call>")
            text = parts[0].strip()
            for tc_block in parts[1:]:
                tc_text = tc_block.split("</tool_call>")[0]
                name = tc_text.split("<arg_key>")[0].strip()
                arguments = {}
                for m in re.finditer(r"<arg_key>(.*?)</arg_key>\s*<arg_value>(.*?)</arg_value>", tc_text, re.DOTALL):
                    arg_name = m.group(1).strip()
                    arg_value = m.group(2).strip()
                    try:
                        arguments[arg_name] = json.loads(arg_value)
                    except (json.JSONDecodeError, ValueError):
                        arguments[arg_name] = arg_value
                tool_calls.append({"function": {"name": name, "arguments": arguments}})

        return ParsedResponse(
            content=text.strip(),
            reasoning_content=reasoning_content,
            tool_calls=tool_calls or None,
        )

    def get_stop_token_ids(self) -> list[int]:
        return [self._endoftext, self._user, self._observation]

    def _render_assistant(self, msg, msg_idx, content, last_user_index, *, emit_special, emit_text):
        reasoning_content = ""
        if isinstance(msg.get("reasoning_content"), str):
            reasoning_content = msg["reasoning_content"]
        elif "</think>" in content:
            before, after = content.split("</think>", 1)
            if "<think>" in before:
                reasoning_content = before.split("<think>")[-1].lstrip("\n")
            else:
                reasoning_content = before.lstrip("\n")
            reasoning_content = reasoning_content.rstrip("\n")
            content = after.lstrip("\n")

        emit_special(self._assistant, msg_idx)

        if msg_idx > last_user_index and reasoning_content:
            emit_text("\n", msg_idx)
            emit_special(self._think, msg_idx)
            emit_text(reasoning_content.strip(), msg_idx)
            emit_special(self._think_end, msg_idx)
        else:
            emit_text("\n", msg_idx)
            emit_special(self._think, msg_idx)
            emit_special(self._think_end, msg_idx)

        # Tool calls — keep content + \n contiguous to preserve BPE merges
        tool_calls = msg.get("tool_calls") or []
        if content.strip() and tool_calls:
            emit_text("\n" + content.strip() + "\n", msg_idx)
        elif content.strip():
            emit_text("\n" + content.strip(), msg_idx)

        for tc in tool_calls:
            func = tc.get("function") or tc
            name = func.get("name", "")
            arguments = func.get("arguments", {})

            if not content.strip():
                emit_text("\n", msg_idx)
            emit_special(self._tool_call_tok, msg_idx)
            emit_text(name + "\n", msg_idx)
            if isinstance(arguments, dict):
                for arg_name, arg_value in arguments.items():
                    emit_special(self._arg_key, msg_idx)
                    emit_text(arg_name, msg_idx)
                    emit_special(self._arg_key_end, msg_idx)
                    emit_text("\n", msg_idx)
                    emit_special(self._arg_value, msg_idx)
                    if isinstance(arg_value, str):
                        emit_text(arg_value, msg_idx)
                    else:
                        emit_text(json.dumps(arg_value, ensure_ascii=False), msg_idx)
                    emit_special(self._arg_value_end, msg_idx)
                    emit_text("\n", msg_idx)
            emit_special(self._tool_call_end_tok, msg_idx)

    def _render_tool(self, messages, msg_idx, content, *, emit_special, emit_text):
        prev_is_tool = msg_idx > 0 and messages[msg_idx - 1].get("role") == "tool"

        if not prev_is_tool:
            emit_special(self._observation, msg_idx)

        emit_text("\n<tool_response>\n" + content + "\n</tool_response>", msg_idx)
