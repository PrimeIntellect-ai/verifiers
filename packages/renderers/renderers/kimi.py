"""Kimi K2.5 Renderer.

Unique format:
- <|im_user|>user<|im_middle|>content<|im_end|>
- <|im_assistant|>assistant<|im_middle|><think></think>content<|im_end|>
- Tool defs: TypeScript namespace in <|im_system|>tool_declare<|im_middle|>...<|im_end|>
- Tool calls: <|tool_calls_section_begin|><|tool_call_begin|><|tool_call_argument_begin|>{json}<|tool_call_end|><|tool_calls_section_end|>
- Tool responses: <|im_system|>tool<|im_middle|>## Return of \\n{content}<|im_end|>
- Always appends generation prompt after last message
"""

from __future__ import annotations

import json
import re
from typing import Any

from transformers.tokenization_utils import PreTrainedTokenizer

from renderers.base import ParsedResponse, RenderedTokens


class KimiRenderer:
    """Deterministic message → token renderer for Kimi K2.5 models."""

    def __init__(self, tokenizer: PreTrainedTokenizer, **kwargs):
        self._tokenizer = tokenizer
        self._im_user = self._token_id("<|im_user|>")
        self._im_assistant = self._token_id("<|im_assistant|>")
        self._im_system = self._token_id("<|im_system|>")
        self._im_middle = self._token_id("<|im_middle|>")
        self._im_end = self._token_id("<|im_end|>")
        self._think = self._token_id("<think>")
        self._think_end = self._token_id("</think>")
        self._tc_section_begin = self._token_id("<|tool_calls_section_begin|>")
        self._tc_section_end = self._token_id("<|tool_calls_section_end|>")
        self._tc_begin = self._token_id("<|tool_call_begin|>")
        self._tc_end = self._token_id("<|tool_call_end|>")
        self._tc_arg_begin = self._token_id("<|tool_call_argument_begin|>")

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

    def _render_tool_definitions(self, tools: list[dict[str, Any]]) -> str:
        """Render tool definitions in Kimi's TypeScript namespace format."""
        text = "# Tools\n\n## functions\nnamespace functions {\n"
        for tool in tools:
            func = tool.get("function", tool)
            name = func.get("name", "")
            desc = func.get("description", "")
            if desc:
                text += "// " + desc + "\n"
            text += "type " + name + " = (_: {\n"
            params = func.get("parameters", {})
            if isinstance(params, dict) and "properties" in params:
                for pname, pfields in params["properties"].items():
                    pdesc = pfields.get("description", "")
                    ptype = pfields.get("type", "any")
                    if pdesc:
                        text += "  // " + pdesc + "\n"
                    text += "  " + pname + ": " + ptype + "\n"
            text += "}) => any;\n"
        text += "}\n"
        return text

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

        # Tool definitions block
        has_tools = tools and len(tools) > 0
        if has_tools:
            emit_special(self._im_system, -1)
            emit_text("tool_declare", -1)
            emit_special(self._im_middle, -1)
            emit_text(self._render_tool_definitions(tools), -1)
            emit_special(self._im_end, -1)

        # Iterate messages
        for i, msg in enumerate(messages):
            role = msg.get("role")
            content = msg.get("content")

            if role == "system":
                emit_special(self._im_system, i)
                emit_text("system", i)
                emit_special(self._im_middle, i)
                emit_text(content or "", i)
                emit_special(self._im_end, i)

            elif role == "user":
                emit_special(self._im_user, i)
                emit_text("user", i)
                emit_special(self._im_middle, i)
                emit_text(content or "", i)
                emit_special(self._im_end, i)

            elif role == "assistant":
                self._render_assistant(msg, i, emit_special=emit_special, emit_text=emit_text)

            elif role == "tool":
                emit_special(self._im_system, i)
                emit_text("tool", i)
                emit_special(self._im_middle, i)
                emit_text("## Return of \n" + (content or ""), i)
                emit_special(self._im_end, i)

        # Always append generation prompt (Kimi template always does this)
        emit_special(self._im_assistant, -1)
        emit_text("assistant", -1)
        emit_special(self._im_middle, -1)
        emit_special(self._think, -1)

        return RenderedTokens(token_ids=tokens, message_indices=indices)

    def render_ids(self, messages, *, tools=None, add_generation_prompt=False):
        return self.render(messages, tools=tools, add_generation_prompt=add_generation_prompt).token_ids

    def parse_response(self, token_ids: list[int]) -> ParsedResponse:
        text = self._tokenizer.decode(token_ids, skip_special_tokens=False)

        # Strip trailing special tokens
        for marker in ["<|im_end|>", "<|im_assistant|>"]:
            text = text.split(marker)[0]

        reasoning_content = None
        if "</think>" in text:
            before, after = text.split("</think>", 1)
            reasoning = before.replace("<think>", "").strip()
            if reasoning:
                reasoning_content = reasoning
            text = after

        tool_calls = None
        if "<|tool_calls_section_begin|>" in text:
            tc_section = text.split("<|tool_calls_section_begin|>")[1].split("<|tool_calls_section_end|>")[0]
            text = text.split("<|tool_calls_section_begin|>")[0].strip()
            tool_calls = []
            for tc_block in tc_section.split("<|tool_call_begin|>")[1:]:
                arg_text = tc_block.split("<|tool_call_argument_begin|>")[-1].split("<|tool_call_end|>")[0].strip()
                if arg_text:
                    try:
                        args = json.loads(arg_text)
                        tool_calls.append({"function": {"name": "", "arguments": args}})
                    except json.JSONDecodeError:
                        pass

        return ParsedResponse(
            content=text.strip(),
            reasoning_content=reasoning_content,
            tool_calls=tool_calls or None,
        )

    def get_stop_token_ids(self) -> list[int]:
        return [self._im_end]

    def _render_assistant(self, msg, msg_idx, *, emit_special, emit_text):
        content = msg.get("content")
        tool_calls = msg.get("tool_calls") or []

        emit_special(self._im_assistant, msg_idx)
        emit_text("assistant", msg_idx)
        emit_special(self._im_middle, msg_idx)
        emit_special(self._think, msg_idx)
        emit_special(self._think_end, msg_idx)

        if content:
            emit_text(content, msg_idx)

        if tool_calls:
            emit_special(self._tc_section_begin, msg_idx)
            for tc in tool_calls:
                func = tc.get("function") or tc
                arguments = func.get("arguments", {})
                emit_special(self._tc_begin, msg_idx)
                emit_special(self._tc_arg_begin, msg_idx)
                args_str = json.dumps(arguments, ensure_ascii=False) if isinstance(arguments, dict) else str(arguments)
                emit_text(args_str, msg_idx)
                emit_special(self._tc_end, msg_idx)
            emit_special(self._tc_section_end, msg_idx)

        emit_special(self._im_end, msg_idx)
