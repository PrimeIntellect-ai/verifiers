"""INTELLECT-3.1 Renderer — hard-coded Python mirroring the INTELLECT chat template.

Similar to Qwen3.5 (same <|im_start|>/<|im_end|>, same <function=name>/<parameter=name> tool calls)
but with key differences:
- Thinking: <think>reasoning</think>\\ncontent (always present, no \\n inside think block)
- Tool definitions: XML-based (<function><name>...</name><parameters>...</parameters></function>)
- Generation prompt: <|im_start|>assistant\\n<think> (no \\n after <think>)
- No last_query_index logic — thinking always present for messages with reasoning_content
"""

from __future__ import annotations

import json
import re
from typing import Any

from transformers.tokenization_utils import PreTrainedTokenizer

from renderers.base import ParsedResponse, RenderedTokens

_DEFAULT_SYSTEM_WITH_TOOLS = "You are INTELLECT-4, a helpful assistant developed by Prime Intellect, that can interact with a computer to solve tasks."

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


class IntellectRenderer:
    """Deterministic message → token renderer for INTELLECT-3.1 models."""

    def __init__(self, tokenizer: PreTrainedTokenizer, **kwargs):
        self._tokenizer = tokenizer
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

    @staticmethod
    def _render_extra_keys(d: dict[str, Any], handled: set[str]) -> str:
        """Render unhandled keys from a dict as XML, matching the Jinja render_extra_keys macro."""
        text = ""
        if not isinstance(d, dict):
            return text
        for key in d:
            if key in handled:
                continue
            val = d[key]
            if isinstance(val, (dict, list)):
                text += "\n<" + key + ">" + json.dumps(val, ensure_ascii=False) + "</" + key + ">"
            else:
                text += "\n<" + key + ">" + str(val) + "</" + key + ">"
        return text

    def _render_tool_definitions(self, tools: list[dict[str, Any]]) -> str:
        """Render tool definitions in INTELLECT's XML format."""
        text = "\n\n# Tools\n\nYou have access to the following functions:\n\n<tools>"
        for tool in tools:
            func = tool.get("function", tool)
            text += "\n<function>"
            text += "\n<name>" + func.get("name", "") + "</name>"
            if "description" in func:
                text += "\n<description>" + func["description"].strip() + "</description>"
            text += "\n<parameters>"
            params = func.get("parameters", {})
            if isinstance(params, dict) and "properties" in params:
                for pname, pfields in params["properties"].items():
                    text += "\n<parameter>"
                    text += "\n<name>" + pname + "</name>"
                    if "type" in pfields:
                        text += "\n<type>" + str(pfields["type"]) + "</type>"
                    if "description" in pfields:
                        text += "\n<description>" + pfields["description"].strip() + "</description>"
                    text += self._render_extra_keys(pfields, {"name", "type", "description"})
                    text += "\n</parameter>"
            text += self._render_extra_keys(params, {"type", "properties"})
            text += "\n</parameters>"
            text += self._render_extra_keys(func, {"type", "name", "description", "parameters"})
            text += "\n</function>"
        text += "\n</tools>"
        text += _TOOLS_INSTRUCTIONS
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

        # Extract system message
        first_is_system = messages[0].get("role") == "system"
        loop_messages = messages[1:] if first_is_system else messages

        # System block
        has_tools = tools and len(tools) > 0
        if first_is_system:
            sys_idx = 0
            emit_special(self._im_start, sys_idx)
            sys_text = "system\n" + messages[0].get("content", "")
            if has_tools:
                sys_text += self._render_tool_definitions(tools)
            emit_text(sys_text, sys_idx)
            emit_special(self._im_end, sys_idx)
            emit_text("\n", sys_idx)
        elif has_tools:
            emit_special(self._im_start, -1)
            sys_text = "system\n" + _DEFAULT_SYSTEM_WITH_TOOLS
            sys_text += self._render_tool_definitions(tools)
            emit_text(sys_text, -1)
            emit_special(self._im_end, -1)
            emit_text("\n", -1)

        # Iterate messages
        for ci, msg in enumerate(loop_messages):
            orig_idx = ci + (1 if first_is_system else 0)
            role = msg.get("role")

            if role == "user" or role == "system":
                emit_special(self._im_start, orig_idx)
                emit_text(role + "\n" + (msg.get("content") or ""), orig_idx)
                emit_special(self._im_end, orig_idx)
                emit_text("\n", orig_idx)

            elif role == "assistant":
                self._render_assistant(msg, orig_idx, emit_special=emit_special, emit_text=emit_text)

            elif role == "tool":
                self._render_tool(loop_messages, ci, orig_idx, emit_special=emit_special, emit_text=emit_text)

        if add_generation_prompt:
            emit_special(self._im_start, -1)
            emit_text("assistant\n", -1)
            emit_special(self._think, -1)

        return RenderedTokens(token_ids=tokens, message_indices=indices)

    def render_ids(self, messages, *, tools=None, add_generation_prompt=False):
        return self.render(messages, tools=tools, add_generation_prompt=add_generation_prompt).token_ids

    def parse_response(self, token_ids: list[int]) -> ParsedResponse:
        text = self._tokenizer.decode(token_ids, skip_special_tokens=False)
        for marker in ["<|im_end|>", "<|endoftext|>"]:
            text = text.split(marker)[0]

        reasoning_content = None
        if "</think>" in text:
            before, after = text.split("</think>", 1)
            if "<think>" in before:
                reasoning_content = before.split("<think>")[-1].strip()
            else:
                reasoning_content = before.strip()
            text = after.lstrip("\n")

        tool_calls = None
        if "<tool_call>" in text:
            tool_calls = []
            parts = text.split("<tool_call>")
            text = parts[0].strip()
            for tc_block in parts[1:]:
                tc_text = tc_block.split("</tool_call>")[0]
                name_match = re.search(r"<function=([^>]+)>", tc_text)
                if not name_match:
                    continue
                name = name_match.group(1)
                arguments = {}
                for param_match in re.finditer(
                    r"<parameter=([^>]+)>\n?(.*?)\n?</parameter>", tc_text, re.DOTALL
                ):
                    arg_name = param_match.group(1)
                    arg_value = param_match.group(2).strip()
                    try:
                        arguments[arg_name] = json.loads(arg_value)
                    except (json.JSONDecodeError, ValueError):
                        arguments[arg_name] = arg_value
                tool_calls.append({"function": {"name": name, "arguments": arguments}})

        return ParsedResponse(
            content=text.strip(),
            reasoning_content=reasoning_content if reasoning_content else None,
            tool_calls=tool_calls or None,
        )

    def get_stop_token_ids(self) -> list[int]:
        return [self._im_end, self._endoftext]

    def _render_assistant(self, msg, msg_idx, *, emit_special, emit_text):
        content = msg.get("content") or ""
        has_tool_calls = msg.get("tool_calls") and len(msg["tool_calls"]) > 0
        has_reasoning = "reasoning_content" in msg

        emit_special(self._im_start, msg_idx)

        # Build text before tool calls, keeping \n contiguous with content for BPE
        tool_calls = msg.get("tool_calls") or []

        if has_reasoning:
            reasoning = (msg.get("reasoning_content") or "").strip()
            emit_text("assistant\n", msg_idx)
            if reasoning:
                emit_special(self._think, msg_idx)
                emit_text(reasoning, msg_idx)
                emit_special(self._think_end, msg_idx)
            else:
                emit_special(self._think, msg_idx)
                emit_special(self._think_end, msg_idx)
            after = "\n" + content.strip() if content.strip() else ""
        else:
            after = "assistant\n" + content.strip() if content.strip() else "assistant\n"

        if tool_calls:
            # \n before <tool_call> must be contiguous with preceding text
            separator = "\n" if (content.strip() or not has_reasoning) else ""
            emit_text(after + separator, msg_idx)
        else:
            emit_text(after, msg_idx)

        for tc in tool_calls:
            func = tc.get("function") or tc
            name = func.get("name", "")
            arguments = func.get("arguments", {})

            emit_text("\n", msg_idx)
            emit_special(self._tool_call, msg_idx)
            emit_text("\n<function=" + name + ">\n", msg_idx)
            if isinstance(arguments, dict):
                for arg_name, arg_value in arguments.items():
                    if isinstance(arg_value, (dict, list)):
                        value_str = json.dumps(arg_value, ensure_ascii=False)
                    else:
                        value_str = str(arg_value)
                    emit_text("<parameter=" + arg_name + ">\n" + value_str + "\n</parameter>\n", msg_idx)
            emit_text("</function>\n", msg_idx)
            emit_special(self._tool_call_end, msg_idx)

        emit_special(self._im_end, msg_idx)
        emit_text("\n", msg_idx)

    def _render_tool(self, loop_messages, ci, msg_idx, *, emit_special, emit_text):
        msg = loop_messages[ci]
        content = msg.get("content") or ""
        prev_is_tool = ci > 0 and loop_messages[ci - 1].get("role") == "tool"
        next_is_tool = ci + 1 < len(loop_messages) and loop_messages[ci + 1].get("role") == "tool"

        if not prev_is_tool:
            emit_special(self._im_start, msg_idx)
            emit_text("user\n", msg_idx)

        emit_special(self._tool_response, msg_idx)
        emit_text("\n" + content + "\n", msg_idx)
        emit_special(self._tool_response_end, msg_idx)
        emit_text("\n", msg_idx)

        if not next_is_tool:
            emit_special(self._im_end, msg_idx)
            emit_text("\n", msg_idx)
