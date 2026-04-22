"""Kimi K2 Renderer — hard-coded Python mirroring the Kimi K2 Jinja chat template.

Key characteristics:
- Role tokens: <|im_user|>, <|im_assistant|>, <|im_system|>
- Separator token: <|im_middle|> between role name and content
- Terminator token: <|im_end|>
- Tool calls wrapped in <|tool_calls_section_begin|>...<|tool_calls_section_end|>
  with individual calls in <|tool_call_begin|>id<|tool_call_argument_begin|>args<|tool_call_end|>
- Tool declaration messages use role="tool_declare" with <|im_system|>tool_declare<|im_middle|>
- Tool results use role="tool" with <|im_system|>{name}<|im_middle|>## Return of {id}\\n
- Thinking uses text tags <think>...</think>; historical messages strip to <think></think>
- Default system message injected if none present
"""

from __future__ import annotations

import json

from transformers.tokenization_utils import PreTrainedTokenizer

from renderers.base import Message, ParsedResponse, RenderedTokens, ToolSpec
from renderers.bridges import chatml_bridge
from renderers.parsing import parse_kimi_k2

_DEFAULT_SYSTEM = "You are Kimi, an AI assistant created by Moonshot AI."


class KimiK2Renderer:
    """Deterministic message → token renderer for Kimi K2 models."""

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        *,
        enable_thinking: bool = True,
    ):
        self._tokenizer = tokenizer
        self._enable_thinking = enable_thinking

        self._im_user = self._token_id("<|im_user|>")
        self._im_assistant = self._token_id("<|im_assistant|>")
        self._im_system = self._token_id("<|im_system|>")
        self._im_middle = self._token_id("<|im_middle|>")
        self._im_end = self._token_id("<|im_end|>")
        self._tool_calls_section_begin = self._token_id("<|tool_calls_section_begin|>")
        self._tool_calls_section_end = self._token_id("<|tool_calls_section_end|>")
        self._tool_call_begin = self._token_id("<|tool_call_begin|>")
        self._tool_call_argument_begin = self._token_id("<|tool_call_argument_begin|>")
        self._tool_call_end = self._token_id("<|tool_call_end|>")

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

    def _ensure_system_message(self, messages: list[Message]) -> list[Message]:
        """Prepend default system message if none present.

        Mirrors the HuggingFace chat template behavior:
        - If messages is empty: return list with just the default system message.
        - If first message is tool_declare and no system follows: insert default
          system after tool_declare.
        - If first message is not system (and not tool_declare): prepend default.
        - Otherwise: return unchanged.
        """
        if not messages:
            return [{"role": "system", "content": _DEFAULT_SYSTEM}]

        first_role = messages[0].get("role")
        if first_role == "tool_declare":
            if len(messages) >= 2 and messages[1].get("role") == "system":
                return messages
            default_sys: Message = {"role": "system", "content": _DEFAULT_SYSTEM}
            return [messages[0], default_sys] + list(messages[1:])
        elif first_role != "system":
            default_sys = {"role": "system", "content": _DEFAULT_SYSTEM}
            return [default_sys] + list(messages)

        return messages

    @staticmethod
    def _extract_thinking(msg: Message, content: str) -> tuple[str, str]:
        """Return (reasoning_content, plain_content) extracted from message.

        Checks msg['reasoning_content'] first, then falls back to parsing
        <think>...</think> out of the content string.
        """
        if isinstance(msg.get("reasoning_content"), str):
            return msg["reasoning_content"], content

        if "</think>" in content:
            before, after = content.split("</think>", 1)
            if "<think>" in before:
                reasoning = before.split("<think>")[-1].lstrip("\n")
            else:
                reasoning = before.lstrip("\n")
            reasoning = reasoning.rstrip("\n")
            return reasoning, after.lstrip("\n")

        return "", content

    def render(
        self,
        messages: list[Message],
        *,
        tools: list[ToolSpec] | None = None,
        add_generation_prompt: bool = False,
    ) -> RenderedTokens:
        if not messages:
            raise ValueError("No messages provided.")

        # Inject tools as tool_declare message + ensure system message
        if tools:
            tools_payload = [{"type": "function", "function": t} for t in tools]
            tools_json = json.dumps(tools_payload, ensure_ascii=False)
            tool_declare_msg: Message = {
                "role": "tool_declare",
                "content": tools_json,
            }
            # Prepend tool_declare if not already present
            if messages[0].get("role") != "tool_declare":
                messages = [tool_declare_msg] + list(messages)
            # else leave as-is (caller already included tool_declare)

        messages = self._ensure_system_message(messages)

        token_ids: list[int] = []
        indices: list[int] = []

        def emit_ids(ids: list[int], msg_idx: int) -> None:
            token_ids.extend(ids)
            indices.extend([msg_idx] * len(ids))

        def emit_special(token_id: int, msg_idx: int) -> None:
            token_ids.append(token_id)
            indices.append(msg_idx)

        def emit_text(text: str, msg_idx: int) -> None:
            emit_ids(self._encode(text), msg_idx)

        # Compute last non-tool-call assistant index to determine thinking preservation
        last_plain_assistant_idx = -1
        for i in range(len(messages) - 1, -1, -1):
            if messages[i]["role"] == "assistant" and not messages[i].get("tool_calls"):
                last_plain_assistant_idx = i
                break

        for i, msg in enumerate(messages):
            role = msg.get("role", "")
            content = msg.get("content") or ""
            if not isinstance(content, str):
                # Flatten list content to text
                parts: list[str] = []
                for part in content:
                    if isinstance(part, dict):
                        if part.get("type") == "text":
                            parts.append(part.get("text", ""))
                        elif part.get("type") == "thinking":
                            parts.append(
                                "<think>" + part.get("thinking", "") + "</think>"
                            )
                    elif isinstance(part, str):
                        parts.append(part)
                content = "".join(parts)

            if role == "system":
                emit_special(self._im_system, i)
                emit_text("system", i)
                emit_special(self._im_middle, i)
                emit_text(content, i)
                emit_special(self._im_end, i)
                emit_text("\n", i)

            elif role == "tool_declare":
                emit_special(self._im_system, i)
                emit_text("tool_declare", i)
                emit_special(self._im_middle, i)
                emit_text(content, i)
                emit_special(self._im_end, i)
                emit_text("\n", i)

            elif role == "user":
                emit_special(self._im_user, i)
                emit_text("user", i)
                emit_special(self._im_middle, i)
                emit_text(content, i)
                emit_special(self._im_end, i)
                emit_text("\n", i)

            elif role == "assistant":
                # Kimi strips reasoning from historical assistant turns and
                # only keeps it for the most-recent plain assistant. Off-by-one
                # here would drop reasoning from the last turn too.
                is_last_turn = (
                    last_plain_assistant_idx == -1 or i >= last_plain_assistant_idx
                )
                self._render_assistant(
                    msg,
                    i,
                    content,
                    is_last_turn=is_last_turn,
                    emit_special=emit_special,
                    emit_text=emit_text,
                )

            elif role == "tool":
                self._render_tool(
                    msg, i, content, emit_special=emit_special, emit_text=emit_text
                )

            else:
                # Unknown role: use system-style formatting
                emit_special(self._im_system, i)
                emit_text(role, i)
                emit_special(self._im_middle, i)
                emit_text(content, i)
                emit_special(self._im_end, i)
                emit_text("\n", i)

        # Generation prompt
        if add_generation_prompt:
            emit_special(self._im_assistant, -1)
            emit_text("assistant", -1)
            emit_special(self._im_middle, -1)

        return RenderedTokens(token_ids=token_ids, message_indices=indices)

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
        return parse_kimi_k2(
            self._tokenizer,
            token_ids,
            stop_ids={self._im_end},
            tool_calls_section_begin_id=self._tool_calls_section_begin,
            tool_calls_section_end_id=self._tool_calls_section_end,
            tool_call_begin_id=self._tool_call_begin,
            tool_call_argument_begin_id=self._tool_call_argument_begin,
            tool_call_end_id=self._tool_call_end,
        )

    def get_stop_token_ids(self) -> list[int]:
        return [self._im_end]

    def bridge_to_next_turn(
        self,
        previous_prompt_ids: list[int],
        previous_completion_ids: list[int],
        new_messages: list[Message],
        *,
        tools: list[ToolSpec] | None = None,
    ) -> list[int] | None:
        return chatml_bridge(
            self, previous_prompt_ids, previous_completion_ids, new_messages, tools=tools
        )

    def _render_assistant(
        self,
        msg: Message,
        msg_idx: int,
        content: str,
        *,
        is_last_turn: bool,
        emit_special,
        emit_text,
    ) -> None:
        reasoning_content, content = self._extract_thinking(msg, content)

        emit_special(self._im_assistant, msg_idx)
        emit_text("assistant", msg_idx)
        emit_special(self._im_middle, msg_idx)

        # Thinking block: preserve for last-turn assistant messages, strip for historical
        if self._enable_thinking and is_last_turn and reasoning_content:
            emit_text("<think>" + reasoning_content + "</think>", msg_idx)
        else:
            emit_text("<think></think>", msg_idx)

        emit_text(content, msg_idx)

        # Tool calls
        tool_calls = msg.get("tool_calls") or []
        if tool_calls:
            emit_special(self._tool_calls_section_begin, msg_idx)
            for idx, tc in enumerate(tool_calls):
                func = tc.get("function") or tc
                name = func.get("name", "")
                arguments = func.get("arguments", {})
                args_str = (
                    json.dumps(arguments, ensure_ascii=False)
                    if not isinstance(arguments, str)
                    else arguments
                )
                # The Kimi template encodes the function name into the
                # tool-call id (``functions.{name}:{idx}``) — that's how its
                # parser recovers the name. An opaque OpenAI-style id
                # (``call_abc123``) would round-trip to a function name of
                # ``call_abc123``, so we only use the caller-provided id if
                # it's in the template's expected shape.
                raw_id = tc.get("id") or ""
                tc_id = raw_id if ":" in raw_id else f"functions.{name}:{idx}"
                emit_special(self._tool_call_begin, msg_idx)
                emit_text(tc_id, msg_idx)
                emit_special(self._tool_call_argument_begin, msg_idx)
                emit_text(args_str, msg_idx)
                emit_special(self._tool_call_end, msg_idx)
            emit_special(self._tool_calls_section_end, msg_idx)

        emit_special(self._im_end, msg_idx)
        emit_text("\n", msg_idx)

    def _render_tool(
        self,
        msg: Message,
        msg_idx: int,
        content: str,
        *,
        emit_special,
        emit_text,
    ) -> None:
        name = msg.get("name") or "tool"
        tool_call_id = msg.get("tool_call_id") or ""

        emit_special(self._im_system, msg_idx)
        emit_text(name, msg_idx)
        emit_special(self._im_middle, msg_idx)
        emit_text(f"## Return of {tool_call_id}\n", msg_idx)
        emit_text(content, msg_idx)
        emit_special(self._im_end, msg_idx)
        emit_text("\n", msg_idx)
