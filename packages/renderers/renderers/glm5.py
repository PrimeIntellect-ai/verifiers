"""GLM-5 Renderer — hard-coded Python mirroring the GLM-5 Jinja chat template.

Key differences from Qwen family:
- Prefix: [gMASK]<sop> before all content
- Role markers: <|system|>, <|user|>, <|assistant|>, <|observation|> (no im_start/im_end)
- No end-of-message token — messages separated by next role marker
- Assistant always emits </think> as separator (even without thinking content)
- Tool calls: <tool_call>name<arg_key>k</arg_key><arg_value>v</arg_value></tool_call>
- Tool responses: <|observation|><tool_response>content</tool_response>
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
)
from renderers.parsing import parse_glm

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
    "<tool_call>{function-name}"
    "<arg_key>{arg-key-1}</arg_key><arg_value>{arg-value-1}</arg_value>"
    "<arg_key>{arg-key-2}</arg_key><arg_value>{arg-value-2}</arg_value>"
    "...</tool_call>"
)


class GLM5Renderer:
    """Deterministic message → token renderer for GLM-5 models."""

    # GLM-5.1 flips this on: even when the most-recent assistant has no
    # reasoning content, the template wraps it with ``<think></think>``
    # instead of just emitting ``</think>`` as a separator. Subclassed in
    # GLM51Renderer; GLM-5 proper keeps this off.
    empty_think_on_last_assistant: bool = False

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
        self._tool_response_tok = self._token_id("<tool_response>")
        self._tool_response_end_tok = self._token_id("</tool_response>")
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
    def _last_user_index(messages: list[Message]) -> int:
        for i in range(len(messages) - 1, -1, -1):
            if messages[i].get("role") == "user":
                return i
        return -1

    @staticmethod
    def _format_tool_spec(tool: ToolSpec) -> str:
        """Serialise a single tool spec to the exact JSON the Jinja template
        emits. GLM-5 just ``tojson``s the dict as passed; GLM-5.1 overrides
        this to unwrap the OpenAI-style ``{"type":"function","function":…}``
        envelope and filter internal-only keys first.
        """
        return json.dumps(tool, ensure_ascii=False)

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

        # ── Prefix ──────────────────────────────────────────────────
        emit_special(self._gmask, -1)
        emit_special(self._sop, -1)

        # ── Tools in system prompt ──────────────────────────────────
        if tools:
            emit_special(self._system, -1)
            tool_text = _TOOLS_HEADER
            for tool in tools:
                tool_text += self._format_tool_spec(tool) + "\n"
            tool_text += _TOOLS_FOOTER
            emit_text(tool_text, -1)

        # ── Compute last_user_index ─────────────────────────────────
        last_ui = self._last_user_index(messages)

        # ── Iterate messages ────────────────────────────────────────
        for i, msg in enumerate(messages):
            role = msg["role"]
            content = self._visible_text(msg.get("content"))

            if role == "system":
                emit_special(self._system, i)
                emit_text(content, i)

            elif role == "user":
                emit_special(self._user, i)
                emit_text(content, i)

            elif role == "assistant":
                self._render_assistant(
                    msg,
                    i,
                    content,
                    last_ui,
                    emit_special=emit_special,
                    emit_text=emit_text,
                )

            elif role == "tool":
                self._render_tool(
                    messages, i, content, emit_special=emit_special, emit_text=emit_text
                )

        # ── Generation prompt ───────────────────────────────────────
        if add_generation_prompt:
            emit_special(self._assistant, -1)
            if self._enable_thinking:
                emit_special(self._think, -1)
            else:
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
        return parse_glm(
            self._tokenizer,
            token_ids,
            stop_ids={self._endoftext, self._user, self._observation},
            think_id=self._think,
            think_end_id=self._think_end,
            tool_call_id=self._tool_call_tok,
            tool_call_end_id=self._tool_call_end_tok,
            arg_key_id=self._arg_key,
            arg_key_end_id=self._arg_key_end,
            arg_value_id=self._arg_value,
            arg_value_end_id=self._arg_value_end,
        )

    def get_stop_token_ids(self) -> list[int]:
        return [self._endoftext, self._user, self._observation]

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

        # GLM has no per-turn close token. An assistant turn ends when the
        # next turn's role marker appears, OR the model emits <|endoftext|>.
        # vLLM includes these in ``stop_token_ids`` so a clean stop leaves
        # one of {endoftext, user, observation} at the tail of
        # previous_completion_ids. Truncation means none is there yet.
        previous_ids = list(previous_prompt_ids) + list(previous_completion_ids)
        stop_ids = {self._endoftext, self._user, self._observation}
        if (
            not previous_ids[len(previous_prompt_ids) :]
            or previous_ids[-1] not in stop_ids
        ):
            # Truncation: synthesise <|endoftext|> as the canonical turn end.
            previous_ids.append(self._endoftext)

        last_prev = previous_ids[-1]

        ext: list[int] = []

        def emit_special(token_id: int, _msg_idx: int = -1) -> None:
            ext.append(token_id)

        def emit_text(text: str, _msg_idx: int = -1) -> None:
            ext.extend(self._encode(text))

        for i, msg in enumerate(new_messages):
            role = msg.get("role")
            content = self._visible_text(msg.get("content"))
            if role == "user":
                # Dedup: model already emitted <|user|> as its stop token.
                if not (i == 0 and last_prev == self._user):
                    emit_special(self._user, i)
                emit_text(content, i)
            elif role == "system":
                emit_special(self._system, i)
                emit_text(content, i)
            elif role == "tool":
                prev_is_tool = i > 0 and new_messages[i - 1].get("role") == "tool"
                if i == 0 and last_prev == self._observation:
                    # Model already emitted <|observation|>; don't repeat.
                    pass
                elif not prev_is_tool:
                    emit_special(self._observation, i)
                emit_special(self._tool_response_tok, i)
                emit_text(content, i)
                emit_special(self._tool_response_end_tok, i)
            else:
                return None

        # Generation prompt — match the gen-prompt branch of ``render()``.
        emit_special(self._assistant, -1)
        if self._enable_thinking:
            emit_special(self._think, -1)
        else:
            emit_special(self._think_end, -1)

        return previous_ids + ext

    def _render_assistant(
        self, msg, msg_idx, content, last_user_index, *, emit_special, emit_text
    ):
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

        include_thinking = (
            self._keep_thinking or msg_idx > last_user_index
        ) and reasoning_content

        if include_thinking:
            emit_special(self._think, msg_idx)
            emit_text(reasoning_content.strip(), msg_idx)
            emit_special(self._think_end, msg_idx)
        elif self.empty_think_on_last_assistant and msg_idx > last_user_index:
            # GLM-5.1: wrap the last assistant with an empty <think></think>
            # even without reasoning, matching the Jinja template.
            emit_special(self._think, msg_idx)
            emit_special(self._think_end, msg_idx)
        else:
            emit_special(self._think_end, msg_idx)

        if content.strip():
            emit_text(content.strip(), msg_idx)

        # Tool calls (directly after content, no newlines)
        tool_calls = msg.get("tool_calls") or []
        for tc in tool_calls:
            func = tc.get("function") or tc
            name = func.get("name", "")
            arguments = func.get("arguments", {})

            emit_special(self._tool_call_tok, msg_idx)
            emit_text(name, msg_idx)
            # OpenAI canonical form: arguments is a JSON string. Parse it so the
            # per-argument rendering below still works.
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except json.JSONDecodeError:
                    arguments = {}
            if isinstance(arguments, dict):
                for arg_name, arg_value in arguments.items():
                    emit_special(self._arg_key, msg_idx)
                    emit_text(arg_name, msg_idx)
                    emit_special(self._arg_key_end, msg_idx)
                    emit_special(self._arg_value, msg_idx)
                    if isinstance(arg_value, str):
                        emit_text(arg_value, msg_idx)
                    else:
                        emit_text(json.dumps(arg_value, ensure_ascii=False), msg_idx)
                    emit_special(self._arg_value_end, msg_idx)
            emit_special(self._tool_call_end_tok, msg_idx)

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

        if not prev_is_tool:
            emit_special(self._observation, msg_idx)

        emit_special(self._tool_response_tok, msg_idx)
        emit_text(content, msg_idx)
        emit_special(self._tool_response_end_tok, msg_idx)


class GLM51Renderer(GLM5Renderer):
    """Deterministic message → token renderer for GLM-5.1 models.

    Diverges from GLM-5 in two places:

    - The most-recent assistant turn is wrapped with an empty
      ``<think></think>`` block even when no ``reasoning_content`` is
      supplied. Historical assistants collapse to just ``</think>``.
    - Tool specs are unwrapped before serialisation: if the caller
      passes the OpenAI ``{"type":"function","function":{…}}`` envelope,
      only the inner ``function`` payload is rendered (minus
      ``defer_loading`` / ``strict`` internal keys).
    """

    empty_think_on_last_assistant = True

    @staticmethod
    def _format_tool_spec(tool: ToolSpec) -> str:
        spec = tool["function"] if "function" in tool else tool
        spec = {k: v for k, v in spec.items() if k not in ("defer_loading", "strict")}
        return json.dumps(spec, ensure_ascii=False)
