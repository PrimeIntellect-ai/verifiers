"""GptOssRenderer — adapter over OpenAI's harmony reference encoder.

Wire format: harmony (channel-based, no BOS). The official renderer ships
in the ``openai-harmony`` package — vLLM uses it directly, so matching
its output guarantees byte-identical tokens with vLLM's serving path
(and with what the model was trained on).

This renderer is a thin adapter that:

  - converts each ``RendererMessage`` to one or more ``openai_harmony``
    messages (assistant tool_calls split into per-call ASSISTANT messages
    with channel=commentary and recipient=functions.<name>; tool results
    routed via TOOL author + recipient=assistant)
  - prepends the canonical SystemContent preamble (model identity,
    knowledge cutoff, current date, reasoning_effort, channel config)
    when ``use_system_prompt`` is set — matches HF's apply_chat_template
    behaviour, which the model was trained against
  - renders one harmony message at a time via ``enc.render(m)`` so each
    token can be attributed to its source caller index. ``enc.render``
    is byte-identical to ``enc.render_conversation`` when concatenated;
    verified empirically.

Special tokens
--------------
<|start|>      message start
<|end|>        message end (non-terminal)
<|return|>     message end (terminal — last assistant turn)
<|call|>       tool call end (terminal)
<|channel|>    followed by channel name
<|message|>    content start
<|constrain|>  followed by constraint (e.g. "json")
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any

from openai_harmony import (
    Conversation,
    DeveloperContent,
    HarmonyEncoding,
    HarmonyEncodingName,
    Message as HarmonyMessage,
    ReasoningEffort,
    Role,
    SystemContent,
    ToolDescription,
    load_harmony_encoding,
)
from transformers.tokenization_utils import PreTrainedTokenizer

from renderers.base import (
    Message,
    ParsedResponse,
    RenderedTokens,
    ToolSpec,
    reject_assistant_in_extension,
    trim_to_turn_close,
)
from renderers.parsing import parse_gpt_oss


def _reasoning_effort(effort: str | None) -> ReasoningEffort:
    if effort is None:
        return ReasoningEffort.MEDIUM
    e = effort.lower()
    if e == "low":
        return ReasoningEffort.LOW
    if e == "medium":
        return ReasoningEffort.MEDIUM
    if e == "high":
        return ReasoningEffort.HIGH
    raise ValueError(f"Unknown reasoning_effort: {effort!r}")


def _content_text(content: Any) -> str:
    """Flatten content (string or list of parts) to plain text."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        out = []
        for p in content:
            if not isinstance(p, dict):
                continue
            ptype = p.get("type")
            if ptype == "text":
                out.append(p.get("text", ""))
            elif ptype == "thinking":
                out.append(p.get("thinking", ""))
        return "".join(out)
    return str(content)


def _tool_to_description(tool: ToolSpec) -> ToolDescription:
    """Convert an OpenAI-format tool spec to a harmony ToolDescription."""
    if tool.get("type") == "function" and isinstance(tool.get("function"), dict):
        fn = tool["function"]
    else:
        fn = tool
    return ToolDescription.new(
        name=fn.get("name", ""),
        description=fn.get("description", ""),
        parameters=fn.get("parameters") or {},
    )


def _arguments_to_str(arguments: Any) -> str:
    if isinstance(arguments, str):
        return arguments
    return json.dumps(arguments, ensure_ascii=False)


class GptOssRenderer:
    """Deterministic message → token renderer for OpenAI gpt-oss (harmony)."""

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        *,
        use_system_prompt: bool = True,
        reasoning_effort: str | None = "medium",
        conversation_start_date: str | None = None,
        knowledge_cutoff: str | None = None,
        model_identity: str | None = None,
    ):
        """Initialise the renderer.

        Args:
            tokenizer: HuggingFace tokenizer.
            use_system_prompt: When True (default), prepend the canonical
                harmony SystemContent preamble. Matches HF's
                apply_chat_template behaviour.
            reasoning_effort: ``"low" | "medium" | "high"``. Default
                ``"medium"`` (matches apply_chat_template).
            conversation_start_date: Optional ISO date for the preamble.
                Defaults to today's date in YYYY-MM-DD form.
            knowledge_cutoff: Optional knowledge cutoff string. Harmony's
                default is built into ``SystemContent.new()``.
            model_identity: Optional override for the model identity line.
        """
        self._tokenizer = tokenizer
        self._enc: HarmonyEncoding = load_harmony_encoding(
            HarmonyEncodingName.HARMONY_GPT_OSS
        )
        self._use_system_prompt = use_system_prompt
        self._reasoning_effort = _reasoning_effort(reasoning_effort)
        self._conversation_start_date = (
            conversation_start_date or datetime.now().strftime("%Y-%m-%d")
        )
        self._knowledge_cutoff = knowledge_cutoff
        self._model_identity = model_identity

        # Cache special-token IDs for the bridge / generation-prompt path.
        self._start = self._token_id("<|start|>")
        self._end = self._token_id("<|end|>")
        self._return = self._token_id("<|return|>")
        self._call = self._token_id("<|call|>")
        self._channel = self._token_id("<|channel|>")
        self._message = self._token_id("<|message|>")
        self._constrain = self._token_id("<|constrain|>")

    # ── token utilities ──────────────────────────────────────────────────────

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

    # ── public interface ─────────────────────────────────────────────────────

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

        def emit(ids: list[int], msg_idx: int) -> None:
            tokens.extend(ids)
            indices.extend([msg_idx] * len(ids))

        # ── Build harmony prefix (system + developer) ───────────────────
        # When tools are present, harmony's conversation-level renderer
        # injects a channel-routing line into the SystemContent block
        # (``Calls to these tools must go to the commentary channel:
        # 'functions'.``). Per-message ``enc.render`` doesn't know about
        # the surrounding tool namespaces, so we render the prefix via
        # ``render_conversation`` to pick that up correctly.
        first_system_idx = next(
            (i for i, m in enumerate(messages) if m.get("role") == "system"),
            None,
        )
        prefix_msgs: list[HarmonyMessage] = []
        if self._use_system_prompt:
            sys_content = SystemContent.new().with_reasoning_effort(
                self._reasoning_effort
            )
            sys_content = sys_content.with_conversation_start_date(
                self._conversation_start_date
            )
            if self._knowledge_cutoff is not None:
                sys_content = sys_content.with_knowledge_cutoff(self._knowledge_cutoff)
            if self._model_identity is not None:
                sys_content = sys_content.with_model_identity(self._model_identity)
            prefix_msgs.append(
                HarmonyMessage.from_role_and_content(Role.SYSTEM, sys_content)
            )
        if first_system_idx is not None or tools:
            dev = DeveloperContent.new()
            if first_system_idx is not None:
                instructions = _content_text(messages[first_system_idx].get("content"))
                if instructions:
                    dev = dev.with_instructions(instructions)
            if tools:
                dev = dev.with_function_tools([_tool_to_description(t) for t in tools])
            prefix_msgs.append(
                HarmonyMessage.from_role_and_content(Role.DEVELOPER, dev)
            )

        if prefix_msgs:
            prefix_tokens = self._enc.render_conversation(
                Conversation.from_messages(prefix_msgs)
            )
            # Attribute the whole prefix block to first_system_idx if the
            # caller supplied a system message (so its content has *some*
            # caller-relative attribution); otherwise to -1 (pure scaffolding).
            prefix_origin = first_system_idx if first_system_idx is not None else -1
            emit(prefix_tokens, prefix_origin)

        # ── Iterate the rest of the messages ────────────────────────────
        last_idx = len(messages) - 1
        for i, msg in enumerate(messages):
            if i == first_system_idx:
                continue  # already emitted as developer
            for hm in self._to_harmony_messages(msg):
                emit(self._enc.render(hm), i)

        # When the conversation ends on an assistant final-channel turn,
        # ``apply_chat_template`` (and ``render_conversation_for_training``)
        # close it with ``<|return|>`` instead of ``<|end|>`` to mark the
        # final assistant turn as terminal. Per-message rendering always
        # uses ``<|end|>``, so patch it here when applicable.
        if (
            not add_generation_prompt
            and last_idx >= 0
            and messages[last_idx].get("role") == "assistant"
            and tokens
            and tokens[-1] == self._end
            and not (messages[last_idx].get("tool_calls"))
        ):
            tokens[-1] = self._return

        # ── Generation prompt: <|start|>assistant<|channel|>analysis<|message|>
        if add_generation_prompt:
            emit([self._start], -1)
            emit(self._encode("assistant"), -1)
            emit([self._channel], -1)
            emit(self._encode("analysis"), -1)
            emit([self._message], -1)

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
        return parse_gpt_oss(
            self._tokenizer,
            token_ids,
            return_id=self._return,
            call_id=self._call,
            start_id=self._start,
            end_id=self._end,
            channel_id=self._channel,
            message_id=self._message,
            constrain_id=self._constrain,
        )

    def get_stop_token_ids(self) -> list[int]:
        return [self._return, self._call]

    def bridge_to_next_turn(
        self,
        previous_prompt_ids: list[int],
        previous_completion_ids: list[int],
        new_messages: list[Message],
        *,
        tools: list[ToolSpec] | None = None,
    ) -> list[int] | None:
        """Per-message harmony bridge.

        Each new message is rendered in isolation via ``enc.render(m)`` —
        same primitive ``render()`` uses — so the bridge can never drift
        from the full re-render.
        """
        if (
            not previous_prompt_ids
            or not new_messages
            or reject_assistant_in_extension(new_messages)
        ):
            return None

        previous_ids = trim_to_turn_close(
            previous_prompt_ids,
            previous_completion_ids,
            {self._return, self._call},
            synthesize_close=self._end,
        )
        if previous_ids is None:
            return None

        ext: list[int] = []
        for msg in new_messages:
            role = msg.get("role")
            if role not in ("tool", "user", "system", "developer"):
                return None
            for hm in self._to_harmony_messages(msg):
                ext.extend(self._enc.render(hm))

        # Generation prompt: <|start|>assistant<|channel|>analysis<|message|>
        ext.append(self._start)
        ext.extend(self._encode("assistant"))
        ext.append(self._channel)
        ext.extend(self._encode("analysis"))
        ext.append(self._message)

        return previous_ids + ext

    # ── message conversion ───────────────────────────────────────────────────

    def _to_harmony_messages(self, msg: Message) -> list[HarmonyMessage]:
        """Convert a single RendererMessage to one or more harmony messages.

        Splits in two cases:
          - assistant with reasoning_content + content → analysis message
            then final message
          - assistant with multiple tool_calls → one ASSISTANT message per
            tool_call, with recipient=functions.<name>, channel=commentary
        """
        role = msg.get("role", "")

        if role == "user":
            return [
                HarmonyMessage.from_role_and_content(
                    Role.USER, _content_text(msg.get("content"))
                )
            ]

        if role == "system" or role == "developer":
            # Caller's system message is normally pulled out as developer
            # Instructions in render(); reaching here means it's a
            # secondary system or an explicit developer message. Render
            # as developer with the content as instructions.
            dev = DeveloperContent.new().with_instructions(
                _content_text(msg.get("content"))
            )
            return [HarmonyMessage.from_role_and_content(Role.DEVELOPER, dev)]

        if role == "tool":
            # Tool result. Author needs the function name; we recover it
            # from `name` (set client-side via _attach_tool_call_names) or
            # from the message dict.
            tool_name = msg.get("name") or "unknown"
            if not tool_name.startswith("functions."):
                tool_name = f"functions.{tool_name}"
            content = _content_text(msg.get("content"))
            tm = HarmonyMessage.from_author_and_content(
                {"role": "tool", "name": tool_name}, content
            )
            tm = tm.with_recipient("assistant").with_channel("commentary")
            return [tm]

        if role == "assistant":
            return self._assistant_to_harmony(msg)

        # Unknown role: render as developer with the raw content.
        dev = DeveloperContent.new().with_instructions(
            _content_text(msg.get("content"))
        )
        return [HarmonyMessage.from_role_and_content(Role.DEVELOPER, dev)]

    def _assistant_to_harmony(self, msg: Message) -> list[HarmonyMessage]:
        """Convert an assistant message to harmony messages.

        Layout:
          - text content (if any)      → final channel
          - each tool_call             → commentary channel,
                                         recipient=functions.<name>

        ``reasoning_content`` is intentionally NOT emitted: harmony strips
        analysis-channel messages from history-style rendering (verified
        against ``HarmonyEncoding.render_conversation`` /
        ``render_conversation_for_training``). Per-turn thinking is only
        relevant for live generation; once a turn closes, its analysis
        block is dropped from context.
        """
        out: list[HarmonyMessage] = []

        content = msg.get("content")
        text_parts: list[str] = []
        if isinstance(content, list):
            for p in content:
                if not isinstance(p, dict):
                    continue
                if p.get("type") == "text":
                    text_parts.append(p.get("text", ""))
                # Thinking parts are dropped (see docstring).
        elif isinstance(content, str):
            text_parts.append(content)
        text = "".join(text_parts)

        if text:
            m = HarmonyMessage.from_role_and_content(Role.ASSISTANT, text)
            m = m.with_channel("final")
            out.append(m)

        tool_calls = msg.get("tool_calls") or []
        for tc in tool_calls:
            fn = tc.get("function") or tc
            name = fn.get("name", "")
            args = _arguments_to_str(fn.get("arguments", {}))
            recipient = name if name.startswith("functions.") else f"functions.{name}"
            m = HarmonyMessage.from_role_and_content(Role.ASSISTANT, args)
            m = m.with_channel("commentary").with_recipient(recipient)
            out.append(m)

        # Empty assistant (no text and no tool_calls) — emit an empty
        # final-channel message so message_indices stays non-empty.
        if not out:
            m = HarmonyMessage.from_role_and_content(Role.ASSISTANT, "")
            m = m.with_channel("final")
            out.append(m)

        return out
