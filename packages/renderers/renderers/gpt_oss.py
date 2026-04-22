"""GptOssRenderer — OpenAI open-source model format (Harmony).

Wire format: channel-based, no BOS token.

Special tokens
--------------
<|start|>      message start
<|end|>        message end (non-terminal)
<|return|>     message end (terminal — last assistant turn)
<|call|>       tool call end (terminal)
<|channel|>    followed by channel name
<|message|>    content start
<|constrain|>  followed by constraint (e.g. "json")

Channels
--------
analysis    chain-of-thought / thinking (hidden from users)
commentary  tool calls and tool results
final       user-facing response text
"""

from __future__ import annotations

import json
from datetime import datetime

from transformers.tokenization_utils import PreTrainedTokenizer

from renderers.base import Message, ParsedResponse, RenderedTokens, ToolSpec
from renderers.bridges import chatml_bridge
from renderers.parsing import parse_gpt_oss

# ---------------------------------------------------------------------------
# TypeScript tool-formatting helpers
# ---------------------------------------------------------------------------


def _json_type_to_typescript(schema: dict) -> str:
    """Convert a JSON Schema type node to a TypeScript type string."""
    if "oneOf" in schema:
        return " | ".join(_json_type_to_typescript(s) for s in schema["oneOf"])
    if "anyOf" in schema:
        return " | ".join(_json_type_to_typescript(s) for s in schema["anyOf"])

    json_type = schema.get("type", "any")

    if isinstance(json_type, list):
        return " | ".join(_json_type_to_typescript({"type": t}) for t in json_type)

    if json_type == "string":
        if "enum" in schema:
            return " | ".join(json.dumps(v) for v in schema["enum"])
        base_type = "string"
    elif json_type in ("number", "integer"):
        base_type = "number"
    elif json_type == "boolean":
        base_type = "boolean"
    elif json_type == "array":
        items_type = _json_type_to_typescript(schema.get("items", {}))
        base_type = f"{items_type}[]"
    elif json_type == "object":
        base_type = _json_schema_to_typescript(schema)
    else:
        base_type = "any"

    if schema.get("nullable"):
        return f"{base_type} | null"
    return base_type


def _json_schema_to_typescript(schema: dict) -> str:
    """Convert a JSON Schema object node to an inline TypeScript type string."""
    if schema.get("type") != "object":
        return "any"

    properties = schema.get("properties", {})
    required = set(schema.get("required", []))

    type_parts = []
    for prop_name, prop_schema in properties.items():
        prop_type = _json_type_to_typescript(prop_schema)
        optional = "" if prop_name in required else "?"
        type_parts.append(f"{prop_name}{optional}: {prop_type}")

    return "{ " + ", ".join(type_parts) + " }"


def _format_tool_definition(tool: ToolSpec) -> str:
    """Format a ToolSpec dict as a Harmony TypeScript-style definition.

    Produces:
        // {description}
        type {name} = (_: { param: type, ... }) => any;
    """
    lines: list[str] = []
    if tool.get("description"):
        lines.append(f"// {tool['description']}")

    params: dict = tool.get("parameters") or {}
    if params.get("type") == "object" and params.get("properties"):
        ts_params = _json_schema_to_typescript(params)
        lines.append(f"type {tool['name']} = (_: {ts_params}) => any;")
    else:
        lines.append(f"type {tool['name']} = (_: {{}}) => any;")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# System-prompt template (matches Tinker reference)
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT_TEMPLATE = (
    "You are ChatGPT, a large language model trained by OpenAI.\n"
    "Knowledge cutoff: 2024-06\n"
    "Current date: {current_date}\n\n"
    "Reasoning: {reasoning_effort}\n\n"
    "# Valid channels: analysis, commentary, final."
    " Channel must be included for every message."
)


# ---------------------------------------------------------------------------
# Renderer
# ---------------------------------------------------------------------------


class GptOssRenderer:
    """Deterministic message → token renderer for OpenAI OSS (Harmony) models."""

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        *,
        use_system_prompt: bool = False,
        reasoning_effort: str | None = None,
    ):
        """Initialise the renderer.

        Args:
            tokenizer: HuggingFace tokenizer.
            use_system_prompt: When True, prepend OpenAI's built-in system
                message (requires ``reasoning_effort`` to be set).
            reasoning_effort: Effort level string (e.g. ``"high"``).  Must be
                provided iff ``use_system_prompt=True``.
        """
        assert use_system_prompt == (reasoning_effort is not None), (
            "reasoning_effort must be set if and only if use_system_prompt=True"
        )
        self._tokenizer = tokenizer
        self._use_system_prompt = use_system_prompt
        self._reasoning_effort = reasoning_effort

        # Cache special-token IDs once
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

        def emit_ids(ids: list[int], msg_idx: int) -> None:
            tokens.extend(ids)
            indices.extend([msg_idx] * len(ids))

        def emit_special(token_id: int, msg_idx: int) -> None:
            tokens.append(token_id)
            indices.append(msg_idx)

        def emit_text(text: str, msg_idx: int) -> None:
            emit_ids(self._encode(text), msg_idx)

        # Prepend internal system message when configured
        effective_messages: list[Message] = list(messages)
        if self._use_system_prompt:
            current_date = datetime.now().strftime("%Y-%m-%d")
            sys_content = _SYSTEM_PROMPT_TEMPLATE.format(
                current_date=current_date,
                reasoning_effort=self._reasoning_effort,
            )
            # Use a sentinel role so _render_message renders it as "system"
            # without applying the developer mapping
            effective_messages = [
                {"role": "_gptoss_internal_system", "content": sys_content}
            ] + effective_messages

        num = len(effective_messages)
        for i, msg in enumerate(effective_messages):
            is_last = i == num - 1
            self._render_message(
                msg,
                i,
                is_last=is_last,
                tools=tools if i == 0 else None,
                emit_special=emit_special,
                emit_text=emit_text,
            )

        # Generation prompt: <|start|>assistant<|channel|>analysis<|message|>
        # (model begins emitting from the analysis channel)
        if add_generation_prompt:
            emit_special(self._start, -1)
            emit_text("assistant", -1)
            emit_special(self._channel, -1)
            emit_text("analysis", -1)
            emit_special(self._message, -1)

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
        return chatml_bridge(
            self, previous_prompt_ids, previous_completion_ids, new_messages, tools=tools
        )

    # ── rendering helpers ────────────────────────────────────────────────────

    def _render_message(
        self,
        msg: Message,
        msg_idx: int,
        *,
        is_last: bool,
        tools: list[ToolSpec] | None,
        emit_special,
        emit_text,
    ) -> None:
        role = msg.get("role", "")

        if role == "tool":
            self._render_tool_result(
                msg, msg_idx, emit_special=emit_special, emit_text=emit_text
            )
            return

        # Map roles
        if role == "_gptoss_internal_system":
            wire_role = "system"
        elif role == "system":
            wire_role = "developer"
        else:
            wire_role = role

        emit_special(self._start, msg_idx)
        emit_text(wire_role, msg_idx)

        if role == "assistant":
            self._render_assistant(
                msg,
                msg_idx,
                is_last=is_last,
                emit_special=emit_special,
                emit_text=emit_text,
            )
        elif role == "system":
            # System messages from user → developer role with "# Instructions" wrapper
            content = msg.get("content") or ""
            if isinstance(content, list):
                content = "".join(
                    p.get("text", "") for p in content if p.get("type") == "text"
                )
            emit_special(self._channel, msg_idx)
            emit_text("final", msg_idx)
            emit_special(self._message, msg_idx)
            emit_text(f"# Instructions\n\n{content}\n\n", msg_idx)
            emit_special(self._end, msg_idx)
        else:
            # user / developer / internal system
            content = msg.get("content") or ""
            if isinstance(content, list):
                content = "".join(
                    p.get("text", "") for p in content if p.get("type") == "text"
                )

            # Inject tool declarations into first developer/system message when tools given
            if tools and role in ("developer", "_gptoss_internal_system", "system"):
                tool_defs = "\n\n".join(_format_tool_definition(t) for t in tools)
                tools_block = (
                    "# Tools\n\n## functions\n\nnamespace functions {\n\n"
                    + tool_defs
                    + "\n\n} // namespace functions"
                )
                content = (content + "\n\n" + tools_block).strip()

            emit_special(self._message, msg_idx)
            emit_text(content, msg_idx)
            emit_special(self._end, msg_idx)

    def _render_assistant(
        self,
        msg: Message,
        msg_idx: int,
        *,
        is_last: bool,
        emit_special,
        emit_text,
    ) -> None:
        """Emit assistant channels: analysis → (commentary tool calls | final text)."""
        content = msg.get("content") or ""
        tool_calls = msg.get("tool_calls") or []

        # Extract thinking / text from content
        thinking = ""
        text = ""
        reasoning_content = msg.get("reasoning_content")
        if isinstance(reasoning_content, str):
            thinking = reasoning_content
            text = content if isinstance(content, str) else ""
        elif isinstance(content, list):
            for part in content:
                if part.get("type") == "thinking":
                    thinking += part.get("thinking", "")
                elif part.get("type") == "text":
                    text += part.get("text", "")
        else:
            text = content

        # Analysis channel (thinking) — always emitted for non-terminal assistant turns
        # and whenever thinking content exists
        if thinking or not is_last:
            emit_special(self._channel, msg_idx)
            emit_text("analysis", msg_idx)
            emit_special(self._message, msg_idx)
            emit_text(thinking, msg_idx)
            emit_special(self._end, msg_idx)
            # Open next assistant block if there is more to emit
            if tool_calls or text or is_last:
                emit_special(self._start, msg_idx)
                emit_text("assistant", msg_idx)
        elif thinking == "" and is_last and not tool_calls:
            # Pure final-channel response with no thinking
            emit_special(self._channel, msg_idx)
            emit_text("final", msg_idx)
            emit_special(self._message, msg_idx)
            emit_text(text, msg_idx)
            emit_special(self._return, msg_idx)
            return

        if tool_calls:
            # If there's a text preamble before tool calls, emit it as commentary
            if text:
                emit_special(self._channel, msg_idx)
                emit_text("commentary", msg_idx)
                emit_special(self._message, msg_idx)
                emit_text(text, msg_idx)
                emit_special(self._end, msg_idx)
                emit_special(self._start, msg_idx)
                emit_text("assistant", msg_idx)

            for tc_idx, tc in enumerate(tool_calls):
                func = tc.get("function") or tc
                name = func.get("name", "")
                arguments = func.get("arguments", {})
                args_str = (
                    json.dumps(arguments, ensure_ascii=False)
                    if not isinstance(arguments, str)
                    else arguments
                )
                emit_text(f" to=functions.{name}", msg_idx)
                emit_special(self._channel, msg_idx)
                emit_text("commentary ", msg_idx)
                emit_special(self._constrain, msg_idx)
                emit_text("json", msg_idx)
                emit_special(self._message, msg_idx)
                emit_text(args_str, msg_idx)
                emit_special(self._call, msg_idx)

                # If more tool calls follow, open the next assistant block
                if tc_idx < len(tool_calls) - 1:
                    emit_special(self._start, msg_idx)
                    emit_text("assistant", msg_idx)
        else:
            # Final channel
            emit_special(self._channel, msg_idx)
            emit_text("final", msg_idx)
            emit_special(self._message, msg_idx)
            emit_text(text, msg_idx)
            if is_last:
                emit_special(self._return, msg_idx)
            else:
                emit_special(self._end, msg_idx)

    def _render_tool_result(
        self,
        msg: Message,
        msg_idx: int,
        *,
        emit_special,
        emit_text,
    ) -> None:
        """Emit a tool result message in Harmony commentary format."""
        tool_name = msg.get("name") or "unknown"
        if not tool_name.startswith("functions."):
            tool_name = f"functions.{tool_name}"

        content = msg.get("content") or ""
        if isinstance(content, list):
            content = "".join(
                p.get("text", "") for p in content if p.get("type") == "text"
            )

        emit_special(self._start, msg_idx)
        emit_text(f"{tool_name} to=assistant", msg_idx)
        emit_special(self._channel, msg_idx)
        emit_text("commentary", msg_idx)
        emit_special(self._message, msg_idx)
        emit_text(content, msg_idx)
        emit_special(self._end, msg_idx)
