"""Kimi K2.5 Renderer — standalone implementation for moonshotai/Kimi-K2.5-Instruct.

Kimi K2.5 shares the same tokenizer and base message format as Kimi K2
(moonshotai/Kimi-K2-Instruct) but adds:

1. Thinking mode via a ``<think>`` prefill in the generation prompt.
2. Multimodal/vision support via ``<|media_begin|>image<|media_content|>...<|media_end|>``.
3. TypeScript-style tool declarations instead of JSON.

Message format (identical to K2):
    <|im_system|>system<|im_middle|>You are Kimi...<|im_end|>
    <|im_user|>user<|im_middle|>Hello<|im_end|>
    <|im_assistant|>assistant<|im_middle|><think>\\n...\\n</think>\\nResponse<|im_end|>

Generation prompt (thinking enabled):
    <|im_assistant|>assistant<|im_middle|><think>

Generation prompt (thinking disabled):
    <|im_assistant|>assistant<|im_middle|><think></think>
"""

from __future__ import annotations

import json
import re
from typing import Any

from transformers.tokenization_utils import PreTrainedTokenizer

from renderers.base import ImagePart, Message, ParsedResponse, RenderedTokens, ToolSpec
from renderers.bridges import chatml_bridge

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_SYSTEM_PROMPT = "You are Kimi, an AI assistant created by Moonshot AI."

# Image wrapper tokens used by K2.5 multimodal format
_IMAGE_PREFIX = "<|media_begin|>image<|media_content|>"
_IMAGE_SUFFIX = "<|media_end|>\n"

# ---------------------------------------------------------------------------
# TypeScript-style tool declaration
# ---------------------------------------------------------------------------

_TS_INDENT = "  "
_TS_FIELD_DELIMITER = ",\n"


def _format_description(description: str, indent: str = "") -> str:
    return "\n".join(
        [f"{indent}// {line}" if line else "" for line in description.split("\n")]
    )


class _BaseType:
    description: str
    constraints: dict[str, Any]

    def __init__(self, extra_props: dict[str, Any], *, allowed_constraint_keys=()):
        self.description = extra_props.get("description", "")
        self.constraints = {
            k: v for k, v in extra_props.items() if k in allowed_constraint_keys
        }

    def to_typescript_style(self, indent: str = "") -> str:
        raise NotImplementedError

    def format_docstring(self, indent: str) -> str:
        lines = []
        if self.description:
            lines.append(_format_description(self.description, indent))
        if self.constraints:
            constraints_str = ", ".join(
                f"{k}: {v}"
                for k, v in sorted(self.constraints.items(), key=lambda kv: kv[0])
            )
            lines.append(f"{indent}// {constraints_str}")
        return "".join(x + "\n" for x in lines)


class _SchemaRegistry:
    def __init__(self):
        self.definitions: dict[str, Any] = {}
        self.has_self_ref = False

    def register_definitions(self, defs: dict[str, Any]) -> None:
        for def_name, def_schema in defs.items():
            self.definitions[def_name] = def_schema

    def resolve_ref(self, ref: str) -> dict[str, Any]:
        if ref == "#":
            self.has_self_ref = True
            return {"$self_ref": True}
        elif ref.startswith("#/$defs/"):
            def_name = ref.split("/")[-1]
            if def_name not in self.definitions:
                raise ValueError(f"Reference not found: {ref}")
            return self.definitions[def_name]
        else:
            raise ValueError(f"Unsupported reference format: {ref}")


class _ScalarType(_BaseType):
    def __init__(self, type_: str, extra_props: dict[str, Any] | None = None):
        self.type_ = type_
        allowed: list[str] = []
        if type_ == "string":
            allowed = ["maxLength", "minLength", "pattern"]
        elif type_ in ("number", "integer"):
            allowed = ["maximum", "minimum"]
        super().__init__(extra_props or {}, allowed_constraint_keys=allowed)

    def to_typescript_style(self, indent: str = "") -> str:
        return "number" if self.type_ == "integer" else self.type_


class _ObjectType(_BaseType):
    def __init__(self, schema: dict[str, Any], registry: _SchemaRegistry | None = None):
        super().__init__(schema)
        self.properties: list[_TypedParam] = []
        self.additional_properties: Any = None
        if not schema:
            return
        if "$defs" in schema and registry:
            registry.register_definitions(schema["$defs"])
        self.additional_properties = schema.get("additionalProperties")
        if isinstance(self.additional_properties, dict):
            self.additional_properties = _parse_type(
                self.additional_properties, registry
            )
        if "properties" not in schema:
            return
        required = set(schema.get("required", []))
        for name, prop in schema["properties"].items():
            self.properties.append(
                _TypedParam(
                    name=name,
                    type_=_parse_type(prop, registry),
                    optional=name not in required,
                    default=prop.get("default") if isinstance(prop, dict) else None,
                )
            )

    def to_typescript_style(self, indent: str = "") -> str:
        required_params = sorted(
            [p for p in self.properties if not p.optional], key=lambda p: p.name
        )
        optional_params = sorted(
            [p for p in self.properties if p.optional], key=lambda p: p.name
        )
        params = required_params + optional_params
        param_strs = [p.to_typescript_style(indent=indent + _TS_INDENT) for p in params]
        if self.additional_properties is not None:
            if self.additional_properties is True:
                ap_type = "any"
            elif self.additional_properties is False:
                ap_type = "never"
            else:
                ap_type = self.additional_properties.to_typescript_style(
                    indent=indent + _TS_INDENT
                )
            param_strs.append(f"{indent + _TS_INDENT}[k: string]: {ap_type}")
        if not param_strs:
            return "{}"
        params_str = _TS_FIELD_DELIMITER.join(param_strs)
        return f"{{\n{params_str}\n{indent}}}"


class _ArrayType(_BaseType):
    def __init__(self, schema: dict[str, Any], registry: _SchemaRegistry | None = None):
        super().__init__(schema, allowed_constraint_keys=("minItems", "maxItems"))
        self.item = (
            _parse_type(schema["items"], registry)
            if schema.get("items")
            else _ScalarType("any")
        )

    def to_typescript_style(self, indent: str = "") -> str:
        docstring = self.item.format_docstring(indent + _TS_INDENT)
        if docstring:
            return (
                "Array<\n"
                + docstring
                + indent
                + _TS_INDENT
                + self.item.to_typescript_style(indent=indent + _TS_INDENT)
                + "\n"
                + indent
                + ">"
            )
        return f"Array<{self.item.to_typescript_style(indent=indent)}>"


class _EnumType(_BaseType):
    def __init__(self, schema: dict[str, Any]):
        super().__init__(schema)
        self.enum = schema["enum"]

    def to_typescript_style(self, indent: str = "") -> str:
        return " | ".join(f'"{e}"' if isinstance(e, str) else str(e) for e in self.enum)


class _AnyOfType(_BaseType):
    def __init__(self, schema: dict[str, Any], registry: _SchemaRegistry | None = None):
        super().__init__(schema)
        self.types = [_parse_type(t, registry) for t in schema["anyOf"]]

    def to_typescript_style(self, indent: str = "") -> str:
        return " | ".join(t.to_typescript_style(indent=indent) for t in self.types)


class _UnionType(_BaseType):
    _MAPPING = {
        "string": "string",
        "number": "number",
        "integer": "number",
        "boolean": "boolean",
        "null": "null",
        "object": "{}",
        "array": "Array<any>",
    }

    def __init__(self, schema: dict[str, Any]):
        super().__init__(schema)
        self.types = [self._MAPPING[t] for t in schema["type"]]

    def to_typescript_style(self, indent: str = "") -> str:
        return " | ".join(self.types)


class _RefType(_BaseType):
    def __init__(self, schema: dict[str, Any], registry: _SchemaRegistry):
        super().__init__(schema)
        ref = schema["$ref"]
        resolved = registry.resolve_ref(ref)
        if resolved.get("$self_ref", False):
            self.ref_name = "parameters"
            self.is_self_ref = True
        else:
            self.ref_name = ref.split("/")[-1]
            self.is_self_ref = False

    def to_typescript_style(self, indent: str = "") -> str:
        return self.ref_name


_ParamType = (
    _ScalarType
    | _ObjectType
    | _ArrayType
    | _EnumType
    | _AnyOfType
    | _UnionType
    | _RefType
)


class _TypedParam:
    def __init__(
        self, name: str, type_: _ParamType, optional: bool = True, default: Any = None
    ):
        self.name = name
        self.type_ = type_
        self.optional = optional
        self.default = default

    def to_typescript_style(self, indent: str = "") -> str:
        comments = self.type_.format_docstring(indent)
        if self.default is not None:
            default_repr = (
                json.dumps(self.default, ensure_ascii=False)
                if not isinstance(self.default, (int, float, bool))
                else repr(self.default)
            )
            comments += f"{indent}// Default: {default_repr}\n"
        opt = "?" if self.optional else ""
        return (
            comments
            + f"{indent}{self.name}{opt}: {self.type_.to_typescript_style(indent=indent)}"
        )


def _parse_type(
    schema: dict[str, Any] | bool, registry: _SchemaRegistry | None = None
) -> _ParamType:
    if isinstance(schema, bool):
        return _ScalarType("any" if schema else "null")
    if "$ref" in schema and registry:
        return _RefType(schema, registry)
    if "anyOf" in schema:
        return _AnyOfType(schema, registry)
    if "enum" in schema:
        return _EnumType(schema)
    if "type" in schema:
        typ = schema["type"]
        if isinstance(typ, list):
            return _UnionType(schema)
        if typ == "object":
            return _ObjectType(schema, registry)
        if typ == "array":
            return _ArrayType(schema, registry)
        return _ScalarType(typ, schema)
    if schema == {}:
        return _ScalarType("any")
    raise ValueError(f"Invalid JSON Schema object: {schema}")


def _function_to_typescript(function: dict[str, Any]) -> str:
    """Convert an OpenAI-format function definition to TypeScript-style string."""
    registry = _SchemaRegistry()
    parameters = function.get("parameters") or {}
    parsed = _ObjectType(parameters, registry)

    interfaces: list[str] = []
    root_interface_name: str | None = None

    if registry.has_self_ref:
        root_interface_name = "parameters"
        params_str = _TS_FIELD_DELIMITER.join(
            p.to_typescript_style(indent=_TS_INDENT) for p in parsed.properties
        )
        params_str = f"\n{params_str}\n" if params_str else ""
        interfaces.append(f"interface {root_interface_name} {{{params_str}}}")

    for def_name, def_schema in registry.definitions.items():
        obj_type = _parse_type(def_schema, registry)
        params_str = obj_type.to_typescript_style()
        description_part = ""
        if desc := def_schema.get("description", ""):
            description_part = _format_description(desc) + "\n"
        interfaces.append(f"{description_part}interface {def_name} {params_str}")

    interface_str = "\n".join(interfaces)
    func_name = function.get("name", "function")
    if root_interface_name:
        type_def = f"type {func_name} = (_: {root_interface_name}) => any;"
    else:
        params_str = parsed.to_typescript_style()
        type_def = f"type {func_name} = (_: {params_str}) => any;"

    description = function.get("description")
    return "\n".join(
        filter(
            bool,
            [
                interface_str,
                (description and _format_description(description)) or "",
                type_def,
            ],
        )
    )


def _encode_tools_typescript(tools: list[ToolSpec]) -> str:
    """Convert a list of ToolSpec dicts to TypeScript-style tool declaration string."""
    if not tools:
        return ""
    functions = []
    for tool in tools:
        func_def = _function_to_typescript(tool)
        if func_def:
            functions.append(func_def)
    if not functions:
        return ""
    functions_str = "\n".join(functions)
    return "# Tools\n\n## functions\nnamespace functions {\n" + functions_str + "\n}\n"


# ---------------------------------------------------------------------------
# Kimi K2.5 response parsing (mirrors K2 format, same token structure)
# ---------------------------------------------------------------------------

_TOOL_CALLS_SECTION_RE = re.compile(
    r"<\|tool_calls_section_begin\|>(.*?)<\|tool_calls_section_end\|>"
    r"|<\|tool_call_section_begin\|>(.*?)<\|tool_call_section_end\|>",
    re.DOTALL,
)
_TOOL_CALL_RE = re.compile(
    r"<\|tool_call_begin\|>\s*([^<]+:\d+)\s*<\|tool_call_argument_begin\|>\s*(.*?)\s*<\|tool_call_end\|>",
    re.DOTALL,
)


def _parse_kimi_k2_response(
    tokenizer,
    token_ids: list[int],
    *,
    stop_ids: set[int],
    think_open_ids: list[int],
    think_close_ids: list[int],
) -> ParsedResponse:
    """Parse Kimi K2/K2.5 completion tokens.

    Strips the stop token, decodes to text, then extracts:
    - reasoning from ``<think>...</think>`` blocks
    - tool calls from ``<|tool_calls_section_begin|>...<|tool_calls_section_end|>``
    """
    # Strip stop token
    ids = list(token_ids)
    for i, t in enumerate(ids):
        if t in stop_ids:
            ids = ids[:i]
            break

    # Decode all tokens (including any special tokens that are text-like)
    text = tokenizer.decode(ids, skip_special_tokens=False) if ids else ""

    # Extract reasoning from <think>...</think>
    reasoning: str | None = None
    if "</think>" in text:
        before, _, after = text.partition("</think>")
        # Strip open <think> tag if present
        reasoning = before.replace("<think>", "").strip("\n")
        text = after.strip("\n")
    elif "<think>" in text:
        # Truncated reasoning (no closing tag)
        _, _, partial = text.partition("<think>")
        return ParsedResponse(
            content="", reasoning_content=partial.strip() or None, tool_calls=None
        )

    # Extract tool calls section
    tool_calls: list[dict[str, Any]] | None = None
    tc_match = _TOOL_CALLS_SECTION_RE.search(text)
    if tc_match:
        text = text[: tc_match.start()]
        tool_section = (
            tc_match.group(1) if tc_match.group(1) is not None else tc_match.group(2)
        )
        parsed_calls = []
        for m in _TOOL_CALL_RE.finditer(tool_section):
            tool_id = m.group(1).strip()
            args_str = m.group(2).strip()
            # Extract function name from "functions.name:index" format
            name_part = tool_id.split(":", 1)[0]
            func_name = name_part.split(".", 1)[1] if "." in name_part else name_part
            try:
                arguments = json.loads(args_str)
            except json.JSONDecodeError:
                arguments = args_str  # preserve raw string if invalid JSON
            parsed_calls.append(
                {
                    "type": "function",
                    "id": tool_id,
                    "function": {"name": func_name, "arguments": arguments},
                }
            )
        if parsed_calls:
            tool_calls = parsed_calls

    return ParsedResponse(
        content=text.strip(),
        reasoning_content=reasoning.strip() if reasoning else None,
        tool_calls=tool_calls,
    )


# ---------------------------------------------------------------------------
# KimiK25Renderer
# ---------------------------------------------------------------------------


class KimiK25Renderer:
    """Deterministic message → token renderer for Kimi K2.5 models.

    Renders to the same ``<|im_*|>`` format as Kimi K2 but adds:
    - Generation prompt prefills ``<think>`` (enable_thinking=True, default) or
      ``<think></think>`` (enable_thinking=False) to control thinking mode.
    - Image content rendering via ``<|media_begin|>image<|media_content|>...<|media_end|>``.
    - TypeScript-style tool declarations instead of JSON.

    The tokenizer should be ``moonshotai/Kimi-K2-Instruct`` (same as K2).
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        *,
        enable_thinking: bool = True,
    ):
        self._tokenizer = tokenizer
        self._enable_thinking = enable_thinking

        # Core structural tokens — all must be single special tokens in the vocab
        self._im_user = self._token_id("<|im_user|>")
        self._im_assistant = self._token_id("<|im_assistant|>")
        self._im_system = self._token_id("<|im_system|>")
        self._im_middle = self._token_id("<|im_middle|>")
        self._im_end = self._token_id("<|im_end|>")

        # Tool call tokens
        self._tool_calls_section_begin = self._token_id("<|tool_calls_section_begin|>")
        self._tool_calls_section_end = self._token_id("<|tool_calls_section_end|>")
        self._tool_call_begin = self._token_id("<|tool_call_begin|>")
        self._tool_call_argument_begin = self._token_id("<|tool_call_argument_begin|>")
        self._tool_call_end = self._token_id("<|tool_call_end|>")

        # Image tokens (optional — only required if image content is present)
        self._media_begin: int | None = self._try_token_id("<|media_begin|>")
        self._media_content: int | None = self._try_token_id("<|media_content|>")
        self._media_end: int | None = self._try_token_id("<|media_end|>")

        # <think> / </think> may be multi-token in K2.5; we encode them as text.
        # We cache the encoded IDs for use in _normalize_response_tokens.
        self._think_open_ids: list[int] = self._encode("<think>")
        self._think_close_ids: list[int] = self._encode("</think>")

        # The stop token for generation
        self._endoftext: int | None = self._try_token_id("<|endoftext|>")

    # ------------------------------------------------------------------
    # Token helpers
    # ------------------------------------------------------------------

    def _token_id(self, token: str) -> int:
        tid = self._tokenizer.convert_tokens_to_ids(token)
        assert isinstance(tid, int) and tid != self._tokenizer.unk_token_id, (
            f"Special token {token!r} not found in tokenizer vocabulary"
        )
        return tid

    def _try_token_id(self, token: str) -> int | None:
        """Return token ID or None if not in vocabulary (used for optional tokens)."""
        tid = self._tokenizer.convert_tokens_to_ids(token)
        if not isinstance(tid, int) or tid == self._tokenizer.unk_token_id:
            return None
        return tid

    def _encode(self, text: str) -> list[int]:
        if not text:
            return []
        return self._tokenizer.encode(text, add_special_tokens=False)

    # ------------------------------------------------------------------
    # Core render
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

        # Auto-inject default system message if absent (mirrors HF chat template)
        messages = self._ensure_system_message(messages)

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

        # ── Tool declaration prefix (TypeScript style, comes first) ──
        if tools:
            # Find the tool_declare slot index for attribution
            tool_declare_idx = next(
                (i for i, m in enumerate(messages) if m.get("role") == "tool_declare"),
                -1,
            )
            tools_ts = _encode_tools_typescript(tools)
            emit_special(self._im_system, tool_declare_idx)
            emit_text("tool_declare", tool_declare_idx)
            emit_special(self._im_middle, tool_declare_idx)
            emit_text(tools_ts, tool_declare_idx)
            emit_special(self._im_end, tool_declare_idx)
            emit_text("\n", tool_declare_idx)

        # ── Iterate messages ─────────────────────────────────────────
        for i, msg in enumerate(messages):
            role = msg.get("role", "")
            if role == "tool_declare":
                # Already emitted above (or will be emitted via tools= path)
                continue

            if role == "system":
                emit_special(self._im_system, i)
                emit_text("system", i)
                emit_special(self._im_middle, i)
                emit_text((msg.get("content") or ""), i)
                emit_special(self._im_end, i)
                emit_text("\n", i)

            elif role == "user":
                emit_special(self._im_user, i)
                emit_text("user", i)
                emit_special(self._im_middle, i)
                self._emit_content(
                    msg.get("content"), i, emit_special, emit_text, emit_ids
                )
                emit_special(self._im_end, i)
                emit_text("\n", i)

            elif role == "assistant":
                self._render_assistant(
                    msg, i, emit_special=emit_special, emit_text=emit_text
                )

            elif role == "tool":
                self._render_tool_response(
                    msg, i, emit_special=emit_special, emit_text=emit_text
                )

            else:
                # Unknown roles use system-style formatting
                emit_special(self._im_system, i)
                emit_text(role, i)
                emit_special(self._im_middle, i)
                emit_text((msg.get("content") or ""), i)
                emit_special(self._im_end, i)
                emit_text("\n", i)

        # ── Generation prompt ────────────────────────────────────────
        if add_generation_prompt:
            emit_special(self._im_assistant, -1)
            emit_text("assistant", -1)
            emit_special(self._im_middle, -1)
            if self._enable_thinking:
                # Prefill open <think> tag to trigger thinking mode
                emit_text("<think>", -1)
            else:
                # Empty <think></think> to disable thinking
                emit_text("<think></think>", -1)

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
        stop_ids: set[int] = {self._im_end}
        if self._endoftext is not None:
            stop_ids.add(self._endoftext)

        # Restore the synthetic <think> prefill if it was stripped by the sampler
        normalized = self._normalize_response_tokens(list(token_ids))

        return _parse_kimi_k2_response(
            self._tokenizer,
            normalized,
            stop_ids=stop_ids,
            think_open_ids=self._think_open_ids,
            think_close_ids=self._think_close_ids,
        )

    def get_stop_token_ids(self) -> list[int]:
        stop = [self._im_end]
        if self._endoftext is not None:
            stop.append(self._endoftext)
        return stop

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

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ensure_system_message(self, messages: list[Message]) -> list[Message]:
        """Auto-inject the default system message if none is present (mirrors HF template)."""
        if not messages:
            return [{"role": "system", "content": _DEFAULT_SYSTEM_PROMPT}]
        first_role = messages[0].get("role", "")
        if first_role == "tool_declare":
            # Check if a system message follows
            if len(messages) >= 2 and messages[1].get("role") == "system":
                return messages
            return [
                messages[0],
                {"role": "system", "content": _DEFAULT_SYSTEM_PROMPT},
            ] + list(messages[1:])
        elif first_role != "system":
            return [{"role": "system", "content": _DEFAULT_SYSTEM_PROMPT}] + list(
                messages
            )
        return messages

    def _emit_content(
        self,
        content: Any,
        msg_idx: int,
        emit_special,
        emit_text,
        emit_ids,
    ) -> None:
        """Emit message content, handling both plain strings and multipart lists."""
        if content is None:
            return
        if isinstance(content, str):
            emit_text(content, msg_idx)
            return
        if isinstance(content, list):
            for part in content:
                if not isinstance(part, dict):
                    continue
                ptype = part.get("type")
                if ptype == "text":
                    emit_text(part.get("text", ""), msg_idx)
                elif ptype == "image":
                    self._emit_image(part, msg_idx, emit_special, emit_text, emit_ids)
                elif ptype == "thinking":
                    # Thinking parts in non-assistant roles are rendered as text
                    thinking = part.get("thinking", "")
                    if thinking:
                        emit_text(f"<think>{thinking}</think>", msg_idx)
                # Other part types are silently skipped

    def _emit_image(
        self,
        part: ImagePart,
        msg_idx: int,
        emit_special,
        emit_text,
        emit_ids,
    ) -> None:
        """Emit an image content part using K2.5 media tokens."""
        image_url: str = part.get("image", "")  # type: ignore[attr-defined]
        if (
            self._media_begin is not None
            and self._media_content is not None
            and self._media_end is not None
        ):
            # Use dedicated special tokens when available
            emit_special(self._media_begin, msg_idx)
            emit_text("image", msg_idx)
            emit_special(self._media_content, msg_idx)
            emit_text(image_url, msg_idx)
            emit_special(self._media_end, msg_idx)
            emit_text("\n", msg_idx)
        else:
            # Fallback: encode the media markers as plain text
            emit_text(f"{_IMAGE_PREFIX}{image_url}{_IMAGE_SUFFIX}", msg_idx)

    def _render_assistant(
        self,
        msg: Message,
        msg_idx: int,
        *,
        emit_special,
        emit_text,
    ) -> None:
        """Render an assistant message with <think>...</think> and optional tool calls."""
        content = msg.get("content")
        reasoning_content: str = ""

        # Extract reasoning from structured content parts or inline <think> tags
        if isinstance(msg.get("reasoning_content"), str):
            reasoning_content = msg["reasoning_content"]
            # content stays as-is (text only)
            if isinstance(content, list):
                text_content = "".join(
                    p.get("text", "")
                    for p in content
                    if isinstance(p, dict) and p.get("type") == "text"
                )
            else:
                text_content = content or ""
        elif isinstance(content, list):
            # Extract thinking + text from content parts
            thinking_parts = [
                p.get("thinking", "")
                for p in content
                if isinstance(p, dict) and p.get("type") == "thinking"
            ]
            text_parts = [
                p.get("text", "")
                for p in content
                if isinstance(p, dict) and p.get("type") == "text"
            ]
            reasoning_content = "".join(thinking_parts)
            text_content = "".join(text_parts)
        elif isinstance(content, str) and "</think>" in content:
            # Inline <think>...</think> in string content
            before, _, after = content.partition("</think>")
            if "<think>" in before:
                reasoning_content = before.split("<think>", 1)[-1]
            else:
                reasoning_content = before
            text_content = after.lstrip("\n")
        else:
            text_content = content or ""

        emit_special(self._im_assistant, msg_idx)
        emit_text("assistant", msg_idx)
        emit_special(self._im_middle, msg_idx)

        # Thinking block (always included in assistant messages, matching K2 format)
        thinking_block = f"<think>{reasoning_content}</think>"
        emit_text(thinking_block, msg_idx)

        # Main text content
        emit_text(text_content, msg_idx)

        # Tool calls section
        tool_calls = msg.get("tool_calls") or []
        if tool_calls:
            emit_special(self._tool_calls_section_begin, msg_idx)
            for tc_idx, tc in enumerate(tool_calls):
                func = tc.get("function") or tc
                func_name = func.get("name", "")
                arguments = func.get("arguments", {})
                args_str = (
                    json.dumps(arguments, ensure_ascii=False)
                    if not isinstance(arguments, str)
                    else arguments
                )
                # Template emits ``tool_call['id']`` verbatim — empty when
                # missing. Round-trip requires caller to pass id in
                # ``functions.{name}:{idx}`` form (Kimi's parser recovers
                # the function name from that field).
                tool_id = tc.get("id") or ""
                emit_special(self._tool_call_begin, msg_idx)
                emit_text(tool_id, msg_idx)
                emit_special(self._tool_call_argument_begin, msg_idx)
                emit_text(args_str, msg_idx)
                emit_special(self._tool_call_end, msg_idx)
            emit_special(self._tool_calls_section_end, msg_idx)

        emit_special(self._im_end, msg_idx)
        emit_text("\n", msg_idx)

    def _render_tool_response(
        self,
        msg: Message,
        msg_idx: int,
        *,
        emit_special,
        emit_text,
    ) -> None:
        """Render a tool result message using the system-style format."""
        # In K2, tool responses use: <|im_system|>name<|im_middle|>## Return of {id}\ncontent<|im_end|>
        role_name = msg.get("name") or "tool"
        tool_call_id = msg.get("tool_call_id", "")
        content = msg.get("content") or ""
        if isinstance(content, list):
            content = "".join(
                p.get("text", "")
                for p in content
                if isinstance(p, dict) and p.get("type") == "text"
            )

        emit_special(self._im_system, msg_idx)
        emit_text(role_name, msg_idx)
        emit_special(self._im_middle, msg_idx)
        if tool_call_id:
            emit_text(f"## Return of {tool_call_id}\n", msg_idx)
        emit_text(content, msg_idx)
        emit_special(self._im_end, msg_idx)
        emit_text("\n", msg_idx)

    def _normalize_response_tokens(self, response: list[int]) -> list[int]:
        """Restore the synthetic ``<think>`` prefill if the sampler stripped it.

        When thinking is enabled the generation prompt ends with a ``<think>``
        prefill. Some samplers strip the prefill from the returned token IDs.
        If the response contains ``</think>`` (encoded tokens) but does NOT
        start with ``<think>`` (encoded tokens), we prepend the ``<think>``
        tokens so the downstream text-based parser sees a complete block.
        """
        if not response:
            return response

        open_ids = self._think_open_ids
        close_ids = self._think_close_ids

        if not open_ids or not close_ids:
            return response

        # Check whether response starts with <think> token sequence
        starts_with_open = response[: len(open_ids)] == open_ids

        # Check whether </think> appears anywhere in the response
        contains_close = any(
            response[j : j + len(close_ids)] == close_ids
            for j in range(len(response) - len(close_ids) + 1)
        )

        if not starts_with_open and contains_close:
            return open_ids + response

        return response
