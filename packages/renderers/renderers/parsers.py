"""Pluggable tool-call and reasoning parsers.

Each parser knows how to pull a specific kind of structured output out of a
completion. They're registered by name so callers can compose a
DefaultRenderer with their own tool / reasoning parsing logic without having
to author a whole Renderer.

ToolParser operates on token ids (most tool formats use special tokens as
delimiters, so matching by id is more reliable than regex on decoded text).

ReasoningParser operates on decoded text (reasoning delimiters are typically
``<think>...</think>`` and the text-vs-token distinction here rarely matters
for the final extraction).
"""

from __future__ import annotations

import json
import re
from typing import Protocol, runtime_checkable


# ── Shared helpers ───────────────────────────────────────────────────


def _find(ids: list[int], target: int, start: int = 0) -> int:
    for i in range(start, len(ids)):
        if ids[i] == target:
            return i
    return -1


def _decode(tokenizer, ids: list[int]) -> str:
    if not ids:
        return ""
    return tokenizer.decode(ids, skip_special_tokens=False)


def _token_id(tokenizer, token: str) -> int | None:
    """Return the id for *token* if present in the vocab, else None."""
    tid = tokenizer.convert_tokens_to_ids(token)
    # HF returns either the unk id or None for missing tokens. Treat the unk id
    # as missing here so we don't accidentally collide on it.
    unk = getattr(tokenizer, "unk_token_id", None)
    if tid is None or tid == unk:
        return None
    return tid


# ── Protocols ────────────────────────────────────────────────────────


@runtime_checkable
class ToolParser(Protocol):
    """Extracts tool calls from completion token ids.

    ``extract`` returns a tuple ``(content_ids, tool_calls)`` where
    ``content_ids`` is the remaining content token ids with the tool-call
    section removed, and ``tool_calls`` is a list of
    ``{"function": {"name": str, "arguments": dict | str}}`` dicts or
    ``None`` when no tool call was found.
    """

    def __init__(self, tokenizer): ...
    def extract(self, token_ids: list[int]) -> tuple[list[int], list[dict] | None]: ...


@runtime_checkable
class ReasoningParser(Protocol):
    """Splits reasoning content from final content.

    ``extract`` returns ``(reasoning_text_or_None, remaining_text)``.
    """

    def __init__(self, tokenizer): ...
    def extract(self, text: str) -> tuple[str | None, str]: ...


# ── Tool parsers ─────────────────────────────────────────────────────


class Qwen3ToolParser:
    """Hermes-style JSON tool calls: ``<tool_call>{"name": .., "arguments": ..}</tool_call>``."""

    def __init__(self, tokenizer):
        self._tokenizer = tokenizer
        self._tc_id = _token_id(tokenizer, "<tool_call>")
        self._tc_end_id = _token_id(tokenizer, "</tool_call>")

    def extract(self, ids: list[int]) -> tuple[list[int], list[dict] | None]:
        if self._tc_id is None:
            return ids, None
        tc_start = _find(ids, self._tc_id)
        if tc_start == -1:
            return ids, None
        content_ids = ids[:tc_start]
        tool_calls: list[dict] = []
        i = tc_start
        while i < len(ids):
            if ids[i] == self._tc_id:
                end = (
                    _find(ids, self._tc_end_id, i + 1)
                    if self._tc_end_id is not None
                    else -1
                )
                if end == -1:
                    end = len(ids)
                tc_text = _decode(self._tokenizer, ids[i + 1 : end]).strip()
                try:
                    parsed = json.loads(tc_text)
                    tool_calls.append(
                        {
                            "function": {
                                "name": parsed.get("name", ""),
                                "arguments": parsed.get("arguments", {}),
                            }
                        }
                    )
                except json.JSONDecodeError:
                    pass
                i = end + 1
            else:
                i += 1
        return content_ids, (tool_calls or None)


class Qwen35ToolParser:
    """XML-style tool calls: ``<tool_call><function=N><parameter=K>V</parameter>...</function></tool_call>``."""

    def __init__(self, tokenizer):
        self._tokenizer = tokenizer
        self._tc_id = _token_id(tokenizer, "<tool_call>")
        self._tc_end_id = _token_id(tokenizer, "</tool_call>")

    def extract(self, ids: list[int]) -> tuple[list[int], list[dict] | None]:
        if self._tc_id is None:
            return ids, None
        tc_start = _find(ids, self._tc_id)
        if tc_start == -1:
            return ids, None
        tool_calls: list[dict] = []
        i = tc_start
        while i < len(ids):
            if ids[i] == self._tc_id:
                end = (
                    _find(ids, self._tc_end_id, i + 1)
                    if self._tc_end_id is not None
                    else -1
                )
                if end == -1:
                    break
                block_text = _decode(self._tokenizer, ids[i + 1 : end])
                name_match = re.search(r"<function=([^>]+)>", block_text)
                if name_match:
                    name = name_match.group(1)
                    arguments: dict = {}
                    for pm in re.finditer(
                        r"<parameter=([^>]+)>\n?(.*?)\n?</parameter>",
                        block_text,
                        re.DOTALL,
                    ):
                        arg_name = pm.group(1)
                        arg_value = pm.group(2).strip()
                        try:
                            arguments[arg_name] = json.loads(arg_value)
                        except (json.JSONDecodeError, ValueError):
                            arguments[arg_name] = arg_value
                    tool_calls.append(
                        {"function": {"name": name, "arguments": arguments}}
                    )
                i = end + 1
            else:
                i += 1
        return ids[:tc_start], (tool_calls or None)


class GlmToolParser:
    """GLM-5/4.5 token-level tool calls with ``<arg_key>``/``<arg_value>`` pairs."""

    def __init__(self, tokenizer):
        self._tokenizer = tokenizer
        self._tc_id = _token_id(tokenizer, "<tool_call>")
        self._tc_end_id = _token_id(tokenizer, "</tool_call>")
        self._ak_id = _token_id(tokenizer, "<arg_key>")
        self._ake_id = _token_id(tokenizer, "</arg_key>")
        self._av_id = _token_id(tokenizer, "<arg_value>")
        self._ave_id = _token_id(tokenizer, "</arg_value>")

    def extract(self, ids: list[int]) -> tuple[list[int], list[dict] | None]:
        if self._tc_id is None:
            return ids, None
        tc_start = _find(ids, self._tc_id)
        if tc_start == -1:
            return ids, None
        tool_calls: list[dict] = []
        i = tc_start
        while i < len(ids):
            if ids[i] == self._tc_id:
                end = (
                    _find(ids, self._tc_end_id, i + 1)
                    if self._tc_end_id is not None
                    else -1
                )
                if end == -1:
                    break
                block = ids[i + 1 : end]
                first_ak = _find(block, self._ak_id) if self._ak_id is not None else -1
                if first_ak == -1:
                    name = _decode(self._tokenizer, block).strip()
                    arguments: dict = {}
                else:
                    name = _decode(self._tokenizer, block[:first_ak]).strip()
                    arguments = {}
                    j = first_ak
                    while j < len(block):
                        if block[j] == self._ak_id:
                            ake = (
                                _find(block, self._ake_id, j + 1)
                                if self._ake_id is not None
                                else -1
                            )
                            if ake == -1:
                                break
                            key = _decode(self._tokenizer, block[j + 1 : ake]).strip()
                            av = (
                                _find(block, self._av_id, ake + 1)
                                if self._av_id is not None
                                else -1
                            )
                            if av == -1:
                                break
                            ave = (
                                _find(block, self._ave_id, av + 1)
                                if self._ave_id is not None
                                else -1
                            )
                            if ave == -1:
                                break
                            val_text = _decode(
                                self._tokenizer, block[av + 1 : ave]
                            ).strip()
                            try:
                                arguments[key] = json.loads(val_text)
                            except (json.JSONDecodeError, ValueError):
                                arguments[key] = val_text
                            j = ave + 1
                        else:
                            j += 1
                tool_calls.append({"function": {"name": name, "arguments": arguments}})
                i = end + 1
            else:
                i += 1
        return ids[:tc_start], (tool_calls or None)


class DeepSeekV3ToolParser:
    """DeepSeek V3 tool calls, delimited by ``<｜tool▁calls▁begin｜> ... <｜tool▁calls▁end｜>``."""

    def __init__(self, tokenizer):
        self._tokenizer = tokenizer
        self._tcs_begin = _token_id(tokenizer, "<｜tool▁calls▁begin｜>")
        self._tcs_end = _token_id(tokenizer, "<｜tool▁calls▁end｜>")
        self._tc_begin = _token_id(tokenizer, "<｜tool▁call▁begin｜>")
        self._tc_end = _token_id(tokenizer, "<｜tool▁call▁end｜>")
        self._sep = _token_id(tokenizer, "<｜tool▁sep｜>")

    def extract(self, ids: list[int]) -> tuple[list[int], list[dict] | None]:
        if self._tcs_begin is None:
            return ids, None
        section_start = _find(ids, self._tcs_begin)
        if section_start == -1:
            return ids, None
        content_ids = ids[:section_start]
        section_end = (
            _find(ids, self._tcs_end, section_start + 1)
            if self._tcs_end is not None
            else -1
        )
        if section_end == -1:
            section_end = len(ids)
        section_ids = ids[section_start + 1 : section_end]

        tool_calls: list[dict] = []
        i = 0
        while i < len(section_ids):
            if self._tc_begin is None or section_ids[i] != self._tc_begin:
                i += 1
                continue
            end = (
                _find(section_ids, self._tc_end, i + 1)
                if self._tc_end is not None
                else -1
            )
            if end == -1:
                end = len(section_ids)
            block_text = _decode(self._tokenizer, section_ids[i + 1 : end])
            # Format: "function<｜tool▁sep｜>{name}\n```json\n{args}\n```"
            name_match = re.search(r"^\s*\w+.*?([A-Za-z0-9_]+)\s*\n", block_text)
            name = name_match.group(1) if name_match else ""
            args: dict | str = {}
            json_match = re.search(r"```json\s*(.*?)\s*```", block_text, re.DOTALL)
            if json_match:
                try:
                    args = json.loads(json_match.group(1))
                except (json.JSONDecodeError, ValueError):
                    args = json_match.group(1)
            tool_calls.append({"function": {"name": name, "arguments": args}})
            i = end + 1

        return content_ids, (tool_calls or None)


# ── Reasoning parsers ────────────────────────────────────────────────


class ThinkTextReasoningParser:
    """Extracts ``<think>...</think>`` from decoded text.

    Works regardless of whether ``<think>`` is a special token or plain text —
    we decode with ``skip_special_tokens=False`` so both render identically.
    """

    def __init__(self, tokenizer):
        self._tokenizer = tokenizer  # unused, for Protocol compat

    def extract(self, text: str) -> tuple[str | None, str]:
        if "</think>" not in text:
            return None, text
        before, _, after = text.partition("</think>")
        if "<think>" in before:
            reasoning = before.split("<think>", 1)[-1]
        else:
            reasoning = before
        # Preserve whitespace on both sides of the `</think>` boundary —
        # chat templates round-trip it verbatim (GLM-4.5 emits reasoning and
        # content back-to-back with no separator), so stripping here causes
        # re-render drift that breaks the trajectory-step extension property.
        return (reasoning or None), after


# ── Registries ───────────────────────────────────────────────────────


TOOL_PARSERS: dict[str, type] = {
    "qwen3": Qwen3ToolParser,
    "qwen3.5": Qwen35ToolParser,
    "glm": GlmToolParser,
    "deepseek_v3": DeepSeekV3ToolParser,
}

REASONING_PARSERS: dict[str, type] = {
    "think": ThinkTextReasoningParser,
}


def get_tool_parser(name: str, tokenizer) -> ToolParser:
    """Look up a tool parser by name and instantiate it with *tokenizer*."""
    if name not in TOOL_PARSERS:
        available = ", ".join(sorted(TOOL_PARSERS))
        raise ValueError(f"Unknown tool_parser {name!r}. Available: {available}.")
    return TOOL_PARSERS[name](tokenizer)


def get_reasoning_parser(name: str, tokenizer) -> ReasoningParser:
    """Look up a reasoning parser by name and instantiate it with *tokenizer*."""
    if name not in REASONING_PARSERS:
        available = ", ".join(sorted(REASONING_PARSERS))
        raise ValueError(f"Unknown reasoning_parser {name!r}. Available: {available}.")
    return REASONING_PARSERS[name](tokenizer)
