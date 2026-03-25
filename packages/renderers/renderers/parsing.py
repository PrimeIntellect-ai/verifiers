"""Shared parsing logic — mirrors vLLM's non-streaming tool call and reasoning parsers.

Each function takes decoded text and returns structured results,
matching vLLM's extract_tool_calls / extract_reasoning behavior exactly.
"""

from __future__ import annotations

import json
import re
from typing import Any

from renderers.base import ParsedResponse

# ── Reasoning extraction (mirrors vLLM reasoning parsers) ────────────


def extract_reasoning_qwen(text: str) -> tuple[str | None, str]:
    """Qwen3/Qwen3.5 reasoning: split on </think>. Mirrors qwen3_reasoning_parser."""
    # Strip <think> if present in generated output
    has_think_tag = "<think>" in text
    parts = text.partition("<think>")
    text = parts[2] if parts[1] else parts[0]

    if "</think>" not in text:
        if has_think_tag:
            # Had <think> but no </think> — truncated reasoning
            return text.strip() or None, ""
        # No thinking markers at all — plain content
        return None, text

    reasoning, _, content = text.partition("</think>")
    return reasoning.strip() or None, content.strip("\n")


def extract_reasoning_glm(text: str) -> tuple[str | None, str]:
    """GLM-5/4.5/4.7 reasoning: split on </think>. Gen prompt starts with <think>."""
    if "</think>" in text:
        before, after = text.split("</think>", 1)
        if "<think>" in before:
            before = before.split("<think>")[-1]
        reasoning = before.strip()
        return reasoning or None, after
    # No </think> — check if <think> present (truncated thinking)
    if "<think>" in text:
        clean = text.replace("<think>", "").strip()
        return clean or None, ""
    return None, text


def extract_reasoning_minimax(text: str) -> tuple[str | None, str]:
    """MiniMax M2: all content before </think> is reasoning (no <think> start token generated)."""
    if "</think>" in text:
        before, after = text.split("</think>", 1)
        if "<think>" in before:
            before = before.split("<think>")[-1]
        reasoning = before.strip("\n").strip()
        return reasoning or None, after.strip("\n")
    # No </think> — if <think> present, truncated
    if "<think>" in text:
        clean = text.replace("<think>", "").strip()
        return clean or None, ""
    return None, text


def extract_reasoning_kimi(text: str) -> tuple[str | None, str]:
    """Kimi K2: reasoning ends at </think> OR <|tool_calls_section_begin|>. Mirrors kimi_k2_reasoning_parser."""
    has_think_tag = "<think>" in text
    # Strip <think> if present
    start = 0
    if has_think_tag:
        start = text.find("<think>") + len("<think>")

    end = text.find("</think>")
    if end != -1:
        reasoning = text[start:end].strip()
        content = text[end + len("</think>"):]
        return reasoning or None, content

    # Implicit end at tool section
    tool_idx = text.find("<|tool_calls_section_begin|>")
    if tool_idx != -1:
        reasoning = text[start:tool_idx].strip()
        return reasoning or None, text[tool_idx:]

    if has_think_tag:
        # Had <think> but no end marker — truncated reasoning
        reasoning = text[start:].strip()
        return reasoning or None, ""

    # No thinking markers — plain content
    return None, text


# ── Tool call extraction (mirrors vLLM tool parsers) ─────────────────

# Hermes (Qwen3): <tool_call>JSON</tool_call>
_HERMES_REGEX = re.compile(r"<tool_call>(.*?)</tool_call>|<tool_call>(.*)", re.DOTALL)


def extract_tool_calls_hermes(text: str) -> tuple[list[dict] | None, str]:
    """Hermes tool parser (Qwen3). Mirrors hermes_tool_parser."""
    if "<tool_call>" not in text:
        return None, text

    content = text[:text.find("<tool_call>")].strip()
    tool_calls = []
    for match in _HERMES_REGEX.findall(text):
        raw = match[0] if match[0] else match[1]
        raw = raw.strip()
        if not raw:
            continue
        try:
            parsed = json.loads(raw)
            tool_calls.append({
                "function": {
                    "name": parsed.get("name", ""),
                    "arguments": parsed.get("arguments", {}),
                }
            })
        except json.JSONDecodeError:
            pass

    return tool_calls or None, content


# Qwen3.5 XML: <tool_call><function=name><parameter=name>val</parameter></function></tool_call>
_QWEN35_FUNC_REGEX = re.compile(r"<function=([^>]+)>", re.DOTALL)
_QWEN35_PARAM_REGEX = re.compile(r"<parameter=([^>]+)>\n?(.*?)\n?</parameter>", re.DOTALL)


def extract_tool_calls_qwen35xml(text: str) -> tuple[list[dict] | None, str]:
    """Qwen3.5 XML tool parser. Mirrors qwen3xml_tool_parser."""
    if "<tool_call>" not in text:
        return None, text

    content = text[:text.find("<tool_call>")].strip()
    tool_calls = []
    for tc_block in text.split("<tool_call>")[1:]:
        if "</tool_call>" in tc_block:
            tc_block = tc_block.split("</tool_call>")[0]
        name_match = _QWEN35_FUNC_REGEX.search(tc_block)
        if not name_match:
            continue
        name = name_match.group(1)
        arguments = {}
        for param_match in _QWEN35_PARAM_REGEX.finditer(tc_block):
            arg_name = param_match.group(1)
            arg_value = param_match.group(2).strip()
            try:
                arguments[arg_name] = json.loads(arg_value)
            except (json.JSONDecodeError, ValueError):
                arguments[arg_name] = arg_value
        tool_calls.append({"function": {"name": name, "arguments": arguments}})

    return tool_calls or None, content


# GLM: <tool_call>name<arg_key>k</arg_key><arg_value>v</arg_value></tool_call>
_GLM_FUNC_CALL_REGEX = re.compile(r"<tool_call>.*?</tool_call>", re.DOTALL)
_GLM47_FUNC_DETAIL_REGEX = re.compile(r"<tool_call>(.*?)(<arg_key>.*?)?</tool_call>", re.DOTALL)
_GLM47_FUNC_ARG_REGEX = re.compile(r"<arg_key>(.*?)</arg_key>(?:\n|\s)*<arg_value>(.*?)</arg_value>", re.DOTALL)


def extract_tool_calls_glm(text: str) -> tuple[list[dict] | None, str]:
    """GLM-4.5/GLM-5/GLM-4.7 tool parser. Mirrors glm4_moe/glm47_moe_tool_parser."""
    if "<tool_call>" not in text:
        return None, text

    content = text[:text.find("<tool_call>")].strip()
    tool_calls = []
    for match in _GLM_FUNC_CALL_REGEX.findall(text):
        detail = _GLM47_FUNC_DETAIL_REGEX.search(match)
        if not detail:
            continue
        tc_name = detail.group(1).strip()
        tc_args_text = detail.group(2) or ""
        pairs = _GLM47_FUNC_ARG_REGEX.findall(tc_args_text)
        arg_dict: dict[str, Any] = {}
        for key, value in pairs:
            key = key.strip()
            value = value.strip()
            try:
                arg_dict[key] = json.loads(value)
            except (json.JSONDecodeError, ValueError):
                arg_dict[key] = value
        tool_calls.append({"function": {"name": tc_name, "arguments": arg_dict}})

    return tool_calls or None, content


# MiniMax: <minimax:tool_call><invoke name="n"><parameter name="k">v</parameter></invoke></minimax:tool_call>
_MINIMAX_TC_REGEX = re.compile(r"<minimax:tool_call>(.*?)</minimax:tool_call>", re.DOTALL)
_MINIMAX_INVOKE_REGEX = re.compile(r'<invoke name=(.*?)</invoke>', re.DOTALL)
_MINIMAX_PARAM_REGEX = re.compile(r'<parameter name=(.*?)</parameter>', re.DOTALL)


def extract_tool_calls_minimax(text: str) -> tuple[list[dict] | None, str]:
    """MiniMax M2 tool parser. Mirrors minimax_m2_tool_parser."""
    if "<minimax:tool_call>" not in text:
        return None, text

    content = text[:text.find("<minimax:tool_call>")].strip()
    tool_calls = []
    for tc_match in _MINIMAX_TC_REGEX.findall(text):
        for invoke_match in _MINIMAX_INVOKE_REGEX.findall(tc_match):
            # Extract name (strip quotes)
            parts = invoke_match.split(">", 1)
            name = parts[0].strip().strip("\"'")
            body = parts[1] if len(parts) > 1 else ""
            arguments = {}
            for param_match in _MINIMAX_PARAM_REGEX.findall(body):
                pparts = param_match.split(">", 1)
                pname = pparts[0].strip().strip("\"'")
                pval = pparts[1].strip() if len(pparts) > 1 else ""
                try:
                    arguments[pname] = json.loads(pval)
                except (json.JSONDecodeError, ValueError):
                    arguments[pname] = pval
            tool_calls.append({"function": {"name": name, "arguments": arguments}})

    return tool_calls or None, content


# Kimi: <|tool_call_begin|>name:0<|tool_call_argument_begin|>args<|tool_call_end|>
_KIMI_TC_REGEX = re.compile(
    r"<\|tool_call_begin\|>\s*(?P<tool_call_id>[^<]+:\d+)\s*<\|tool_call_argument_begin\|>\s*(?P<function_arguments>(?:(?!<\|tool_call_begin\|>).)*?)\s*<\|tool_call_end\|>",
    re.DOTALL,
)


def extract_tool_calls_kimi(text: str) -> tuple[list[dict] | None, str]:
    """Kimi K2 tool parser. Mirrors kimi_k2_tool_parser."""
    if "<|tool_calls_section_begin|>" not in text:
        return None, text

    content = text[:text.find("<|tool_calls_section_begin|>")].strip()
    tool_calls = []
    for match in _KIMI_TC_REGEX.finditer(text):
        function_id = match.group("tool_call_id").strip()
        function_args = match.group("function_arguments").strip()
        # function_id: functions.get_weather:0 or get_weather:0
        function_name = function_id.split(":")[0].split(".")[-1]
        tool_calls.append({
            "function": {
                "name": function_name,
                "arguments": function_args,
            }
        })

    return tool_calls or None, content


# ── Unified parse_response builders ──────────────────────────────────


def build_parsed_response(
    reasoning: str | None,
    content: str,
    tool_calls: list[dict] | None,
) -> ParsedResponse:
    return ParsedResponse(
        content=content.strip() if content else "",
        reasoning_content=reasoning.strip() if reasoning else None,
        tool_calls=tool_calls,
    )
