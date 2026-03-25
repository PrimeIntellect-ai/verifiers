"""Token-level parsing — Tinker-style, operates on token IDs directly.

Finds special token boundaries by scanning token IDs, then decodes only
the text segments between them. No regex on decoded text, no false positives
from content that happens to look like special tokens.
"""

from __future__ import annotations

import json

from renderers.base import ParsedResponse


def _find(ids: list[int], target: int, start: int = 0) -> int:
    """Find index of target in ids, or -1."""
    for i in range(start, len(ids)):
        if ids[i] == target:
            return i
    return -1


def _find_all(ids: list[int], target: int) -> list[int]:
    """Find all indices of target in ids."""
    return [i for i, t in enumerate(ids) if t == target]


def _strip_stop_tokens(ids: list[int], stop_ids: set[int]) -> list[int]:
    """Truncate at first stop token (model shouldn't generate past it)."""
    for i, t in enumerate(ids):
        if t in stop_ids:
            return ids[:i]
    return ids


def _decode(tokenizer, ids: list[int]) -> str:
    """Decode token IDs to text, skipping special tokens."""
    if not ids:
        return ""
    return tokenizer.decode(ids, skip_special_tokens=False)


# ── Qwen3: <tool_call> JSON </tool_call> ────────────────────────────


def parse_qwen3(
    tokenizer,
    token_ids: list[int],
    *,
    stop_ids: set[int],
    tool_call_id: int,
    tool_call_end_id: int,
) -> ParsedResponse:
    """Parse Qwen3 completion tokens. Hermes-style JSON tool calls."""
    ids = _strip_stop_tokens(token_ids, stop_ids)

    # No thinking tokens in Qwen3 gen prompt — model may or may not think
    # Parse from decoded text since <think>/<tool_call> may be multi-token in Qwen3
    # Actually in Qwen3, <tool_call> IS a special token (151657)
    # So we can find it by token ID

    # Find tool calls by token ID
    tc_start = _find(ids, tool_call_id)
    if tc_start != -1:
        content_ids = ids[:tc_start]
        # Extract all tool call blocks
        tool_calls = []
        i = tc_start
        while i < len(ids):
            if ids[i] == tool_call_id:
                end = _find(ids, tool_call_end_id, i + 1)
                if end == -1:
                    end = len(ids)
                tc_text = _decode(tokenizer, ids[i + 1 : end]).strip()
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
    else:
        content_ids = ids
        tool_calls = None

    text = _decode(tokenizer, content_ids)
    # Extract reasoning from text (Qwen3 doesn't have <think> as special token)
    reasoning = None
    if "</think>" in text:
        before, _, after = text.partition("</think>")
        reasoning = before.replace("<think>", "").strip("\n").strip()
        text = after.strip("\n")

    return ParsedResponse(
        content=text.strip(),
        reasoning_content=reasoning or None,
        tool_calls=tool_calls or None,
    )


# ── Qwen3.5: <tool_call> <function=name> <parameter=name> v </parameter> </function> </tool_call>


def parse_qwen35(
    tokenizer,
    token_ids: list[int],
    *,
    stop_ids: set[int],
    think_id: int,
    think_end_id: int,
    tool_call_id: int,
    tool_call_end_id: int,
) -> ParsedResponse:
    """Parse Qwen3.5 completion tokens. XML-style tool calls, token-level thinking."""
    ids = _strip_stop_tokens(token_ids, stop_ids)

    # Thinking: find </think> by token ID
    reasoning = None
    think_end = _find(ids, think_end_id)
    if think_end != -1:
        # Everything before </think> is reasoning
        reasoning_ids = ids[:think_end]
        # Strip <think> if present at start
        reasoning_ids = [t for t in reasoning_ids if t != think_id]
        reasoning = _decode(tokenizer, reasoning_ids).strip()
        ids = ids[think_end + 1 :]
    elif think_id in set(ids):
        # <think> present but no </think> — truncated reasoning
        think_start = _find(ids, think_id)
        reasoning = _decode(tokenizer, ids[think_start + 1 :]).strip()
        return ParsedResponse(
            content="", reasoning_content=reasoning or None, tool_calls=None
        )

    # Tool calls by token ID
    tc_start = _find(ids, tool_call_id)
    if tc_start != -1:
        content_text = _decode(tokenizer, ids[:tc_start]).strip()
        tool_calls = _parse_xml_tool_calls(
            tokenizer, ids[tc_start:], tool_call_id, tool_call_end_id
        )
    else:
        content_text = _decode(tokenizer, ids).strip()
        tool_calls = None

    return ParsedResponse(
        content=content_text,
        reasoning_content=reasoning or None,
        tool_calls=tool_calls or None,
    )


def _parse_xml_tool_calls(
    tokenizer, ids: list[int], tc_id: int, tc_end_id: int
) -> list[dict]:
    """Parse Qwen3.5-style XML tool calls from token IDs."""
    import re

    tool_calls = []
    i = 0
    while i < len(ids):
        if ids[i] == tc_id:
            end = _find(ids, tc_end_id, i + 1)
            if end == -1:
                break
            block_text = _decode(tokenizer, ids[i + 1 : end])
            name_match = re.search(r"<function=([^>]+)>", block_text)
            if name_match:
                name = name_match.group(1)
                arguments = {}
                for pm in re.finditer(
                    r"<parameter=([^>]+)>\n?(.*?)\n?</parameter>", block_text, re.DOTALL
                ):
                    arg_name = pm.group(1)
                    arg_value = pm.group(2).strip()
                    try:
                        arguments[arg_name] = json.loads(arg_value)
                    except (json.JSONDecodeError, ValueError):
                        arguments[arg_name] = arg_value
                tool_calls.append({"function": {"name": name, "arguments": arguments}})
            i = end + 1
        else:
            i += 1
    return tool_calls


# ── GLM-5/4.7/4.5: <tool_call> name <arg_key>k</arg_key> <arg_value>v</arg_value> </tool_call>


def parse_glm(
    tokenizer,
    token_ids: list[int],
    *,
    stop_ids: set[int],
    think_id: int,
    think_end_id: int,
    tool_call_id: int,
    tool_call_end_id: int,
    arg_key_id: int,
    arg_key_end_id: int,
    arg_value_id: int,
    arg_value_end_id: int,
) -> ParsedResponse:
    """Parse GLM completion tokens. Token-level thinking + arg_key/arg_value tool calls."""
    ids = _strip_stop_tokens(token_ids, stop_ids)

    # Thinking by token ID
    reasoning = None
    think_end = _find(ids, think_end_id)
    if think_end != -1:
        reasoning_ids = ids[:think_end]
        reasoning_ids = [t for t in reasoning_ids if t != think_id]
        reasoning = _decode(tokenizer, reasoning_ids).strip()
        ids = ids[think_end + 1 :]
    elif think_id in set(ids):
        think_start = _find(ids, think_id)
        reasoning = _decode(tokenizer, ids[think_start + 1 :]).strip()
        return ParsedResponse(
            content="", reasoning_content=reasoning or None, tool_calls=None
        )

    # Tool calls by token ID
    tc_start = _find(ids, tool_call_id)
    if tc_start != -1:
        content_text = _decode(tokenizer, ids[:tc_start]).strip()
        tool_calls = _parse_glm_tool_calls(
            tokenizer,
            ids[tc_start:],
            tool_call_id,
            tool_call_end_id,
            arg_key_id,
            arg_key_end_id,
            arg_value_id,
            arg_value_end_id,
        )
    else:
        content_text = _decode(tokenizer, ids).strip()
        tool_calls = None

    return ParsedResponse(
        content=content_text,
        reasoning_content=reasoning or None,
        tool_calls=tool_calls or None,
    )


def _parse_glm_tool_calls(
    tokenizer, ids, tc_id, tc_end_id, ak_id, ake_id, av_id, ave_id
) -> list[dict]:
    """Parse GLM-style tool calls: name + arg_key/arg_value pairs, all by token ID."""
    tool_calls = []
    i = 0
    while i < len(ids):
        if ids[i] == tc_id:
            end = _find(ids, tc_end_id, i + 1)
            if end == -1:
                break
            block = ids[i + 1 : end]
            # Name is everything before first <arg_key>
            first_ak = _find(block, ak_id)
            if first_ak == -1:
                name = _decode(tokenizer, block).strip()
                arguments = {}
            else:
                name = _decode(tokenizer, block[:first_ak]).strip()
                arguments = {}
                j = first_ak
                while j < len(block):
                    if block[j] == ak_id:
                        ake = _find(block, ake_id, j + 1)
                        if ake == -1:
                            break
                        key = _decode(tokenizer, block[j + 1 : ake]).strip()
                        av = _find(block, av_id, ake + 1)
                        if av == -1:
                            break
                        ave = _find(block, ave_id, av + 1)
                        if ave == -1:
                            break
                        val_text = _decode(tokenizer, block[av + 1 : ave]).strip()
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
    return tool_calls


# ── MiniMax: <minimax:tool_call> ... </minimax:tool_call> ────────────


def parse_minimax(
    tokenizer,
    token_ids: list[int],
    *,
    stop_ids: set[int],
    think_id: int,
    think_end_id: int,
    tool_call_id: int,
    tool_call_end_id: int,
) -> ParsedResponse:
    """Parse MiniMax M2 completion tokens."""
    ids = _strip_stop_tokens(token_ids, stop_ids)

    # Thinking: </think> by token ID. MiniMax doesn't generate <think> start.
    reasoning = None
    think_end = _find(ids, think_end_id)
    if think_end != -1:
        reasoning_ids = ids[:think_end]
        reasoning_ids = [t for t in reasoning_ids if t != think_id]
        reasoning = _decode(tokenizer, reasoning_ids).strip()
        ids = ids[think_end + 1 :]
    elif think_id in set(ids):
        think_start = _find(ids, think_id)
        reasoning = _decode(tokenizer, ids[think_start + 1 :]).strip()
        return ParsedResponse(
            content="", reasoning_content=reasoning or None, tool_calls=None
        )

    # Tool calls by token ID
    tc_start = _find(ids, tool_call_id)
    if tc_start != -1:
        content_text = _decode(tokenizer, ids[:tc_start]).strip()
        # Decode the tool call blocks and parse with regex (invoke/parameter are text, not tokens)
        tool_calls = []
        i = tc_start
        while i < len(ids):
            if ids[i] == tool_call_id:
                end = _find(ids, tool_call_end_id, i + 1)
                if end == -1:
                    break
                block_text = _decode(tokenizer, ids[i + 1 : end])
                import re

                for invoke_match in re.finditer(
                    r'<invoke name="([^"]+)">(.*?)</invoke>', block_text, re.DOTALL
                ):
                    name = invoke_match.group(1)
                    body = invoke_match.group(2)
                    arguments = {}
                    for pm in re.finditer(
                        r'<parameter name="([^"]+)">(.*?)</parameter>', body, re.DOTALL
                    ):
                        pname = pm.group(1)
                        pval = pm.group(2).strip()
                        try:
                            arguments[pname] = json.loads(pval)
                        except (json.JSONDecodeError, ValueError):
                            arguments[pname] = pval
                    tool_calls.append(
                        {"function": {"name": name, "arguments": arguments}}
                    )
                i = end + 1
            else:
                i += 1
    else:
        content_text = _decode(tokenizer, ids).strip()
        tool_calls = None

    return ParsedResponse(
        content=content_text,
        reasoning_content=reasoning or None,
        tool_calls=tool_calls or None,
    )
