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


# ── DeepSeek V3: <｜tool▁calls▁begin｜>...<｜tool▁calls▁end｜> + text <think> tags ──


def parse_deepseek_v3(
    tokenizer,
    token_ids: list[int],
    *,
    stop_ids: set[int],
    tool_calls_begin_id: int,
    tool_calls_end_id: int,
    tool_call_begin_id: int,
    tool_call_end_id: int,
    tool_sep_id: int,
) -> ParsedResponse:
    """Parse DeepSeek V3 completion tokens.

    Thinking is embedded as plain text <think>...</think> tags (not special tokens).
    Tool calls are delimited by special tokens:
        <｜tool▁calls▁begin｜>
          <｜tool▁call▁begin｜>function<｜tool▁sep｜>{name}\\n```json\\n{args}\\n```<｜tool▁call▁end｜>
        <｜tool▁calls▁end｜>
    """
    ids = _strip_stop_tokens(token_ids, stop_ids)

    # ── Tool calls ──────────────────────────────────────────────────
    tc_section_start = _find(ids, tool_calls_begin_id)
    if tc_section_start != -1:
        content_ids = ids[:tc_section_start]
        tool_calls = _parse_deepseek_tool_calls(
            tokenizer,
            ids[tc_section_start:],
            tool_calls_begin_id,
            tool_calls_end_id,
            tool_call_begin_id,
            tool_call_end_id,
            tool_sep_id,
        )
    else:
        content_ids = ids
        tool_calls = None

    text = _decode(tokenizer, content_ids)

    # ── Thinking from text tags ────────────────────────────────────
    reasoning = None
    if "</think>" in text:
        before, _, after = text.partition("</think>")
        reasoning = before.replace("<think>", "").lstrip("\n").rstrip("\n").strip()
        text = after.lstrip("\n")

    return ParsedResponse(
        content=text.strip(),
        reasoning_content=reasoning or None,
        tool_calls=tool_calls or None,
    )


def _parse_deepseek_tool_calls(
    tokenizer,
    ids: list[int],
    tc_begin_id: int,
    tc_end_id: int,
    call_begin_id: int,
    call_end_id: int,
    sep_id: int,
) -> list[dict] | None:
    """Parse DeepSeek V3-style tool calls from token IDs.

    Each individual tool call is delimited by <｜tool▁call▁begin｜> ... <｜tool▁call▁end｜>.
    Inside, <｜tool▁sep｜> separates the call type (e.g. "function") from the
    function name and JSON arguments block.
    """
    import re

    tool_calls: list[dict] = []

    # Find the outer section boundaries.
    section_start = _find(ids, tc_begin_id)
    if section_start == -1:
        return None
    section_end = _find(ids, tc_end_id, section_start + 1)
    if section_end == -1:
        section_end = len(ids)

    section_ids = ids[section_start + 1 : section_end]

    i = 0
    while i < len(section_ids):
        if section_ids[i] == call_begin_id:
            end = _find(section_ids, call_end_id, i + 1)
            if end == -1:
                end = len(section_ids)

            call_ids = section_ids[i + 1 : end]

            # Find <｜tool▁sep｜> to split type from name+args.
            sep_pos = _find(call_ids, sep_id)
            if sep_pos == -1:
                # Malformed — skip.
                i = end + 1
                continue

            # Everything after <｜tool▁sep｜> is the name and args block.
            after_sep_ids = call_ids[sep_pos + 1 :]
            after_sep_text = _decode(tokenizer, after_sep_ids).strip()

            # Extract function name and JSON arguments.
            # Format: "{name}\n```json\n{args}\n```"
            # But we also gracefully handle raw JSON without the code fence.
            name = ""
            args_str = ""

            # Try to split on first newline to get name, then find JSON.
            newline_pos = after_sep_text.find("\n")
            if newline_pos != -1:
                name = after_sep_text[:newline_pos].strip()
                rest = after_sep_text[newline_pos + 1 :].strip()
                # Strip optional ```json ... ``` fence.
                fence_match = re.match(r"```(?:json)?\s*([\s\S]*?)\s*```$", rest)
                if fence_match:
                    args_str = fence_match.group(1).strip()
                else:
                    args_str = rest
            else:
                # No newline — treat entire text as name, no args.
                name = after_sep_text

            # Parse arguments as JSON.
            try:
                arguments = json.loads(args_str) if args_str else {}
            except json.JSONDecodeError:
                arguments = args_str  # preserve raw string on failure

            tool_calls.append({"function": {"name": name, "arguments": arguments}})
            i = end + 1
        else:
            i += 1

    return tool_calls if tool_calls else None


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


# ── Kimi K2: <|tool_calls_section_begin|> ... <|tool_calls_section_end|> ────


def parse_kimi_k2(
    tokenizer,
    token_ids: list[int],
    *,
    stop_ids: set[int],
    tool_calls_section_begin_id: int,
    tool_calls_section_end_id: int,
    tool_call_begin_id: int,
    tool_call_argument_begin_id: int,
    tool_call_end_id: int,
) -> ParsedResponse:
    """Parse Kimi K2 completion tokens.

    Thinking is encoded as text tags <think>...</think>.
    Tool calls use section/call-level special tokens.
    Tool call IDs are in format ``functions.name:index``.
    """
    ids = _strip_stop_tokens(token_ids, stop_ids)

    # ── Tool calls ────────────────────────────────────────────────
    section_start = _find(ids, tool_calls_section_begin_id)
    if section_start != -1:
        content_ids = ids[:section_start]
        section_end = _find(ids, tool_calls_section_end_id, section_start + 1)
        if section_end == -1:
            section_end = len(ids)
        section_ids = ids[section_start + 1 : section_end]
        tool_calls = _parse_kimi_k2_tool_calls(
            tokenizer,
            section_ids,
            tool_call_begin_id,
            tool_call_argument_begin_id,
            tool_call_end_id,
        )
    else:
        content_ids = ids
        tool_calls = None

    # ── Thinking from text tags ───────────────────────────────────
    text = _decode(tokenizer, content_ids)
    reasoning: str | None = None
    if "</think>" in text:
        before, _, after = text.partition("</think>")
        raw_think = before.replace("<think>", "", 1)
        reasoning = raw_think.strip("\n").strip() or None
        text = after.strip("\n")
    elif "<think>" in text:
        # Truncated thinking (no closing tag)
        raw_think = text.split("<think>", 1)[1]
        reasoning = raw_think.strip("\n").strip() or None
        return ParsedResponse(
            content="",
            reasoning_content=reasoning,
            tool_calls=None,
        )

    return ParsedResponse(
        content=text.strip(),
        reasoning_content=reasoning,
        tool_calls=tool_calls or None,
    )


def _parse_kimi_k2_tool_calls(
    tokenizer,
    ids: list[int],
    tc_begin_id: int,
    tc_arg_begin_id: int,
    tc_end_id: int,
) -> list[dict]:
    """Parse individual Kimi K2 tool calls from the section token IDs.

    Format per call:
        <|tool_call_begin|>{id}<|tool_call_argument_begin|>{json_args}<|tool_call_end|>

    The ``id`` is in format ``functions.name:index``; the function name is
    extracted by stripping the ``functions.`` prefix and ``:index`` suffix.
    """
    tool_calls: list[dict] = []
    i = 0
    while i < len(ids):
        if ids[i] == tc_begin_id:
            # Find <|tool_call_argument_begin|>
            arg_begin = _find(ids, tc_arg_begin_id, i + 1)
            if arg_begin == -1:
                break
            # Find <|tool_call_end|>
            tc_end = _find(ids, tc_end_id, arg_begin + 1)
            if tc_end == -1:
                tc_end = len(ids)

            raw_id = _decode(tokenizer, ids[i + 1 : arg_begin]).strip()
            args_str = _decode(tokenizer, ids[arg_begin + 1 : tc_end]).strip()

            # Extract function name from "functions.name:index"
            name_part = raw_id.split(":", 1)[0]
            if "." in name_part:
                _, func_name = name_part.split(".", 1)
            else:
                func_name = name_part

            try:
                arguments = json.loads(args_str)
            except json.JSONDecodeError:
                arguments = args_str

            tool_calls.append(
                {
                    "id": raw_id,
                    "type": "function",
                    "function": {
                        "name": func_name,
                        "arguments": arguments,
                    },
                }
            )
            i = tc_end + 1
        else:
            i += 1
    return tool_calls


# ── GptOss (Harmony): <|start|>role<|channel|>ch<|message|>content<|end|/return|/call|>


def parse_gpt_oss(
    tokenizer,
    token_ids: list[int],
    *,
    return_id: int,
    call_id: int,
    start_id: int,
    end_id: int,
    channel_id: int,
    message_id: int,
    constrain_id: int,
) -> ParsedResponse:
    """Parse GptOss (Harmony) completion tokens.

    Finds the earliest terminal token (<|return|> or <|call|>), then walks the
    token stream block-by-block to extract:

    - analysis channel              → reasoning_content
    - final channel                 → content
    - commentary with to=functions.*  → tool_calls (JSON arguments)
    - commentary without recipient  → content (preamble text)
    """
    import re

    # Locate terminal tokens
    return_pos = _find(token_ids, return_id)
    call_pos = _find(token_ids, call_id)

    if return_pos == -1 and call_pos == -1:
        # No terminal found — decode everything as raw content
        return ParsedResponse(
            content=_decode(tokenizer, token_ids),
            reasoning_content=None,
            tool_calls=None,
        )

    # Pick earliest terminal
    if return_pos == -1:
        stop_pos = call_pos
    elif call_pos == -1:
        stop_pos = return_pos
    else:
        stop_pos = min(return_pos, call_pos)

    ids = token_ids[:stop_pos]

    reasoning_parts: list[str] = []
    content_parts: list[str] = []
    tool_calls: list[dict] = []

    i = 0
    while i < len(ids):
        if ids[i] != start_id:
            i += 1
            continue

        # Find <|message|> that terminates this block's header
        msg_pos = _find(ids, message_id, i + 1)
        if msg_pos == -1:
            break

        # Header: tokens between <|start|> and <|message|>
        header_ids = ids[i + 1 : msg_pos]
        header_text = _decode(tokenizer, header_ids)

        # Body: tokens from after <|message|> up to next <|start|> or <|end|>
        body_start = msg_pos + 1
        next_start = _find(ids, start_id, body_start)
        next_end = _find(ids, end_id, body_start)

        if next_start == -1 and next_end == -1:
            body_end = len(ids)
        elif next_start == -1:
            body_end = next_end
        elif next_end == -1:
            body_end = next_start
        else:
            body_end = min(next_start, next_end)

        body_text = _decode(tokenizer, ids[body_start:body_end])

        # Extract channel: token after <|channel|> in header_ids
        channel = _gptoss_extract_after_token(tokenizer, header_ids, channel_id)

        # Extract recipient: "to=..." field in header text
        recipient_match = re.search(r"to=([^\s<]+)", header_text)
        recipient = recipient_match.group(1) if recipient_match else None

        if recipient and recipient.startswith("functions."):
            tool_name = recipient[len("functions."):]
            try:
                arguments = json.loads(body_text)
            except json.JSONDecodeError:
                arguments = body_text  # preserve raw string on failure
            tool_calls.append({
                "function": {
                    "name": tool_name,
                    "arguments": arguments,
                }
            })
        elif channel == "analysis":
            reasoning_parts.append(body_text)
        elif channel == "final":
            content_parts.append(body_text)
        elif channel == "commentary":
            # Commentary without a tool recipient is a user-visible preamble
            content_parts.append(body_text)

        # Advance: skip body + optional <|end|>
        i = body_end
        if i < len(ids) and ids[i] == end_id:
            i += 1

    reasoning = "".join(reasoning_parts).strip() or None
    content = "".join(content_parts).strip()

    return ParsedResponse(
        content=content,
        reasoning_content=reasoning,
        tool_calls=tool_calls or None,
    )


def _gptoss_extract_after_token(
    tokenizer,
    header_ids: list[int],
    marker_id: int,
) -> str | None:
    """Return the first decoded word appearing after marker_id in header_ids."""
    pos = _find(header_ids, marker_id)
    if pos == -1:
        return None
    after = _decode(tokenizer, header_ids[pos + 1:]).strip()
    # Take first whitespace-delimited word (channel name)
    return after.split()[0] if after else None
