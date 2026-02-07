import json
import re
import uuid
from typing import cast

from verifiers.types import (
    ChatCompletion,
    ChatMessage,
    Completion,
    Messages,
    MessageType,
    ModelResponse,
    TrajectoryStepTokens,
)


async def parse_response_tokens(
    response: ModelResponse, message_type: MessageType, max_seq_len: int | None = None
) -> TrajectoryStepTokens | None:
    if message_type == "chat":
        assert isinstance(response, ChatCompletion)
        assert len(response.choices) == 1, "Response should always have one choice"
        if not hasattr(response.choices[0], "token_ids"):
            return None
        if not hasattr(response, "prompt_token_ids"):
            return None
        if not hasattr(response.choices[0], "logprobs"):
            return None
        if response.choices[0].logprobs is None:
            return None
        has_logprobs_obj = (
            hasattr(response.choices[0].logprobs, "content")
            and response.choices[0].logprobs.content is not None
        )
        has_logprobs_dict = (
            isinstance(response.choices[0].logprobs, dict)
            and "content" in response.choices[0].logprobs.keys()
            and response.choices[0].logprobs["content"] is not None
        )
        if not (has_logprobs_obj or has_logprobs_dict):
            return None
        prompt_ids = getattr(response, "prompt_token_ids")
        prompt_mask = [0] * len(prompt_ids)
        completion_ids = getattr(response.choices[0], "token_ids")
        completion_mask = [1] * len(completion_ids)
        if has_logprobs_obj:
            assert response.choices[0].logprobs.content is not None
            logprobs_content = response.choices[0].logprobs.content
            completion_logprobs = [token.logprob for token in logprobs_content]
        else:
            assert isinstance(response.choices[0].logprobs, dict)
            logprobs_content = response.choices[0].logprobs["content"]
            completion_logprobs = [token["logprob"] for token in logprobs_content]
    elif message_type == "completion":
        assert isinstance(response, Completion)
        if not hasattr(response.choices[0], "prompt_token_ids"):
            return None
        if not hasattr(response.choices[0], "token_ids"):
            return None
        if not hasattr(response.choices[0], "logprobs"):
            return None
        if response.choices[0].logprobs is None:
            return None
        if not hasattr(response.choices[0].logprobs, "token_logprobs"):
            return None
        prompt_ids = getattr(response.choices[0], "prompt_token_ids")
        prompt_mask = [0] * len(prompt_ids)
        completion_ids = getattr(response.choices[0], "token_ids")
        completion_mask = [1] * len(completion_ids)
        completion_logprobs = getattr(response.choices[0].logprobs, "token_logprobs")
    if max_seq_len is not None:
        prompt_len = len(prompt_ids)
        completion_len = len(completion_ids)
        overlong_prompt = prompt_len > max_seq_len
        if overlong_prompt:
            is_truncated = True
            prompt_ids = prompt_ids[:max_seq_len]
            prompt_mask = prompt_mask[:max_seq_len]
            completion_ids = []
            completion_mask = []
            completion_logprobs = []
        elif prompt_len + completion_len > max_seq_len:
            is_truncated = True
            completion_ids = completion_ids[: max_seq_len - prompt_len]
            completion_mask = completion_mask[: max_seq_len - prompt_len]
            completion_logprobs = completion_logprobs[: max_seq_len - prompt_len]
        else:
            is_truncated = False
    else:
        overlong_prompt = False
        is_truncated = False
    return TrajectoryStepTokens(
        prompt_ids=prompt_ids,
        prompt_mask=prompt_mask,
        completion_ids=completion_ids,
        completion_mask=completion_mask,
        completion_logprobs=completion_logprobs,
        overlong_prompt=overlong_prompt,
        is_truncated=is_truncated,
    )


def _extract_json_object_from_pos(text: str, start: int) -> tuple[dict | None, int]:
    """Extract a single JSON object starting at start; return (parsed, end_pos)."""
    depth = 0
    i = start
    n = len(text)
    if i >= n or text[i] != "{":
        return None, i
    depth = 1
    i += 1
    in_string = False
    escape = False
    quote = None
    while i < n and depth > 0:
        c = text[i]
        if escape:
            escape = False
            i += 1
            continue
        if c == "\\" and in_string:
            escape = True
            i += 1
            continue
        if not in_string:
            if c in ("'", '"'):
                in_string = True
                quote = c
                i += 1
                continue
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    try:
                        obj = json.loads(text[start : i + 1])
                        return obj, i + 1
                    except json.JSONDecodeError:
                        return None, i + 1
        else:
            if c == quote:
                in_string = False
        i += 1
    return None, i


def parse_tool_calls_from_content(text: str) -> list[dict]:
    """
    Parse tool calls from reasoning or content when the API did not populate
    message.tool_calls (e.g. GLM 4.7 with tool calls embedded in reasoning).

    Looks for JSON objects with "name" and "arguments" (or "parameters"),
    as used by Hermes-style and GLM 4.7 / vLLM reasoning parsers.
    """
    if not text or not isinstance(text, str):
        return []
    out: list[dict] = []
    # Find potential start of tool-call JSON: {"name": or "name":
    pattern = re.compile(r'\{\s*"name"\s*:\s*"([^"]+)"\s*,\s*"(?:arguments|parameters)"\s*:')
    for m in pattern.finditer(text):
        start = m.start()
        obj, _ = _extract_json_object_from_pos(text, start)
        if not obj or not isinstance(obj, dict):
            continue
        name = obj.get("name")
        args = obj.get("arguments") or obj.get("parameters")
        if not name or not isinstance(name, str):
            continue
        if isinstance(args, dict):
            args_str = json.dumps(args)
        elif isinstance(args, str):
            args_str = args
        else:
            args_str = "{}"
        out.append({
            "id": f"call_{uuid.uuid4().hex[:24]}",
            "type": "function",
            "function": {"name": name, "arguments": args_str},
        })
    return out


async def parse_response_messages(
    response: ModelResponse, message_type: MessageType
) -> Messages:
    response_text = ""
    if message_type == "chat":
        assert isinstance(response, ChatCompletion)
        if response.choices and response.choices[0].message:
            message = response.choices[0].message
            # Also check reasoning field (for vLLM --reasoning-parser)
            response_text = message.content or getattr(message, "reasoning", None) or ""
        response_message: dict[str, object] = {
            "role": "assistant",
            "content": response_text,
        }
        top_level_tool_calls = getattr(
            response.choices[0].message, "tool_calls", None
        ) if response.choices and response.choices[0].message else None
        if top_level_tool_calls:
            response_message["tool_calls"] = [
                tool_call.model_dump() for tool_call in top_level_tool_calls
            ]
        else:
            # Parse tool calls from reasoning/content (e.g. GLM 4.7 with tool content in reasoning)
            parsed_from_content = parse_tool_calls_from_content(response_text)
            if parsed_from_content:
                response_message["tool_calls"] = parsed_from_content
        completion_messages = list[ChatMessage]([cast(ChatMessage, response_message)])
    else:
        assert isinstance(response, Completion)
        if response.choices and response.choices[0]:
            response_text = response.choices[0].text or ""
        completion_messages = str(response_text)
    return completion_messages


async def parse_is_truncated(
    response: ModelResponse, message_type: MessageType
) -> bool:
    if message_type == "chat":
        assert isinstance(response, ChatCompletion)
        assert len(response.choices) == 1, "Response should always have one choice"
        return response.choices[0].finish_reason == "length"
    elif message_type == "completion":
        assert isinstance(response, Completion)
        assert len(response.choices) == 1, "Response should always have one choice"
        return response.choices[0].finish_reason == "length"
    else:
        return False
