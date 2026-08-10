"""The OpenAI Responses dialect (codex and friends).

Request parsing walks the `input` items, folding each run of assistant-side items (reasoning /
assistant message / function or custom tool call) into one typed assistant message; response
parsing reads the `output` items. Relay-only: the eval client forwards the program's bytes to a
`/responses` endpoint and this dialect parses a copy for the trace. Server-side statefulness
(`previous_response_id`) is not emulated — the endpoint owns it.
"""

import json
from collections import deque

from openai.types.responses import (
    ResponseUsage,
)
from pydantic import BaseModel, ConfigDict

from verifiers.v1.dialects.base import (
    Dialect,
    StreamParser,
    capability_notice,
    is_remote_url,
    iter_sse_reverse,
)
from verifiers.v1.types import (
    AssistantMessage,
    ContentPart,
    FinishReason,
    ImageUrlContentPart,
    ImageUrlSource,
    Messages,
    Response,
    Sampling,
    SamplingConfig,
    SystemMessage,
    TextContentPart,
    Tool,
    ToolCall,
    ToolMessage,
    Usage,
    UserMessage,
)

FINAL_EVENTS = ("response.completed", "response.incomplete", "response.failed")
# Byte markers for the terminal event types above, in both compact and spaced JSON, so the
# interception server can cheaply spot the turn-ending event without parsing each delta.
_TERMINAL_MARKERS = tuple(
    marker.encode()
    for event in FINAL_EVENTS
    for marker in (f'"type":"{event}"', f'"type": "{event}"')
)
# Sampling knobs the eval owns, in this format's shape (Responses uses `max_output_tokens`).
_SAMPLING_KEYS = frozenset({"temperature", "top_p", "max_output_tokens", "max_tokens"})
# Provider-native tools execute outside the rollout's network policy, so unknown tool
# types fail closed and only client-executed types are allowlisted here.
_CLIENT_TOOL_TYPES = frozenset(
    {
        "function",
        "custom",
        "local_shell",
        "apply_patch",
        "computer",
        "computer_use_preview",
    }
)
_CLIENT_TOOL_CHOICES = _CLIENT_TOOL_TYPES | {
    "namespace",
    "tool_search",
    "shell",
}
_SAFE_INPUT_TYPES = frozenset(
    {
        "input_text",
        "input_file",
        "input_image",
        "computer_screenshot",
        "output_text",
        "refusal",
        "computer_call",
        "function_call",
        "custom_tool_call",
        "reasoning",
        "compaction",
        "tool_search_call",
        "local_shell_call",
        "local_shell_call_output",
        "shell_call",
        "shell_call_output",
        "apply_patch_call",
        "apply_patch_call_output",
        "compaction_trigger",
    }
)


class ProviderUsageInputTokensDetails(BaseModel):
    """Permissive input token details: OpenAI-compatible providers may omit fields
    the pinned SDK declares required (e.g. ``cache_write_tokens``)."""

    model_config = ConfigDict(extra="allow")
    cache_write_tokens: int | None = None
    cached_tokens: int | None = None


class ProviderUsageOutputTokensDetails(BaseModel):
    """Permissive output token details: providers may omit ``reasoning_tokens``."""

    model_config = ConfigDict(extra="allow")
    reasoning_tokens: int | None = None


class ProviderUsage(ResponseUsage):
    """Responses usage with optional detail objects for OpenAI-compatible providers."""

    input_tokens_details: ProviderUsageInputTokensDetails | None = None
    output_tokens_details: ProviderUsageOutputTokensDetails | None = None


class OpenAIResponse(BaseModel):
    """Permissive parse-only view of a Responses object: `extra='allow'` keeps it a plain dict
    for the trace (read via `model_dump`), so a strict SDK model can't crash the rollout on a
    provider/SDK enum skew (e.g. a value the pinned `openai` rejects)."""

    model_config = ConfigDict(extra="allow")
    usage: ProviderUsage | None = None


def parse_content(content) -> str | list[ContentPart]:
    if isinstance(content, str):
        return content
    parts: list[ContentPart] = []
    for part in content or []:
        kind = part.get("type")
        if kind in ("input_text", "output_text"):
            parts.append(TextContentPart(text=part.get("text", "")))
        elif kind == "input_image":
            parts.append(
                ImageUrlContentPart(
                    image_url=ImageUrlSource(url=part.get("image_url", ""))
                )
            )
    return parts


def _tools_capability(tools, path: str) -> str | None:
    for index, tool in enumerate(tools or []):
        item_path = f"{path}[{index}]"
        if not isinstance(tool, dict):
            return item_path
        kind = tool.get("type")
        if kind in _CLIENT_TOOL_TYPES:
            continue
        if kind == "namespace":
            if capability := _tools_capability(tool.get("tools"), f"{item_path}.tools"):
                return capability
            continue
        if kind == "tool_search" and tool.get("execution") == "client":
            continue
        environment = tool.get("environment")
        if (
            kind == "shell"
            and isinstance(environment, dict)
            and environment.get("type") == "local"
        ):
            continue
        return f"{item_path}.type"
    return None


def _content_capability(value, path: str) -> str | None:
    if isinstance(value, list):
        for index, item in enumerate(value):
            if capability := _content_capability(item, f"{path}[{index}]"):
                return capability
        return None
    if not isinstance(value, dict):
        return None
    kind = value.get("type")
    caller = value.get("caller")
    if isinstance(caller, dict) and caller.get("type") == "program":
        return f"{path}.caller.type"
    if kind == "input_file":
        if value.get("file_id"):
            return f"{path}.file_id"
        if is_remote_url(value.get("file_url")):
            return f"{path}.file_url"
    if kind in ("input_image", "computer_screenshot"):
        if value.get("file_id"):
            return f"{path}.file_id"
        if is_remote_url(value.get("image_url")):
            return f"{path}.image_url"
    if kind == "reasoning" and value.get("id") and not value.get("encrypted_content"):
        return f"{path}.id"
    if kind == "tool_search_call" and value.get("execution") != "client":
        return f"{path}.execution"
    if kind == "shell_call":
        environment = value.get("environment")
        if not (isinstance(environment, dict) and environment.get("type") == "local"):
            return f"{path}.environment"
    if kind == "additional_tools":
        return _tools_capability(value.get("tools"), f"{path}.tools")
    if kind == "tool_search_output":
        if value.get("execution") != "client":
            return f"{path}.execution"
        return _tools_capability(value.get("tools"), f"{path}.tools")
    if kind in (
        "computer_call_output",
        "function_call_output",
        "custom_tool_call_output",
    ):
        return _content_capability(value.get("output"), f"{path}.output")
    if kind in (None, "message") and "role" in value and "content" in value:
        return _content_capability(value.get("content"), f"{path}.content")
    if kind in _SAFE_INPUT_TYPES:
        return None
    return f"{path}.type"


def fold_assistant(items: list[dict]) -> AssistantMessage:
    """One run of assistant-side items -> one typed assistant message."""
    content = ""
    reasoning: list[str] = []
    calls: list[ToolCall] = []
    for item in items:
        if item.get("type") == "reasoning":
            reasoning += [s.get("text", "") for s in item.get("summary") or []]
            reasoning += [c.get("text", "") for c in item.get("content") or []]
        elif item.get("type") in ("function_call", "custom_tool_call"):
            calls.append(
                ToolCall(
                    id=item.get("call_id", ""),
                    name=item.get("name", ""),
                    arguments=item.get("arguments", item.get("input", "")),
                )
            )
        else:  # an assistant message item
            raw = item.get("content")
            content += (
                raw
                if isinstance(raw, str)
                else "".join(
                    p.get("text", "")
                    for p in raw or []
                    if p.get("type") in ("input_text", "output_text")
                )
            )
    return AssistantMessage(
        content=content or None,
        reasoning_content="\n".join(r for r in reasoning if r) or None,
        tool_calls=calls or None,
        provider_state=items,
    )


def response_from_wire(response: OpenAIResponse) -> Response:
    """An OpenAI Responses object -> a vf `Response` (its `output` items folded into one
    assistant message)."""
    data = response.model_dump()
    content = ""
    reasoning: list[str] = []
    calls: list[ToolCall] = []
    for item in data.get("output") or []:
        kind = item.get("type")
        if kind == "message":
            content += "".join(
                p.get("text", "")
                for p in item.get("content") or []
                if p.get("type") == "output_text"
            )
        elif kind == "reasoning":
            reasoning += [s.get("text", "") for s in item.get("summary") or []]
            reasoning += [c.get("text", "") for c in item.get("content") or []]
        elif kind in ("function_call", "custom_tool_call"):
            calls.append(
                ToolCall(
                    id=item.get("call_id", ""),
                    name=item.get("name", ""),
                    arguments=item.get("arguments", item.get("input", "")),
                )
            )
    tool_calls = calls or None
    finish: FinishReason = (
        "length"
        if data.get("status") == "incomplete"
        else ("tool_calls" if tool_calls else "stop")
    )
    usage = None
    if response.usage:
        provider_usage = response.usage
        input_details = provider_usage.input_tokens_details
        output_details = provider_usage.output_tokens_details
        cached = input_details.cached_tokens if input_details else None
        # Responses input_tokens includes cache hits; vf keeps the buckets disjoint.
        usage = Usage(
            prompt_tokens=provider_usage.input_tokens - (cached or 0),
            completion_tokens=provider_usage.output_tokens,
            cached_input_tokens=cached,
            reasoning_tokens=output_details.reasoning_tokens
            if output_details
            else None,
            cost=getattr(provider_usage, "cost", None),
        )
    return Response(
        id=data.get("id", ""),
        created=data.get("created_at", 0),
        model=data.get("model", ""),
        message=AssistantMessage(
            content=content or None,
            reasoning_content="\n".join(r for r in reasoning if r) or None,
            tool_calls=tool_calls,
            provider_state=data.get("output"),
        ),
        finish_reason=finish,
        usage=usage,
    )


class ResponsesStreamParser(StreamParser):
    """Retain only the complete terminal response event and trailing DONE event."""

    def __init__(self) -> None:
        self.events: deque[bytes] = deque(maxlen=2)
        self.feed = self.events.append
        self.terminal_events: tuple[bytes, ...] | None = None

    def on_done(self) -> None:
        # Freeze the terminal tail before later relay chunks can evict it.
        self.terminal_events = tuple(self.events)

    def finish(self) -> Response:
        events = self.terminal_events or self.events
        for event in iter_sse_reverse(b"".join(events)):
            if event.get("type") in FINAL_EVENTS:
                return response_from_wire(
                    OpenAIResponse.model_validate(event["response"])
                )
        raise ValueError("Responses stream ended without a terminal event")


class ResponsesDialect(Dialect[dict, OpenAIResponse]):
    sampling_fields = frozenset(
        {
            "temperature",
            "top_p",
            "max_output_tokens",
            "max_tool_calls",
            "reasoning",
            "text",
            "tool_choice",
            "parallel_tool_calls",
            "top_logprobs",
            "truncation",
        }
    )
    routes = ("/v1/responses",)
    upstream_path = "/responses"
    response_type = OpenAIResponse

    def external_capability(self, body: dict) -> str | None:
        if body.get("previous_response_id") is not None:
            return "previous_response_id"
        if body.get("conversation") is not None:
            return "conversation"
        prompt = body.get("prompt")
        if isinstance(prompt, dict):
            variables = prompt.get("variables")
            if isinstance(variables, dict):
                for index, value in enumerate(variables.values()):
                    if capability := _content_capability(
                        value, f"prompt.variables[{index}]"
                    ):
                        return capability
            if prompt.get("id"):
                return "prompt.id"
        elif prompt is not None:
            return "prompt"
        if capability := _content_capability(body.get("input"), "input"):
            return capability
        choice = body.get("tool_choice")
        if isinstance(choice, str):
            if choice not in ("none", "auto", "required"):
                return "tool_choice"
        elif choice is not None:
            if not isinstance(choice, dict):
                return "tool_choice"
            kind = choice.get("type")
            if kind == "allowed_tools":
                if capability := _tools_capability(
                    choice.get("tools"), "tool_choice.tools"
                ):
                    return capability
            elif kind not in _CLIENT_TOOL_CHOICES:
                return "tool_choice.type"
        return _tools_capability(body.get("tools"), "tools")

    def mediate_external_capabilities(self, body: dict) -> tuple[dict, list[str]]:
        mediated = dict(body)
        capabilities: list[str] = []

        for field in ("previous_response_id", "conversation"):
            if mediated.pop(field, None) is not None:
                capabilities.append(field)

        prompt = mediated.get("prompt")
        if prompt is not None:
            capability = self.external_capability({"prompt": prompt})
            if capability is not None:
                capabilities.append(capability)
                mediated.pop("prompt")

        raw_input = mediated.get("input")
        if isinstance(raw_input, list):
            safe_input = []
            for item_index, item in enumerate(raw_input):
                item_path = f"input[{item_index}]"
                if not isinstance(item, dict):
                    safe_input.append(item)
                    continue
                item = dict(item)

                content_field = None
                if item.get("type") in (
                    "function_call_output",
                    "custom_tool_call_output",
                    "computer_call_output",
                ):
                    content_field = "output"
                elif item.get("type") in (None, "message") and "content" in item:
                    content_field = "content"

                content = item.get(content_field) if content_field else None
                if isinstance(content, list):
                    safe_content = []
                    for part_index, part in enumerate(content):
                        part_path = f"{item_path}.{content_field}[{part_index}]"
                        capability = _content_capability(part, part_path)
                        if capability is None:
                            safe_content.append(part)
                        else:
                            capabilities.append(capability)
                            safe_content.append(
                                {
                                    "type": "input_text",
                                    "text": capability_notice([capability]),
                                }
                            )
                    item[content_field] = safe_content

                capability = _content_capability(item, item_path)
                if capability is None:
                    safe_input.append(item)
                else:
                    capabilities.append(capability)
                    safe_input.append(
                        {
                            "type": "message",
                            "role": "user",
                            "content": [
                                {
                                    "type": "input_text",
                                    "text": capability_notice([capability]),
                                }
                            ],
                        }
                    )
            mediated["input"] = safe_input
        elif capability := _content_capability(raw_input, "input"):
            capabilities.append(capability)
            mediated["input"] = capability_notice([capability])

        tools = []
        for index, tool in enumerate(mediated.get("tools") or []):
            capability = _tools_capability([tool], "tools")
            if capability is None:
                tools.append(tool)
            else:
                capabilities.append(
                    capability.replace("tools[0]", f"tools[{index}]", 1)
                )
        if len(tools) != len(mediated.get("tools") or []):
            mediated["tools"] = tools

        choice = mediated.get("tool_choice")
        if choice is not None and (
            capability := self.external_capability({"tool_choice": choice})
        ):
            capabilities.append(capability)
            mediated.pop("tool_choice")

        if capabilities:
            notice = capability_notice(capabilities)
            instructions = mediated.get("instructions")
            mediated["instructions"] = (
                f"{instructions}\n\n{notice}"
                if isinstance(instructions, str)
                else notice
            )
            if mediated.get("input") is None:
                mediated["input"] = notice
        return mediated, capabilities

    def is_terminal_event(self, chunk: bytes) -> bool:
        # A Responses client (e.g. codex) ends its turn on `response.completed`, before the
        # trailing `[DONE]`, so the turn-ending event is the final event, not the sentinel.
        return any(marker in chunk for marker in _TERMINAL_MARKERS)

    def parse_sampling(self, body: dict) -> Sampling:
        settings = {k: v for k, v in body.items() if k in self.sampling_fields}
        # Lift `reasoning.effort` onto the typed knob; keep any other reasoning keys
        # (e.g. `summary`) as the wire sent them.
        if isinstance(reasoning := settings.get("reasoning"), dict):
            reasoning = dict(reasoning)
            if reasoning.get("effort"):
                settings["reasoning_effort"] = reasoning.pop("effort")
            if reasoning:
                settings["reasoning"] = reasoning
            else:
                settings.pop("reasoning")
        if "max_output_tokens" in settings:
            settings["max_tokens"] = settings.pop("max_output_tokens")
        return Sampling.model_validate(settings)

    def parse_request(self, body: dict) -> tuple[Messages, list[Tool] | None]:
        prompt: Messages = []
        if instructions := body.get("instructions"):
            prompt.append(SystemMessage(content=instructions))
        raw = body.get("input")
        items = (
            [{"role": "user", "content": raw}] if isinstance(raw, str) else raw or []
        )
        run: list[dict] = []  # the current run of assistant-side items
        for item in items:
            role = item.get("role")
            assistant = (
                role == "assistant"
                or role is None
                and not (item.get("type") or "").endswith(("_output", "_response"))
            )
            if run and not assistant:
                prompt.append(fold_assistant(run))
                run = []
            if assistant:
                run.append(item)
            elif item.get("type") in (
                "function_call_output",
                "custom_tool_call_output",
            ):
                output = item.get("output")
                content = (
                    parse_content(output)
                    if isinstance(output, (str, list))
                    else json.dumps(output)
                )
                prompt.append(
                    ToolMessage(
                        tool_call_id=item.get("call_id", ""),
                        content=content,
                    )
                )
            elif item.get("role") in ("system", "developer"):
                prompt.append(SystemMessage(content=parse_content(item.get("content"))))
            else:
                prompt.append(UserMessage(content=parse_content(item.get("content"))))
        if run:
            prompt.append(fold_assistant(run))
        tools = [
            Tool(
                name=t["name"],
                description=t.get("description") or "",
                parameters=t.get("parameters") or {},
                strict=t.get("strict"),
            )
            for t in body.get("tools") or []
            if t.get("type") == "function"
        ] or None
        return prompt, tools

    def parse_response(self, response: OpenAIResponse) -> Response:
        return response_from_wire(response)

    def stream_parser(self) -> StreamParser:
        return ResponsesStreamParser()

    def apply_overrides(self, body: dict, model: str, sampling: SamplingConfig) -> dict:
        # Preserve native fields except the eval's model + sampling, mapped to the Responses shape
        # (`max_tokens` -> `max_output_tokens`); sampling is authoritative.
        s = sampling.model_dump(exclude_none=True)
        name = model.rsplit("/", 1)[-1]
        reasoning_model = (
            name.startswith(("gpt-5", "o1", "o3", "o4"))
            and "-chat" not in name
            and ("/" not in model or model.startswith("openai/"))
        )
        overrides: dict = {"model": model}
        if reasoning_model:
            # Preserve opaque reasoning state so it can be replayed on the next turn.
            include = list(body.get("include") or [])
            if "reasoning.encrypted_content" not in include:
                include.append("reasoning.encrypted_content")
            overrides["include"] = include
        if "temperature" in s:
            overrides["temperature"] = s["temperature"]
        if "top_p" in s:
            overrides["top_p"] = s["top_p"]
        if "max_tokens" in s:
            overrides["max_output_tokens"] = s["max_tokens"]
        reasoning = dict(body.get("reasoning") or {})
        if reasoning_model:
            # Summaries provide the trace's readable reasoning text.
            reasoning = {"summary": "auto", **reasoning}
        if "reasoning_effort" in s:
            reasoning["effort"] = s["reasoning_effort"]
        if reasoning:
            overrides["reasoning"] = reasoning
        steered = {
            k: v
            for k, v in body.items()
            if k not in _SAMPLING_KEYS and k not in overrides
        }
        return {**steered, **overrides}
