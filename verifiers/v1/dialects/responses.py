"""The OpenAI Responses dialect (codex and friends).

Request parsing walks the `input` items, folding each run of assistant-side items (reasoning /
assistant message / function or custom tool call) into one typed assistant message; response
parsing reads the `output` items. Relay-only: the eval client forwards the program's bytes to a
`/responses` endpoint and this dialect parses a copy for the trace. Server-side statefulness
(`previous_response_id`) is not emulated — the endpoint owns it.
"""

import json
import re
from collections import deque

from openai.types.responses.response_usage import (
    InputTokensDetails,
    OutputTokensDetails,
    ResponseUsage,
)
from pydantic import BaseModel, ConfigDict

from verifiers.v1.configs.runtime import NetworkPolicyConfig
from verifiers.v1.dialects.base import (
    Dialect,
    RawRequest,
    StreamParser,
    append_user_notice,
    blocked_url,
    parse_sse_event,
    provider_allowed_domains,
)
from verifiers.v1.errors import model_error
from verifiers.v1.types import (
    AssistantMessage,
    ContentPart,
    FinishReason,
    ImageUrlContentPart,
    ImageUrlSource,
    Messages,
    Request,
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
# Client tools return calls to the harness; every other type may execute at the provider.
_CLIENT_TOOL_TYPES = (
    "function",
    "custom",
    "local_shell",
    "apply_patch",
    "computer",
    "computer_use_preview",
)
_CLIENT_TOOL_CHOICES = (*_CLIENT_TOOL_TYPES, "namespace", "tool_search", "shell")
_WEB_SEARCH_TOOL_TYPE = re.compile(r"web_search(?:_\d{4}_\d{2}_\d{2})?").fullmatch
_SAFE_INPUT_TYPES = (
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
)
TEXT_TOOL_OUTPUT_TYPES = ("function_call_output", "custom_tool_call_output")
BLANK_PNG = (
    "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJ"
    "AAAADUlEQVR42mNk+M/wHwAF/gL+Xw4AAAAASUVORK5CYII="
)


class ProviderUsageInputTokensDetails(InputTokensDetails):
    """Permissive input token details: OpenAI-compatible providers may omit fields
    the pinned SDK declares required (e.g. ``cache_write_tokens``)."""

    cache_write_tokens: int | None = None
    cached_tokens: int | None = None


class ProviderUsageOutputTokensDetails(OutputTokensDetails):
    """Permissive output token details: providers may omit ``reasoning_tokens``."""

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


def mediate_tools(
    tools, path: str, policy: NetworkPolicyConfig
) -> tuple[list[dict], list[str]]:
    if tools is not None and not isinstance(tools, list):
        return [], [path]
    mediated = []
    capabilities = []
    for index, tool in enumerate(tools or []):
        item_path = f"{path}[{index}]"
        if not isinstance(tool, dict):
            capabilities.append(item_path)
            continue
        kind = tool.get("type")
        if isinstance(kind, str) and _WEB_SEARCH_TOOL_TYPE(kind):
            raw_filters = tool.get("filters")
            if raw_filters is None or isinstance(raw_filters, dict):
                requested = raw_filters.get("allowed_domains") if raw_filters else None
                domains = provider_allowed_domains(policy, requested)
            else:
                domains = []
            if domains:
                filters = dict(raw_filters) if isinstance(raw_filters, dict) else {}
                filters["allowed_domains"] = domains
                mediated.append({**tool, "filters": filters})
                continue
            capabilities.append(f"{item_path}.type")
            continue
        if kind in _CLIENT_TOOL_TYPES:
            mediated.append(tool)
            continue
        if kind == "namespace":
            nested, removed = mediate_tools(
                tool.get("tools"), f"{item_path}.tools", policy
            )
            capabilities.extend(removed)
            if nested:
                mediated.append({**tool, "tools": nested})
            continue
        if kind == "tool_search" and tool.get("execution") == "client":
            mediated.append(tool)
            continue
        environment = tool.get("environment")
        if (
            kind == "shell"
            and isinstance(environment, dict)
            and environment.get("type") == "local"
        ):
            mediated.append(tool)
            continue
        capabilities.append(f"{item_path}.type")
    return mediated, capabilities


def blocked_content_path(value, path: str, policy: NetworkPolicyConfig) -> str | None:
    if isinstance(value, list):
        for index, item in enumerate(value):
            if blocked := blocked_content_path(item, f"{path}[{index}]", policy):
                return blocked
        return None
    if not isinstance(value, dict):
        return None

    kind = value.get("type")
    caller = value.get("caller")
    if caller is not None and not (
        isinstance(caller, dict) and caller.get("type") == "direct"
    ):
        return f"{path}.caller.type"
    if kind == "input_file":
        if value.get("file_id"):
            return f"{path}.file_id"
        if "file_url" in value:
            if not isinstance(value["file_url"], str) or blocked_url(
                value["file_url"], policy
            ):
                return f"{path}.file_url"
        elif not isinstance(value.get("file_data"), str):
            return f"{path}.file_data"
    elif kind in ("input_image", "computer_screenshot"):
        if value.get("file_id"):
            return f"{path}.file_id"
        if not isinstance(value.get("image_url"), str) or blocked_url(
            value["image_url"], policy
        ):
            return f"{path}.image_url"

    if kind == "reasoning" and value.get("id") and not value.get("encrypted_content"):
        return f"{path}.id"
    if kind == "item_reference" or kind is None and set(value) == {"id"}:
        return f"{path}.id"

    if kind == "tool_search_call" and value.get("execution") != "client":
        return f"{path}.execution"
    if kind == "shell_call":
        environment = value.get("environment")
        if not (isinstance(environment, dict) and environment.get("type") == "local"):
            return f"{path}.environment"
    if kind in ("additional_tools", "tool_search_output"):
        if kind == "tool_search_output" and value.get("execution") != "client":
            return f"{path}.execution"
        _, removed = mediate_tools(value.get("tools"), f"{path}.tools", policy)
        return removed[0] if removed else None

    if kind in (
        "computer_call_output",
        "function_call_output",
        "custom_tool_call_output",
    ):
        return blocked_content_path(value.get("output"), f"{path}.output", policy)
    if kind in (None, "message") and "role" in value and "content" in value:
        return blocked_content_path(value["content"], f"{path}.content", policy)
    return None if kind in _SAFE_INPUT_TYPES else f"{path}.type"


def mediate_content(value, path: str, policy: NetworkPolicyConfig):
    if not isinstance(value, list):
        blocked = blocked_content_path(value, path, policy)
        return ("", [blocked]) if blocked else (value, [])

    mediated = []
    capabilities = []
    for index, part in enumerate(value):
        if blocked := blocked_content_path(part, f"{path}[{index}]", policy):
            capabilities.append(blocked)
            continue
        mediated.append(part)
    return mediated, capabilities


def fold_assistant(items: list[dict] | None) -> AssistantMessage:
    """Assistant-side Responses items -> one typed assistant message."""
    content = ""
    reasoning: list[str] = []
    calls: list[ToolCall] = []
    for item in items or []:
        kind = item.get("type")
        if kind == "reasoning":
            reasoning += [s.get("text", "") for s in item.get("summary") or []]
            reasoning += [c.get("text", "") for c in item.get("content") or []]
        elif kind in ("function_call", "custom_tool_call"):
            calls.append(
                ToolCall(
                    id=item.get("call_id", ""),
                    type="custom" if kind == "custom_tool_call" else "function",
                    name=item.get("name", ""),
                    arguments=item.get("arguments", item.get("input", "")),
                )
            )
        else:
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
    # Copy the output snapshot without traversing echoed tools or request settings.
    data = response.model_dump(
        include={"id", "created_at", "model", "status", "error", "output"}
    )
    status = data.get("status")
    if status not in (None, "completed", "incomplete"):
        error = data.get("error") or {}
        code = error.get("code") if isinstance(error, dict) else None
        message = error.get("message") if isinstance(error, dict) else None
        detail = ": ".join(str(value) for value in (status, code, message) if value)
        status_code = (
            429
            if code in ("rate_limit_exceeded", "rate_limit_error")
            else 400
            if code in ("invalid_prompt", "context_length_exceeded")
            else 502
        )
        raise model_error(
            f"upstream Responses request did not complete: {detail}",
            status_code=status_code,
        )
    message = fold_assistant(data.get("output"))
    finish: FinishReason = (
        "length"
        if data.get("status") == "incomplete"
        else ("tool_calls" if message.tool_calls else "stop")
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
        message=message,
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
        for raw in reversed(events):
            event = parse_sse_event(raw)
            if event and event.get("type") in FINAL_EVENTS:
                response = response_from_wire(
                    OpenAIResponse.model_validate(event["response"])
                )
                response.raw = event["response"]
                return response
        raise ValueError("Responses stream ended without a terminal event")


class ResponsesDialect(Dialect[OpenAIResponse]):
    sampling_fields = frozenset(
        {
            "temperature",
            "top_p",
            "max_output_tokens",
            "max_tool_calls",
            "reasoning",
            "service_tier",
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

    def mediate_external_capabilities(
        self, body: RawRequest, policy: NetworkPolicyConfig
    ) -> tuple[RawRequest, list[str]]:
        mediated = body
        capabilities: list[str] = []

        for field in ("previous_response_id", "conversation"):
            if mediated.pop(field, None) is not None:
                capabilities.append(field)

        if mediated.pop("prompt", None) is not None:
            capabilities.append("prompt")
        if mediated.pop("plugins", None) is not None:
            capabilities.append("plugins")

        raw_input = mediated.get("input")
        if isinstance(raw_input, list):
            safe_input = []
            for item_index, item in enumerate(raw_input):
                item_path = f"input[{item_index}]"
                if not isinstance(item, dict):
                    safe_input.append(item)
                    continue
                kind = item.get("type")
                if kind in ("additional_tools", "tool_search_output"):
                    if blocked := blocked_content_path(
                        {**item, "tools": []}, item_path, policy
                    ):
                        capabilities.append(blocked)
                        continue
                    item["tools"], removed = mediate_tools(
                        item.get("tools"), f"{item_path}.tools", policy
                    )
                    capabilities.extend(removed)
                    if kind == "tool_search_output" or item["tools"]:
                        safe_input.append(item)
                    continue
                content_field = None
                if kind in TEXT_TOOL_OUTPUT_TYPES:
                    content_field = "output"
                elif kind in (None, "message") and "content" in item:
                    content_field = "content"

                if content_field:
                    content, removed = mediate_content(
                        item.get(content_field),
                        f"{item_path}.{content_field}",
                        policy,
                    )
                    capabilities.extend(removed)
                    if removed:
                        item[content_field] = content or ""

                scan = {**item, content_field: []} if content_field else item
                blocked = blocked_content_path(scan, item_path, policy)
                if blocked is None:
                    safe_input.append(item)
                else:
                    capabilities.append(blocked)
                    if kind == "computer_call_output" and blocked.startswith(
                        f"{item_path}.output"
                    ):
                        item["output"] = {
                            "type": "computer_screenshot",
                            "image_url": BLANK_PNG,
                        }
                        safe_input.append(item)
            mediated["input"] = safe_input
        elif blocked := blocked_content_path(raw_input, "input", policy):
            capabilities.append(blocked)
            mediated["input"] = []

        tools, tool_capabilities = mediate_tools(mediated.get("tools"), "tools", policy)
        capabilities.extend(tool_capabilities)
        if "tools" in mediated:
            mediated["tools"] = tools
            if not tools:
                mediated.pop("tool_choice", None)

        choice = mediated.get("tool_choice")
        valid_choice = choice is None or (
            isinstance(choice, str) and choice in ("none", "auto", "required")
        )
        if isinstance(choice, dict):
            kind = choice.get("type")
            valid_choice = (
                kind in _CLIENT_TOOL_CHOICES
                or isinstance(kind, str)
                and _WEB_SEARCH_TOOL_TYPE(kind)
            ) and any(
                tool.get("type") == kind
                and ("name" not in choice or tool.get("name") == choice["name"])
                for tool in tools
            )
            if kind == "allowed_tools":
                choice_tools, choice_capabilities = mediate_tools(
                    choice.get("tools"), "tool_choice.tools", policy
                )
                valid_choice = not choice_capabilities
                mediated["tool_choice"] = {**choice, "tools": choice_tools}
        if not valid_choice:
            capabilities.append("tool_choice")
            mediated.pop("tool_choice")

        input_items = mediated.get("input")
        if not isinstance(input_items, list):
            input_items = (
                []
                if input_items is None
                else [{"role": "user", "content": input_items}]
            )
        append_user_notice(input_items, text_type="input_text", message_type="message")
        mediated["input"] = input_items
        return mediated, capabilities

    def is_terminal_event(self, chunk: bytes) -> bool:
        # A Responses client (e.g. codex) ends its turn on `response.completed`, before the
        # trailing `[DONE]`, so the turn-ending event is the final event, not the sentinel.
        return any(marker in chunk for marker in _TERMINAL_MARKERS)

    def parse_sampling(self, body: RawRequest) -> Sampling:
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

    def parse_request(self, body: RawRequest) -> Request:
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
            elif item.get("type") in TEXT_TOOL_OUTPUT_TYPES:
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
        return Request(messages=prompt, tools=tools)

    def parse_response(self, response: OpenAIResponse) -> Response:
        return response_from_wire(response)

    def rewrite_request(self, body: dict, before: Request, after: Request) -> None:
        original = [
            m for m in before.messages if isinstance(m, (UserMessage, ToolMessage))
        ]
        rewritten = [
            m for m in after.messages if isinstance(m, (UserMessage, ToolMessage))
        ]
        items = body.get("input")
        if isinstance(items, str):
            if original != rewritten:
                message = rewritten[0]
                content = (
                    message.content
                    if isinstance(message.content, str)
                    else [
                        {"type": "input_text", "text": part.text}
                        if isinstance(part, TextContentPart)
                        else {
                            "type": "input_image",
                            "image_url": part.image_url.url,
                        }
                        for part in message.content
                    ]
                )
                body["input"] = (
                    content
                    if isinstance(content, str)
                    else [{"role": "user", "content": content}]
                )
            return

        targets = []
        for item in items or []:
            role = item.get("role")
            kind = item.get("type") or ""
            assistant = role == "assistant" or (
                role is None and not kind.endswith(("_output", "_response"))
            )
            if assistant or role in ("system", "developer"):
                continue
            targets.append(item)
        for item, old, new in zip(targets, original, rewritten, strict=True):
            if old == new:
                continue
            content = (
                new.content
                if isinstance(new.content, str)
                else [
                    {"type": "input_text", "text": part.text}
                    if isinstance(part, TextContentPart)
                    else {
                        "type": "input_image",
                        "image_url": part.image_url.url,
                    }
                    for part in new.content
                ]
            )
            item["output" if isinstance(new, ToolMessage) else "content"] = content

    def rewrite_response(self, raw: dict, text: str) -> None:
        original = next(
            (
                item
                for item in raw.get("output") or []
                if isinstance(item, dict) and item.get("type") == "message"
            ),
            {},
        )
        raw["output"] = [
            {
                "type": "message",
                "id": original.get("id") or "msg_intercepted",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": text, "annotations": []}],
            }
        ]
        raw.update(status="completed", error=None, incomplete_details=None)
        raw.pop("required_action", None)
        if "output_text" in raw:
            raw["output_text"] = text

    def stream_events(self, raw: dict) -> list[bytes]:
        item = raw["output"][0]
        part = item["content"][0]
        common = {"output_index": 0, "item_id": item["id"], "content_index": 0}
        head = {
            **raw,
            "status": "in_progress",
            "output": [],
            "completed_at": None,
        }
        events = [
            ("response.created", {"response": head}),
            (
                "response.output_item.added",
                {"output_index": 0, "item": {**item, "content": []}},
            ),
            (
                "response.content_part.added",
                {**common, "part": {**part, "text": ""}},
            ),
            ("response.output_text.delta", {**common, "delta": part["text"]}),
            ("response.output_text.done", {**common, "text": part["text"]}),
            ("response.content_part.done", {**common, "part": part}),
            ("response.output_item.done", {"output_index": 0, "item": item}),
            ("response.completed", {"response": raw}),
        ]
        return [
            *(
                f"data: {json.dumps({'type': kind, 'sequence_number': i, **data})}\n\n".encode()
                for i, (kind, data) in enumerate(events)
            ),
            b"data: [DONE]\n\n",
        ]

    def stream_parser(self) -> StreamParser:
        return ResponsesStreamParser()

    def apply_overrides(
        self, body: RawRequest, model: str, sampling: SamplingConfig
    ) -> RawRequest:
        # Preserve native fields except the eval's model + sampling, mapped to the Responses shape
        # (`max_tokens` -> `max_output_tokens`); sampling is authoritative.
        s = sampling.wire_args()
        max_tokens = s.pop("max_tokens", None)
        reasoning_effort = s.pop("reasoning_effort", None)
        sampling_reasoning = s.pop("reasoning", None)
        name = model.rsplit("/", 1)[-1]
        reasoning_model = (
            name.startswith(("gpt-5", "o1", "o3", "o4"))
            and "-chat" not in name
            and ("/" not in model or model.startswith("openai/"))
        )
        overrides: dict = {**s, "model": model}
        if reasoning_model:
            # Preserve opaque reasoning state so it can be replayed on the next turn.
            include = list(overrides.get("include") or body.get("include") or [])
            if "reasoning.encrypted_content" not in include:
                include.append("reasoning.encrypted_content")
            overrides["include"] = include
        if max_tokens is not None:
            overrides["max_output_tokens"] = max_tokens
        reasoning = {
            **({"summary": "auto"} if reasoning_model else {}),
            **dict(body.get("reasoning") or {}),
            **dict(sampling_reasoning or {}),
        }
        if reasoning_effort is not None:
            reasoning["effort"] = reasoning_effort
        if reasoning:
            overrides["reasoning"] = reasoning
        steered = {
            k: v
            for k, v in body.items()
            if k not in _SAMPLING_KEYS and k not in overrides
        }
        return {**steered, **overrides}
