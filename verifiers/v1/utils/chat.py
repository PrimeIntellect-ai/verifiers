"""SDK stream assembly shared by the interception server and bundled chat programs."""

from dataclasses import dataclass, field
from typing import Any, cast

from openai import APIConnectionError
from openai.lib.streaming.chat import ChatCompletionStreamState
from openai.types.chat import ChatCompletion, ChatCompletionChunk


@dataclass
class ChatCompletionAccumulator:
    state: ChatCompletionStreamState = field(default_factory=ChatCompletionStreamState)
    completion: ChatCompletion | None = None
    reasoning_details: dict[int, list[dict]] = field(default_factory=dict)

    def feed(self, chunk: ChatCompletionChunk) -> None:
        for choice in chunk.choices:
            # Providers repeat the role and reasoning identities across chunks. The
            # SDK concatenates strings, so only reasoning text should be accumulated.
            choice.delta.role = None
            for call in choice.delta.tool_calls or []:
                if call.type is None:
                    if call.function is not None:
                        call.type = "function"
                    elif getattr(call, "custom", None) is not None:
                        cast(Any, call).type = "custom"
            details = (choice.delta.model_extra or {}).pop("reasoning_details", None)
            accumulated = self.reasoning_details.setdefault(choice.index, [])
            for detail in details or []:
                previous = accumulated[-1] if accumulated else {}
                kind = detail.get("type")
                content_field = {
                    "reasoning.summary": "summary",
                    "reasoning.text": "text",
                }.get(kind)
                if (
                    content_field
                    and kind == previous.get("type")
                    and all(
                        previous.get(key) is None
                        or detail.get(key) is None
                        or previous[key] == detail[key]
                        for key in ("id", "index", "format")
                    )
                ):
                    previous[content_field] = (previous.get(content_field) or "") + (
                        detail.get(content_field) or ""
                    )
                    for key in ("id", "index", "signature", "format"):
                        if previous.get(key) is None and detail.get(key) is not None:
                            previous[key] = detail[key]
                else:
                    accumulated.append(dict(detail))
        usage = chunk.usage or (self.completion.usage if self.completion else None)
        self.state.handle_chunk(chunk)
        self.completion = self.state.current_completion_snapshot
        self.completion.usage = usage
        for choice in self.completion.choices:
            choice.message.role = "assistant"
            if details := self.reasoning_details.get(choice.index):
                cast(Any, choice.message).reasoning_details = details
        # Some providers repeat whole tool headers. An unchanged ID identifies
        # those repeats; leave other name/namespace fragments to the SDK.
        for choice in chunk.choices:
            calls = self.completion.choices[choice.index].message.tool_calls or []
            for delta in choice.delta.tool_calls or []:
                call = calls[delta.index]
                if not delta.id or call.id != delta.id * 2:
                    continue
                call.id = delta.id
                native = getattr(call, call.type)
                incoming = delta.model_dump(exclude_none=True).get(call.type, {})
                for key in ("name", "namespace"):
                    if value := incoming.get(key):
                        if isinstance(native, dict):
                            if native.get(key) == value * 2:
                                native[key] = value
                        elif getattr(native, key, None) == value * 2:
                            setattr(native, key, value)


async def read_chat_completion(stream) -> ChatCompletion:
    accumulator = ChatCompletionAccumulator()
    async with stream:
        async for chunk in stream:
            accumulator.feed(chunk)
    completion = accumulator.completion
    if (
        completion is None
        or not completion.choices
        or any(choice.finish_reason is None for choice in completion.choices)
    ):
        raise APIConnectionError(
            message="Model stream ended before a completion finished",
            request=stream.response.request,
        )
    return completion
