"""The proxy client: forward the program's request to the provider 1:1.

`ProxyClient` (the default) forwards the program's request body verbatim to an OpenAI-
compatible endpoint and parses the provider's response into a vf `Response` (via the harness's
`Dialect`) for the trace — carrying the raw response on `Response.raw` so the interception
server hands it back to the program untouched, no field lost to a typed round-trip.

Also holds the vf -> wire serializers (`message_to_wire` / `serialize_completion`) the renderer
and the interception server (user-sim injection) need, plus `model_error` (shared error
mapping). The wire -> vf parsing lives in `dialects` (per native format).
"""

import httpx
from openai import AsyncOpenAI, OpenAIError

from verifiers.v1.clients.client import Client
from verifiers.v1.clients.dialects import Dialect
from verifiers.v1.errors import ModelError, OverlongPromptError
from verifiers.v1.types import Message, Messages, Response, SamplingConfig, Tool

_CONTEXT_LENGTH_PHRASES = (
    "this model's maximum context length is",
    "is longer than the model's context length",
    "is longer than the maximum model length",
    "exceeds the model's context length",
    "exceed the configured limit",
    "exceeds the configured limit",
    "exceeded model",
    "prompt_too_long",
    "context length",
    "maximum model length",
)


def model_error(e: OpenAIError) -> ModelError:
    """Map a provider client error to our error type, distinguishing an overlong prompt
    from any other model-call failure (auth, rate limit, a genuine bad request, ...)."""
    text = str(e).casefold()
    if any(phrase in text for phrase in _CONTEXT_LENGTH_PHRASES):
        return OverlongPromptError(str(e))
    return ModelError(str(e))


def _content_to_wire(content):
    """Plain text passes through; a content-part list becomes OpenAI wire dicts (so the
    provider / renderer sees the native `image_url` shape)."""
    if isinstance(content, str):
        return content
    return [part.model_dump() for part in content]


def message_to_wire(message: Message) -> dict:
    """A vf message -> the OpenAI chat wire dict. Used to build the renderer's request and to
    inject a user simulator's turns into the wire history (the proxy forwards verbatim, so it
    doesn't use this)."""
    if message.role == "assistant":
        wire: dict = {"role": "assistant", "content": message.content}
        # Reasoning models (DeepSeek V4, Kimi K2 Thinking, ...) require the prior turns'
        # `reasoning_content` sent back as a message-level field; carry it when present.
        if message.reasoning_content is not None:
            wire["reasoning_content"] = message.reasoning_content
        if message.tool_calls:
            wire["tool_calls"] = [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {"name": call.name, "arguments": call.arguments},
                }
                for call in message.tool_calls
            ]
        return wire
    if message.role == "tool":
        return {
            "role": "tool",
            "tool_call_id": message.tool_call_id,
            "content": message.content,
        }
    return {"role": message.role, "content": _content_to_wire(message.content)}


def tool_to_wire(tool: Tool) -> dict:
    function: dict = {
        "name": tool.name,
        "description": tool.description,
        "parameters": tool.parameters,
    }
    if tool.strict is not None:
        function["strict"] = tool.strict
    return {"type": "function", "function": function}


def serialize_completion(response: Response, model: str) -> dict:
    """A vf `Response` -> an OpenAI chat.completion dict the program's SDK expects. Used by the
    renderer (which generates a `Response` and has no raw wire to relay); the proxy returns the
    provider's raw dict instead."""
    message: dict = {"role": "assistant", "content": response.message.content}
    if response.message.reasoning_content is not None:
        message["reasoning_content"] = response.message.reasoning_content
    if response.message.tool_calls:
        message["tool_calls"] = [
            {
                "id": c.id,
                "type": "function",
                "function": {"name": c.name, "arguments": c.arguments},
            }
            for c in response.message.tool_calls
        ]
    usage = (
        {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens,
        }
        if response.usage
        else None
    )
    return {
        "id": response.id or "vf-intercept",
        "object": "chat.completion",
        "created": response.created,
        "model": response.model or model,
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": response.finish_reason or "stop",
            }
        ],
        "usage": usage,
    }


class ProxyClient(Client):
    """The default client: forward the program's request 1:1 to an OpenAI-compatible endpoint,
    parse the provider's response into a vf `Response` (via the harness's dialect) for the
    trace, and carry the raw response on `Response.raw` so it reaches the program untouched.
    `prompt`/`tools` are already in `body` and unused here — they're on the signature only so
    the renderer (which must tokenize) can translate the typed prompt instead of forwarding."""

    def __init__(self, openai: AsyncOpenAI) -> None:
        self.openai = openai

    async def get_response(
        self,
        body: dict,
        dialect: Dialect,
        prompt: Messages,
        model: str,
        sampling_args: SamplingConfig,
        tools: list[Tool] | None = None,
    ) -> Response:
        # Forward verbatim; the eval owns model + sampling (override whatever the program set).
        upstream = {
            **body,
            "model": model,
            **sampling_args.model_dump(exclude_none=True),
        }
        try:
            resp = await self.openai.post(
                "/chat/completions", cast_to=httpx.Response, body=upstream
            )
        except OpenAIError as e:
            raise model_error(e) from e
        raw = resp.json()
        response = dialect.parse_response(dialect.response_type.model_validate(raw))
        response.raw = raw  # the program gets the provider's bytes back 1:1
        return response

    async def close(self) -> None:
        await self.openai.close()
