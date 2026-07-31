"""Renderer client: client-side tokenization via the `renderers` package.

A drop-in alternative to the chat-completions client: instead of sending messages
as JSON text, it renders them to token ids with a HF chat template and calls a
vLLM `/inference/v1/generate` engine, so every response carries token ids +
sampling logprobs (recorded on the trace's per-turn `tokens`) for training. It
reuses the chat client's wire translation (message/tool shapes are the same), and
needs a running vLLM engine.
"""

import asyncio
import contextlib
import json
import logging
from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

from openai import OpenAIError
from renderers import OverlongPromptError as RendererOverlongPromptError
from renderers import RenderedTokens, RendererConfig

from verifiers.v1.clients.client import SESSION_ID_HEADER, Client
from verifiers.v1.configs.client import TrainClientConfig, build_async_openai
from verifiers.v1.dialects import FINISH_REASONS, ChatDialect, Dialect, parse_tools
from verifiers.v1.dialects.chat import message_to_wire
from verifiers.v1.errors import OverlongPromptError, model_error
from verifiers.v1.graph import PendingTurn
from verifiers.v1.types import (
    AssistantMessage,
    FinishReason,
    KeptTokens,
    Response,
    SamplingConfig,
    Tool,
    ToolCall,
    TurnTokens,
    Usage,
)

logger = logging.getLogger(__name__)


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
    """A vf `Response` -> an OpenAI chat.completion dict the program's SDK expects. The renderer
    sets this on `Response.raw` (it generates, so has no provider response to relay)."""
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
    usage: dict | None = None
    if response.usage:
        # Usage is validated earlier in the pipeline; building its wire dict directly saves time.
        usage = {
            "completion_tokens": response.usage.completion_tokens,
            "prompt_tokens": response.usage.input_tokens,
            "total_tokens": response.usage.total_tokens,
        }
        if response.usage.reasoning_tokens is not None:
            usage["completion_tokens_details"] = {
                "reasoning_tokens": response.usage.reasoning_tokens
            }
        if response.usage.cached_input_tokens is not None:
            usage["prompt_tokens_details"] = {
                "cached_tokens": response.usage.cached_input_tokens
            }
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


def response_from_generate(
    result: dict, model: str, bridged_turn: PendingTurn | None = None
) -> Response:
    """Parse a `renderers.client.generate` result dict into a typed `Response`,
    mirroring the chat client's `response_from_wire` (plus the token encoding)."""
    finish: FinishReason = (
        result["finish_reason"]
        if result.get("finish_reason") in FINISH_REASONS
        else None
    )
    tool_calls = [
        ToolCall(
            id=tc.id or f"call_{i}",
            name=tc.name,
            arguments=tc.arguments
            if isinstance(tc.arguments, str)
            else json.dumps(tc.arguments or {}),
        )
        for i, tc in enumerate(result.get("tool_calls") or [])
        if getattr(tc, "name", None)
    ] or None
    prompt_ids = result.get("prompt_ids") or []
    completion_ids = result.get("completion_ids") or []
    # Per-message token spans (the renderer's attribution) let the trace graph store each
    # message's tokens once; carried transiently on TurnTokens and consumed by turn.commit().
    attribution = result.get("prompt_attribution")
    if attribution is None:
        message_spans = None
    elif bridged_turn is not None:
        message_spans = bridged_turn.prompt_message_spans(attribution)
    else:
        message_spans = attribution.message_token_spans()
    return Response(
        id=result.get("request_id", ""),
        created=0,
        model=model,
        message=AssistantMessage(
            content=result.get("content") or None,
            reasoning_content=result.get("reasoning_content"),
            tool_calls=tool_calls,
        ),
        finish_reason=finish,
        # /inference/v1/generate returns exact token ids but no usage details, so the
        # completion's reasoning-token subset is unknown.
        usage=Usage(
            prompt_tokens=len(prompt_ids), completion_tokens=len(completion_ids)
        ),
        # generate() returns owned, typed lists. Skip revalidation here to avoid copying
        # million-token contexts synchronously on the event loop.
        tokens=TurnTokens.model_construct(
            prompt_ids=prompt_ids,
            completion_ids=completion_ids,
            completion_logprobs=result.get("completion_logprobs") or [],
            message_spans=message_spans,
            is_content=attribution.is_content if attribution is not None else None,
            multi_modal_data=result.get("multi_modal_data"),
            routed_experts=result.get("routed_experts"),
            kept_tokens=KeptTokens(**kept)
            if (kept := result.get("kept_tokens"))
            else None,
        ),
    )


def _is_valid_incremental_tail(messages: list[dict[str, Any]]) -> bool:
    """Renderer bridges may extend sampled assistant turns with tool calls and/or a new user."""
    if not messages:
        return False
    roles = []
    for message in messages:
        role = message.get("role")
        roles.append(role if isinstance(role, str) else None)
    if roles[-1] == "user":
        return all(role == "tool" for role in roles[:-1])
    return all(role == "tool" for role in roles)


def _has_multimodal_content(messages) -> bool:
    for message in messages:
        content = getattr(message, "content", None)
        if not isinstance(content, list):
            continue
        if any(getattr(part, "type", None) == "image_url" for part in content):
            return True
    return False


@dataclass
class _RendererSlot:
    """One tokenizer, and the rollouts currently sharing it. A `size=1` pool rather than a
    bare renderer: it carries the lock that makes concurrent renders on one tokenizer safe,
    and `renderers` offloads its work to a thread only for a pool."""

    renderer: Any
    load: int = 0


class ElasticRendererPool:
    """Renderers grown on demand: one warmed up front, then `multiplex` rollouts per
    tokenizer — the renderer-side counterpart to `ElasticInterceptionPool`.

    Sizing a pool up front means paying for tokenizers a run may never need while every
    rollout that arrives before the build waits on all of them. Warming one and growing
    from there costs a single tokenizer at startup, and only a run that actually reaches
    `multiplex` concurrent rollouts pays for a second.

    Renderers carry no rollout state, so a pool is keyed by what builds it — `shared` hands
    every client with the same (model, config, template kwargs, multiplex) the same pool.
    With one client per rollout, owning one each would put a tokenizer behind every rollout."""

    _shared: dict[tuple, "ElasticRendererPool"] = {}

    def __init__(
        self,
        renderer_model: str,
        config: RendererConfig | None,
        *,
        chat_template_kwargs: Mapping[str, Any] | None = None,
        multiplex: int,
    ) -> None:
        self.renderer_model = renderer_model
        self.config = config
        self.chat_template_kwargs = chat_template_kwargs
        self.multiplex = multiplex
        self.slots: list[_RendererSlot] = []
        self._lock = asyncio.Lock()
        self._warm_task: asyncio.Task[_RendererSlot] | None = None

    @classmethod
    def shared(
        cls,
        renderer_model: str,
        config: RendererConfig | None,
        *,
        chat_template_kwargs: Mapping[str, Any] | None = None,
        multiplex: int,
    ) -> "ElasticRendererPool":
        """The process-wide pool for these construction inputs, warming its first renderer
        on the way. Same key shape as v0's `RendererClient._shared_pools`: renderers owns
        config resolution, so only pools whose build inputs differ are kept apart."""
        key = (
            renderer_model,
            config.model_dump_json() if config is not None else None,
            json.dumps(dict(chat_template_kwargs), sort_keys=True)
            if chat_template_kwargs
            else None,
            multiplex,
        )
        pool = cls._shared.get(key)
        if pool is None:
            pool = cls._shared[key] = cls(
                renderer_model,
                config,
                chat_template_kwargs=chat_template_kwargs,
                multiplex=multiplex,
            )
        pool.warm()
        return pool

    def warm(self) -> None:
        """Start building the first renderer, if nothing has yet. Called when a client is
        built so the tokenizer loads while the rollout is still provisioning, rather than
        in front of its first turn. A no-op off the event loop (tests, sync construction) —
        `acquire` builds on demand anyway."""
        if self._warm_task is not None or self.slots:
            return
        try:
            self._warm_task = asyncio.get_running_loop().create_task(self._grow())
        except RuntimeError:
            pass

    async def _grow(self) -> _RendererSlot:
        """Load one more tokenizer, on a thread — `create_renderer_pool` is seconds of
        blocking work. Callers hold `_lock`, so exactly one grows at a time."""
        from renderers import create_renderer_pool

        kwargs: dict[str, Any] = {"size": 1}
        if self.chat_template_kwargs:
            kwargs["chat_template_kwargs"] = self.chat_template_kwargs
        renderer = await asyncio.to_thread(
            create_renderer_pool, self.renderer_model, self.config, **kwargs
        )
        slot = _RendererSlot(renderer)
        self.slots.append(slot)
        logger.info(
            "renderer pool: %d renderer(s), multiplex=%d",
            len(self.slots),
            self.multiplex,
        )
        return slot

    @asynccontextmanager
    async def acquire(self) -> AsyncIterator[Any]:
        """A renderer to render this turn with, growing the pool when every one already
        carries `multiplex` rollouts. The slot is held for the turn, so `load` counts
        rollouts in flight rather than renders in progress."""
        if self._warm_task is not None:
            # Shielded: a cancelled acquire must not cancel the build every other rollout
            # is waiting on. A failed warm falls through to growing under the lock, where
            # the error reaches a caller instead of vanishing into a stray task.
            with contextlib.suppress(Exception):
                await asyncio.shield(self._warm_task)
            self._warm_task = None
        async with self._lock:
            slot = next((s for s in self.slots if s.load < self.multiplex), None)
            if slot is None:
                slot = await self._grow()
            slot.load += 1
        try:
            yield slot.renderer
        finally:
            slot.load -= 1


class TrainClient(Client):
    """Renders prompts to token ids and calls a vLLM `/inference/v1/generate` engine.

    One client per rollout: it owns its engine connection and takes a slot on the elastic
    renderer pool for each turn. Building the client warms the pool's first tokenizer, so
    the load happens while the rollout provisions rather than in front of its first turn.
    The pool itself is shared across clients — see `ElasticRendererPool`."""

    def __init__(self, config: TrainClientConfig) -> None:
        self.config = config
        self.client = build_async_openai(config)
        # The per-request model is only known at call time; a config that pins the renderer
        # model can warm now, which is every training run (prime-rl always pins it).
        if config.renderer_model_name is not None:
            self._pool_for(config.renderer_model_name)

    def _pool_for(self, renderer_model: str, chat_template_kwargs=None):
        return ElasticRendererPool.shared(
            renderer_model,
            self.config.renderer,
            chat_template_kwargs=chat_template_kwargs,
            multiplex=self.config.multiplex,
        )

    async def get_response(
        self,
        dialect: Dialect,
        body: dict,
        model: str,
        sampling_args: SamplingConfig,
        session_id: str | None = None,
        turn: PendingTurn | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> Response:
        # The renderer tokenizes the typed prompt for training (it needs per-token ids + logprobs
        # back), so it can't forward the raw request — it parses `body` via the dialect and renders
        # it with a chat template. It leaves `Response.raw` unset; the interception server serializes
        # its `Response` for the program instead of relaying provider bytes.
        if not isinstance(dialect, ChatDialect):
            # The renderer renders a chat template, so it's only validated for chat-completions
            # input; other dialects' semantics (Responses reasoning items, Anthropic thinking) may
            # not round-trip faithfully through chat-template tokenization. Refuse them explicitly.
            raise NotImplementedError(
                f"The renderer client only supports the chat-completions dialect, got "
                f"{type(dialect).__name__}. Use the proxy client for this dialect, or add "
                f"renderer support for it."
            )
        # Intercepted turns already own the typed prompt, so only their tools need parsing here.
        if turn is not None:
            prompt = turn.prompt
            tools = parse_tools(body.get("tools"))
        else:
            prompt, tools = dialect.parse_request(body)
        from renderers.client import _maybe_offload, generate

        wire_tools = [tool_to_wire(t) for t in tools] if tools else None
        wire_messages = (
            [message_to_wire(m) for m in turn.tail] if turn is not None else []
        )
        prompt_ids: list[int] | None = None
        multi_modal_data = None
        prompt_attribution: RenderedTokens | None = None
        raw_sampling = sampling_args.model_dump(exclude_none=True)
        sampling_params: dict[str, Any] = dict(
            raw_sampling.pop("extra_body", None) or {}
        )
        chat_template_kwargs = sampling_params.pop("chat_template_kwargs", None)
        sampling_params.update(raw_sampling)
        pool = self._pool_for(
            self.config.renderer_model_name or model,
            chat_template_kwargs=chat_template_kwargs,
        )
        bridged_turn: PendingTurn | None = None

        async with pool.acquire() as renderer:
            # Only build the (O(context)) previous-turn token ids once the cheap guards pass — a
            # multimodal prompt or a tail that isn't a clean `[tool*, user?]` extension can't bridge.
            can_bridge = (
                turn is not None
                and not _has_multimodal_content(prompt)
                and _is_valid_incremental_tail(wire_messages)
            )
            previous_ids = turn.previous_token_ids() if can_bridge else None
            if previous_ids is not None:
                previous_prompt_ids, previous_completion_ids = previous_ids

                def bridge():
                    return renderer.bridge_to_next_turn(
                        previous_prompt_ids,
                        previous_completion_ids,
                        wire_messages,
                        tools=wire_tools,
                    )

                bridged = await _maybe_offload(renderer, bridge)
                if bridged is not None:
                    prompt_ids = bridged.token_ids
                    multi_modal_data = bridged.multi_modal_data
                    prompt_attribution = bridged
                    bridged_turn = turn
                    sampling_params["routed_experts_prompt_start"] = max(
                        len(previous_prompt_ids) + len(previous_completion_ids) - 1,
                        0,
                    )

            # Bridged prompt ids bypass rendering; only fallback needs the full wire prompt.
            if prompt_ids is None:
                wire_messages = [message_to_wire(m) for m in prompt]

            try:
                result = await generate(
                    client=self.client,
                    renderer=renderer,
                    messages=wire_messages,
                    model=model,
                    prompt_ids=prompt_ids,
                    multi_modal_data=multi_modal_data,
                    prompt_attribution=prompt_attribution,
                    tools=wire_tools,
                    sampling_params=sampling_params,
                    extra_headers={SESSION_ID_HEADER: session_id}
                    if session_id
                    else None,
                )
            except RendererOverlongPromptError as e:
                raise OverlongPromptError(str(e)) from e
            except OpenAIError as e:
                raise model_error(e) from e
        response = response_from_generate(result, model, bridged_turn)
        # No provider response to relay (we generated), so serialize one for the program; the
        # interception server hands `Response.raw` back regardless of client.
        response.raw = serialize_completion(response, model)
        return response

    async def close(self) -> None:
        await self.client.close()
