"""Train client: renders prompts to token ids and calls a vLLM generate endpoint."""

import asyncio
import json
import logging
import threading
from collections.abc import AsyncIterator, Callable, Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any, ClassVar, TypeVar

from openai import OpenAIError
from renderers import OverlongPromptError as RendererOverlongPromptError
from renderers import RenderedTokens, Renderer, RendererConfig
from renderers.base import is_multimodal

from verifiers.v1.clients.base import build_async_openai
from verifiers.v1.clients.client import SESSION_ID_HEADER, Client
from verifiers.v1.configs.client import TrainClientConfig
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

T = TypeVar("T")


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


@dataclass
class RendererSlot:
    """One renderer and the rollouts currently holding it. Encoding mutates a fast
    tokenizer's state, so `run` serializes encode-side work (render, bridge) on a thread;
    decode-side work is a pure read and needs neither the lock nor the hop."""

    renderer: Renderer
    load: int = 0
    lock: threading.Lock = field(default_factory=threading.Lock)

    async def run(self, fn: Callable[[], T]) -> T:
        def locked() -> T:
            with self.lock:
                return fn()

        return await asyncio.to_thread(locked)


class ElasticRendererPool:
    """Process-shared renderers, multiplexed and auto-growing. A pool object is a cheap
    per-client view: the renderers themselves are the shared state, keyed by what builds
    them, so every client with the same build inputs works one list."""

    _renderers: ClassVar[dict[tuple, list[RendererSlot]]] = {}
    """The process's renderers, keyed by build inputs. Unlike the owned interception pool,
    renderers have no owner to inject them (clients are built per rollout) and must outlive
    loops and clients, so the shared state lives on the type — and needs no `start`/`stop`:
    a renderer is pure memory."""

    _locks: ClassVar[dict[tuple, tuple[asyncio.AbstractEventLoop, asyncio.Lock]]] = {}
    """Per-key single-flight lock for `grow`, minted fresh on a loop change: an asyncio
    lock binds to the loop that first awaits it, while renderers outlive loops."""

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
        self.key = (
            renderer_model,
            config.model_dump_json() if config is not None else None,
            json.dumps(dict(chat_template_kwargs), sort_keys=True)
            if chat_template_kwargs
            else None,
        )
        self.renderers = self._renderers.setdefault(self.key, [])

    def warm(self) -> None:
        """Start building the first renderer if none exists, so the tokenizer loads while
        the rollout provisions rather than in front of its first turn. A no-op off the
        event loop — `acquire` builds on demand anyway."""
        if self.renderers:
            return
        try:
            task = asyncio.get_running_loop().create_task(self.grow())
        except RuntimeError:
            return
        # A failed warm is not an event: acquire retries the build and surfaces the error.
        task.add_done_callback(lambda t: t.cancelled() or t.exception())

    async def grow(self) -> RendererSlot:
        """A renderer with spare capacity — reuse one under `multiplex`, else load one
        more tokenizer on a thread. Single-flight per key: concurrent cold acquires wait
        for one build instead of stacking tokenizers."""
        loop = asyncio.get_running_loop()
        bound = self._locks.get(self.key)
        if bound is None or bound[0] is not loop:
            bound = self._locks[self.key] = (loop, asyncio.Lock())
        async with bound[1]:
            for slot in self.renderers:
                if slot.load < self.multiplex:
                    return slot
            from renderers import create_renderer
            from renderers.base import load_tokenizer

            def build():
                return create_renderer(
                    load_tokenizer(self.renderer_model),
                    self.config,
                    chat_template_kwargs=self.chat_template_kwargs,
                )

            slot = RendererSlot(await asyncio.to_thread(build))
            self.renderers.append(slot)
            logger.info(
                "renderer pool: %d renderer(s), multiplex=%d",
                len(self.renderers),
                self.multiplex,
            )
            return slot

    @asynccontextmanager
    async def acquire(self) -> AsyncIterator[RendererSlot]:
        """A renderer to render this turn with, growing the pool when every one already
        carries `multiplex` rollouts. The slot is held for the turn, so `load` counts
        rollouts in flight rather than renders in progress."""
        slot = await self.grow()
        slot.load += 1
        try:
            yield slot
        finally:
            slot.load -= 1


class TrainClient(Client):
    """Renders prompts to token ids and calls a vLLM `/inference/v1/generate` engine.

    One client per rollout: it owns its engine connection and takes a slot on the shared
    `ElasticRendererPool` for each turn."""

    def __init__(self, config: TrainClientConfig) -> None:
        self.config = config
        self.client = build_async_openai(config)
        # The per-request model is only known at call time; a config that pins the renderer
        # model can warm now, which is every training run (prime-rl always pins it).
        if config.renderer_model_name is not None:
            ElasticRendererPool(
                config.renderer_model_name,
                config.renderer,
                multiplex=config.multiplex,
            ).warm()

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
        from renderers.client import generate

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
        pool = ElasticRendererPool(
            self.config.renderer_model_name or model,
            self.config.renderer,
            chat_template_kwargs=chat_template_kwargs,
            multiplex=self.config.multiplex,
        )
        bridged_turn: PendingTurn | None = None

        async with pool.acquire() as slot:
            renderer = slot.renderer
            # Only build the (O(context)) previous-turn token ids once the cheap guards pass — a
            # tail that isn't a clean `[tool*, user?]` extension can't bridge.
            can_bridge = turn is not None and _is_valid_incremental_tail(wire_messages)
            previous_ids = turn.previous_token_ids() if can_bridge else None
            if previous_ids is not None:
                previous_prompt_ids, previous_completion_ids = previous_ids

                def bridge():
                    kwargs: dict[str, Any] = {"tools": wire_tools}
                    if is_multimodal(renderer):
                        kwargs["previous_multi_modal_data"] = (
                            turn.previous_multi_modal_data()
                        )
                    return renderer.bridge_to_next_turn(
                        previous_prompt_ids,
                        previous_completion_ids,
                        wire_messages,
                        **kwargs,
                    )

                bridged = await slot.run(bridge)
                if bridged is not None:
                    prompt_ids = bridged.token_ids
                    multi_modal_data = bridged.multi_modal_data
                    prompt_attribution = bridged
                    bridged_turn = turn
                    sampling_params["routed_experts_prompt_start"] = max(
                        len(previous_prompt_ids) + len(previous_completion_ids) - 1,
                        0,
                    )

            # Render here (encode-side, so through the slot) rather than inside `generate`:
            # handed prebuilt prompt_ids, generate's own renderer touches are decode-side
            # and stop-id reads, safe on a bare renderer without lock or thread hop.
            if prompt_ids is None:
                wire_messages = [message_to_wire(m) for m in prompt]
                rendered = await slot.run(
                    lambda: renderer.render(
                        wire_messages, tools=wire_tools, add_generation_prompt=True
                    )
                )
                prompt_ids = rendered.token_ids
                multi_modal_data = rendered.multi_modal_data
                prompt_attribution = rendered

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
