import asyncio
from collections.abc import Mapping
from typing import Any, Optional, cast

from openai import AsyncOpenAI, BaseModel
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionAssistantMessageParam,
)
from openai.types.chat.chat_completion_message_function_tool_call_param import (
    ChatCompletionMessageFunctionToolCallParam,
    Function,
)

from verifiers.clients.openai_chat_completions_client import (
    OpenAIChatCompletionsClient,
    OpenAIChatMessage,
    OpenAIChatMessages,
    OpenAIChatResponse,
    OpenAITool,
    handle_openai_overlong_prompt,
)
from verifiers.types import (
    RendererTransport,
    SamplingArgs,
    State,
    normalize_renderer_transport,
)

# Sentinel returned by transports that don't tokenize over HTTP. Lets callers
# route around the legacy /tokenize body shape without changing the signature.
_DEFAULT_TRANSPORT: RendererTransport = "vllm"


def _has_multimodal_content(messages) -> bool:
    """Check if any message contains multimodal content (images, audio).

    Works with both plain dicts (OpenAIChatMessages) and Pydantic models
    (Messages stored in trajectory steps) since both support .get().
    """
    for msg in messages:
        content = msg.get("content") if hasattr(msg, "get") else None
        if isinstance(content, list):
            for part in content:
                if hasattr(part, "get") and part.get("type") in (
                    "image_url",
                    "input_audio",
                ):
                    return True
    return False


def _get_role(msg) -> str | None:
    return msg.get("role") if hasattr(msg, "get") else getattr(msg, "role", None)


def _is_valid_env_tail(messages: list) -> bool:
    """Validate that messages follow env response patterns:
    all tool messages, with optionally a single user message last."""
    if not messages:
        return False
    for msg in messages[:-1]:
        if _get_role(msg) != "tool":
            return False
    return _get_role(messages[-1]) in ("tool", "user")


# copy from vllm/entrypoints/openai/protocol.py
class TokenizeResponse(BaseModel):
    count: int
    max_model_len: int
    tokens: list[int]
    token_strs: Optional[list[str]] = None


class OpenAIChatCompletionsTokenClient(OpenAIChatCompletionsClient):
    """Token-in/token-out chat client.

    Two transports share this class:

    * ``vllm`` (default): the historical TITO surface that
      posts to vLLM's ``/v1/chat/completions/tokens`` and uses the engine's
      ``/tokenize`` for bridge-token computation. This is what vanilla vLLM
      ``>=0.20`` exposes.
    * ``dynamo``: posts pre-tokenized prompts to Dynamo's standard
      ``/v1/chat/completions`` route with ``nvext.token_data`` carrying the
      stitched ``prompt_ids``. Bridge tokenization runs locally via the
      ``renderers`` package (no ``/tokenize`` round-trip) since Dynamo
      doesn't expose vLLM's token routes. Selection is via
      ``ClientConfig.renderer_transport``; same field the renderer client
      consults so a single config option drives both clients consistently.
    """

    @property
    def token_client(self) -> AsyncOpenAI:
        """Strips trailing /v1 from the OpenAI client."""
        base_url = str(self.client.base_url).rstrip("/")
        if base_url.endswith("/v1"):
            base_url = base_url[:-3]
        return self.client.with_options(base_url=base_url)

    @property
    def renderer_transport(self) -> RendererTransport:
        """Wire-shape selector. ``ClientConfig.renderer_transport`` if set,
        else the default vLLM TITO shape. Mirrors the same field used by
        ``RendererClient`` so backend selection stays in one place."""
        return normalize_renderer_transport(
            getattr(self._config, "renderer_transport", _DEFAULT_TRANSPORT)
            if self._config is not None
            else _DEFAULT_TRANSPORT,
        )

    def _get_renderer(self, model: str):
        """Lazy, per-model renderer cache. Used only by the ``dynamo``
        transport for client-side tokenization and stop-token resolution.

        Loaded on first use and reused across calls so we pay the
        ``AutoTokenizer.from_pretrained`` cost once. The renderer's
        underlying tokenizer is HuggingFace fast-tokenizer-backed, so the
        wrapping ``asyncio.to_thread`` calls in ``tokenize()`` get real
        parallelism (the Rust encode releases the GIL).
        """
        cache: dict[str, Any] = self.__dict__.setdefault("_renderer_cache", {})
        if model in cache:
            return cache[model]
        try:
            from renderers import create_renderer
            from transformers import AutoTokenizer
        except ImportError as exc:  # pragma: no cover - dependency surface
            raise ImportError(
                "OpenAIChatCompletionsTokenClient with renderer_transport="
                "'dynamo' requires the 'renderers' and 'transformers' "
                "packages. Install via `pip install verifiers[renderers]` or add "
                "renderers + transformers to your environment."
            ) from exc
        tokenizer = AutoTokenizer.from_pretrained(model)
        renderer_name = (
            getattr(self._config, "renderer", "auto")
            if self._config is not None
            else "auto"
        )
        renderer = create_renderer(tokenizer, renderer=renderer_name or "auto")
        cache[model] = renderer
        return renderer

    @handle_openai_overlong_prompt
    async def get_native_response(
        self,
        prompt: OpenAIChatMessages,
        model: str,
        sampling_args: SamplingArgs,
        tools: list[OpenAITool] | None = None,
        **kwargs,
    ) -> OpenAIChatResponse:
        def normalize_sampling_args(sampling_args: SamplingArgs):
            sampling_args = dict(sampling_args)
            if "max_tokens" in sampling_args:
                sampling_args["max_completion_tokens"] = sampling_args.pop("max_tokens")
            sampling_args["logprobs"] = True

            if self.renderer_transport == "dynamo":
                extra_body: dict[str, Any] = {
                    "nvext": {"extra_fields": ["engine_data"]}
                }
            else:
                extra_body = {"return_token_ids": True}

            if "extra_body" in sampling_args:
                merged = {**sampling_args["extra_body"]}
                if "nvext" in merged and "nvext" in extra_body:
                    merged_nvext = merged.get("nvext")
                    extra_nvext = extra_body.get("nvext")
                    base = (
                        dict(merged_nvext) if isinstance(merged_nvext, Mapping) else {}
                    )
                    inc = dict(extra_nvext) if isinstance(extra_nvext, Mapping) else {}
                    base_extra_fields = list(base.get("extra_fields") or [])
                    inc_extra_fields = list(inc.get("extra_fields") or [])
                    extra_fields = list(
                        dict.fromkeys(base_extra_fields + inc_extra_fields)
                    )
                    merged["nvext"] = {**base, **inc, "extra_fields": extra_fields}
                    sampling_args["extra_body"] = {
                        **{k: v for k, v in extra_body.items() if k != "nvext"},
                        **merged,
                    }
                else:
                    sampling_args["extra_body"] = {**merged, **extra_body}
            else:
                sampling_args["extra_body"] = extra_body
            return {k: v for k, v in sampling_args.items() if v is not None}

        sampling_args = normalize_sampling_args(sampling_args)
        state = cast(State, kwargs.pop("state"))
        extra_headers = kwargs.pop("extra_headers", None)
        # Use standard /chat/completions for: (1) first turn (no prior tokens
        # to stitch), or (2) conversations that contain multimodal content in
        # any turn.  vLLM ≤0.16's /tokenize doesn't run the multimodal
        # processor, so image placeholders stay collapsed (1 token instead of
        # N) and token-stitching (TITO) produces broken prompts.  Falling back
        # to message-based inference (MITO) lets vLLM handle expansion
        # correctly on every turn.
        has_multimodal = _has_multimodal_content(prompt) or any(
            _has_multimodal_content(step["prompt"]) for step in state["trajectory"]
        )
        if len(state["trajectory"]) == 0 or has_multimodal:
            return await super().get_native_response(
                prompt, model, sampling_args, tools, extra_headers=extra_headers
            )
        # The bridge tokenize calls inside get_prompt_ids must run under the
        # same chat-template config as the engine's actual generation,
        # otherwise the bridge tokens won't line up with what vLLM streamed
        # (e.g. GLM-5.1's `clear_thinking` flag changes the rendering of past
        # assistants — and of the dummy assistant we use for the bridge —
        # which can break the bridge prefix property).
        # `extra_body` is guaranteed by normalize_sampling_args above;
        # `chat_template_kwargs` is rollout-configured and may be absent.
        chat_template_kwargs = sampling_args["extra_body"].get(
            "chat_template_kwargs", {}
        )
        prompt_ids = await self.get_prompt_ids(
            state, prompt, tools, chat_template_kwargs=chat_template_kwargs
        )
        if prompt_ids is None:
            # Reaching this branch means we have a non-empty trajectory but
            # could not stitch — surface it loudly so ops catches regressions.
            self.logger.warning(
                f"TITO fell back to MITO on turn {len(state['trajectory']) + 1}"
            )
            return await super().get_native_response(
                prompt, model, sampling_args, tools, extra_headers=extra_headers
            )

        if self.renderer_transport == "dynamo":
            return await self._post_dynamo_chat_completions(
                prompt=prompt,
                prompt_ids=prompt_ids,
                model=model,
                tools=tools,
                sampling_args=sampling_args,
                extra_headers=extra_headers,
            )

        extra_body = sampling_args.pop("extra_body", {})
        body = dict(
            model=model,
            messages=prompt,
            tools=tools,
            tokens=prompt_ids,
            **sampling_args,
            **extra_body,
        )

        return await self.client.post(
            "/chat/completions/tokens",
            body=body,
            cast_to=ChatCompletion,
            options={"headers": extra_headers} if extra_headers else {},
        )

    async def _post_dynamo_chat_completions(
        self,
        prompt: OpenAIChatMessages,
        prompt_ids: list[int],
        model: str,
        tools: list[OpenAITool] | None,
        sampling_args: dict,
        extra_headers: Mapping[str, str] | None,
    ) -> OpenAIChatResponse:
        """Post stitched prompt_ids to Dynamo's chat-completions route.

        The engine sees ``nvext.token_data`` and skips tokenization. Response
        token IDs come back through ``nvext.engine_data.completion_token_ids``
        and are grafted onto the standard token fields by
        ``OpenAIChatCompletionsClient.from_native_response``.
        """
        renderer = self._get_renderer(model)
        stop_token_ids = list(renderer.get_stop_token_ids())

        extra_body = dict(sampling_args.pop("extra_body", {}) or {})

        nvext = dict(extra_body.pop("nvext", None) or {})
        nvext["token_data"] = prompt_ids
        priority = sampling_args.get("priority", extra_body.get("priority"))
        if priority is not None:
            nvext["agent_hints"] = {"priority": priority}

        body: dict[str, Any] = {
            "model": model,
            "messages": [{"role": "user", "content": "(token-in mode)"}],
            "stream": False,
            "logprobs": True,
            "stop_token_ids": stop_token_ids,
            "nvext": nvext,
        }
        if tools:
            body["tools"] = tools

        # Promote sampling fields that Dynamo's chat-completions surface
        # accepts directly. Anything else stays in extra_body and rides as
        # an unrecognized passthrough field (validate.rs:104 allowlist).
        promotable = (
            "max_completion_tokens",
            "max_tokens",
            "temperature",
            "top_p",
            "top_k",
            "min_p",
            "seed",
            "n",
            "repetition_penalty",
            "min_tokens",
            "top_logprobs",
            "stop",
        )
        for key in promotable:
            value = sampling_args.get(key, extra_body.get(key))
            if value is not None:
                body[key] = value

        # Pass any remaining unhandled extra_body keys straight through (e.g.
        # cache_salt, return_token_ids). Dynamo's PASSTHROUGH_EXTRA_FIELDS
        # allowlist accepts these without rejection.
        passthrough = {
            k: v
            for k, v in extra_body.items()
            if k not in promotable and v is not None and k not in body
        }
        body.update(passthrough)

        return await self.client.post(
            "/chat/completions",
            body=body,
            cast_to=ChatCompletion,
            options={"headers": extra_headers} if extra_headers else {},
        )

    async def get_prompt_ids(
        self,
        state: State,
        prompt_messages: OpenAIChatMessages,
        oai_tools: list[OpenAITool] | None,
        chat_template_kwargs: dict | None = None,
    ) -> list[int] | None:
        """
        Build prompt_ids for the next turn by stitching engine tokens with
        bridge tokens for the environment response.

        The engine's prev_turn_ids are preserved exactly (no retokenization),
        guaranteeing KV cache reuse via vLLM's prefix caching. Only the bridge
        tokens (env response + generation prompt) are new.

        Returns None to fall back to MITO when stitching is not possible.
        """

        def normalize_for_comparison(value: Any) -> Any:
            if hasattr(value, "model_dump"):
                return normalize_for_comparison(value.model_dump())
            if isinstance(value, Mapping):
                normalized = {
                    str(key): normalize_for_comparison(val)
                    for key, val in value.items()
                }
                # Treat content=None and content="" as equivalent: tool-call-only
                # assistant messages can be serialized either way depending on the
                # upstream pipeline (e.g., reasoning parsers strip text content
                # to "" while other paths leave it as None). Coerce to None so
                # prefix-match equality is unaffected.
                if normalized.get("content") == "":
                    normalized["content"] = None
                # Drop None-valued keys so model_dump's exhaustive view (which
                # carries e.g. thinking_blocks=None on AssistantMessage) is
                # equivalent to to_native_prompt's slimmer view (which omits
                # the field entirely). Without this, vf.Message-shaped input
                # never matches the to_native_prompt-normalized step messages,
                # which breaks the prefix match for MultiTurnEnv rollouts.
                normalized = {k: v for k, v in normalized.items() if v is not None}
                return normalized
            if isinstance(value, list):
                return [normalize_for_comparison(item) for item in value]
            return value

        async def find_largest_prefix_match() -> tuple[list[int], bool, int] | None:
            """Scan trajectory backwards for the step whose messages form the
            longest prefix of prompt_messages. Returns
            (token_ids, is_truncated, prefix_len) or None."""
            normalized_prompt_messages = normalize_for_comparison(prompt_messages)
            best_prefix_len = -1
            best_step = None
            for step in reversed(state["trajectory"]):
                step_tokens = step["tokens"]
                if step_tokens is None:
                    continue
                step_messages = cast(Any, [*step["prompt"], *step["completion"]])
                step_prompt_messages, _ = await self.to_native_prompt(step_messages)
                normalized_step_messages = normalize_for_comparison(
                    step_prompt_messages
                )
                prefix_len = len(normalized_step_messages)
                if prefix_len <= 0:
                    continue
                if prefix_len <= best_prefix_len:
                    continue
                if prefix_len > len(normalized_prompt_messages):
                    continue
                if normalized_prompt_messages[:prefix_len] != normalized_step_messages:
                    continue
                best_prefix_len = prefix_len
                best_step = step
                if best_prefix_len == len(normalized_prompt_messages):
                    break

            if best_step is None:
                return None
            best_step_tokens = best_step["tokens"]
            prev_turn_ids = (
                best_step_tokens["prompt_ids"] + best_step_tokens["completion_ids"]
            )
            # Check both seq_len overflow (from token parsing) and max_tokens
            # truncation (from vLLM finish_reason="length").
            is_truncated = best_step_tokens.get("is_truncated", False) or (
                best_step.get("response") is not None
                and getattr(best_step["response"].message, "is_truncated", False)
            )
            return prev_turn_ids, is_truncated, best_prefix_len

        match = await find_largest_prefix_match()
        if match is None:
            return None

        prev_turn_ids, is_truncated, prefix_len = match

        # Truncated completions have no stop token — can't reliably stitch.
        if is_truncated:
            self.logger.debug("TITO: truncated completion, falling back to MITO")
            return None

        # The env messages are everything after the prefix match.
        env_messages: OpenAIChatMessages = list(prompt_messages[prefix_len:])
        if not _is_valid_env_tail(env_messages):
            return None

        # Extract the bridge tokens using a minimal dual-tokenization that
        # avoids the problematic assistant message entirely. We tokenize:
        #   (a) [dummy_assistant, env_messages...]  with gen=True
        #   (b) [dummy_assistant]                   with gen=False
        # The bridge = (a)[cut_point:] where cut_point accounts for the gap
        # between the engine's stop token and the template's inter-turn separator.
        #
        # Using a dummy assistant message ensures the inter-turn separator between
        # assistant and env response is correct, while avoiding template behaviors
        # that depend on the assistant being the last message (e.g., Qwen3's
        # context-dependent think block injection with add_generation_prompt=False).
        # Collect tool_call_ids from leading tool messages so the dummy
        # assistant satisfies chat-template validation ("tool message must
        # follow an assistant message with a tool call").
        tool_call_ids: list[str] = []
        for msg in env_messages:
            if _get_role(msg) != "tool":
                break
            tc_id = (
                msg.get("tool_call_id")
                if hasattr(msg, "get")
                else getattr(msg, "tool_call_id", None)
            )
            if tc_id:
                tool_call_ids.append(tc_id)

        # GLM-5.1's chat template only renders `<think>{rc}</think>` for an
        # assistant when `reasoning_content` ends up *defined*. The cascade
        # that defines it from position (`idx > last_user_index`) flips when
        # env_messages ends in a user message, breaking the bridge prefix
        # property. Setting reasoning_content="" forces branch 1 of the
        # cascade so the dummy renders identically across env-tail shapes.
        if tool_call_ids:
            dummy_assistant: OpenAIChatMessage = ChatCompletionAssistantMessageParam(
                role="assistant",
                reasoning_content="",  # type: ignore[typeddict-unknown-key]
                tool_calls=[
                    ChatCompletionMessageFunctionToolCallParam(
                        id=tc_id,
                        type="function",
                        function=Function(name="f", arguments="{}"),
                    )
                    for tc_id in tool_call_ids
                ],
            )
        else:
            dummy_assistant: OpenAIChatMessage = ChatCompletionAssistantMessageParam(
                role="assistant",
                reasoning_content="",  # type: ignore[typeddict-unknown-key]
                content="x",
            )

        # Forward the rollout's chat_template_kwargs so the bridge is
        # rendered under the same template config as the engine's stream.
        forwarded_ctk = (
            {"chat_template_kwargs": dict(chat_template_kwargs)}
            if chat_template_kwargs
            else {}
        )

        try:
            bridge_full_ids = await self.tokenize(
                messages=[dummy_assistant] + env_messages,
                tools=oai_tools,
                model=state["model"],
                extra_kwargs=dict(forwarded_ctk),
            )
            bridge_base_ids = await self.tokenize(
                messages=[dummy_assistant],
                tools=oai_tools,
                model=state["model"],
                extra_kwargs=dict(add_generation_prompt=False, **forwarded_ctk),
            )
        except Exception:
            self.logger.debug("TITO: bridge tokenization failed, falling back to MITO")
            return None

        # Verify the base is a prefix of the full (sanity check)
        if bridge_full_ids[: len(bridge_base_ids)] != bridge_base_ids:
            self.logger.debug(
                "TITO: bridge prefix property broken, falling back to MITO"
            )
            return None

        # The base ends at the template-rendered stop token + inter-turn separator.
        # The engine's prev_turn_ids ends at just the stop token.
        # The gap = tokens the template adds after the stop token (e.g., \n for Qwen).
        # We include the gap in the bridge so it covers everything after the stop token.
        #
        # Find the gap by locating the stop token in bridge_base_ids.
        # The stop token is the last completion_ids token from the matched step.
        stop_token_id = prev_turn_ids[-1]
        gap = 0
        for i in range(len(bridge_base_ids) - 1, -1, -1):
            if bridge_base_ids[i] == stop_token_id:
                gap = len(bridge_base_ids) - i - 1
                break

        bridge_ids = bridge_full_ids[len(bridge_base_ids) - gap :]

        # Handle stop tokens that double as role markers (e.g., GLM's <|observation|>):
        # if the bridge starts with the stop token that's already at the end of
        # prev_turn_ids, skip it to avoid duplication.
        if bridge_ids and bridge_ids[0] == stop_token_id:
            bridge_ids = bridge_ids[1:]

        return prev_turn_ids + list(bridge_ids)

    async def tokenize(
        self,
        messages: str | OpenAIChatMessages,
        tools: list[OpenAITool] | None,
        model: str,
        extra_kwargs: dict | None = None,
        **kwargs,
    ) -> list[int]:
        """Tokenize messages.

        ``dynamo`` transport: tokenizes locally via the
        ``renderers`` package, no network call. Runs on a worker thread so
        the event loop stays free; HuggingFace fast tokenizers release the
        GIL during the Rust encode pass.

        Default transport: posts to vLLM's ``/tokenize`` route on the
        host root.
        """
        if extra_kwargs is None:
            extra_kwargs = {}

        if self.renderer_transport == "dynamo":
            return await self._local_tokenize(
                messages=messages,
                tools=tools,
                model=model,
                extra_kwargs=extra_kwargs,
            )

        if isinstance(messages, str):
            body = dict(
                model=model,
                prompt=messages,
                **extra_kwargs,
            )
            tokenize_response = await self.token_client.post(
                "/tokenize", body=body, cast_to=TokenizeResponse
            )
        else:
            body = dict(
                model=model,
                messages=messages,
                tools=tools,
                **extra_kwargs,
            )
            tokenize_response = await self.token_client.post(
                "/tokenize", body=body, cast_to=TokenizeResponse
            )
        return tokenize_response.tokens

    async def _local_tokenize(
        self,
        messages: str | OpenAIChatMessages,
        tools: list[OpenAITool] | None,
        model: str,
        extra_kwargs: dict,
    ) -> list[int]:
        """Local in-process tokenization for the dynamo transport.

        Bridge tokenization under TITO calls this twice per turn (once for
        ``add_generation_prompt=True`` and once for ``False``). Both calls
        go through the same renderer, so the chat-template + tool-call
        normalization is consistent with whatever Dynamo's worker would
        produce server-side.
        """
        renderer = self._get_renderer(model)

        def _render() -> list[int]:
            if isinstance(messages, str):
                tokenizer = getattr(renderer, "tokenizer", None)
                if tokenizer is None:
                    raise RuntimeError(
                        "Renderer for model %r does not expose a tokenizer; "
                        "cannot tokenize a raw string under dynamo." % model
                    )
                # Strip BOS for parity with vLLM /tokenize (which never
                # prepends a BOS for raw-prompt tokenize requests).
                encoded = tokenizer(messages, add_special_tokens=False)
                return list(encoded["input_ids"])

            add_generation_prompt = bool(
                extra_kwargs.get("add_generation_prompt", True)
            )
            return list(
                renderer.render_ids(
                    cast(Any, list(messages)),
                    tools=cast(Any, tools),
                    add_generation_prompt=add_generation_prompt,
                )
            )

        return await asyncio.to_thread(_render)
