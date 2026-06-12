from collections.abc import Mapping
from typing import Any, Optional, cast

from openai import AsyncOpenAI, BaseModel
from openai.types.chat import (
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
from verifiers.types import RendererTransport, SamplingArgs, State
from verifiers.utils.client_utils import (
    post_chat_completion_with_routed_experts_sidecar,
)

# Sentinel for the default (legacy vLLM) transport. Lets callers route
# around the legacy /tokenize body shape without changing the signature.
_DEFAULT_TRANSPORT: RendererTransport = "vllm"

# vLLM/prime-only sampling keys Dynamo's strict validator rejects — scrubbed
# from every dynamo request body (both MITO and TITO paths).
_DYNAMO_DROP_KEYS = frozenset(
    {"return_token_ids", "spaces_between_special_tokens", "priority"}
)


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


# copy from vllm/entrypoints/openai/protocol.py
class TokenizeResponse(BaseModel):
    count: int
    max_model_len: int
    tokens: list[int]
    token_strs: Optional[list[str]] = None


class OpenAIChatCompletionsTokenClient(OpenAIChatCompletionsClient):
    """Token-in / token-out chat client.

    Two transports share this class, selected via
    ``ClientConfig.renderer_transport``:

    * ``vllm`` (default): vLLM's TITO surface.
      Posts to ``/v1/chat/completions/tokens`` with ``tokens=prompt_ids``
      and uses the server's ``/tokenize`` endpoint for bridge tokens.
      Requires vLLM ``>=0.20``.

    * ``dynamo``: Dynamo's standard ``/v1/chat/completions``
      route with ``nvext.token_data=prompt_ids``. Server-side response
      token IDs come back via ``response.nvext.engine_data.*``
      (`OpenAIChatCompletionsClient.from_native_response` grafts them
      onto the OpenAI-shaped response). Bridge tokens are computed
      locally via the model's HuggingFace fast tokenizer — no
      ``/tokenize`` HTTP round-trip — since Dynamo doesn't expose vLLM's
      token routes.
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
        else the default vLLM TITO surface. Mirrors the same field used by
        ``RendererClient`` so backend selection stays in one place."""
        return cast(
            RendererTransport,
            getattr(self._config, "renderer_transport", _DEFAULT_TRANSPORT)
            if self._config is not None
            else _DEFAULT_TRANSPORT,
        )

    def _get_local_tokenizer(self, model: str):
        """Lazy, per-model HF fast tokenizer for the ``dynamo``
        transport. Bridge tokens are stitched locally — no ``/tokenize``
        round-trip. Cached so we pay the ``AutoTokenizer.from_pretrained``
        cost once.
        """
        # Honor the explicit tokenizer override (renderer_model_name) so model
        # aliases don't break bridge stitching; fall back to the served model.
        override = (
            getattr(self._config, "renderer_model_name", None)
            if self._config is not None
            else None
        )
        model = override or model
        cache: dict[str, Any] = self.__dict__.setdefault("_tokenizer_cache", {})
        if model in cache:
            return cache[model]
        try:
            from transformers import AutoTokenizer  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover - dependency surface
            raise ImportError(
                "OpenAIChatCompletionsTokenClient with "
                "renderer_transport='dynamo' requires "
                "`transformers`. Install with `pip install transformers`."
            ) from exc
        cache[model] = AutoTokenizer.from_pretrained(model)
        return cache[model]

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

            # Transport-specific opt-ins. Both transports get response-side
            # token IDs, just via different fields:
            #
            #   * vllm (vLLM): `extra_body.return_token_ids=True`
            #     tells vLLM to set the non-standard `choices[0].token_ids` and
            #     `response.prompt_token_ids` fields. `parse_tokens` reads them
            #     directly.
            #
            #   * dynamo: `nvext.extra_fields=["engine_data"]`
            #     tells Dynamo's response builder to emit `response.nvext`
            #     `engine_data.{completion_token_ids, completion_logprobs,
            #     prompt_token_ids}`. `from_native_response` grafts
            #     this onto the OpenAI-shaped response so `parse_tokens`
            #     works unmodified. `return_token_ids` is dropped because
            #     Dynamo's strict validator rejects it.
            if self.renderer_transport == "dynamo":
                extra_body: dict[str, Any] = {
                    "nvext": {"extra_fields": ["engine_data"]}
                }
            else:
                extra_body = {"return_token_ids": True}

            if "extra_body" in sampling_args:
                merged = {**sampling_args["extra_body"]}
                # Merge nvext.extra_fields cumulatively rather than overwriting,
                # so caller-provided extra_fields (e.g. "timing", "worker_id")
                # coexist with our "engine_data" opt-in.
                if "nvext" in merged and "nvext" in extra_body:
                    base = dict(merged.get("nvext") or {})
                    inc = dict(extra_body.get("nvext") or {})
                    base_ef = list(base.get("extra_fields") or [])
                    inc_ef = list(inc.get("extra_fields") or [])
                    merged_ef = list(dict.fromkeys(base_ef + inc_ef))
                    merged_nvext = {**base, **inc, "extra_fields": merged_ef}
                    merged["nvext"] = merged_nvext
                    sampling_args["extra_body"] = {
                        **{k: v for k, v in extra_body.items() if k != "nvext"},
                        **merged,
                    }
                else:
                    sampling_args["extra_body"] = {**merged, **extra_body}
            else:
                sampling_args["extra_body"] = extra_body
            if self.renderer_transport == "dynamo":
                # Drop vLLM/prime-only keys Dynamo rejects from both top-level
                # args and extra_body, so MITO + TITO paths send a clean body.
                eb = sampling_args.get("extra_body")
                if isinstance(eb, dict):
                    for k in _DYNAMO_DROP_KEYS:
                        eb.pop(k, None)
                for k in _DYNAMO_DROP_KEYS:
                    sampling_args.pop(k, None)
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
            return await self._post_dynamo(
                prompt=prompt,
                prompt_ids=prompt_ids,
                model=model,
                tools=tools,
                sampling_args=sampling_args,
                extra_headers=extra_headers,
            )

        extra_body = sampling_args.pop("extra_body", {})
        body = {
            "model": model,
            "messages": prompt,
            "tools": tools,
            "tokens": prompt_ids,
            **sampling_args,
            **extra_body,
        }

        return await post_chat_completion_with_routed_experts_sidecar(
            self.client,
            "/chat/completions/tokens",
            body=body,
            extra_headers=extra_headers,
        )

    async def _post_dynamo(
        self,
        prompt: OpenAIChatMessages,
        prompt_ids: list[int],
        model: str,
        tools: list[OpenAITool] | None,
        sampling_args: dict,
        extra_headers: Mapping[str, str] | None,
    ) -> OpenAIChatResponse:
        """Post stitched ``prompt_ids`` to Dynamo's chat-completions route.

        The engine sees ``nvext.token_data`` and skips its own tokenization,
        so the placeholder ``messages`` value stays small regardless of
        trajectory length. Response token IDs come back via
        ``response.nvext.engine_data.completion_token_ids`` and are grafted
        onto ``choices[0].token_ids`` by
        ``OpenAIChatCompletionsClient.from_native_response`` so the rest of
        the pipeline reads them via the standard openai SDK attribute path.
        """
        extra_body = dict(sampling_args.pop("extra_body", {}) or {})

        # nvext.token_data is the canonical pre-tokenized-prompt channel.
        # Merge with caller-provided nvext (extra_fields etc.) rather than
        # overwriting it. normalize_sampling_args already injected
        # extra_fields=["engine_data"] into extra_body.nvext, so this just
        # adds token_data to that same dict.
        caller_nvext = dict(extra_body.pop("nvext", None) or {})
        caller_nvext["token_data"] = prompt_ids
        nvext = caller_nvext

        body: dict[str, Any] = {
            "model": model,
            "messages": [{"role": "user", "content": ""}],
            "stream": False,
            "nvext": nvext,
        }
        if tools:
            body["tools"] = tools

        # Forward the full normalized sampling_args (parity with the vLLM path,
        # which spreads all of sampling_args), then remaining extra_body keys —
        # minus vLLM-only keys Dynamo's strict validator rejects (return_token_ids).
        # Unknown keys ride through the dynamo frontend's PASSTHROUGH_EXTRA_FIELDS.
        vllm_only = _DYNAMO_DROP_KEYS
        for source in (sampling_args, extra_body):
            for key, value in source.items():
                if value is None or key in vllm_only or key in body:
                    continue
                body[key] = value

        # Use the sidecar-aware post (same as the vLLM TITO + MITO paths) so any
        # routed_experts blob is streamed, not JSON-parsed. dynamo opts into
        # extra_fields=["engine_data"] only, so routed_experts is normally absent.
        return await post_chat_completion_with_routed_experts_sidecar(
            self.client,
            "/chat/completions",
            body=body,
            extra_headers=extra_headers,
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
                # (what MultiTurnEnv produces after maybe_normalize_messages)
                # never matches the to_native_prompt-normalized step messages,
                # which breaks the prefix match and forces TITO to fall back
                # to MITO every turn-2+.
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
        if not env_messages:
            return None
        env_roles = [
            msg.get("role") if hasattr(msg, "get") else getattr(msg, "role", None)
            for msg in env_messages
        ]
        if any(role != "tool" for role in env_roles[:-1]) or env_roles[-1] not in (
            "tool",
            "user",
        ):
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
            role = (
                msg.get("role") if hasattr(msg, "get") else getattr(msg, "role", None)
            )
            if role != "tool":
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
        """Tokenize messages for bridge-token computation.

        Dispatched by ``renderer_transport``:

        * ``vllm`` (default): POST to vLLM's ``/tokenize`` route.
        * ``dynamo``: local HF fast-tokenizer call. Dynamo doesn't
          expose ``/tokenize``; running locally also saves two HTTP RTTs per
          turn (the bridge computes both ``add_generation_prompt=True`` and
          ``False`` views). The HF Rust encode releases the GIL so the
          ``asyncio.to_thread`` wrap gives the event loop real parallelism.
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
        """Local in-process tokenization for the ``dynamo`` transport.

        Bridge tokenization under TITO calls this twice per turn (once for
        ``add_generation_prompt=True`` and once for ``False``). Both runs
        execute in a worker thread so the event loop stays free; HF fast
        tokenizers release the GIL during the Rust encode pass.
        """
        import asyncio

        add_generation_prompt = bool(extra_kwargs.get("add_generation_prompt", True))
        chat_template_kwargs = dict(extra_kwargs.get("chat_template_kwargs") or {})

        # Load the tokenizer inside the worker thread: a cache miss runs the
        # synchronous AutoTokenizer.from_pretrained, which must not block the loop.
        if isinstance(messages, str):
            def _encode_text() -> list[int]:
                tokenizer = self._get_local_tokenizer(model)
                return list(tokenizer.encode(messages, add_special_tokens=False))
            return await asyncio.to_thread(_encode_text)

        def _encode_chat() -> list[int]:
            tokenizer = self._get_local_tokenizer(model)
            ids = tokenizer.apply_chat_template(
                messages,
                tools=tools,
                add_generation_prompt=add_generation_prompt,
                tokenize=True,
                **chat_template_kwargs,
            )
            if hasattr(ids, "input_ids"):
                ids = ids.input_ids
            if ids and isinstance(ids[0], list):
                ids = ids[0]
            return [int(t) for t in ids]

        return await asyncio.to_thread(_encode_chat)
