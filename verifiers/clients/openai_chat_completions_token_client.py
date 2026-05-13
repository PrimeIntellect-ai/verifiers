from collections.abc import Mapping
from typing import Any, Optional, cast

import httpx
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
from verifiers.types import SamplingArgs, State


TTT_CONTROL_KEYS = {
    "ttt_enabled",
    "ttt_learner_url",
    "ttt_window_seq_len",
    "ttt_train_prompt_lora",
    "ttt_train_completion_lora",
    "ttt_require_exact_token_ids",
    "ttt_completion_lora_trains_initial_prompt",
    "ttt_prompt_lora_trains_environment_responses",
    "ttt_cache_salt_includes_adapter",
    "ttt_request_timeout_s",
}


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
    """Wrapper for custom vLLM route /v1/chat/completions/tokens via AsyncOpenAI client."""

    @property
    def token_client(self) -> AsyncOpenAI:
        """Strips trailing /v1 from the OpenAI client."""
        base_url = str(self.client.base_url).rstrip("/")
        if base_url.endswith("/v1"):
            base_url = base_url[:-3]
        return self.client.with_options(base_url=base_url)

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
            extra_body = dict(return_token_ids=True)
            if "extra_body" in sampling_args:
                sampling_args["extra_body"] = {
                    **sampling_args["extra_body"],
                    **extra_body,
                }
            else:
                sampling_args["extra_body"] = extra_body
            return {k: v for k, v in sampling_args.items() if v is not None}

        sampling_args = normalize_sampling_args(sampling_args)
        state = cast(State, kwargs.pop("state"))
        extra_headers = kwargs.pop("extra_headers", None)
        ttt_options = self._pop_ttt_options(sampling_args)
        ttt_enabled = bool(ttt_options.get("ttt_enabled"))
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
        if has_multimodal:
            if ttt_enabled and ttt_options.get("ttt_require_exact_token_ids", True):
                raise ValueError("TTT online LoRA requires exact token ids; multimodal token fallback is unsupported.")
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
        if len(state["trajectory"]) == 0 and not ttt_enabled:
            return await super().get_native_response(
                prompt, model, sampling_args, tools, extra_headers=extra_headers
            )

        if len(state["trajectory"]) == 0:
            prompt_ids = await self.tokenize(
                messages=prompt,
                tools=tools,
                model=state["model"],
                extra_kwargs={"chat_template_kwargs": dict(chat_template_kwargs)}
                if chat_template_kwargs
                else None,
            )
            new_prompt_ids = list(prompt_ids)
        elif ttt_enabled:
            prompt_match = await self.get_prompt_ids_with_new_tokens(
                state, prompt, tools, chat_template_kwargs=chat_template_kwargs
            )
            if prompt_match is None:
                prompt_ids = None
                new_prompt_ids = []
            else:
                prompt_ids, new_prompt_ids = prompt_match
        else:
            prompt_ids = await self.get_prompt_ids(
                state, prompt, tools, chat_template_kwargs=chat_template_kwargs
            )
            new_prompt_ids = []

        if prompt_ids is None:
            # Reaching this branch means we have a non-empty trajectory but
            # could not stitch — surface it loudly so ops catches regressions.
            if ttt_enabled and ttt_options.get("ttt_require_exact_token_ids", True):
                raise ValueError("TTT online LoRA could not stitch exact prompt token ids.")
            self.logger.warning(
                f"TITO fell back to MITO on turn {len(state['trajectory']) + 1}"
            )
            return await super().get_native_response(
                prompt, model, sampling_args, tools, extra_headers=extra_headers
            )

        generation_prompt_ids = prompt_ids
        if ttt_enabled:
            window_seq_len = int(ttt_options.get("ttt_window_seq_len") or 0)
            max_completion_tokens = int(sampling_args.get("max_completion_tokens") or 0)
            prompt_window = max(window_seq_len - max_completion_tokens, 1) if max_completion_tokens else window_seq_len
            if prompt_window > 0 and len(generation_prompt_ids) > prompt_window:
                generation_prompt_ids = generation_prompt_ids[-prompt_window:]

        adapter_name = model
        ttt_prepare: dict[str, Any] | None = None
        if ttt_enabled:
            ttt_prepare = await self._ttt_prepare_turn(
                state=state,
                model=model,
                prompt_ids=generation_prompt_ids,
                new_prompt_ids=new_prompt_ids,
                options=ttt_options,
            )
            adapter_name = str(ttt_prepare["adapter_name"])

        extra_body = sampling_args.pop("extra_body", {})
        if ttt_enabled and ttt_options.get("ttt_cache_salt_includes_adapter", True):
            salt = extra_body.get("cache_salt")
            adapter_salt = adapter_name
            extra_body["cache_salt"] = f"{salt}:ttt:{adapter_salt}" if salt is not None else f"ttt:{adapter_salt}"
        body = dict(
            model=adapter_name,
            messages=prompt,
            tools=tools,
            tokens=generation_prompt_ids,
            **sampling_args,
            **extra_body,
        )

        response = await self.client.post(
            "/chat/completions/tokens",
            body=body,
            cast_to=ChatCompletion,
            options={"headers": extra_headers} if extra_headers else {},
        )
        if ttt_enabled:
            await self._ttt_complete_turn(
                state=state,
                model=model,
                response=response,
                prepare=ttt_prepare or {},
                options=ttt_options,
            )
        return response

    def _pop_ttt_options(self, sampling_args: dict[str, Any]) -> dict[str, Any]:
        extra_body = dict(sampling_args.get("extra_body") or {})
        options: dict[str, Any] = {}
        for key in TTT_CONTROL_KEYS:
            if key in extra_body:
                options[key] = extra_body.pop(key)
        sampling_args["extra_body"] = extra_body
        return options

    async def get_prompt_ids_with_new_tokens(
        self,
        state: State,
        prompt_messages: OpenAIChatMessages,
        oai_tools: list[OpenAITool] | None,
        chat_template_kwargs: dict | None = None,
    ) -> tuple[list[int], list[int]] | None:
        stitched = await self._get_prompt_ids_and_bridge(
            state, prompt_messages, oai_tools, chat_template_kwargs=chat_template_kwargs
        )
        if stitched is None:
            return None
        return stitched[0], stitched[1]

    async def get_prompt_ids(
        self,
        state: State,
        prompt_messages: OpenAIChatMessages,
        oai_tools: list[OpenAITool] | None,
        chat_template_kwargs: dict | None = None,
    ) -> list[int] | None:
        stitched = await self._get_prompt_ids_and_bridge(
            state, prompt_messages, oai_tools, chat_template_kwargs=chat_template_kwargs
        )
        if stitched is None:
            return None
        return stitched[0]

    async def _get_prompt_ids_and_bridge(
        self,
        state: State,
        prompt_messages: OpenAIChatMessages,
        oai_tools: list[OpenAITool] | None,
        chat_template_kwargs: dict | None = None,
    ) -> tuple[list[int], list[int], int] | None:
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

        return prev_turn_ids + list(bridge_ids), list(bridge_ids), prefix_len

    async def _ttt_prepare_turn(
        self,
        state: State,
        model: str,
        prompt_ids: list[int],
        new_prompt_ids: list[int],
        options: dict[str, Any],
    ) -> dict[str, Any]:
        learner_url = str(options.get("ttt_learner_url") or "").rstrip("/")
        if not learner_url:
            raise ValueError("ttt_learner_url must be set when ttt_enabled=true.")
        turn_idx = len(state["trajectory"])
        session_id = str(state.get("trajectory_id"))
        token_role = "completion_initial_prompt" if turn_idx == 0 else "prompt_environment"
        train_ids = list(new_prompt_ids)
        if token_role == "completion_initial_prompt" and (
            not options.get("ttt_train_completion_lora", True)
            or not options.get("ttt_completion_lora_trains_initial_prompt", True)
        ):
            train_ids = []
        if token_role == "prompt_environment" and (
            not options.get("ttt_train_prompt_lora", True)
            or not options.get("ttt_prompt_lora_trains_environment_responses", True)
        ):
            train_ids = []
        payload = {
            "session_id": session_id,
            "turn_idx": turn_idx,
            "model": model,
            "prompt_ids": prompt_ids,
            "new_prompt_ids": train_ids,
            "token_role": token_role,
        }
        timeout = float(options.get("ttt_request_timeout_s") or 120.0)
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(f"{learner_url}/prepare_turn", json=payload)
            response.raise_for_status()
            prepared = cast(dict[str, Any], response.json())
        prepared["prompt_ids"] = list(prompt_ids)
        prepared["new_prompt_ids"] = list(new_prompt_ids)
        return prepared

    async def _ttt_complete_turn(
        self,
        state: State,
        model: str,
        response: Any,
        prepare: dict[str, Any],
        options: dict[str, Any],
    ) -> None:
        learner_url = str(options.get("ttt_learner_url") or "").rstrip("/")
        if not learner_url:
            raise ValueError("ttt_learner_url must be set when ttt_enabled=true.")
        raw_completion_ids = getattr(response.choices[0], "token_ids", None)
        if raw_completion_ids is None and options.get("ttt_require_exact_token_ids", True):
            raise ValueError("TTT online LoRA requires exact completion token ids, but the response omitted them.")
        completion_ids = list(raw_completion_ids or [])
        completion_logprobs = self._extract_completion_logprobs(response)
        completion_train_ids = completion_ids if options.get("ttt_train_completion_lora", True) else []
        payload = {
            "session_id": str(state.get("trajectory_id")),
            "turn_idx": len(state["trajectory"]),
            "model": model,
            "completion_ids": completion_train_ids,
            "completion_logprobs": completion_logprobs,
            "prepare_version": prepare.get("version"),
        }
        timeout = float(options.get("ttt_request_timeout_s") or 120.0)
        async with httpx.AsyncClient(timeout=timeout) as client:
            http_response = await client.post(f"{learner_url}/complete_turn", json=payload)
            http_response.raise_for_status()
            complete = cast(dict[str, Any], http_response.json())

        trace = state.setdefault("ttt_trace", [])
        entry = {
            "turn_idx": len(state["trajectory"]),
            "session_id": str(state.get("trajectory_id")),
            "model": model,
            "adapter_name": prepare.get("adapter_name"),
            "adapter_path": prepare.get("adapter_path"),
            "base_step": prepare.get("base_step"),
            "prompt_token_count": prepare.get("trained_token_count", 0),
            "prompt_token_role": prepare.get("token_role"),
            "prompt_ids": list(prepare.get("prompt_ids") or []),
            "new_prompt_ids": list(prepare.get("new_prompt_ids") or []),
            "completion_token_count": len(completion_ids),
            "completion_ids": completion_ids,
            "completion_logprobs": completion_logprobs,
            "prepare": prepare,
            "complete": complete,
        }
        trace.append(entry)
        if isinstance(complete.get("final_prompt_adapter"), dict):
            state["ttt_final_prompt_adapter"] = complete["final_prompt_adapter"]

    def _extract_completion_logprobs(self, response: Any) -> list[float]:
        choice = response.choices[0]
        logprobs = getattr(choice, "logprobs", None)
        if logprobs is None:
            return []
        if hasattr(logprobs, "content") and logprobs.content is not None:
            return [float(token.logprob) for token in logprobs.content]
        if isinstance(logprobs, dict) and logprobs.get("content") is not None:
            return [float(token["logprob"]) for token in logprobs["content"]]
        return []

    async def tokenize(
        self,
        messages: str | OpenAIChatMessages,
        tools: list[OpenAITool] | None,
        model: str,
        extra_kwargs: dict | None = None,
        **kwargs,
    ) -> list[int]:
        """Tokenize messages using the vLLM /tokenize API."""
        if extra_kwargs is None:
            extra_kwargs = {}
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
