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
from verifiers.types import SamplingArgs, State


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
        lineage_key = kwargs.pop("lineage_key", None)
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
        # Multi-agent envs pass lineage_key explicitly via
        # Environment.get_model_response so prefix matching only consults
        # steps authored by the same speaker. Keep the state-based lookup
        # as a migration fallback for older callers.
        if lineage_key is None and isinstance(state, dict):
            lineage_key = state.get("_lineage_key")
        prompt_ids = await self.get_prompt_ids(
            state, prompt, tools, lineage_key=lineage_key
        )
        if prompt_ids is None:
            return await super().get_native_response(
                prompt, model, sampling_args, tools, extra_headers=extra_headers
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

    async def get_prompt_ids(
        self,
        state: State,
        prompt_messages: OpenAIChatMessages,
        oai_tools: list[OpenAITool] | None,
        *,
        lineage_key: str | None = None,
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
                return {
                    str(key): normalize_for_comparison(val)
                    for key, val in value.items()
                }
            if isinstance(value, list):
                return [normalize_for_comparison(item) for item in value]
            return value

        async def find_largest_prefix_match() -> tuple[list[int], bool, int] | None:
            """Scan trajectory backwards for the step whose messages form the
            longest prefix of prompt_messages. Returns
            (token_ids, is_truncated, prefix_len) or None.

            When ``lineage_key`` is provided, only steps whose
            ``extras["member_id"]`` matches are considered. This keeps
            per-speaker prefix caches from colliding in multi-agent
            rollouts where speakers' prompts share a dataset prefix but
            diverge in history."""
            normalized_prompt_messages = normalize_for_comparison(prompt_messages)
            best_prefix_len = -1
            best_step = None
            for step in reversed(state["trajectory"]):
                step_tokens = step["tokens"]
                if step_tokens is None:
                    continue
                if lineage_key is not None:
                    extras = step.get("extras") or {}
                    if extras.get("member_id") != lineage_key:
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

        if tool_call_ids:
            dummy_assistant: OpenAIChatMessage = ChatCompletionAssistantMessageParam(
                role="assistant",
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
                role="assistant", content="x"
            )

        try:
            bridge_full_ids = await self.tokenize(
                messages=[dummy_assistant] + env_messages,
                tools=oai_tools,
                model=state["model"],
            )
            bridge_base_ids = await self.tokenize(
                messages=[dummy_assistant],
                tools=oai_tools,
                model=state["model"],
                extra_kwargs=dict(add_generation_prompt=False),
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
        extra_kwargs: dict = {},
        **kwargs,
    ) -> list[int]:
        """Tokenize messages using the vLLM /tokenize API."""
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
