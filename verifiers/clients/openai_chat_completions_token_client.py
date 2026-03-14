from collections.abc import Mapping
from typing import Any, Optional, cast

from openai import AsyncOpenAI, BaseModel
from openai.types.chat import ChatCompletion

from verifiers.clients.openai_chat_completions_client import (
    OpenAIChatCompletionsClient,
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
                prompt, model, sampling_args, tools
            )
        prompt_ids = await self.get_prompt_ids(state, prompt, tools)
        if prompt_ids is None:
            return await super().get_native_response(
                prompt, model, sampling_args, tools
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
        )

    async def get_prompt_ids(
        self,
        state: State,
        prompt_messages: OpenAIChatMessages,
        oai_tools: list[OpenAITool] | None,
    ) -> list[int] | None:
        """
        Build prompt_ids (token prompt) corresponding to prompt_messages. We assume
        that this method is called *before* making the model response from
        prompt_messages, i.e. the previous turn's prompt and completion do not yet
        include the environment response and next turn's model response.

        Returns None when no trajectory step has a message-level prefix match with
        prompt_messages.
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

        async def find_largest_prefix_match_tokens() -> list[int] | None:
            """Scan trajectory backwards for the step whose messages form the longest
            prefix of prompt_messages. Returns that step's token IDs, or None."""
            normalized_prompt_messages = normalize_for_comparison(prompt_messages)
            best_prefix_len = -1
            best_step_tokens = None
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
                best_step_tokens = step_tokens
                if best_prefix_len == len(normalized_prompt_messages):
                    break

            if best_step_tokens is None:
                return None
            return best_step_tokens["prompt_ids"] + best_step_tokens["completion_ids"]

        prev_turn_ids = await find_largest_prefix_match_tokens()
        if prev_turn_ids is None:
            return None

        # Tokenize the full prompt including and excluding env response to
        # derive just the env_response_ids
        full_ids_with_env_response = await self.tokenize(
            messages=prompt_messages,
            tools=oai_tools,
            model=state["model"],
        )
        full_ids_without_env_response = await self.tokenize(
            messages=prompt_messages[:-1],  # TODO: fix multi env-response
            tools=oai_tools,
            model=state["model"],
            extra_kwargs=dict(add_generation_prompt=False),
        )
        env_response_ids = full_ids_with_env_response[
            len(full_ids_without_env_response) :
        ]

        prompt_ids = prev_turn_ids + env_response_ids

        if prompt_ids != full_ids_with_env_response:
            import difflib

            from transformers import AutoTokenizer

            tok = AutoTokenizer.from_pretrained(state["model"])
            tito_text = tok.decode(prompt_ids)
            mito_text = tok.decode(full_ids_with_env_response)
            diff = difflib.unified_diff(
                mito_text.splitlines(keepends=True),
                tito_text.splitlines(keepends=True),
                fromfile="MITO",
                tofile="TITO",
            )
            self.logger.debug(
                "TITO/MITO mismatch, falling back to MITO:\n%s", "".join(diff)
            )
            return None

        return prompt_ids

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
