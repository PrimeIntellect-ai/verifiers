from __future__ import annotations

from verifiers.clients.openai_chat_completions_client import (
    OpenAIChatCompletionsClient,
    OpenAIChatMessages,
    OpenAIChatResponse,
    OpenAITool,
    handle_openai_overlong_prompt,
)
from verifiers.types import SamplingArgs


class NeMoRLChatCompletionsClient(OpenAIChatCompletionsClient):
    """Client for NeMo Gym model servers that embed token IDs in the message dict.

    NeMo Gym's model servers (e.g., vllm_model) return token information as
    extra fields on the chat completion message:
        - message.prompt_token_ids: list[int]
        - message.generation_token_ids: list[int | str]
        - message.generation_log_probs: list[float]

    Verifiers' parse_tokens() (called by from_native_response) expects:
        - response.prompt_token_ids (top-level attribute)
        - choice.token_ids (choice-level attribute)
        - choice.logprobs.content (list of {token, logprob, top_logprobs} dicts)

    This client relocates the fields after calling the standard OpenAI chat
    completions endpoint, so the rest of the verifiers pipeline works unchanged.
    """

    @handle_openai_overlong_prompt
    async def get_native_response(
        self,
        prompt: OpenAIChatMessages,
        model: str,
        sampling_args: SamplingArgs,
        tools: list[OpenAITool] | None = None,
        **kwargs,
    ) -> OpenAIChatResponse:
        # Standard OpenAI chat completions call to the NeMo Gym model server.
        # The model server handles logprobs, tokenization, and token ID extraction
        # internally. Extra fields survive because the OpenAI SDK's ChatCompletion
        # and ChatCompletionMessage models use extra="allow".
        response = await super().get_native_response(
            prompt, model, sampling_args, tools, **kwargs
        )

        choice = response.choices[0]
        message = choice.message

        # Extract token data from the message's extra fields.
        # These are set by the NeMo Gym model server when
        # return_token_id_information is enabled.
        prompt_token_ids = getattr(message, "prompt_token_ids", None)
        generation_token_ids = getattr(message, "generation_token_ids", None)
        generation_log_probs = getattr(message, "generation_log_probs", None)

        if prompt_token_ids is None and generation_token_ids is None:
            # No token data from the model server; pass through unchanged.
            # parse_tokens() will return None, which is correct for inference-only.
            return response

        # Normalize string token IDs to ints.
        # The vllm_model server may return strings after removeprefix("token_id:").
        if prompt_token_ids and isinstance(prompt_token_ids[0], str):
            prompt_token_ids = [int(tid) for tid in prompt_token_ids]
        if generation_token_ids and isinstance(generation_token_ids[0], str):
            generation_token_ids = [int(tid) for tid in generation_token_ids]

        # Relocate to where parse_tokens() expects them.
        if prompt_token_ids is not None:
            setattr(response, "prompt_token_ids", prompt_token_ids)
        if generation_token_ids is not None:
            setattr(choice, "token_ids", generation_token_ids)

        # Reconstruct logprobs structure from token IDs + log probs.
        # parse_tokens() handles both Pydantic ChoiceLogprobs objects and plain
        # dicts, so a dict is sufficient and avoids an import.
        if generation_token_ids and generation_log_probs:
            choice.logprobs = {
                "content": [
                    {"token": f"token_id:{tid}", "logprob": lp, "top_logprobs": []}
                    for tid, lp in zip(generation_token_ids, generation_log_probs)
                ]
            }

        return response
