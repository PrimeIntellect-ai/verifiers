from __future__ import annotations

from typing import Any, cast

from verifiers.clients.openai_chat_completions_client import (
    OpenAIChatCompletionsClient,
    OpenAIChatMessages,
    OpenAIChatResponse,
    OpenAITool,
    handle_openai_overlong_prompt,
)
from verifiers.types import (
    AssistantMessage,
    Messages,
    Response,
    SamplingArgs,
    Tool,
)


class NeMoRLChatCompletionsClient(OpenAIChatCompletionsClient):
    """
    Client for NeMo Gym vllm_model server.
    Formats requests properly for NeMo RL's retokenization fix.
    Implements token ids request for retokenization fix in NeMo RL and formats properly for verifiers.
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
        """Move NeMo Gym's message-level token id fields to where verifiers' `parse_tokens()` expects them."""
        response = await super().get_native_response(
            prompt, model, sampling_args, tools, **kwargs
        )
        choice = response.choices[0]
        message = choice.message

        prompt_token_ids = getattr(message, "prompt_token_ids", None)
        generation_token_ids = getattr(message, "generation_token_ids", None)
        generation_log_probs = getattr(message, "generation_log_probs", None)

        if prompt_token_ids is None and generation_token_ids is None:
            return response

        if prompt_token_ids is not None:
            response.prompt_token_ids = prompt_token_ids
        if generation_token_ids is not None:
            choice.token_ids = generation_token_ids
        if generation_token_ids and generation_log_probs:
            choice.logprobs = {
                "content": [
                    {"token": f"token_id:{tid}", "logprob": lp, "top_logprobs": []}
                    for tid, lp in zip(generation_token_ids, generation_log_probs)
                ]
            }

        return response

    async def get_response(
        self,
        prompt: Messages,
        model: str,
        sampling_args: SamplingArgs,
        tools: list[Tool] | None = None,
        **kwargs,
    ) -> Response:
        """Annotate prior assistant messages with their trajectory tokens before delegating to the parent client."""
        state = kwargs.get("state")
        if state is not None:
            _attach_trajectory_tokens_to_prompt(prompt, state)
        return await super().get_response(prompt, model, sampling_args, tools, **kwargs)

    async def to_native_prompt(
        self, messages: Messages
    ) -> tuple[OpenAIChatMessages, dict[str, Any]]:
        """Serialize per-assistant token extras onto the outgoing native chat dicts so the server receives them."""
        native_messages, extras = await super().to_native_prompt(messages)
        for msg, native_msg in zip(messages, native_messages):
            if not isinstance(msg, AssistantMessage):
                continue
            native_dict = cast("dict[str, Any]", native_msg)
            for key in (
                "prompt_token_ids",
                "generation_token_ids",
                "generation_log_probs",
            ):
                val = getattr(msg, key, None)
                if val is not None:
                    native_dict[key] = list(val)
        return native_messages, extras


def _attach_trajectory_tokens_to_prompt(
    prompt: Messages, state: dict[str, Any]
) -> None:
    """Attach each past assistant message's token ids from the trajectory."""
    trajectory = state.get("trajectory") or []
    if not trajectory:
        return
    indices = [i for i, m in enumerate(prompt) if isinstance(m, AssistantMessage)]
    step_tokens = [step.get("tokens") for step in trajectory]
    n = min(len(indices), len(step_tokens))
    for i, tokens in zip(indices[-n:], step_tokens[-n:]):
        if tokens is None:
            continue
        msg = prompt[i]
        if (prompt_ids := tokens.get("prompt_ids")) is not None:
            msg.prompt_token_ids = list(prompt_ids)
        if (completion_ids := tokens.get("completion_ids")) is not None:
            msg.generation_token_ids = list(completion_ids)
        if (completion_logprobs := tokens.get("completion_logprobs")) is not None:
            msg.generation_log_probs = list(completion_logprobs)
