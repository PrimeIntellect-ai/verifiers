import logging
import time
from abc import abstractmethod

from openai import AsyncOpenAI

from verifiers.envs.environment import Environment
from verifiers.types import (
    ChatCompletion,
    ChatMessage,
    Completion,
    Info,
    Messages,
    SamplingArgs,
    State,
)
from verifiers.utils.async_utils import maybe_await

logger = logging.getLogger(__name__)


class MultiTurnEnv(Environment):
    def __init__(self, max_turns: int = -1, **kwargs):
        super().__init__(**kwargs)
        self.max_turns = max_turns

    async def prompt_too_long(self, state: State) -> bool:
        return state.get("prompt_too_long", False)

    async def max_turns_reached(self, state: State) -> bool:
        """Check if the maximum number of turns has been reached."""
        return state["turn"] >= self.max_turns and self.max_turns > 0

    async def setup_state(self, state: State, **kwargs) -> State:
        return state

    async def is_completed(self, messages: Messages, state: State, **kwargs) -> bool:
        """When overriding, call self.max_turns_reached(state) to check if turn limit reached."""
        max_turns_reached = await self.max_turns_reached(state)
        prompt_too_long = await self.prompt_too_long(state)
        return max_turns_reached or prompt_too_long

    @abstractmethod
    async def env_response(
        self, messages: Messages, state: State, **kwargs
    ) -> tuple[Messages, State]:
        """
        Generate a response from the environment (messages, state).
        """
        pass

    async def get_context_messages(self, state: State) -> Messages:
        return state["prompt"] + state["completion"]

    async def rollout(
        self,
        client: AsyncOpenAI,
        model: str,
        prompt: Messages,
        completion: Messages | None = None,
        answer: str = "",
        state: State = {},
        task: str = "default",
        info: Info | None = None,
        example_id: int = 0,
        sampling_args: SamplingArgs | None = None,
        **kwargs,
    ) -> tuple[Messages, State]:
        """
        Generate a multi-turn rollout with the environment (messages, state).
        """
        completion = completion or await self.init_completion()
        info = info if info is not None else {}
        is_completed = False
        state = state or await self.init_state(
            prompt, completion, answer, task, info, example_id
        )
        start_time = time.time()
        state = await maybe_await(self.setup_state, state, **kwargs)
        if self.message_type == "chat":
            assert isinstance(state["prompt"], list)
            assert isinstance(state["completion"], list)
        else:
            assert isinstance(state["prompt"], str)
            assert isinstance(state["completion"], str)
            state["responses_start_idx"] = []
        while not is_completed:
            context_messages = await self.get_context_messages(state)
            if await maybe_await(self.is_completed, context_messages, state, **kwargs):
                is_completed = True
                break
            response = await self.get_model_response(
                client,
                model,
                context_messages,
                oai_tools=info.get("oai_tools", None),
                sampling_args=sampling_args,
                message_type=self.message_type,
                initial_prompt=len(state["responses"]) == 0,
                **kwargs,
            )
            if response is not None and response.id == "overlong-prompt":
                state["prompt_too_long"] = True
                break
            state["responses"].append(response)
            response_text: str = ""
            if self.message_type == "chat":
                assert isinstance(context_messages, list)
                assert isinstance(response, ChatCompletion)
                if response.choices and response.choices[0].message:
                    response_text = response.choices[0].message.content or ""
                response_message: ChatMessage = {
                    "role": "assistant",
                    "content": response_text,
                }
                if (
                    response.choices
                    and response.choices[0].message
                    and response.choices[0].message.tool_calls
                ):
                    tool_calls = response.choices[0].message.tool_calls
                    response_message["tool_calls"] = [  # type: ignore
                        tool_call.model_dump() for tool_call in tool_calls
                    ]
                state["completion"].append(response_message)
            else:
                assert isinstance(response, Completion)
                state["responses_start_idx"].append(len(completion))
                if response.choices and response.choices[0]:
                    response_text = response.choices[0].text or ""
                state["completion"] += response_text
            context_messages = await self.get_context_messages(state)
            state["turn"] += 1
            if await maybe_await(self.is_completed, context_messages, state, **kwargs):
                is_completed = True
                end_time = time.time()
                state["timing"]["generation_ms"] = (end_time - start_time) * 1000
                state["timing"]["total_ms"] = (end_time - start_time) * 1000
            else:
                env_msgs, state = await maybe_await(
                    self.env_response, context_messages, state, **kwargs
                )
                if self.message_type == "chat":
                    assert isinstance(env_msgs, list)
                    state["completion"] += env_msgs
                else:
                    assert isinstance(env_msgs, str)
                    state["completion"] += env_msgs
        return state["completion"], state
