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

logger = logging.getLogger("verifiers.envs.multiturn_env")


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

    async def rollout(
        self,
        client: AsyncOpenAI,
        model: str,
        prompt: Messages,
        answer: str = "",
        task: str = "default",
        info: Info | None = None,
        sampling_args: SamplingArgs | None = None,
        **kwargs,
    ) -> tuple[Messages, State]:
        """
        Generate a multi-turn rollout with the environment (messages, state).
        """
        info = info or {}
        is_completed = False
        state = {
            "id": 0,  # TODO: add id
            "prompt": prompt,
            "completion": [],
            "answer": answer,
            "task": task,
            "info": info,
            "responses": [],
            "turn": 0,
            "timing": {
                "generation_ms": 0.0,
                "scoring_ms": 0.0,
                "total_ms": 0.0,
            },
        }
        start_time = time.time()
        state = await maybe_await(self.setup_state, state, **kwargs)
        if self.message_type == "chat":
            assert isinstance(prompt, list)
            completion = []
        else:
            assert isinstance(prompt, str)
            completion = ""
            state["responses_start_idx"] = []
        rollout = list(prompt) if not isinstance(prompt, str) else prompt
        while not is_completed:
            if await maybe_await(self.is_completed, rollout, state, **kwargs):
                is_completed = True
                break
            response = await self.get_model_response(
                client,
                model,
                rollout,
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
            if self.message_type == "chat":
                assert isinstance(rollout, list)
                assert isinstance(completion, list)
                assert isinstance(response, ChatCompletion)
                response_text: str = ""
                if response.choices and response.choices[0].message:
                    response_text = response.choices[0].message.content or ""
                response_message: ChatMessage = {
                    "role": "assistant",
                    "content": response_text,
                }
                if response.choices and response.choices[0].message and response.choices[0].message.tool_calls:
                    response_message["tool_calls"] = response.choices[0].message.tool_calls #type:ignore
                rollout.append(response_message)
                completion.append(response_message)
            else:
                assert isinstance(rollout, str)
                assert isinstance(completion, str)
                assert isinstance(response, Completion)
                state["responses_start_idx"].append(len(completion))
                response_text: str = ""
                if response.choices and response.choices[0]:
                    response_text = response.choices[0].text or ""
                rollout += response_text
                completion += response_text
            state["turn"] += 1
            if await maybe_await(self.is_completed, rollout, state, **kwargs):
                is_completed = True
                end_time = time.time()
                state["timing"]["generation_ms"] = (end_time - start_time) * 1000
                state["timing"]["total_ms"] = (end_time - start_time) * 1000
            else:
                env_msgs, state = await maybe_await(
                    self.env_response, rollout, state, **kwargs
                )
                if self.message_type == "chat":
                    assert isinstance(env_msgs, list)
                    assert isinstance(rollout, list)
                    assert isinstance(completion, list)
                    rollout += env_msgs
                    completion += env_msgs
                else:
                    assert isinstance(env_msgs, str)
                    assert isinstance(rollout, str)
                    assert isinstance(completion, str)
                    rollout += env_msgs
                    completion += env_msgs
        return completion, state
