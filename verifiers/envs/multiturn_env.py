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
    def __init__(
        self,
        max_turns: int = -1,
        exclude_think: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.max_turns = max_turns
        self.exclude_think = exclude_think

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

    @staticmethod
    def _process_assistant_message(msg: ChatMessage) -> ChatMessage:
        import re
        from copy import deepcopy

        def _strip_prefix_up_to_close(text: str) -> str:
            return re.sub(r"(?s)^.*</think>", "", text).lstrip()

        new_msg: ChatMessage = deepcopy(msg)
        new_msg["role"] = msg.get("role", "assistant")

        content = msg.get("content")
        if content is None:
            new_msg["content"] = ""
            return new_msg

        if "</think>" in content:
            new_msg["content"] = _strip_prefix_up_to_close(content)
        else:
            new_msg["content"] = content

        return new_msg

    async def get_context_messages(self, state: State) -> Messages:
        if not self.exclude_think:
            return state["prompt"] + state["completion"]

        prompt_msgs = state["prompt"]
        completion_msgs = state["completion"]

        processed_completion: list[ChatMessage] = []
        for m in completion_msgs:
            role = m.get("role")
            if role == "assistant":
                processed_completion.append(self._process_assistant_message(m))
            else:
                processed_completion.append(m)

        return prompt_msgs + processed_completion

    async def rollout(
        self,
        client: AsyncOpenAI,
        model: str,
        prompt: Messages,
        completion: Messages | None = None,
        answer: str = "",
        state: State | None = None,
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
        info = info or {}
        is_completed = False
        state = state or await self.init_state(
            prompt, completion, answer, task, info, example_id
        )
        track_step_scores: bool = bool(kwargs.get("track_step_scores", False))
        if track_step_scores and "step_scores" not in state:
            state["step_scores"] = []
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
            # 1. Build current context and check early termination
            context_messages = await self.get_context_messages(state)
            if await maybe_await(self.is_completed, context_messages, state, **kwargs):
                is_completed = True
                break

            # 2. Model response for this turn
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

            # 2a. Append assistant message to completion
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
                    response_message["tool_calls"] = response.choices[0].message.tool_calls  # type: ignore
                state["completion"].append(response_message)
            else:
                assert isinstance(response, Completion)
                # track where this assistant response starts in the running text
                state["responses_start_idx"].append(len(state["completion"]))
                if response.choices and response.choices[0]:
                    response_text = response.choices[0].text or ""
                state["completion"] += response_text

            # 3) Environment feedback for THIS turn
            #    Use latest context that includes the assistant message
            context_messages = await self.get_context_messages(state)
            env_msgs, state = await maybe_await(
                self.env_response, context_messages, state, **kwargs
            )
            if self.message_type == "chat":
                assert isinstance(env_msgs, list)
                state["completion"] += env_msgs
            else:
                assert isinstance(env_msgs, str)
                state["completion"] += env_msgs

            # 4) Now compute per-turn score after env feedback is appended
            if track_step_scores:
                try:
                    rs = await self.rubric.score_rollout(
                        prompt=state["prompt"],
                        completion=state["completion"],
                        answer=state.get("answer", ""),
                        state=state,
                        task=state.get("task", "default"),
                        info=state.get("info", {}),
                        example_id=state.get("example_id", example_id),
                        **kwargs,
                    )
                    state.setdefault("step_scores", []).append(float(rs.reward))
                except Exception as e:
                    logger.error(f"Error computing step score: {e}")
                    # state.setdefault("step_scores", []).append(0.0)
                    raise RuntimeError(f"Step score computation failed: {e}")

            # 5) Prepare for next turn
            state["turn"] += 1
            context_messages = await self.get_context_messages(state)
            if await maybe_await(self.is_completed, context_messages, state, **kwargs):
                is_completed = True
                end_time = time.time()
                state["timing"]["generation_ms"] = (end_time - start_time) * 1000
                state["timing"]["total_ms"] = (end_time - start_time) * 1000
                break
        return state["completion"], state
