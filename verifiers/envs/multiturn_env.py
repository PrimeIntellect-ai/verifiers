import asyncio
import logging
import time
from abc import abstractmethod

import verifiers as vf
from verifiers.clients import Client
from verifiers.types import (
    Messages,
    Response,
    RolloutInput,
    SamplingArgs,
    State,
    TimeSpan,
    TrajectoryStep,
)
from verifiers.utils.message_utils import (
    concat_messages,
    maybe_normalize_messages,
)
from verifiers.utils.response_utils import (
    parse_response_message,
    parse_response_tokens,
)

logger = logging.getLogger(__name__)


class MultiTurnMonitorRubric(vf.Rubric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_metric(self.num_turns)

    async def num_turns(self, state: State) -> int:
        return len(state["trajectory"])


class MultiTurnEnv(vf.Environment):
    def __init__(
        self,
        max_turns: int = -1,
        timeout_seconds: float | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.max_turns = max_turns
        self.timeout_seconds = timeout_seconds
        self.max_total_completion_tokens: int = -1

        self.add_rubric(MultiTurnMonitorRubric())

    def set_max_total_completion_tokens(self, max_total_completion_tokens: int) -> None:
        """Set the maximum total completion tokens for this environment."""
        self.max_total_completion_tokens = max_total_completion_tokens

    @abstractmethod
    async def env_response(
        self, messages: Messages, state: State, **kwargs
    ) -> Messages:
        """
        Generate a response from the environment.
        """
        pass

    @vf.stop(priority=100)  # always check for errors first
    async def has_error(self, state: State, **kwargs) -> bool:
        return state.get("error") is not None

    @vf.stop
    async def prompt_too_long(self, state: State) -> bool:
        return state.get("prompt_too_long", False)

    @vf.stop
    async def max_turns_reached(self, state: State) -> bool:
        return len(state["trajectory"]) >= self.max_turns and self.max_turns > 0

    def mark_timed_out(self, state: State) -> None:
        state["timed_out"] = True
        state["is_completed"] = True
        state["stop_condition"] = "timeout_reached"

    @vf.stop
    async def max_total_completion_tokens_reached(self, state: State) -> bool:
        if self.max_total_completion_tokens <= 0:
            return False
        usage = self.get_state_usage(state)
        if usage is None:
            return False
        return usage["output_tokens"] >= self.max_total_completion_tokens

    @vf.stop
    async def has_final_env_response(self, state: State) -> bool:
        """Check if env_response signaled termination via final_env_response."""
        return state.get("final_env_response") is not None

    async def setup_state(self, state: State) -> State | None:
        """Override to add environment-specific state fields. Mutate state in place."""
        return state

    async def get_prompt_messages(self, state: State) -> Messages:
        """Build prompt messages for the current turn."""
        if len(state["trajectory"]) == 0:
            return state["prompt"]
        prev_turn_prompt = state["trajectory"][-1]["prompt"]
        prev_turn_completion = state["trajectory"][-1]["completion"]
        messages = concat_messages([prev_turn_prompt, prev_turn_completion])
        env_response = await self.env_response(messages, state)
        env_response = maybe_normalize_messages(env_response, field_name="env_response")
        return concat_messages([messages, env_response])

    async def render_completion(self, state: State):
        """Override for rollouts with non-linear message sequences."""
        if len(state["trajectory"]) == 0:
            state["completion"] = []
            return
        last_prompt = state["trajectory"][-1]["prompt"]
        last_completion = state["trajectory"][-1]["completion"]
        full_conversation = concat_messages([last_prompt, last_completion])
        if state.get("final_env_response"):
            final_resp = state["final_env_response"]
            final_resp = maybe_normalize_messages(
                final_resp, field_name="final_env_response"
            )
            full_conversation = concat_messages([full_conversation, final_resp])
        prompt_messages = state["prompt"]
        state["completion"] = full_conversation[len(prompt_messages) :]

    @vf.cleanup(priority=100)
    async def render_state(self, state: State) -> None:
        """Render core rollout fields before user cleanup handlers run."""
        state["timing"].generation.end = time.time()
        await self.render_completion(state)

    async def add_trajectory_step(self, state: State, trajectory_step: TrajectoryStep):
        """Override to set intermediate rewards, advantages, or extra metadata."""
        state["trajectory"].append(trajectory_step)

    async def _finalize_rollout(self, state: State) -> None:
        """Finalize rollout state and run cleanup handlers exactly once."""
        await self.cleanup(state)

    async def add_model_response(
        self,
        state: State,
        prompt_messages: Messages,
        response: Response,
    ):
        completion_messages = await parse_response_message(response)
        tokens = await parse_response_tokens(response, self.max_seq_len)
        response_is_truncated = response.message.is_truncated or False
        is_truncated = response_is_truncated or (
            tokens is not None and bool(tokens.get("is_truncated"))
        )
        trajectory_step = TrajectoryStep(
            prompt=prompt_messages,
            completion=completion_messages,
            response=response,
            tokens=tokens,
            reward=None,
            advantage=None,
            is_truncated=is_truncated,
            trajectory_id=state["trajectory_id"],
            extras={},
        )
        await self.add_trajectory_step(state, trajectory_step)

    async def rollout(
        self,
        input: RolloutInput,
        client: Client,
        model: str,
        sampling_args: SamplingArgs | None = None,
    ) -> State:
        state = await self.init_state(input, client, model, sampling_args)

        async def rollout_loop() -> None:
            nonlocal state
            state["timing"].generation.start = time.time()
            state["timing"].setup.start = time.time()
            try:
                setup_state = await self.setup_state(state)
                if setup_state is not None:
                    state = setup_state
            except vf.Error as e:
                state["error"] = e
            finally:
                state["timing"].setup.end = time.time()
            while not await self.is_completed(state):
                try:
                    timing = state["timing"]
                    start_time = time.time()
                    prompt_messages = await self.get_prompt_messages(state)
                    end_time = time.time()
                    # First iteration has no preceding env_response; skip recording.
                    if state["trajectory"]:
                        timing.env.spans.append(
                            TimeSpan(start=start_time, end=end_time)
                        )

                    prompt_messages = maybe_normalize_messages(
                        prompt_messages, field_name="prompt_messages"
                    )
                    if state.get("final_env_response") is not None:
                        continue

                    start_time = time.time()
                    response = await self.get_model_response(state, prompt_messages)
                    end_time = time.time()
                    timing.model.spans.append(TimeSpan(start=start_time, end=end_time))
                    await self.add_model_response(state, prompt_messages, response)
                except vf.Error as e:
                    if isinstance(e, vf.OverlongPromptError):
                        state["prompt_too_long"] = True
                        state["is_truncated"] = True
                    else:
                        state["error"] = e

        try:
            await asyncio.wait_for(rollout_loop(), timeout=self.timeout_seconds)
        except asyncio.TimeoutError:
            self.mark_timed_out(state)
        finally:
            await self._finalize_rollout(state)
        return state
