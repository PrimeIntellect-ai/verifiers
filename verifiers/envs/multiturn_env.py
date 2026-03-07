import asyncio
import logging
from abc import abstractmethod
from typing import final

import verifiers as vf
from verifiers.clients import Client
from verifiers.types import (
    Messages,
    Response,
    RolloutInput,
    SamplingArgs,
    State,
    TrajectoryStep,
)
from verifiers.utils.message_utils import concat_messages, normalize_messages
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
    def __init__(self, max_turns: int = -1, **kwargs):
        super().__init__(**kwargs)
        self.max_turns = max_turns

        self.add_rubric(MultiTurnMonitorRubric())

    @abstractmethod
    async def env_response(
        self, messages: Messages, state: State, **kwargs
    ) -> Messages | str:
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

    @vf.stop
    async def has_final_env_response(self, state: State) -> bool:
        """Check if env_response signaled termination via final_env_response."""
        return state.get("final_env_response") is not None

    async def setup_state(self, state: State) -> State:
        """Override to add environment-specific state fields."""
        return state

    async def get_prompt_messages(self, state: State) -> Messages:
        """Override for rollouts with non-linear message sequences."""
        if len(state["trajectory"]) == 0:
            msgs = normalize_messages(state["prompt"], field_name="state.prompt")
            logger.info(
                "[MITO-DEBUG] get_prompt_messages turn=0 msg_count=%d",
                len(msgs),
            )
            return msgs
        prev_turn_prompt = normalize_messages(
            state["trajectory"][-1]["prompt"], field_name="trajectory.prompt"
        )
        prev_turn_completion = normalize_messages(
            state["trajectory"][-1]["completion"], field_name="trajectory.completion"
        )
        messages = concat_messages([prev_turn_prompt, prev_turn_completion])
        env_response = await self.env_response(messages, state)
        env_response_messages = normalize_messages(
            env_response, field_name="env_response"
        )
        result = concat_messages([messages, env_response_messages])
        logger.info(
            "[MITO-DEBUG] get_prompt_messages turn=%d msg_count=%d env_response_msgs=%d",
            len(state["trajectory"]),
            len(result),
            len(env_response_messages),
        )
        return result

    async def render_completion(self, state: State):
        """Override for rollouts with non-linear message sequences."""
        if len(state["trajectory"]) == 0:
            state["completion"] = []
            return
        last_prompt = normalize_messages(
            state["trajectory"][-1]["prompt"], field_name="trajectory.prompt"
        )
        last_completion = normalize_messages(
            state["trajectory"][-1]["completion"], field_name="trajectory.completion"
        )
        full_conversation = concat_messages([last_prompt, last_completion])
        if state.get("final_env_response"):
            full_conversation = concat_messages(
                [
                    full_conversation,
                    normalize_messages(
                        state["final_env_response"], field_name="final_env_response"
                    ),
                ]
            )
        prompt_messages = normalize_messages(state["prompt"], field_name="state.prompt")
        state["completion"] = full_conversation[len(prompt_messages) :]

    async def add_trajectory_step(self, state: State, trajectory_step: TrajectoryStep):
        """Override to set intermediate rewards, advantages, or extra metadata."""
        state["trajectory"].append(trajectory_step)

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

        # MITO-DEBUG: log response details
        content = response.message.content
        content_len = len(content) if isinstance(content, str) else 0
        has_reasoning = response.message.reasoning_content is not None
        reasoning_len = len(response.message.reasoning_content) if has_reasoning else 0
        prompt_ids_len = 0
        completion_ids_len = 0
        if tokens is not None:
            prompt_ids_len = len(tokens.get("prompt_ids", []))
            completion_ids_len = len(tokens.get("completion_ids", []))
        logger.info(
            "[MITO-DEBUG] add_model_response turn=%d content_len=%d "
            "has_reasoning=%s reasoning_len=%d "
            "prompt_ids_len=%d completion_ids_len=%d is_truncated=%s",
            len(state["trajectory"]),
            content_len,
            has_reasoning,
            reasoning_len,
            prompt_ids_len,
            completion_ids_len,
            is_truncated,
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

        # MITO-DEBUG: inline extension check against previous step
        if len(state["trajectory"]) >= 2:
            prev_tokens = state["trajectory"][-2].get("tokens")
            curr_tokens = state["trajectory"][-1].get("tokens")
            if prev_tokens is not None and curr_tokens is not None:
                prev_prompt = prev_tokens.get("prompt_ids", [])
                prev_completion = prev_tokens.get("completion_ids", [])
                curr_prompt = curr_tokens.get("prompt_ids", [])
                expected_prefix = list(prev_prompt) + list(prev_completion)
                prefix_len = len(expected_prefix)
                actual_prefix = list(curr_prompt[:prefix_len])
                extension_holds = actual_prefix == expected_prefix

                turn = len(state["trajectory"]) - 1
                prev_prompt_len = len(prev_prompt)
                prev_completion_len = len(prev_completion)
                curr_prompt_len = len(curr_prompt)

                if not extension_holds:
                    # Classify the break
                    if curr_prompt_len < prefix_len:
                        break_type = "TRUNCATION"
                        mismatch_pos = curr_prompt_len
                        section = "N/A"
                    else:
                        # Find first mismatch position
                        mismatch_pos = -1
                        for i in range(min(len(expected_prefix), len(actual_prefix))):
                            if expected_prefix[i] != actual_prefix[i]:
                                mismatch_pos = i
                                break
                        if mismatch_pos == -1:
                            mismatch_pos = min(len(expected_prefix), len(actual_prefix))

                        if mismatch_pos < prev_prompt_len:
                            break_type = "PROMPT_MUTATION"
                            section = f"prompt[{mismatch_pos}/{prev_prompt_len}]"
                        else:
                            break_type = "COMPLETION_MUTATION"
                            comp_offset = mismatch_pos - prev_prompt_len
                            section = f"completion[{comp_offset}/{prev_completion_len}]"

                    W = 10
                    logger.warning(
                        "[MITO-DEBUG] EXTENSION BREAK at turn=%d type=%s "
                        "prev_prompt_len=%d prev_completion_len=%d "
                        "expected_prefix_len=%d curr_prompt_len=%d "
                        "mismatch_pos=%d section=%s "
                        "expected[pos-%d:pos+%d]=%s actual[pos-%d:pos+%d]=%s",
                        turn,
                        break_type,
                        prev_prompt_len,
                        prev_completion_len,
                        prefix_len,
                        curr_prompt_len,
                        mismatch_pos,
                        section,
                        W,
                        W,
                        expected_prefix[max(0, mismatch_pos - W) : mismatch_pos + W]
                        if mismatch_pos >= 0
                        else "N/A",
                        W,
                        W,
                        actual_prefix[max(0, mismatch_pos - W) : mismatch_pos + W]
                        if mismatch_pos >= 0
                        else "N/A",
                    )

                    # For first extension check (turn=1), log boundary regions to spot
                    # prompt/completion junction issues
                    if turn == 1:
                        logger.warning(
                            "[MITO-DEBUG] turn=1 detail: "
                            "prev_prompt_tail=%s prev_completion_head=%s "
                            "prev_completion_tail=%s "
                            "curr_prompt_at_boundary=%s",
                            prev_prompt[-W:] if prev_prompt else "[]",
                            list(prev_completion[:W]) if prev_completion else "[]",
                            list(prev_completion[-W:]) if prev_completion else "[]",
                            list(curr_prompt[prev_prompt_len - W : prev_prompt_len + W])
                            if curr_prompt_len > prev_prompt_len
                            else "[]",
                        )
                else:
                    logger.info(
                        "[MITO-DEBUG] extension OK at turn=%d "
                        "prev_prompt_len=%d prev_completion_len=%d curr_prompt_len=%d",
                        turn,
                        prev_prompt_len,
                        prev_completion_len,
                        curr_prompt_len,
                    )

    @final
    async def rollout(
        self,
        input: RolloutInput,
        client: Client,
        model: str,
        sampling_args: SamplingArgs | None = None,
    ) -> State:
        state = await self.init_state(input, client, model, sampling_args)
        try:
            try:
                state = await self.setup_state(state)
            except vf.Error as e:
                state["error"] = e
            # checks all @vf.stop methods, runs all @vf.cleanup methods if any are True
            turn_idx = 0
            while not await self.is_completed(state):
                try:
                    logger.info(
                        "[MITO-DEBUG] rollout turn=%d starting, trajectory_len=%d",
                        turn_idx,
                        len(state["trajectory"]),
                    )
                    prompt_messages = await self.get_prompt_messages(state)
                    if state.get("final_env_response") is not None:
                        continue
                    response = await self.get_model_response(state, prompt_messages)
                    await self.add_model_response(state, prompt_messages, response)
                    turn_idx += 1
                except vf.Error as e:
                    if isinstance(e, vf.OverlongPromptError):
                        state["prompt_too_long"] = True
                        state["is_truncated"] = True
                    else:
                        state["error"] = e
            logger.info(
                "[MITO-DEBUG] rollout complete, total_turns=%d trajectory_len=%d",
                turn_idx,
                len(state["trajectory"]),
            )
            await self.render_completion(state)
            return state
        except asyncio.CancelledError:
            await self._cleanup(state)
            raise
