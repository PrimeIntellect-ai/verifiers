import re
from abc import abstractmethod
from typing import cast

import verifiers as vf
from verifiers.types import ChatMessage, Messages, State, TrajectoryStep
from verifiers.utils.message_utils import concat_messages


def other_player_id(player_one_id: str, player_two_id: str, current_id: str) -> str:
    return player_two_id if current_id == player_one_id else player_one_id


class MultiAgentEnv(vf.MultiTurnEnv):
    """
    Multi-turn environment that supports interleaved trajectories per actor id.
    trajectory_id is the actor id for the current model call.
    """

    async def setup_state(self, state: State) -> State:
        state = await super().setup_state(state)
        state["trajectory_id"] = self.get_initial_actor_id(state)
        return state

    @abstractmethod
    def get_initial_actor_id(self, state: State) -> str:
        pass

    @abstractmethod
    def flip_trajectory_id(self, state: State) -> None:
        pass

    def last_step_for_trajectory_id(
        self, state: State, trajectory_id: str
    ) -> TrajectoryStep | None:
        for step in reversed(state["trajectory"]):
            if step["trajectory_id"] == trajectory_id:
                return step
        return None

    def messages_for_trajectory_id(self, state: State, trajectory_id: str) -> Messages:
        step = self.last_step_for_trajectory_id(state, trajectory_id)
        if step is None:
            return state["prompt"]
        return concat_messages([step["prompt"], step["completion"]])

    async def get_prompt_messages(self, state: State) -> Messages:
        if len(state["trajectory"]) == 0:
            return state["prompt"]
        self.flip_trajectory_id(state)
        messages = self.messages_for_trajectory_id(state, state["trajectory_id"])
        env_response = await self.env_response(messages, state)
        return concat_messages([messages, env_response])


class AlternatingTwoAgentEnv(MultiAgentEnv):
    """Simple pattern: alternate the active actor every environment response."""

    def __init__(
        self,
        player_one_id: str = "player_a",
        player_two_id: str = "player_b",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.player_one_id = player_one_id
        self.player_two_id = player_two_id

    async def setup_state(self, state: State) -> State:
        state = await super().setup_state(state)
        state["turn_count"] = state.get("turn_count", 0)
        return state

    def get_initial_actor_id(self, state: State) -> str:
        return self.player_one_id

    def flip_trajectory_id(self, state: State) -> None:
        current_id = state["trajectory_id"]
        state["trajectory_id"] = other_player_id(
            self.player_one_id, self.player_two_id, current_id
        )

    async def env_response(
        self, messages: Messages, state: State, **kwargs
    ) -> Messages:
        state["turn_count"] += 1
        actor_id = state["trajectory_id"]
        return [
            cast(
                ChatMessage,
                {
                    "role": "user",
                    "content": f"Turn {state['turn_count']}: {actor_id}, respond.",
                },
            )
        ]


class TwoPlayerZeroSumGameEnv(MultiAgentEnv):
    """
    Example zero-sum game pattern.

    Game: players alternate taking 1 or 2 tokens. The player taking the last token
    wins (+1) and the opponent loses (-1). All game state transitions happen in
    env_response, which also flips state["trajectory_id"] between players.
    """

    def __init__(
        self,
        starting_tokens: int = 7,
        player_one_id: str = "player_a",
        player_two_id: str = "player_b",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.starting_tokens = starting_tokens
        self.player_one_id = player_one_id
        self.player_two_id = player_two_id

    async def setup_state(self, state: State) -> State:
        state = await super().setup_state(state)
        state["tokens_remaining"] = state.get("tokens_remaining", self.starting_tokens)
        state["scores"] = state.get(
            "scores",
            {self.player_one_id: 0.0, self.player_two_id: 0.0},
        )
        return state

    def get_initial_actor_id(self, state: State) -> str:
        return self.player_one_id

    def flip_trajectory_id(self, state: State) -> None:
        current_id = state["trajectory_id"]
        state["trajectory_id"] = other_player_id(
            self.player_one_id, self.player_two_id, current_id
        )

    @staticmethod
    def completion_text(completion: Messages) -> str:
        if isinstance(completion, str):
            return completion
        return str(completion[-1]["content"])

    def extract_take_action(self, step: TrajectoryStep) -> int | None:
        text = self.completion_text(step["completion"])
        match = re.search(r"\b([12])\b", text)
        if match is None:
            return None
        return int(match.group(1))

    def user_message(self, content: str) -> ChatMessage:
        return cast(ChatMessage, {"role": "user", "content": content})

    def finalize_game(
        self,
        state: State,
        winner_id: str,
        loser_id: str,
        reason: str,
    ) -> Messages:
        scores = state["scores"]
        scores[winner_id] += 1.0
        scores[loser_id] -= 1.0
        summary = (
            f"Game over. {reason} Winner: {winner_id}. "
            f"Final score: {winner_id}={scores[winner_id]}, {loser_id}={scores[loser_id]}."
        )
        final_message = self.user_message(summary)
        state["final_env_response"] = [final_message]
        return state["final_env_response"]

    async def env_response(
        self, messages: Messages, state: State, **kwargs
    ) -> Messages:
        actor_id = state["trajectory_id"]
        previous_actor_id = other_player_id(
            self.player_one_id, self.player_two_id, actor_id
        )
        previous_actor_step = self.last_step_for_trajectory_id(state, previous_actor_id)
        if previous_actor_step is None:
            return [
                self.user_message(
                    f"{actor_id}, take 1 or 2 tokens. "
                    f"Tokens remaining: {state['tokens_remaining']}."
                )
            ]

        action = self.extract_take_action(previous_actor_step)
        if action is None:
            return self.finalize_game(
                state=state,
                winner_id=actor_id,
                loser_id=previous_actor_id,
                reason=f"{previous_actor_id} made an invalid move.",
            )

        tokens_remaining = state["tokens_remaining"]
        if action > tokens_remaining:
            return self.finalize_game(
                state=state,
                winner_id=actor_id,
                loser_id=previous_actor_id,
                reason=(
                    f"{previous_actor_id} tried to take {action} token(s) with only "
                    f"{tokens_remaining} remaining."
                ),
            )

        tokens_remaining -= action
        state["tokens_remaining"] = tokens_remaining
        if tokens_remaining == 0:
            return self.finalize_game(
                state=state,
                winner_id=previous_actor_id,
                loser_id=actor_id,
                reason=f"{previous_actor_id} took the last token.",
            )

        return [
            self.user_message(
                f"{previous_actor_id} took {action} token(s). "
                f"Tokens remaining: {tokens_remaining}. "
                f"{actor_id}, take 1 or 2 tokens."
            )
        ]
