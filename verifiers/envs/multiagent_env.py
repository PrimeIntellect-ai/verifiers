import re
from abc import abstractmethod
from typing import cast

import verifiers as vf
from verifiers.types import ChatMessage, Messages, State, TrajectoryStep
from verifiers.utils.message_utils import concat_messages


def other_player_id(player_one_id: str, player_two_id: str, current_id: str) -> str:
    if current_id == player_one_id:
        return player_two_id
    if current_id == player_two_id:
        return player_one_id
    raise ValueError(f"Unknown player id: {current_id}")


class MultiAgentEnv(vf.MultiTurnEnv):
    """
    Multi-turn environment that supports interleaved trajectories per actor.

    The active actor is tracked in state["actor_id"]. For rollout compatibility, this
    is mirrored to state["trajectory_id"] before model calls. Prompt construction
    selects the latest trajectory step with the active actor id, then appends
    env_response(state).
    """

    async def setup_state(self, state: State) -> State:
        state = await super().setup_state(state)
        actor_id = state.get("actor_id", state["trajectory_id"])
        self.set_actor_id(state, actor_id)
        return state

    def get_actor_id(self, state: State) -> str:
        actor_id = state.get("actor_id")
        trajectory_id = state.get("trajectory_id")
        if actor_id is None:
            actor_id = trajectory_id
        elif trajectory_id is not None and trajectory_id != actor_id:
            # Backward compatibility for callers that still mutate trajectory_id.
            actor_id = trajectory_id
        if actor_id is None:
            raise ValueError("actor_id is missing from state")
        self.set_actor_id(state, actor_id)
        return actor_id

    def set_actor_id(self, state: State, actor_id: str) -> None:
        state["actor_id"] = actor_id
        state["trajectory_id"] = actor_id

    def last_step_for_actor_id(
        self, state: State, actor_id: str
    ) -> TrajectoryStep | None:
        for step in reversed(state.get("trajectory", [])):
            if step.get("trajectory_id") == actor_id:
                return step
        return None

    def messages_for_actor_id(self, state: State, actor_id: str) -> Messages:
        step = self.last_step_for_actor_id(state, actor_id)
        if step is None:
            return state["prompt"]
        return concat_messages([step["prompt"], step["completion"]])

    @abstractmethod
    async def env_response(self, state: State, **kwargs) -> Messages:
        """
        Generate the next environment message(s) for the active actor.

        Implementations can update state["actor_id"] via set_actor_id(...) to switch
        active actors.
        """
        pass

    async def get_prompt_messages(self, state: State) -> Messages:
        if len(state["trajectory"]) == 0:
            self.get_actor_id(state)
            return state["prompt"]

        current_actor_id = self.get_actor_id(state)
        messages = self.messages_for_actor_id(state, current_actor_id)
        env_response = await self.env_response(state)

        # env_response can switch actors for the next model call.
        updated_actor_id = self.get_actor_id(state)
        if updated_actor_id != current_actor_id:
            messages = self.messages_for_actor_id(state, updated_actor_id)

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
        self.set_actor_id(state, self.player_one_id)
        return state

    async def env_response(self, state: State, **kwargs) -> Messages:
        current_id = self.get_actor_id(state)
        next_id = other_player_id(self.player_one_id, self.player_two_id, current_id)
        state["turn_count"] += 1
        self.set_actor_id(state, next_id)
        return [
            cast(
                ChatMessage,
                {
                    "role": "user",
                    "content": f"Turn {state['turn_count']}: {next_id}, respond.",
                },
            )
        ]


class TwoPlayerZeroSumGameEnv(MultiAgentEnv):
    """
    Example zero-sum game pattern.

    Game: players alternate taking 1 or 2 tokens. The player taking the last token
    wins (+1) and the opponent loses (-1). All game state transitions happen in
    env_response, which also flips state["actor_id"] between players.
    """

    def __init__(
        self,
        starting_tokens: int = 7,
        player_one_id: str = "player_a",
        player_two_id: str = "player_b",
        **kwargs,
    ):
        if starting_tokens < 1:
            raise ValueError("starting_tokens must be >= 1")
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
        self.set_actor_id(state, self.player_one_id)
        return state

    @staticmethod
    def completion_text(completion: Messages) -> str:
        if isinstance(completion, str):
            return completion
        if len(completion) == 0:
            return ""
        content = completion[-1].get("content", "")
        if isinstance(content, str):
            return content
        return str(content)

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

    async def env_response(self, state: State, **kwargs) -> Messages:
        actor_id = self.get_actor_id(state)
        actor_step = self.last_step_for_actor_id(state, actor_id)
        if actor_step is None:
            return [
                self.user_message(
                    f"{actor_id}, take 1 or 2 tokens. "
                    f"Tokens remaining: {state['tokens_remaining']}."
                )
            ]

        opponent_id = other_player_id(self.player_one_id, self.player_two_id, actor_id)
        action = self.extract_take_action(actor_step)
        if action is None:
            return self.finalize_game(
                state=state,
                winner_id=opponent_id,
                loser_id=actor_id,
                reason=f"{actor_id} made an invalid move.",
            )

        tokens_remaining = state["tokens_remaining"]
        if action > tokens_remaining:
            return self.finalize_game(
                state=state,
                winner_id=opponent_id,
                loser_id=actor_id,
                reason=(
                    f"{actor_id} tried to take {action} token(s) with only "
                    f"{tokens_remaining} remaining."
                ),
            )

        tokens_remaining -= action
        state["tokens_remaining"] = tokens_remaining
        if tokens_remaining == 0:
            return self.finalize_game(
                state=state,
                winner_id=actor_id,
                loser_id=opponent_id,
                reason=f"{actor_id} took the last token.",
            )

        self.set_actor_id(state, opponent_id)
        return [
            self.user_message(
                f"{actor_id} took {action} token(s). "
                f"Tokens remaining: {tokens_remaining}. "
                f"{opponent_id}, take 1 or 2 tokens."
            )
        ]
