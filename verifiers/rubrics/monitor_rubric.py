from typing import Callable

from verifiers.rubrics.rubric import Rubric
from verifiers.types import State

StateKey = str
RenamedStateKey = tuple[StateKey, str]
RenamedTransformedStateKey = tuple[StateKey, str, Callable[..., float]]


class MonitorRubric(Rubric):
    """Simple rubric that reads values from the state for logging."""

    def __init__(
        self,
        state_keys: list[StateKey | RenamedStateKey | RenamedTransformedStateKey]
        | None = None,
    ):
        self.state_keys: list[
            StateKey | RenamedStateKey | RenamedTransformedStateKey
        ] = state_keys or []

        reward_funcs = []
        for state_key in self.state_keys:
            if isinstance(state_key, str):
                reward_func = self.get_read_from_state(state_key)
            else:
                reward_func = self.get_read_from_state(*state_key)  # type: ignore
            reward_funcs.append(reward_func)
        reward_weights = [0.0] * len(self.state_keys)  # only for logging

        # pass them to parent class
        super().__init__(funcs=reward_funcs, weights=reward_weights)

    def get_read_from_state(
        self,
        key: str,
        name: str | None = None,
        transform: Callable[..., float] = float,
    ) -> Callable:
        """Create a reward function that reads from the state."""

        async def read_from_state(state: State) -> float:
            key_parts = key.split(".")
            for key_part in key_parts[:-1]:
                state = state.get(key_part, {})
            value = state.get(key_parts[-1], 0.0)
            return transform(value)

        read_from_state.__name__ = name if name is not None else key

        return read_from_state
