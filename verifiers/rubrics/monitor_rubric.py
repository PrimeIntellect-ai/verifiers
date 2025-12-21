from typing import Callable

from verifiers.rubrics.rubric import Rubric
from verifiers.types import State


class MonitorRubric(Rubric):
    """Simple rubric that reads values from the state for logging."""

    def __init__(
        self,
        state_keys: list[str] | None = None,
        transforms: list[Callable[..., float] | None] | None = None,
    ):
        self.state_keys = state_keys or []
        self.transforms = transforms or []
        assert len(self.transforms) == len(self.state_keys), (
            "Number of transforms must match number of state keys"
        )

        reward_funcs = []
        for key, transform in zip(self.state_keys, self.transforms):
            reward_funcs.append(self.get_read_from_state(key, transform or float))
        reward_weights = [0.0] * len(self.state_keys)  # only for logging

        # pass them to parent class
        super().__init__(funcs=reward_funcs, weights=reward_weights)

    async def get_read_from_state(
        self, key: str, transform: Callable[..., float]
    ) -> Callable:
        """Create a reward function that reads from the state."""

        async def read_from_state(state: State) -> float:
            return transform(state.get(key, 0.0))

        read_from_state.__name__ = key

        return read_from_state
