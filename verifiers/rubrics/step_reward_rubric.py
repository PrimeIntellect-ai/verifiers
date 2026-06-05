from typing import cast

import verifiers as vf
from verifiers.types import State
from verifiers.utils.step_reward_utils import apply_step_advantages


def _sum_step_rewards(state: State) -> float:
    return state.total_step_reward()


class StepRewardRubric(vf.Rubric):
    """Rubric that aggregates step-level rewards with discounted advantages."""

    def __init__(self, gamma: float = 1.0, weight: float = 1.0, **kwargs):
        super().__init__(funcs=[_sum_step_rewards], weights=[weight], **kwargs)
        self.gamma = gamma

    async def score_group(self, states: list[State]):
        num_states = len(states)
        if num_states == 0:
            return

        for state in states:
            state["reward"] = _sum_step_rewards(state)
            state["metrics"] = {"step_reward_sum": cast(float, state["reward"])}

        apply_step_advantages(states, gamma=self.gamma)
