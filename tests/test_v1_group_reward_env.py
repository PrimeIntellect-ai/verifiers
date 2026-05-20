from typing import cast

import pytest

from verifiers.clients import Client
from verifiers.types import RolloutInput

from environments.hello_group_reward_v1.hello_group_reward_v1 import (
    GroupRewardEnvConfig,
    load_environment,
)


@pytest.mark.asyncio
async def test_hello_group_reward_v1_scores_full_group_lifecycle() -> None:
    env = load_environment(config=GroupRewardEnvConfig(taskset={"num_examples": 1}))
    assert env.requires_group_rollouts
    assert env.provides_advantages

    row = cast(RolloutInput, env.taskset.get_dataset()[0])
    states = await env._run_group_states(
        [row, row, row, row],
        cast(Client, object()),
        "unused-model",
        {},
    )

    assert len(states) == 4
    by_candidate = {state["candidate_id"]: state for state in states}
    exact = by_candidate["exact"]
    off_topic = by_candidate["off-topic"]

    assert exact["group_summary"]["rank"] == 1
    assert exact["group_summary"]["best_candidate_id"] == "exact"
    assert exact["metrics"]["relative_group_reward"] == 1.0
    assert off_topic["metrics"]["relative_group_reward"] == 0.0
    assert exact["reward"] > off_topic["reward"]
    assert all(state["group_cleaned"] is True for state in states)
    assert sum(float(state["advantage"]) for state in states) == pytest.approx(0.0)
    assert all("runtime_id" not in state.get("runtime", {}) for state in states)
