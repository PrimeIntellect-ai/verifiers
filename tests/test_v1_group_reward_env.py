import asyncio
from typing import cast

import pytest

import verifiers.v1 as vf
from verifiers.clients import Client
from verifiers.types import ClientConfig
from verifiers.types import RolloutInput

from environments.hello_group_reward_v1.hello_group_reward_v1 import taskset as module
from verifiers.v1.loaders import load_environment_from_components


@pytest.mark.asyncio
async def test_hello_group_reward_v1_scores_full_group_lifecycle() -> None:
    env = load_environment_from_components(
        module, {"config": {"taskset": {"num_examples": 1}}}
    )
    assert env.requires_group_rollouts
    assert env.provides_advantages

    row = cast(RolloutInput, env.taskset.get_dataset()[0])
    model = vf.ModelConfig(client=ClientConfig(), model="unused-model")
    env.harness.load_model_client = lambda _: vf.ModelClient(
        config=model, client=cast(Client, object())
    )

    async def close_model_client(_: vf.ModelClient) -> None:
        return None

    env.harness.close_model_client = close_model_client
    base_task = env.taskset.to_task(row)
    tasks, states = await env.taskset.init_group(base_task, 4)
    states = list(
        await asyncio.gather(
            *[
                env.run_rollout(task, model=model, state=state)
                for task, state in zip(tasks, states, strict=True)
            ]
        )
    )
    states = await env.score_group(tasks, states)

    assert len(states) == 4
    by_candidate = {state.extras["candidate_id"]: state for state in states}
    exact = by_candidate["exact"]
    off_topic = by_candidate["off-topic"]

    assert exact.metrics["group_rank"] == 1.0
    assert exact.metrics["relative_group_reward"] == 1.0
    assert off_topic.metrics["relative_group_reward"] == 0.0
    assert exact.reward > off_topic.reward
    assert sum(float(state.advantage) for state in states) == pytest.approx(0.0)
    assert all(state.transcript for state in states)
    assert all("runtime_id" not in state.metadata for state in states)
