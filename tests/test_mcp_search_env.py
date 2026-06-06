from environments.mcp_search_env_v1.mcp_search_env_v1 import taskset as module

import pytest
import verifiers.v1 as vf
from verifiers.v1.loaders import load_environment_from_components


def test_mcp_search_env_is_v1_only() -> None:
    env = load_environment_from_components(
        module, {"config": {"taskset": {"max_turns": 4}}}
    )

    assert isinstance(env, vf.Env)
    assert isinstance(env.taskset, vf.Taskset)
    assert isinstance(env.harness, vf.Harness)
    assert not hasattr(module, "load_environment")
    assert not hasattr(module, "load_v1_environment")
    assert env.taskset.config.max_turns == 4


def test_mcp_search_env_preserves_harness_config() -> None:
    env = load_environment_from_components(
        module, {"config": {"harness": {"max_turns": 7}}}
    )

    assert env.harness.config.max_turns == 7


def test_mcp_search_default_taskset_has_stable_non_doc_fixture() -> None:
    rows = list(module.load_tasks())

    assert len(rows) >= 10
    assert len({row["answer"] for row in rows}) == len(rows)
    assert all(row["max_turns"] == 6 for row in rows)
    assert all("document" not in str(row["question"]).lower() for row in rows)


def test_mcp_search_taskset_accepts_v1_taskset_config() -> None:
    env = load_environment_from_components(
        module, {"config": {"taskset": {"max_turns": 3}}}
    )
    tasks = list(env.taskset)

    assert env.taskset.config.max_turns == 3
    assert all(task.max_turns == 3 for task in tasks)


@pytest.mark.asyncio
async def test_mcp_search_reward_handles_missing_assistant() -> None:
    task = module.MCPSearchTask(
        query="expected",
        question="find expected",
        answer="expected",
    )
    taskset = module.MCPSearchTaskset(module.MCPSearchTasksetConfig())
    state = vf.State(task_id=task.task_id)
    assert await taskset.exact_title_reward(task, state) == 0.0
    state.add_turn(
        vf.Turn(prompt=task.prompt, completion=[vf.UserMessage(content="expected")])
    )
    assert await taskset.exact_title_reward(task, state) == 0.0
