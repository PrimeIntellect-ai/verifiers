import importlib.util
import inspect
from pathlib import Path
from typing import Any

import pytest
import verifiers.v1 as vf


def _load_mcp_search_module() -> Any:
    module_path = (
        Path(__file__).resolve().parent.parent
        / "environments"
        / "mcp_search_env"
        / "mcp_search_env.py"
    )
    spec = importlib.util.spec_from_file_location("test_mcp_search_env", module_path)
    assert spec is not None
    assert spec.loader is not None

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_mcp_search_env_is_v1_only() -> None:
    module = _load_mcp_search_module()

    env = module.load_environment(config=vf.EnvConfig(), max_turns=4)

    assert isinstance(env, vf.Env)
    assert isinstance(env.taskset, vf.Taskset)
    assert isinstance(env.harness, vf.Harness)
    assert "v1" not in inspect.signature(module.load_environment).parameters
    assert not hasattr(module, "load_v1_environment")
    assert not (Path(module.__file__).parent / "mcp_search_v1.py").exists()
    assert env.taskset.config.max_turns == 4


def test_mcp_search_default_taskset_has_stable_non_doc_fixture() -> None:
    module = _load_mcp_search_module()

    rows = module.load_taskset().rows()

    assert len(rows) >= 10
    assert len({row["answer"] for row in rows}) == len(rows)
    assert all(row["max_turns"] == 6 for row in rows)
    assert all("document" not in str(row["question"]).lower() for row in rows)


def test_mcp_search_taskset_accepts_v1_taskset_config() -> None:
    module = _load_mcp_search_module()

    env = module.load_environment(
        config=vf.EnvConfig(taskset={"max_turns": 3}),
    )
    rows = env.taskset.rows()

    assert env.taskset.config.max_turns == 3
    assert all(row["max_turns"] == 3 for row in rows)


@pytest.mark.asyncio
async def test_mcp_search_reward_handles_missing_assistant() -> None:
    module = _load_mcp_search_module()

    task = vf.Task({"answer": "expected"})
    assert await module.exact_title_reward(task, vf.State({"completion": []})) == 0.0
    assert (
        await module.exact_title_reward(
            task,
            vf.State({"completion": [{"role": "user", "content": "expected"}]}),
        )
        == 0.0
    )
