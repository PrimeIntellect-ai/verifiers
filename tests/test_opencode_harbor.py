from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any, cast

import pytest

import verifiers.v1 as vf


def _load_opencode_module() -> Any:
    module_path = (
        Path(__file__).resolve().parent.parent
        / "environments"
        / "opencode_harbor"
        / "opencode_harbor.py"
    )
    spec = importlib.util.spec_from_file_location(
        "test_opencode_harbor_module", module_path
    )
    assert spec is not None
    assert spec.loader is not None

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_load_environment_uses_v1_taskset_and_harness() -> None:
    module = _load_opencode_module()

    env = module.load_environment()

    assert isinstance(env, vf.Env)
    assert isinstance(env.taskset, vf.HarborTaskset)
    assert isinstance(env.harness, vf.OpenCode)
    assert isinstance(env.harness.config, vf.OpenCodeConfig)
    assert not hasattr(module, "OpenCodeHarborHarnessConfig")
    assert Path(env.taskset.tasks) == Path(module.__file__).parent / "tasks"
    assert env.harness.config.max_turns == 4
    assert env.harness.config.disabled_tools == ["webfetch", "question"]

    program = cast(dict[str, object], env.harness.program)
    mcp_setup = cast(dict[str, object], program["tools"])["mcp"]
    assert '"webfetch": false' in cast(str, mcp_setup)
    assert '"question": false' in cast(str, mcp_setup)
    assert '"read": false' not in cast(str, mcp_setup)


def test_load_environment_accepts_v1_taskset_and_harness_config(
    tmp_path: Path,
) -> None:
    module = _load_opencode_module()

    env = module.load_environment(
        config=vf.EnvConfig(
            taskset={
                "tasks": str(tmp_path),
                "task_names": ["task-a"],
                "cpu_cores": 1.5,
            },
            harness={
                "agent_workdir": "/workspace",
                "disabled_tools": ["webfetch"],
                "max_turns": 2,
            },
        )
    )

    assert Path(env.taskset.tasks) == tmp_path
    assert env.taskset.task_names == ["task-a"]
    assert env.taskset.cpu_cores == 1.5
    assert env.harness.config.agent_workdir == "/workspace"
    assert env.harness.config.max_turns == 2

    program = cast(dict[str, object], env.harness.program)
    command = cast(list[object], program["command"])
    mcp_setup = cast(dict[str, object], program["tools"])["mcp"]
    assert "/workspace" in cast(str, command[2])
    assert '"webfetch": false' in cast(str, mcp_setup)
    assert '"question": false' not in cast(str, mcp_setup)


def test_dataset_shortcuts_select_task_names() -> None:
    module = _load_opencode_module()

    env = module.load_environment(dataset="terminal-bench-sample")

    assert env.taskset.task_names == module.TERMINAL_BENCH_SAMPLE_TASKS


def test_dataset_rejects_explicit_task_names() -> None:
    module = _load_opencode_module()

    with pytest.raises(ValueError, match="dataset.*task_names"):
        module.load_environment(
            dataset="terminal-bench-sample",
            task_names=["hello-world"],
        )
