import importlib.util
import sys
from pathlib import Path
from typing import Any, cast

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
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_load_environment_uses_v1_taskset_and_harness() -> None:
    module = _load_opencode_module()

    env = module.load_environment(config=module.OpenCodeHarborEnvConfig())

    assert isinstance(env, vf.Env)
    assert isinstance(env.taskset, vf.HarborTaskset)
    assert isinstance(env.harness, vf.OpenCode)
    assert isinstance(env.harness.config, vf.OpenCodeConfig)
    assert not hasattr(module, "OpenCodeHarborHarnessConfig")
    assert not hasattr(module, "TERMINAL_BENCH_SAMPLE_TASKS")
    assert env.taskset.resolve_tasks_root() == Path(module.__file__).parent / "tasks"
    assert env.harness.config.max_turns == 4
    assert env.harness.config.disabled_tools == vf.OpenCodeConfig().disabled_tools
    assert "webfetch" in env.harness.config.disabled_tools
    assert "question" in env.harness.config.disabled_tools

    program = cast(dict[str, object], env.harness.program)
    mcp_setup = cast(dict[str, object], program["channels"])["mcp"]
    assert '"webfetch": false' in cast(str, mcp_setup)
    assert '"question": false' in cast(str, mcp_setup)


def test_load_environment_accepts_v1_taskset_and_harness_config() -> None:
    module = _load_opencode_module()

    env = module.load_environment(
        config=module.OpenCodeHarborEnvConfig(
            taskset={
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

    assert env.taskset.resolve_tasks_root() == Path(module.__file__).parent / "tasks"
    assert env.taskset.task_names == ["task-a"]
    assert env.taskset.cpu_cores == 1.5
    assert env.harness.config.agent_workdir == "/workspace"
    assert env.harness.config.max_turns == 2

    program = cast(dict[str, object], env.harness.program)
    command = cast(list[object], program["command"])
    mcp_setup = cast(dict[str, object], program["channels"])["mcp"]
    assert "/workspace" in cast(str, command[2])
    assert '"webfetch": false' in cast(str, mcp_setup)
    assert '"question": false' not in cast(str, mcp_setup)


def test_pyproject_does_not_define_unsupported_harness_defaults() -> None:
    module = _load_opencode_module()
    pyproject = Path(module.__file__).parent / "pyproject.toml"

    assert "[tool.verifiers.harness]" not in pyproject.read_text()
