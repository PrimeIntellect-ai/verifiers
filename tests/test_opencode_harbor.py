import importlib.util
import sys
from pathlib import Path
from typing import Any

import verifiers.v1 as vf
from harnesses import OpenCode, OpenCodeConfig
from tasksets import HarborTaskset


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

    env = module.load_environment(
        config=vf.EnvConfig(
            taskset=module.HarborTasksetConfig(),
            harness=module.OpenCodeConfig(),
        )
    )

    assert isinstance(env, vf.Env)
    assert isinstance(env.taskset, HarborTaskset)
    assert isinstance(env.harness, OpenCode)
    assert isinstance(env.harness.config, OpenCodeConfig)
    assert not hasattr(module, "OpenCodeHarborHarnessConfig")
    assert not hasattr(module, "TERMINAL_BENCH_SAMPLE_TASKS")
    assert env.taskset.config.bundle_package == module.__name__
    task = next(iter(env.taskset))
    assert Path(task.task_dir).parent == Path(module.__file__).parent / "tasks"
    assert env.harness.config.max_turns == 4
    assert env.harness.config.disabled_tools == OpenCodeConfig().disabled_tools
    assert "webfetch" in env.harness.config.disabled_tools
    assert "question" in env.harness.config.disabled_tools

    command = env.harness.command(task, vf.State(task_id=task.task_id))
    assert '"webfetch": false' in command[2]
    assert '"question": false' in command[2]


def test_load_environment_accepts_v1_taskset_and_harness_config() -> None:
    module = _load_opencode_module()

    env = module.load_environment(
        config=vf.EnvConfig(
            taskset=module.HarborTasksetConfig(
                task_names=["hello-world"],
                task_runtime={"cpu_cores": 1.5},
            ),
            harness=module.OpenCodeConfig(
                cwd="/workspace",
                disabled_tools=["webfetch"],
                max_turns=2,
            ),
        )
    )

    assert env.taskset.config.bundle_package == module.__name__
    task = next(iter(env.taskset))
    assert task.task_dir == str(Path(module.__file__).parent / "tasks" / "hello-world")
    assert env.taskset.config.task_names == ["hello-world"]
    assert env.taskset.config.task_runtime["cpu_cores"] == 1.5
    assert env.harness.config.cwd == "/workspace"
    assert env.harness.config.max_turns == 2

    command = env.harness.command(task, vf.State(task_id=task.task_id))
    assert '"webfetch": false' in command[2]
    assert '"question": false' not in command[2]


def test_pyproject_does_not_define_unsupported_harness_defaults() -> None:
    module = _load_opencode_module()
    pyproject = Path(module.__file__).parent / "pyproject.toml"

    assert "[tool.verifiers.harness]" not in pyproject.read_text()
