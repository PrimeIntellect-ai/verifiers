import importlib
import sys
from pathlib import Path
from typing import Any

import pytest
import verifiers.v1 as vf
from harnesses import OpenCode, OpenCodeConfig
from tasksets import HarborTaskset
from verifiers.v1.loaders import load_environment_from_components


def _load_opencode_modules(monkeypatch: pytest.MonkeyPatch) -> tuple[Any, Any]:
    env_dir = (
        Path(__file__).resolve().parent.parent / "environments" / "opencode_harbor_v1"
    )
    monkeypatch.syspath_prepend(str(env_dir))
    for name in (
        "opencode_harbor_v1",
        "opencode_harbor_v1.taskset",
        "opencode_harbor_v1.harness",
    ):
        sys.modules.pop(name, None)
    return (
        importlib.import_module("opencode_harbor_v1"),
        importlib.import_module("opencode_harbor_v1.taskset"),
    )


def test_load_environment_uses_v1_taskset_and_harness(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package, module = _load_opencode_modules(monkeypatch)

    env = load_environment_from_components(
        package,
        {
            "config": {
                "taskset": {},
                "harness": {},
            }
        },
    )

    assert isinstance(env, vf.Env)
    assert isinstance(env.taskset, HarborTaskset)
    assert isinstance(env.harness, OpenCode)
    assert isinstance(env.harness.config, OpenCodeConfig)
    assert not hasattr(module, "OpenCodeHarborHarnessConfig")
    assert not hasattr(module, "TERMINAL_BENCH_SAMPLE_TASKS")
    assert env.taskset.config.source == "package"
    assert env.taskset.config.dataset == "opencode_harbor_v1"
    task = next(iter(env.taskset))
    assert Path(task.task_dir).parent == Path(module.__file__).parent / "tasks"
    assert env.harness.config.max_turns == 4
    assert env.harness.config.disabled_tools == OpenCodeConfig().disabled_tools
    assert "webfetch" in env.harness.config.disabled_tools
    assert "question" in env.harness.config.disabled_tools

    command = env.harness.command(task, vf.State(task_id=task.task_id))
    assert '"webfetch": false' in command[2]
    assert '"question": false' in command[2]


def test_load_environment_accepts_v1_taskset_and_harness_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package, module = _load_opencode_modules(monkeypatch)

    env = load_environment_from_components(
        package,
        {
            "config": {
                "taskset": {
                    "tasks": ["hello-world"],
                },
                "harness": {
                    "cwd": "/workspace",
                    "disabled_tools": ["webfetch"],
                    "max_turns": 2,
                },
            }
        },
    )

    assert env.taskset.config.source == "package"
    assert env.taskset.config.dataset == "opencode_harbor_v1"
    assert isinstance(env.harness, OpenCode)
    task = next(iter(env.taskset))
    assert task.task_dir == str(Path(module.__file__).parent / "tasks" / "hello-world")
    assert env.taskset.config.tasks == ["hello-world"]
    assert env.harness.config.cwd == "/workspace"
    assert env.harness.config.max_turns == 2

    command = env.harness.command(task, vf.State(task_id=task.task_id))
    assert '"webfetch": false' in command[2]
    assert '"question": false' not in command[2]


def test_pyproject_does_not_define_unsupported_harness_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, module = _load_opencode_modules(monkeypatch)
    pyproject = Path(module.__file__).parents[1] / "pyproject.toml"

    assert "[tool.verifiers.harness]" not in pyproject.read_text()
