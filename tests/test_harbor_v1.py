import importlib
import sys
from pathlib import Path
from typing import Any

import verifiers.v1 as vf
from harnesses import OpenCode
from tasksets import HarborTaskset
from verifiers.v1.loaders import load_environment_from_components


def _load_harbor_modules(monkeypatch: Any) -> tuple[Any, Any]:
    env_dir = Path(__file__).resolve().parent.parent / "environments" / "harbor_v1"
    monkeypatch.syspath_prepend(str(env_dir))
    for name in ("harbor_v1", "harbor_v1.taskset", "harbor_v1.harness"):
        sys.modules.pop(name, None)
    return (
        importlib.import_module("harbor_v1"),
        importlib.import_module("harbor_v1.taskset"),
    )


def test_harbor_v1_loads_thin_taskset_harness_package(monkeypatch: Any) -> None:
    package, module = _load_harbor_modules(monkeypatch)

    env = load_environment_from_components(package, {"config": {}})

    assert isinstance(env, vf.Env)
    assert isinstance(env.taskset, HarborTaskset)
    assert isinstance(env.harness, OpenCode)
    assert env.taskset.id == "harbor"
    assert env.taskset.config.source == "package"
    assert env.taskset.config.dataset == "harbor_v1"
    task = next(iter(env.taskset))
    assert task.task_name == "hello-world"
    assert task.image == "python:3.11-slim"
    assert Path(task.task_dir).parent == Path(module.__file__).parent / "tasks"


def test_harbor_v1_allows_local_dataset_override(
    monkeypatch: Any, tmp_path: Path
) -> None:
    package, _ = _load_harbor_modules(monkeypatch)
    task_dir = tmp_path / "tasks" / "local-task"
    task_dir.mkdir(parents=True)
    (task_dir / "instruction.md").write_text("write ok\n")
    (task_dir / "task.toml").write_text(
        "[environment]\n"
        'docker_image = "python:3.11-slim"\n'
        "[verifier]\n"
        "timeout_sec = 5\n"
    )

    env = load_environment_from_components(
        package,
        {
            "config": {
                "taskset": {
                    "source": "local",
                    "dataset": str(tmp_path / "tasks"),
                }
            }
        },
    )

    task = next(iter(env.taskset))
    assert task.task_name == "local-task"
    assert task.task_dir == str(task_dir.resolve())
