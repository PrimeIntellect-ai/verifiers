from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import pytest

import verifiers.v1 as vf
from verifiers.v1.packages.harnesses.pi import pi_mcp_json, pi_models_json


def write_harbor_task(root: Path, name: str = "task-a") -> Path:
    task_dir = root / name
    (task_dir / "tests").mkdir(parents=True)
    (task_dir / "solution").mkdir()
    (task_dir / "instruction.md").write_text("Write hello to /app/hello.txt\n")
    (task_dir / "task.toml").write_text(
        """
version = "1.0"

[environment]
docker_image = "ubuntu:24.04"
cpus = 1
memory = "2G"
storage = "8G"

[agent]
timeout_sec = 600

[verifier]
timeout_sec = 300
""".strip()
    )
    (task_dir / "tests" / "test.sh").write_text("echo 1 > /logs/verifier/reward.txt")
    (task_dir / "solution" / "solve.sh").write_text("echo hello > /app/hello.txt")
    return task_dir


def test_harbor_taskset_loads_local_tasks_with_program_patch(tmp_path: Path) -> None:
    write_harbor_task(tmp_path)

    taskset = vf.HarborTaskset(tasks=tmp_path)
    task = next(iter(taskset))

    assert task["taskset_id"] == "harbor"
    assert task["task_name"] == "task-a"
    assert task["prompt"] == [
        {"role": "user", "content": "Write hello to /app/hello.txt"}
    ]
    assert task["sandbox"]["image"] == "ubuntu:24.04"
    assert task["sandbox"]["memory_gb"] == 2.0
    assert task["sandbox"]["disk_size_gb"] == 8.0
    assert task["sandbox"]["command_timeout"] == 600
    assert task["harbor"]["test_timeout"] == 300.0
    assert task["program"]["files"] == {
        "/task/instruction.md": {"task": "instruction"},
        "/task/task.toml": {"task": "task_toml"},
    }
    assert task["program"]["env"]["HARBOR_TASK_NAME"] == "task-a"
    assert task["program"]["env"]["AGENT_WORKDIR"] == "/app"


def test_harbor_taskset_accepts_single_task_dir(tmp_path: Path) -> None:
    task_dir = write_harbor_task(tmp_path, "only-task")

    taskset = vf.HarborTaskset(tasks=task_dir)

    assert [task["task_name"] for task in taskset] == ["only-task"]


def test_harbor_taskset_constructs_env_with_opencode(tmp_path: Path) -> None:
    write_harbor_task(tmp_path)

    env = vf.Env(taskset=vf.HarborTaskset(tasks=tmp_path), harness=vf.OpenCode())

    row = env.get_dataset()[0]
    task = env.taskset.to_task(row)
    assert task["task_name"] == "task-a"
    assert isinstance(env.harness, vf.OpenCode)
    assert "task_dir" not in cast(dict[str, object], env.harness.program)


def test_packaged_harbor_and_opencode_imports_are_reexported() -> None:
    from verifiers.v1.packages.harnesses import OpenCode, Pi
    from verifiers.v1.packages.tasksets import HarborTaskset

    assert vf.OpenCode is OpenCode
    assert vf.Pi is Pi
    assert vf.HarborTaskset is HarborTaskset


def test_pi_harness_writes_intercepted_model_and_mcp_config() -> None:
    models = json.loads(
        pi_models_json(
            {
                "base_url": "http://127.0.0.1:1/rollout/key/v1",
                "api_key": "secret",
                "api_client_type": "openai_chat_completions",
                "model": "openai/gpt-5.4-mini",
            }
        )
    )
    mcp = json.loads(pi_mcp_json())

    provider = models["providers"]["verifiers"]
    assert provider["baseUrl"] == "http://127.0.0.1:1/rollout/key/v1"
    assert provider["api"] == "openai-completions"
    assert provider["apiKey"] == "secret"
    assert provider["models"] == [{"id": "model", "name": "openai/gpt-5.4-mini"}]
    assert mcp["mcpServers"]["verifiers-tools"]["command"] == "python3"


def test_task_program_merges_into_command_program_without_collisions() -> None:
    harness = vf.CLIHarness(
        command=["tool"],
        sandbox=True,
        files={"/harness.txt": "harness"},
        setup="echo harness",
        tools={"mcp": "echo harness tools"},
        env={"HARNESS": "1"},
        artifacts={"log": {"path": "/logs/harness.log", "format": "text"}},
        program={"args": ["--base"]},
    )
    task = vf.Task(
        {
            "prompt": [],
            "program": {
                "files": {"/task/instruction.md": "task"},
                "setup": "echo task",
                "env": {"TASK": "1"},
                "artifacts": {"task_log": {"path": "/logs/task.log", "format": "text"}},
                "args": ["--task"],
            },
        }
    ).freeze()

    program = harness.task_merged_program(
        cast(dict[str, object], harness.program), task, kind="command"
    )

    assert program["files"] == {
        "/harness.txt": "harness",
        "/task/instruction.md": "task",
    }
    assert program["setup"] == ["echo harness", "echo task"]
    assert program["tools"] == {"mcp": "echo harness tools"}
    assert program["env"] == {"HARNESS": "1", "TASK": "1"}
    assert program["args"] == ["--base", "--task"]
    assert program["artifacts"] == {
        "log": {"path": "/logs/harness.log", "format": "text"},
        "task_log": {"path": "/logs/task.log", "format": "text"},
    }


def test_task_program_rejects_harness_owned_keys() -> None:
    harness = vf.CLIHarness(command=["tool"], sandbox=True)
    task = vf.Task({"prompt": [], "program": {"command": ["other"]}}).freeze()

    with pytest.raises(ValueError, match="task.program can only define"):
        harness.task_merged_program(
            cast(dict[str, object], harness.program), task, kind="command"
        )


def test_task_program_rejects_colliding_upload_paths() -> None:
    harness = vf.CLIHarness(
        command=["tool"], sandbox=True, files={"/task/instruction.md": "harness"}
    )
    task = vf.Task(
        {"prompt": [], "program": {"files": {"/task/instruction.md": "task"}}}
    ).freeze()

    with pytest.raises(ValueError, match="define the same keys"):
        harness.task_merged_program(
            cast(dict[str, object], harness.program), task, kind="command"
        )
