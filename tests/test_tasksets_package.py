"""Coverage for optional tasksets package builders."""

import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, call, patch

import pytest
from datasets import Dataset


PACKAGE_DIR = Path(__file__).resolve().parent.parent / "packages" / "tasksets"
HARNESSES_DIR = Path(__file__).resolve().parent.parent / "packages" / "harnesses"
SWEBENCH_PRO_DIR = (
    Path(__file__).resolve().parent.parent / "environments" / "swebench_pro"
)
TERMINAL_BENCH_2_DIR = (
    Path(__file__).resolve().parent.parent / "environments" / "terminal_bench_2"
)
sys.path.insert(0, str(PACKAGE_DIR))
sys.path.insert(0, str(HARNESSES_DIR))
sys.path.insert(0, str(SWEBENCH_PRO_DIR))
sys.path.insert(0, str(TERMINAL_BENCH_2_DIR))

import tasksets  # noqa: E402  # ty: ignore[unresolved-import]
from harnesses import Harness, build_harness_from_config  # noqa: E402  # ty: ignore[unresolved-import]
from swebench_pro import SWEBenchProEnv  # noqa: E402  # ty: ignore[unresolved-import]
from terminal_bench_2 import TerminalBench2Env  # noqa: E402  # ty: ignore[unresolved-import]
from tasksets.harbor import HarborTaskSet  # noqa: E402  # ty: ignore[unresolved-import]
from tasksets.harbor import HarborMCPHealthcheck  # noqa: E402  # ty: ignore[unresolved-import]
from tasksets.swe_bench import (  # noqa: E402  # ty: ignore[unresolved-import]
    SWEBenchProTaskSet,
    build_swebench_image_name,
)
from tasksets.terminal_bench import TerminalBench2TaskSet  # noqa: E402  # ty: ignore[unresolved-import]


def _sample_swebench_pro_dataset() -> Dataset:
    return Dataset.from_list(
        [
            {
                "repo": "NodeBB/NodeBB",
                "instance_id": "instance_NodeBB__NodeBB-abc123-v1",
                "base_commit": "deadbeef",
                "patch": "diff --git a/a.py b/a.py\n",
                "problem_statement": "Fix the failing forum setting.",
                "version": "v1",
                "dockerhub_tag": "nodebb.nodebb-NodeBB__NodeBB-abc123-v1",
                "fail_to_pass": '["test_a"]',
                "pass_to_pass": '["test_b"]',
                "difficulty": "hard",
            }
        ]
    )


def _write_terminal_bench_task(root: Path) -> Path:
    task_dir = root / "hello-world"
    (task_dir / "environment").mkdir(parents=True)
    (task_dir / "tests").mkdir()
    (task_dir / "solution").mkdir()
    (task_dir / "instruction.md").write_text(
        "Create hello.txt in the current directory."
    )
    (task_dir / "task.toml").write_text(
        """\
version = "1.0"

[metadata]
author_name = "Test Author"
author_email = "test@example.com"
difficulty = "easy"
category = "file-operations"
tags = ["file-operations"]

[verifier]
timeout_sec = 60.0

[agent]
timeout_sec = 360.0

[environment]
docker_image = "example/hello-world:latest"
cpus = 2
memory = "4G"
storage = "10G"
"""
    )
    (task_dir / "environment" / "Dockerfile").write_text(
        "FROM example/hello-world:latest\n"
    )
    (task_dir / "tests" / "test.sh").write_text(
        "pytest $TEST_DIR/test_outputs.py -rA\n"
    )
    (task_dir / "tests" / "test_outputs.py").write_text(
        "from pathlib import Path\n\n"
        "def test_hello():\n"
        "    assert Path('/app/hello.txt').read_text().strip() == 'Hello, world!'\n"
    )
    (task_dir / "solution" / "solve.sh").write_text(
        "#!/bin/bash\nprintf 'Hello, world!\\n' > /app/hello.txt\n"
    )
    return task_dir


def test_harbor_taskset_loads_task_directory():
    with tempfile.TemporaryDirectory() as tmpdir:
        task_dir = Path(tmpdir) / "hello"
        task_dir.mkdir()
        (task_dir / "task.toml").write_text(
            '[environment]\ndocker_image = "python:3.11"\n'
        )
        (task_dir / "instruction.md").write_text("Say hello.")

        taskset = tasksets.HarborTaskSet(dataset_path=tmpdir)
        row = taskset.get_dataset()[0]

    assert row["task"] == "hello"
    assert row["prompt"] == [{"role": "user", "content": "Say hello."}]
    assert row["info"]["docker_image"] == "python:3.11"
    assert taskset.get_sandbox_spec(row["info"]).timeout_minutes == 75
    assert taskset.get_test_timeout_seconds(row["info"]) is None


def test_harbor_taskset_parses_runtime_sandbox_spec_fields():
    with tempfile.TemporaryDirectory() as tmpdir:
        task_dir = Path(tmpdir) / "hello"
        task_dir.mkdir()
        (task_dir / "task.toml").write_text(
            """\
[agent]
timeout_sec = 120.0

[verifier]
timeout_sec = 60.0

[environment]
docker_image = "python:3.12"
start_command = "sleep infinity"
cpus = 3
memory = "5G"
storage = "11G"
gpus = 1
gpu_type = "L4"
vm = true
environment_vars = { FOO = "bar" }
"""
        )
        (task_dir / "instruction.md").write_text("Say hello.")
        taskset = tasksets.HarborTaskSet(dataset_path=tmpdir)
        row = taskset.get_dataset()[0]

    runtime = taskset.get_runtime_spec(row["info"])
    assert runtime.sandbox.image == "python:3.12"
    assert runtime.sandbox.start_command == "sleep infinity"
    assert runtime.sandbox.cpu_cores == 3
    assert runtime.sandbox.memory_gb == 5
    assert runtime.sandbox.disk_size_gb == 11
    assert runtime.sandbox.gpu_count == 1
    assert runtime.sandbox.gpu_type == "L4"
    assert runtime.sandbox.vm is True
    assert runtime.sandbox.environment_vars == {"FOO": "bar"}
    assert runtime.agent_timeout_seconds == 120.0
    assert runtime.test_timeout_seconds == 60.0


def test_harbor_taskset_extracts_mcp_tools():
    with tempfile.TemporaryDirectory() as tmpdir:
        task_dir = Path(tmpdir) / "hello"
        task_dir.mkdir()
        (task_dir / "task.toml").write_text(
            """\
[environment]
docker_image = "python:3.11"

[[environment.mcp_servers]]
name = "svc"
transport = "streamable-http"
url = "http://svc-host:8000/mcp"
headers = { Authorization = "Bearer token" }
"""
        )
        (task_dir / "instruction.md").write_text("Say hello.")
        taskset = tasksets.HarborTaskSet(dataset_path=tmpdir)
        row = taskset.get_dataset()[0]
        tools = taskset.get_tools(row["info"])

    assert tools.mcp_servers == [
        {
            "name": "svc",
            "transport": "streamable-http",
            "url": "http://svc-host:8000/mcp",
            "headers": {"Authorization": "Bearer token"},
        }
    ]


@pytest.mark.asyncio
async def test_harbor_taskset_prepares_external_mcp_urls():
    with tempfile.TemporaryDirectory() as tmpdir:
        task_dir = Path(tmpdir) / "hello"
        task_dir.mkdir()
        (task_dir / "task.toml").write_text(
            """\
[environment]
docker_image = "python:3.11"

[[environment.mcp_servers]]
name = "remote"
transport = "streamable-http"
url = "https://mcp.example.com/mcp"
"""
        )
        (task_dir / "instruction.md").write_text("Say hello.")
        taskset = tasksets.HarborTaskSet(dataset_path=tmpdir)
        row = taskset.get_dataset()[0]
        state = {"sandbox_id": "sbx", "info": row["info"]}

        tools = await taskset.prepare_tools(state, taskset.get_tools(row["info"]))

    assert tools.mcp_servers[0]["url"] == "https://mcp.example.com/mcp"
    assert tools.env_vars == {"HARBOR_MCP_REMOTE_URL": "https://mcp.example.com/mcp"}


@pytest.mark.asyncio
async def test_harbor_taskset_starts_managed_mcp_servers():
    with tempfile.TemporaryDirectory() as tmpdir:
        task_dir = Path(tmpdir) / "hello"
        task_dir.mkdir()
        (task_dir / "task.toml").write_text(
            """\
[environment]
docker_image = "python:3.11"

[[environment.mcp_servers]]
name = "svc"
transport = "streamable-http"
url = "http://svc-host:8000/mcp"
"""
        )
        (task_dir / "instruction.md").write_text("Say hello.")
        taskset = tasksets.HarborTaskSet(
            dataset_path=tmpdir,
            mcp_launch_commands={"svc": "python /mcp/server.py"},
            mcp_healthcheck=HarborMCPHealthcheck(retries=1, interval_sec=0),
        )
        row = taskset.get_dataset()[0]
        sandbox_client = SimpleNamespace(
            execute_command=AsyncMock(
                return_value=SimpleNamespace(exit_code=0, stdout="", stderr="")
            ),
            start_background_job=AsyncMock(return_value=SimpleNamespace(job_id="job")),
            get_background_job=AsyncMock(
                return_value=SimpleNamespace(completed=False, stderr="", exit_code=None)
            ),
        )
        state = {
            "sandbox_id": "sbx",
            "sandbox_client": sandbox_client,
            "info": row["info"],
        }

        tools = await taskset.prepare_tools(state, taskset.get_tools(row["info"]))

    assert tools.mcp_servers[0]["url"] == "http://127.0.0.1:8000/mcp"
    assert tools.env_vars == {"HARBOR_MCP_SVC_URL": "http://127.0.0.1:8000/mcp"}
    sandbox_client.start_background_job.assert_awaited_once()
    assert "svc-host" in sandbox_client.execute_command.await_args_list[0].args[1]
    assert "/proc/net/tcp" in sandbox_client.execute_command.await_args_list[1].args[1]


@pytest.mark.asyncio
async def test_harbor_taskset_runs_environment_setup():
    with tempfile.TemporaryDirectory() as tmpdir:
        task_dir = Path(tmpdir) / "hello"
        (task_dir / "environment").mkdir(parents=True)
        (task_dir / "task.toml").write_text(
            '[agent]\ntimeout_sec = 42\n[environment]\ndocker_image = "python:3.11"\n'
        )
        (task_dir / "instruction.md").write_text("Say hello.")
        (task_dir / "environment" / "setup.sh").write_text("#!/bin/bash\ntrue\n")
        taskset = tasksets.HarborTaskSet(dataset_path=tmpdir)
        state = {
            "sandbox_id": "sbx",
            "sandbox_client": SimpleNamespace(
                upload_file=AsyncMock(),
                execute_command=AsyncMock(
                    return_value=SimpleNamespace(exit_code=0, stdout="", stderr="")
                ),
                run_background_job=AsyncMock(
                    return_value=SimpleNamespace(exit_code=0, stdout="", stderr="")
                ),
            ),
            "info": taskset.get_dataset()[0]["info"],
        }

        await taskset.setup(state)

    execute_command = state["sandbox_client"].execute_command
    assert execute_command.await_args_list[-1:] == [
        call(
            "sbx",
            "rm -rf /environment && tar -xzf /tmp/harbor_environment.tar.gz -C / && rm /tmp/harbor_environment.tar.gz",
            working_dir=None,
            timeout=900,
        ),
    ]
    state["sandbox_client"].run_background_job.assert_awaited_once_with(
        sandbox_id="sbx",
        command="bash /environment/setup.sh && rm -rf /environment",
        working_dir=None,
        timeout=900,
    )


@pytest.mark.asyncio
async def test_harbor_taskset_runs_tests_as_background_job():
    with tempfile.TemporaryDirectory() as tmpdir:
        _write_terminal_bench_task(Path(tmpdir))
        taskset = tasksets.HarborTaskSet(dataset_path=tmpdir)
        row = taskset.get_dataset()[0]
        sandbox_client = SimpleNamespace(
            upload_file=AsyncMock(),
            execute_command=AsyncMock(
                side_effect=[
                    SimpleNamespace(exit_code=0, stdout="", stderr=""),
                    SimpleNamespace(exit_code=0, stdout="1\n", stderr=""),
                ]
            ),
            run_background_job=AsyncMock(
                return_value=SimpleNamespace(exit_code=0, stdout="", stderr="")
            ),
        )
        state = {
            "sandbox_id": "sbx",
            "sandbox_client": sandbox_client,
            "info": row["info"],
        }

        reward = await taskset.run_tests(
            sandbox_client=sandbox_client,
            sandbox_id="sbx",
            state=state,
            test_timeout=3600,
        )

    assert reward == "1"
    sandbox_client.run_background_job.assert_awaited_once_with(
        sandbox_id="sbx",
        command="bash test.sh",
        working_dir="/tests",
        timeout=3600,
    )
    assert sandbox_client.execute_command.await_args_list[0] == call(
        "sbx",
        "mkdir -p /oracle /tests && tar -xzf /tmp/harbor_tests.tar.gz -C / && rm /tmp/harbor_tests.tar.gz",
        working_dir=None,
        timeout=900,
    )


def test_build_swebench_image_name_normalizes_instance_id():
    image = build_swebench_image_name(
        "pytest-dev__pytest-12345",
        namespace="swebench",
        arch="x86_64",
        tag="latest",
    )
    assert image == "swebench/sweb.eval.x86_64.pytest-dev_1776_pytest-12345:latest"


def test_swebench_pro_taskset_generates_harbor_tasks():
    with patch(
        "tasksets.swe_bench.swe_bench.load_dataset",
        return_value=_sample_swebench_pro_dataset(),
    ):
        taskset = SWEBenchProTaskSet(max_examples=1)

    row = taskset.get_dataset()[0]
    assert row["task"] == "instance_NodeBB__NodeBB-abc123-v1"
    assert row["info"]["FAIL_TO_PASS"] == ["test_a"]
    assert row["info"]["PASS_TO_PASS"] == ["test_b"]
    assert row["info"]["config"]["agent"]["harness"]["agent"] == "openclaw"
    assert row["info"]["config"]["environment"]["docker_image"] == (
        "jefzda/sweap-images:nodebb.nodebb-NodeBB__NodeBB-abc123-v1"
    )

    task_dir = Path(row["info"]["task_dir"])
    assert (task_dir / "instruction.md").exists()
    assert (task_dir / "task.toml").exists()
    assert (task_dir / "tests" / "test.sh").exists()
    assert (task_dir / "solution" / "solve.sh").exists()


def test_terminal_bench_2_taskset_uses_harbor_task_directory():
    with tempfile.TemporaryDirectory() as tmpdir:
        _write_terminal_bench_task(Path(tmpdir))
        taskset = TerminalBench2TaskSet(
            dataset_path=tmpdir,
            task_ids=["hello-world"],
        )
        row = taskset.get_dataset()[0]

        assert row["task"] == "hello-world"
        assert row["prompt"] == [
            {"role": "user", "content": "Create hello.txt in the current directory."}
        ]
        assert row["info"]["config"]["environment"]["docker_image"] == (
            "example/hello-world:latest"
        )
        assert taskset.get_sandbox_spec(row["info"]).timeout_minutes == 21
        assert taskset.get_agent_timeout_seconds(row["info"]) == 360.0
        assert taskset.get_test_timeout_seconds(row["info"]) == 60.0

        task_dir = Path(row["info"]["task_dir"])
        assert (task_dir / "instruction.md").exists()
        assert (task_dir / "tests" / "test.sh").exists()
        assert (task_dir / "solution" / "solve.sh").exists()
        assert TerminalBench2TaskSet.setup is HarborTaskSet.setup
        assert TerminalBench2TaskSet.run_tests is HarborTaskSet.run_tests


def test_swebench_pro_env_uses_harness_config_workdir_for_taskset():
    with patch(
        "tasksets.swe_bench.swe_bench.load_dataset",
        return_value=_sample_swebench_pro_dataset(),
    ):
        env = SWEBenchProEnv(
            max_examples=1,
            harness_config={
                "transport": "acp",
                "agent": "openclaw",
                "cwd": "/testbed",
            },
        )

    assert env.taskset.agent_workdir == "/testbed"
    assert "OPENCLAW_AGENT_WORKDIR=/testbed" in env.harness.run_command


def test_swebench_pro_env_accepts_legacy_top_level_harness_args():
    with patch(
        "tasksets.swe_bench.swe_bench.load_dataset",
        return_value=_sample_swebench_pro_dataset(),
    ):
        env = SWEBenchProEnv(
            harness="codex",
            limit=1,
            codex_reasoning_effort="medium",
            codex_reasoning_summary="auto",
        )

    assert len(env.taskset.get_dataset()) == 1
    assert env.harness.instruction_path == "/codex/prompt.txt"
    assert "model_reasoning_effort=medium" in env.harness.run_command
    assert "model_reasoning_summary=auto" in env.harness.run_command


def test_swebench_pro_env_accepts_generic_harness_object():
    harness = Harness(run_command="echo ok", instruction_path="/custom/task.md")
    with patch(
        "tasksets.swe_bench.swe_bench.load_dataset",
        return_value=_sample_swebench_pro_dataset(),
    ):
        env = SWEBenchProEnv(harness=harness, limit=1)

    assert env.harness is harness
    assert env.harness.instruction_path == "/custom/task.md"


def test_swebench_pro_env_accepts_generic_harness_table():
    with patch(
        "tasksets.swe_bench.swe_bench.load_dataset",
        return_value=_sample_swebench_pro_dataset(),
    ):
        env = SWEBenchProEnv(
            harness={
                "factory": "harnesses.codex.codex_harness",
                "reasoning_effort": "medium",
            },
            limit=1,
        )

    assert env.harness.instruction_path == "/codex/prompt.txt"
    assert "model_reasoning_effort=medium" in env.harness.run_command


def test_terminal_bench_2_env_uses_generic_harness_config():
    with tempfile.TemporaryDirectory() as tmpdir:
        _write_terminal_bench_task(Path(tmpdir))
        env = TerminalBench2Env(
            dataset_path=tmpdir,
            limit=1,
            harness="opencode",
            opencode_disabled_tools=["question"],
        )

        assert len(env.taskset.get_dataset()) == 1
        assert env.harness.instruction_path == "/opencode/prompt.txt"
        assert '"question": false' in env.harness.run_command


def test_generic_harness_factory_uses_convention_and_aliases():
    harness = build_harness_from_config(
        {
            "transport": "acp",
            "agent": "opencode",
            "cwd": "/workspace",
            "disabled_tools": ["question"],
        }
    )

    assert harness.instruction_path == "/opencode/prompt.txt"
    assert "cd /workspace" in harness.run_command
    assert '"question": false' in harness.run_command


def test_generic_harness_factory_strips_harness_name_prefixes():
    harness = build_harness_from_config(
        {
            "harness": "opencode",
            "opencode_disabled_tools": ["question", "task"],
        }
    )

    assert '"question": false' in harness.run_command
    assert '"task": false' in harness.run_command


def test_generic_harness_factory_strips_short_harness_prefixes():
    harness = build_harness_from_config(
        {
            "harness": "claude-code",
            "agent_max_turns": 80,
            "claude_reasoning_effort": "high",
            "max_thinking_tokens": 32000,
        }
    )

    assert "CLAUDE_ARGS+=(--max-turns 80)" in harness.run_command
    assert "CLAUDE_ARGS+=(--effort high)" in harness.run_command
    assert "export MAX_THINKING_TOKENS=32000" in harness.run_command


def test_generic_harness_factory_accepts_harness_object():
    harness = Harness(run_command="echo ok")
    assert build_harness_from_config(harness) is harness


def test_generic_harness_factory_accepts_string_name():
    harness = build_harness_from_config("codex")
    assert harness.instruction_path == "/codex/prompt.txt"


def test_generic_harness_factory_accepts_nested_harness_table():
    harness = build_harness_from_config(
        {
            "harness": {
                "factory": "harnesses.codex.codex_harness",
                "reasoning_effort": "medium",
            },
        }
    )

    assert harness.instruction_path == "/codex/prompt.txt"
    assert "model_reasoning_effort=medium" in harness.run_command


def test_generic_harness_factory_accepts_name_alias():
    harness = build_harness_from_config(
        {
            "name": "codex",
            "cwd": "/workspace",
        }
    )

    assert harness.instruction_path == "/codex/prompt.txt"
    assert "CODEX_AGENT_WORKDIR=/workspace" in harness.run_command


def test_generic_harness_factory_accepts_nested_legacy_agent_config():
    harness = build_harness_from_config(
        {
            "agent": {
                "timeout_sec": 120,
                "harness": {
                    "agent": "terminus-2",
                    "cwd": "/workspace",
                    "parser_name": "xml",
                },
            },
        }
    )

    assert harness.instruction_path == "/task/instruction.md"
    assert "--parser-name xml" in harness.run_command
    assert "--max-turns" in harness.run_command
