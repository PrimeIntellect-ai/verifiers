from __future__ import annotations

import json
import logging
import tarfile
import tempfile
from pathlib import Path
from typing import Any

from verifiers.utils.import_utils import load_toml

import verifiers.v1 as vf

logger = logging.getLogger("opencode_harbor.v1")

TERMINAL_BENCH_SAMPLE_TASKS = [
    "build-cython-ext",
    "chess-best-move",
    "configure-git-webserver",
    "fix-code-vulnerability",
    "log-summary-date-ranges",
    "polyglot-c-py",
    "qemu-alpine-ssh",
    "qemu-startup",
    "regex-log",
    "sqlite-with-gcov",
]

DEFAULT_DISABLED_TOOLS = ["webfetch", "question"]


def _build_opencode_config(
    disabled_tools: list[str] | None = None,
    system_prompt_path: str | None = None,
) -> str:
    config: dict[str, Any] = {
        "${SCHEMA_DOLLAR}schema": "https://opencode.ai/config.json",
        "provider": {
            "intercepted": {
                "npm": "@ai-sdk/openai-compatible",
                "name": "Intercepted",
                "options": {
                    "baseURL": "$OPENAI_BASE_URL",
                    "apiKey": "intercepted",
                    "timeout": 600000,
                },
                "models": {
                    "model": {
                        "name": "Intercepted Model",
                        "modalities": {"input": ["text"], "output": ["text"]},
                    }
                },
            }
        },
        "model": "intercepted/model",
    }
    if system_prompt_path or disabled_tools:
        build_config: dict[str, Any] = {}
        if system_prompt_path:
            build_config["prompt"] = "{file:" + system_prompt_path + "}"
        if disabled_tools:
            build_config["tools"] = {tool: False for tool in disabled_tools}
        config["agent"] = {"build": build_config}
    return json.dumps(config, indent=2)


def _build_run_command(
    agent_workdir: str,
    disabled_tools: list[str] | None = None,
    has_system_prompt: bool = False,
) -> str:
    system_prompt_sandbox_path = "/opencode/prompt.txt" if has_system_prompt else None
    config_json = _build_opencode_config(disabled_tools, system_prompt_sandbox_path)
    return f"""
set -e
apt-get update && apt-get install -y curl
curl -fsSL https://opencode.ai/install | bash
export PATH="$HOME/.opencode/bin:$PATH"
mkdir -p ~/.config/opencode /logs/agent /opencode /task {agent_workdir}
SCHEMA_DOLLAR='$'
cat > ~/.config/opencode/opencode.json << EOFCONFIG
{config_json}
EOFCONFIG
cd {agent_workdir}
opencode run "$(cat /task/instruction.md)" 2>&1 | tee /logs/agent/opencode.txt
"""


def parse_gb(value: object, default: float) -> float:
    if value is None:
        return default
    if isinstance(value, int | float):
        return float(value)
    text = str(value).strip().lower()
    if text.endswith("gb"):
        text = text[:-2]
    elif text.endswith("g"):
        text = text[:-1]
    elif text.endswith("mb"):
        return float(text[:-2]) / 1024
    elif text.endswith("m"):
        return float(text[:-1]) / 1024
    return float(text)


def harbor_source(
    dataset_path: str | Path,
    tasks: list[str] | None = None,
    docker_image: str = "python:3.11-slim",
    cpu_cores: int = 2,
    memory_gb: int = 4,
    disk_size_gb: int = 10,
    timeout_minutes: int = 120,
):
    dataset_dir = Path(dataset_path)
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset path not found: {dataset_dir}")
    selected_tasks = set(tasks or [])
    rows = []
    for task_dir in sorted(dataset_dir.iterdir()):
        if not task_dir.is_dir():
            continue
        if selected_tasks and task_dir.name not in selected_tasks:
            continue
        task_toml_path = task_dir / "task.toml"
        instruction_path = task_dir / "instruction.md"
        if not task_toml_path.exists() or not instruction_path.exists():
            logger.warning("Skipping %s: missing task.toml or instruction.md", task_dir)
            continue
        with open(task_toml_path, "rb") as f:
            config = load_toml(f)
        environment = config.get("environment", {}) or {}
        instruction = instruction_path.read_text().strip()
        rows.append(
            {
                "example_id": len(rows),
                "task_name": task_dir.name,
                "instruction": instruction,
                "task_toml": task_toml_path.read_text(),
                "task_dir": str(task_dir),
                "prompt": [{"role": "user", "content": instruction}],
                "sandbox": {
                    "image": environment.get("docker_image") or docker_image,
                    "cpu_cores": float(environment.get("cpus") or cpu_cores),
                    "memory_gb": parse_gb(environment.get("memory"), memory_gb),
                    "disk_size_gb": parse_gb(environment.get("storage"), disk_size_gb),
                    "timeout_minutes": timeout_minutes,
                    "scope": "rollout",
                },
                "info": {
                    "config": config,
                    "task_dir": str(task_dir),
                    "docker_image": environment.get("docker_image"),
                },
            }
        )
    if not rows:
        raise ValueError(f"No valid Harbor tasks found in {dataset_dir}")
    return rows


async def upload_harbor_tests(client, sandbox_id: str, task_dir: Path) -> None:
    solution_dir = task_dir / "solution"
    tests_dir = task_dir / "tests"
    with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as tmp_file:
        tar_path = Path(tmp_file.name)
    try:
        with tarfile.open(tar_path, "w:gz") as tar:
            if solution_dir.exists():
                for item in solution_dir.iterdir():
                    tar.add(item, arcname=f"oracle/{item.name}")
            if tests_dir.exists():
                for item in tests_dir.iterdir():
                    tar.add(item, arcname=f"tests/{item.name}")
        remote_tar = "/tmp/harbor_tests.tar.gz"
        await client.upload_file(sandbox_id, remote_tar, str(tar_path))
        await client.execute_command(
            sandbox_id,
            f"mkdir -p /oracle /tests && tar -xzf {remote_tar} -C / && rm {remote_tar}",
            timeout=900,
        )
    finally:
        tar_path.unlink(missing_ok=True)


@vf.reward(weight=1.0)
async def harbor_reward(task, state) -> float:
    if state.get("error") is not None:
        return 0.0
    sandbox_id = state.get("sandbox_id")
    if not isinstance(sandbox_id, str):
        return 0.0
    task_dir = Path(str(task["task_dir"]))
    from prime_sandboxes import AsyncSandboxClient

    client = AsyncSandboxClient()
    try:
        await upload_harbor_tests(client, sandbox_id, task_dir)
        result = await client.execute_command(
            sandbox_id,
            "bash test.sh",
            working_dir="/tests",
            timeout=300,
        )
        state["harbor_tests"] = {
            "returncode": result.exit_code,
            "stdout": result.stdout or "",
            "stderr": result.stderr or "",
        }
        reward_result = await client.execute_command(
            sandbox_id,
            "if [ -s /logs/verifier/reward.txt ]; then cat /logs/verifier/reward.txt; "
            "elif [ -s /logs/verifier/reward.json ]; then cat /logs/verifier/reward.json; fi",
        )
    except Exception as e:
        state["harbor_error"] = str(e)
        return 0.0
    finally:
        await client.aclose()
    reward_text = str(reward_result.stdout or "").strip()
    if not reward_text:
        return 0.0
    try:
        return float(reward_text)
    except ValueError:
        try:
            data = json.loads(reward_text)
        except json.JSONDecodeError:
            return 0.0
        return float(data.get("reward", 0.0))


def load_taskset(
    dataset_path: str | Path,
    tasks: list[str] | None = None,
    docker_image: str = "python:3.11-slim",
    cpu_cores: int = 2,
    memory_gb: int = 4,
    disk_size_gb: int = 10,
    timeout_minutes: int = 120,
    config=None,
):
    return vf.Taskset(
        source=lambda: harbor_source(
            dataset_path=dataset_path,
            tasks=tasks,
            docker_image=docker_image,
            cpu_cores=cpu_cores,
            memory_gb=memory_gb,
            disk_size_gb=disk_size_gb,
            timeout_minutes=timeout_minutes,
        ),
        rewards=[harbor_reward],
        config=config,
    )


def load_harness(
    agent_workdir: str = "/app",
    system_prompt_path: str | Path | None = None,
    disabled_tools: list[str] | None = None,
    max_turns: int = 4,
    config=None,
):
    prompt_text = None
    if system_prompt_path is not None:
        prompt_text = Path(system_prompt_path).read_text()
    files: dict[str, object] = {
        "/task/instruction.md": {"task": "instruction"},
        "/task/task.toml": {"task": "task_toml"},
    }
    if prompt_text is not None:
        files["/opencode/prompt.txt"] = prompt_text
    return vf.Harness(
        program={
            "sandbox": True,
            "command": [
                "bash",
                "-lc",
                _build_run_command(
                    agent_workdir=agent_workdir,
                    disabled_tools=disabled_tools,
                    has_system_prompt=prompt_text is not None,
                ),
            ],
            "files": files,
            "artifacts": {
                "opencode_log": {"path": "/logs/agent/opencode.txt", "format": "text"}
            },
        },
        sandbox={
            "image": "python:3.11-slim",
            "workdir": agent_workdir,
            "scope": "rollout",
            "timeout_minutes": 120,
            "command_timeout": 900,
            "network_access": True,
        },
        max_turns=max_turns,
        config=config,
    )


def load_v1_environment(
    dataset_path: str | Path = Path(__file__).parent / "tasks",
    dataset: str | None = None,
    tasks: list[str] | None = None,
    agent_workdir: str = "/app",
    docker_image: str = "python:3.11-slim",
    system_prompt_path: str | Path | None = Path(__file__).parent / "prompt.txt",
    disabled_tools: list[str] | None = None,
    timeout_seconds: float = 900.0,
    cpu_cores: int = 2,
    memory_gb: int = 4,
    disk_size_gb: int = 10,
    timeout_minutes: int = 120,
    max_turns: int = 4,
) -> vf.Env:
    _ = timeout_seconds
    if dataset and tasks:
        raise ValueError("Cannot specify both 'dataset' and 'tasks'")
    if dataset:
        if dataset == "terminal-bench-sample":
            tasks = TERMINAL_BENCH_SAMPLE_TASKS
        elif dataset == "terminal-bench":
            tasks = [
                path.name
                for path in sorted(Path(dataset_path).iterdir())
                if path.is_dir()
            ]
        else:
            raise ValueError(
                "dataset must be 'terminal-bench' or 'terminal-bench-sample'."
            )
    disabled_tools = (
        list(DEFAULT_DISABLED_TOOLS) if disabled_tools is None else disabled_tools
    )
    return vf.Env(
        taskset=load_taskset(
            dataset_path=dataset_path,
            tasks=tasks,
            docker_image=docker_image,
            cpu_cores=cpu_cores,
            memory_gb=memory_gb,
            disk_size_gb=disk_size_gb,
            timeout_minutes=timeout_minutes,
        ),
        harness=load_harness(
            agent_workdir=agent_workdir,
            system_prompt_path=system_prompt_path,
            disabled_tools=disabled_tools,
            max_turns=max_turns,
        ),
    )
