from pathlib import Path

import verifiers as vf
from verifiers.envs.experimental.modules.harnesses import OpenCode
from verifiers.envs.experimental.modules.tasksets import HarborTaskset

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

DATASETS = {
    "terminal-bench-sample": TERMINAL_BENCH_SAMPLE_TASKS,
}


def _read_system_prompt(system_prompt_path: str | Path | None) -> str | None:
    if system_prompt_path is None:
        return None
    path = Path(system_prompt_path)
    if not path.exists():
        raise FileNotFoundError(f"System prompt file not found: {path}")
    return path.read_text()


def load_environment(
    dataset_path: str | Path = Path(__file__).parents[1] / "opencode_harbor" / "tasks",
    dataset: str | None = None,
    tasks: list[str] | None = None,
    agent_workdir: str = "/app",
    docker_image: str = "python:3.11-slim",
    system_prompt_path: str | Path | None = Path(__file__).parents[1]
    / "opencode_harbor"
    / "prompt.txt",
    disabled_tools: list[str] | None = None,
    timeout_seconds: float = 900.0,
    cpu_cores: int = 2,
    memory_gb: int = 4,
    disk_size_gb: int = 10,
    timeout_minutes: int = 120,
    max_turns: int = 10,
) -> vf.Environment:
    if dataset and tasks:
        raise ValueError("Cannot specify both 'dataset' and 'tasks'")
    if dataset:
        if dataset not in DATASETS:
            raise ValueError(
                f"Unknown dataset '{dataset}'. Available: {', '.join(DATASETS.keys())}"
            )
        tasks = DATASETS[dataset]
    if disabled_tools is None:
        disabled_tools = ["webfetch", "question"]

    taskset = HarborTaskset(
        path=dataset_path,
        tasks=tasks,
        agent_workdir=agent_workdir,
    )
    harness = OpenCode(
        agent_workdir=agent_workdir,
        system_prompt=_read_system_prompt(system_prompt_path),
        disabled_tools=disabled_tools,
        sandbox=vf.SandboxSpec(
            image=docker_image,
            cpu_cores=cpu_cores,
            memory_gb=memory_gb,
            disk_size_gb=disk_size_gb,
            timeout_minutes=timeout_minutes,
        ),
        timeout_seconds=timeout_seconds,
        max_turns=max_turns,
    )
    return vf.Env(taskset=taskset, harness=harness)
