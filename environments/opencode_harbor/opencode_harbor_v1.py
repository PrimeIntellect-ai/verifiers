from __future__ import annotations

from pathlib import Path

import verifiers.v1 as vf

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


def load_taskset(
    dataset_path: str | Path,
    tasks: list[str] | None = None,
    docker_image: str = "python:3.11-slim",
    cpu_cores: float = 2.0,
    memory_gb: float = 4.0,
    disk_size_gb: float = 10.0,
    timeout_minutes: int = 120,
    config=None,
):
    return vf.HarborTaskset(
        tasks=dataset_path,
        task_names=tasks,
        docker_image=docker_image,
        cpu_cores=cpu_cores,
        memory_gb=memory_gb,
        disk_size_gb=disk_size_gb,
        timeout_minutes=timeout_minutes,
        config=config,
    )


def load_harness(
    agent_workdir: str = "/app",
    system_prompt_path: str | Path | None = Path(__file__).parent / "prompt.txt",
    disabled_tools: list[str] | None = None,
    max_turns: int = 4,
    config=None,
):
    return vf.OpenCode(
        agent_workdir=agent_workdir,
        system_prompt=(
            Path(system_prompt_path).read_text()
            if system_prompt_path is not None
            else None
        ),
        disabled_tools=disabled_tools,
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
    cpu_cores: float = 2.0,
    memory_gb: float = 4.0,
    disk_size_gb: float = 10.0,
    timeout_minutes: int = 120,
    max_turns: int = 4,
) -> vf.Env:
    _ = timeout_seconds
    if dataset and tasks:
        raise ValueError("Cannot specify both 'dataset' and 'tasks'.")
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
