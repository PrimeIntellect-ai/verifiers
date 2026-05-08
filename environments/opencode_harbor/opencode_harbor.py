from __future__ import annotations

from pathlib import Path

import verifiers.v1 as vf

_TASKS_DIR = Path(__file__).parent / "tasks"

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
    config: vf.TasksetConfig | None = None,
    tasks: str | Path | None = None,
    task_names: list[str] | None = None,
    dataset: str | None = None,
    docker_image: str | None = None,
    cpu_cores: float | None = None,
    memory_gb: float | None = None,
    disk_size_gb: float | None = None,
    timeout_minutes: int | None = None,
    agent_timeout_seconds: float | None = None,
    verifier_timeout_seconds: float | None = None,
    workdir: str | None = None,
    task_dir: str | None = None,
    scope: str | None = None,
    env: dict[str, object] | None = None,
) -> vf.HarborTaskset:
    config = vf.HarborTasksetConfig(
        config,
        tasks=str(tasks) if tasks is not None else None,
        task_names=task_names,
        docker_image=docker_image,
        cpu_cores=cpu_cores,
        memory_gb=memory_gb,
        disk_size_gb=disk_size_gb,
        timeout_minutes=timeout_minutes,
        agent_timeout_seconds=agent_timeout_seconds,
        verifier_timeout_seconds=verifier_timeout_seconds,
        workdir=workdir,
        task_dir=task_dir,
        scope=scope,
        env=env,
    )
    tasks_root = tasks if tasks is not None else config.tasks or _TASKS_DIR
    selected_task_names = _dataset_task_names(
        dataset=dataset,
        tasks_root=tasks_root,
        task_names=config.task_names,
    )
    return vf.HarborTaskset(
        tasks=tasks_root,
        task_names=selected_task_names,
        config=config,
    )


def load_harness(
    config: vf.HarnessConfig | None = None,
) -> vf.OpenCode:
    return vf.OpenCode(config=config)


def load_environment(
    config: vf.EnvConfig | None = None,
    tasks: str | Path | None = None,
    task_names: list[str] | None = None,
    dataset: str | None = None,
    docker_image: str | None = None,
    cpu_cores: float | None = None,
    memory_gb: float | None = None,
    disk_size_gb: float | None = None,
    timeout_minutes: int | None = None,
    agent_timeout_seconds: float | None = None,
    verifier_timeout_seconds: float | None = None,
    workdir: str | None = None,
    task_dir: str | None = None,
    scope: str | None = None,
    env: dict[str, object] | None = None,
) -> vf.Env:
    config = vf.EnvConfig(
        config,
        taskset=vf.HarborTasksetConfig(
            tasks=str(tasks) if tasks is not None else None,
            task_names=task_names,
            docker_image=docker_image,
            cpu_cores=cpu_cores,
            memory_gb=memory_gb,
            disk_size_gb=disk_size_gb,
            timeout_minutes=timeout_minutes,
            agent_timeout_seconds=agent_timeout_seconds,
            verifier_timeout_seconds=verifier_timeout_seconds,
            workdir=workdir,
            task_dir=task_dir,
            scope=scope,
            env=env,
        ),
    )
    return vf.Env(
        taskset=load_taskset(config=config.taskset, dataset=dataset),
        harness=load_harness(config=config.harness),
    )


def _dataset_task_names(
    dataset: str | None,
    tasks_root: object,
    task_names: list[str] | None,
) -> list[str] | None:
    if dataset is None:
        return task_names
    if task_names:
        raise ValueError("Cannot specify both 'dataset' and 'task_names'.")
    if dataset == "terminal-bench-sample":
        return TERMINAL_BENCH_SAMPLE_TASKS
    if dataset == "terminal-bench":
        return [
            path.name
            for path in sorted(Path(str(tasks_root)).iterdir())
            if path.is_dir()
        ]
    raise ValueError("dataset must be 'terminal-bench' or 'terminal-bench-sample'.")
