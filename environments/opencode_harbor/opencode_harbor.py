from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import cast

import verifiers.v1 as vf

_TASKS_DIR = Path(__file__).parent / "tasks"
_DEFAULT_DISABLED_TOOLS = ["webfetch", "question"]

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
    config: vf.HarborTasksetConfig | Mapping[str, object] | None = None,
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
    config = vf.HarborTasksetConfig.from_config(
        config,
        **_taskset_overrides(
            tasks=tasks,
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
    tasks_root = tasks if tasks is not None else config.tasks or _TASKS_DIR
    selected_task_names = _dataset_task_names(
        dataset=dataset,
        tasks_root=tasks_root,
        task_names=config.task_names,
    )
    return vf.HarborTaskset(
        tasks=cast(str | Path, tasks_root),
        task_names=selected_task_names,
        config=config,
    )


def load_harness(
    config: vf.OpenCodeConfig | Mapping[str, object] | None = None,
) -> vf.OpenCode:
    if _has_disabled_tools(config):
        return vf.OpenCode(config=config)
    return vf.OpenCode(config=config, disabled_tools=list(_DEFAULT_DISABLED_TOOLS))


def _has_disabled_tools(
    config: vf.OpenCodeConfig | Mapping[str, object] | None,
) -> bool:
    if isinstance(config, vf.OpenCodeConfig):
        return "disabled_tools" in config.model_fields_set
    return isinstance(config, Mapping) and "disabled_tools" in config


def load_environment(
    config: vf.EnvConfig | Mapping[str, object] | None = None,
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
    config = vf.EnvConfig.from_config(
        config,
        taskset=vf.HarborTasksetConfig.from_config(
            **_taskset_overrides(
                tasks=tasks,
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
        ),
    )
    return vf.Env(
        taskset=load_taskset(
            config=cast(
                vf.HarborTasksetConfig | Mapping[str, object] | None, config.taskset
            ),
            dataset=dataset,
        ),
        harness=load_harness(
            config=cast(vf.OpenCodeConfig | Mapping[str, object] | None, config.harness)
        ),
    )


def _taskset_overrides(
    *,
    tasks: str | Path | None = None,
    task_names: list[str] | None = None,
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
) -> dict[str, object]:
    overrides: dict[str, object] = {}
    if tasks is not None:
        overrides["tasks"] = str(tasks)
    if task_names is not None:
        overrides["task_names"] = task_names
    if docker_image is not None:
        overrides["docker_image"] = docker_image
    if cpu_cores is not None:
        overrides["cpu_cores"] = cpu_cores
    if memory_gb is not None:
        overrides["memory_gb"] = memory_gb
    if disk_size_gb is not None:
        overrides["disk_size_gb"] = disk_size_gb
    if timeout_minutes is not None:
        overrides["timeout_minutes"] = timeout_minutes
    if agent_timeout_seconds is not None:
        overrides["agent_timeout_seconds"] = agent_timeout_seconds
    if verifier_timeout_seconds is not None:
        overrides["verifier_timeout_seconds"] = verifier_timeout_seconds
    if workdir is not None:
        overrides["workdir"] = workdir
    if task_dir is not None:
        overrides["task_dir"] = task_dir
    if scope is not None:
        overrides["scope"] = scope
    if env is not None:
        overrides["env"] = env
    return overrides


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
