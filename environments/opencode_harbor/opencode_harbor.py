from __future__ import annotations

from pathlib import Path

import verifiers.v1 as vf

_TASKS_DIR = Path(__file__).parent / "tasks"
_SYSTEM_PROMPT_PATH = Path(__file__).parent / "prompt.txt"

DEFAULT_DISABLED_TOOLS = ("webfetch", "question")
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


class OpenCodeHarborHarnessConfig(vf.HarnessConfig):
    agent_workdir: str = "/app"
    system_prompt_path: str | None = None
    disabled_tools: list[str] | None = None
    max_turns: int = 4


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
    config = vf.HarborTasksetConfig(config)
    tasks_root = tasks if tasks is not None else config.tasks or _TASKS_DIR
    selected_task_names = task_names if task_names is not None else config.task_names
    selected_task_names = _dataset_task_names(
        dataset=dataset,
        tasks_root=tasks_root,
        task_names=selected_task_names,
    )
    return vf.HarborTaskset(
        tasks=tasks_root,
        task_names=selected_task_names,
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
        config=config,
    )


def load_harness(
    config: vf.HarnessConfig | None = None,
    agent_workdir: str | None = None,
    system_prompt_path: str | Path | None = None,
    disabled_tools: list[str] | tuple[str, ...] | None = None,
    max_turns: int | None = None,
) -> vf.OpenCode:
    config = OpenCodeHarborHarnessConfig(
        config,
        agent_workdir=agent_workdir,
        system_prompt_path=str(system_prompt_path)
        if system_prompt_path is not None
        else None,
        disabled_tools=list(disabled_tools) if disabled_tools is not None else None,
        max_turns=max_turns,
    )
    system_prompt = config.system_prompt
    if system_prompt is None and config.system_prompt_path is not None:
        system_prompt = Path(config.system_prompt_path).read_text()
    return vf.OpenCode(
        agent_workdir=config.agent_workdir,
        system_prompt=system_prompt,
        disabled_tools=config.disabled_tools,
        max_turns=config.max_turns,
        config=config,
    )


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
    agent_workdir: str | None = None,
    system_prompt_path: str | Path | None = _SYSTEM_PROMPT_PATH,
    disabled_tools: list[str] | tuple[str, ...] | None = DEFAULT_DISABLED_TOOLS,
    max_turns: int | None = 4,
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
        harness=OpenCodeHarborHarnessConfig(
            agent_workdir=agent_workdir,
            system_prompt_path=str(system_prompt_path)
            if system_prompt_path is not None
            else None,
            disabled_tools=list(disabled_tools) if disabled_tools is not None else None,
            max_turns=max_turns,
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
