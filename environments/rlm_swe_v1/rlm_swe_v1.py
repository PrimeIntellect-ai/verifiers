from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import verifiers.v1 as vf

_SKILLS_DIR = Path(__file__).parent / "skills"
_TASKS_DIR = Path(__file__).parent / "tasks"


def load_taskset(
    config: vf.TasksetConfig | None = None,
    tasks: str | Path | None = None,
    task_names: list[str] | None = None,
    cache_dir: str | Path | None = None,
    refresh: bool | None = None,
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
    if tasks is None and config.tasks is None and _TASKS_DIR.is_dir():
        tasks = _TASKS_DIR
    return vf.HarborTaskset(
        tasks=tasks,
        task_names=task_names,
        cache_dir=cache_dir,
        refresh=refresh,
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
    workdir: str = "/app",
    gh_token: str | None = None,
    rlm_tools: list[str] | None = None,
    skills: str | Path | None = _SKILLS_DIR if _SKILLS_DIR.is_dir() else None,
    **rlm_kwargs: Any,
) -> vf.RLM:
    token = gh_token or os.environ.get("GH_TOKEN")
    tools = rlm_tools if rlm_tools is not None else ["bash", "edit"]
    return vf.RLM(
        workdir=workdir,
        gh_token=token,
        rlm_tools=tools,
        skills=skills,
        config=config,
        **rlm_kwargs,
    )


def load_environment(
    config: vf.EnvConfig | None = None,
    tasks: str | Path | None = None,
    task_names: list[str] | None = None,
    cache_dir: str | Path | None = None,
    refresh: bool | None = None,
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
    instruction_path: str | None = None,
    rlm_repo_url: str | None = None,
    rlm_ref: str | None = None,
    rlm_max_turns: int | None = None,
    rlm_exec_timeout: int | None = None,
    rlm_max_depth: int | None = None,
    summarize_at_tokens: int | tuple[int, int] | list[int] | None = None,
    include_sub_rlm_trajectories: bool | None = None,
    append_to_system_prompt: str | None = None,
    local_checkout: str | Path | None = None,
    gh_token: str | None = None,
    rlm_tools: list[str] | None = None,
    rlm_env: dict[str, object] | None = None,
    skills: str | Path | None = _SKILLS_DIR if _SKILLS_DIR.is_dir() else None,
) -> vf.Env:
    config = vf.EnvConfig(
        config,
        taskset=vf.HarborTasksetConfig(
            tasks=tasks,
            task_names=task_names,
            cache_dir=str(cache_dir) if cache_dir is not None else None,
            refresh=refresh,
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
    taskset = load_taskset(config=config.taskset)
    rlm_kwargs = {
        key: value
        for key, value in {
            "instruction_path": instruction_path,
            "rlm_repo_url": rlm_repo_url,
            "rlm_ref": rlm_ref,
            "rlm_max_turns": rlm_max_turns,
            "rlm_exec_timeout": rlm_exec_timeout,
            "rlm_max_depth": rlm_max_depth,
            "summarize_at_tokens": summarize_at_tokens,
            "include_sub_rlm_trajectories": include_sub_rlm_trajectories,
            "append_to_system_prompt": append_to_system_prompt,
            "local_checkout": local_checkout,
            "rlm_env": rlm_env,
        }.items()
        if value is not None
    }
    harness = load_harness(
        config=config.harness,
        workdir=taskset.workdir,
        gh_token=gh_token,
        rlm_tools=rlm_tools,
        skills=skills,
        **rlm_kwargs,
    )
    return vf.Env(taskset=taskset, harness=harness)
