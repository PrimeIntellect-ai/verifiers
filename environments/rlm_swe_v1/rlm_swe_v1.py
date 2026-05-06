from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import verifiers.v1 as vf

_SKILLS_DIR = Path(__file__).parent / "skills"


def load_taskset(
    config=None,
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
    config=None,
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


def load_environment(config=None, **kwargs: Any) -> vf.Env:
    config_obj = config or {}
    taskset_config = getattr(config_obj, "taskset", None) or (
        config_obj.get("taskset") if isinstance(config_obj, dict) else None
    )
    harness_config = getattr(config_obj, "harness", None) or (
        config_obj.get("harness") if isinstance(config_obj, dict) else None
    )
    taskset = load_taskset(taskset_config, **taskset_kwargs(kwargs))
    harness = load_harness(harness_config, **harness_kwargs(kwargs, taskset))
    return vf.Env(taskset=taskset, harness=harness)


def taskset_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    keys = {
        "tasks",
        "task_names",
        "cache_dir",
        "refresh",
        "docker_image",
        "cpu_cores",
        "memory_gb",
        "disk_size_gb",
        "timeout_minutes",
        "agent_timeout_seconds",
        "verifier_timeout_seconds",
        "workdir",
        "task_dir",
        "scope",
        "env",
    }
    return {key: kwargs[key] for key in keys if key in kwargs}


def harness_kwargs(kwargs: dict[str, Any], taskset: vf.HarborTaskset) -> dict[str, Any]:
    harness_keys = {
        "instruction_path",
        "rlm_repo_url",
        "rlm_ref",
        "rlm_max_turns",
        "rlm_exec_timeout",
        "rlm_max_depth",
        "summarize_at_tokens",
        "include_sub_rlm_trajectories",
        "append_to_system_prompt",
        "local_checkout",
        "gh_token",
        "rlm_tools",
        "rlm_env",
        "skills",
    }
    values = {key: kwargs[key] for key in harness_keys if key in kwargs}
    values.setdefault("workdir", taskset.workdir)
    return values
