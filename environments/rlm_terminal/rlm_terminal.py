import os
from pathlib import Path

from verifiers.envs.experimental.composable import ComposableEnv
from verifiers.envs.experimental.composable.harnesses import rlm_harness
from verifiers.envs.experimental.composable.tasksets.harbor import (
    make_terminal_lego_taskset,
)

DEFAULT_APPEND_TO_SYSTEM_PROMPT = """You are solving a terminal task in /app.
Use the available shell and editor tools to inspect and modify files.
Leave the final answer as files or commands' effects in the workspace; do not only describe the solution.
"""


def load_environment(
    dataset_path: str | Path | None = None,
    task_names: list[str] | str | None = None,
    image_map_path: str | Path | None = None,
    exclusion_path: str | Path | None = None,
    hf_repo_id: str = "SWE-Lego/Terminal-Lego-15k",
    hf_revision: str | None = None,
    image_ref_field: str = "full_image_path",
    filter_fn: str | None = None,
    rlm_tools: list[str] | str | None = None,
    rlm_max_turns: int = 12,
    rlm_exec_timeout: int = 300,
    rlm_max_depth: int = 0,
    summarize_at_tokens: int | tuple[int, int] | list[int] | None = None,
    include_sub_rlm_trajectories: bool = False,
    append_to_system_prompt: str | None = DEFAULT_APPEND_TO_SYSTEM_PROMPT,
    rlm_repo_url: str = "github.com/PrimeIntellect-ai/rlm-harness.git",
    rlm_ref: str = "main",
    local_checkout: str | Path | None = None,
    gh_token: str | None = None,
    timeout_seconds: float = 1800.0,
    max_turns: int | None = None,
    keep_sandbox_for_scoring: bool = True,
    sandbox_wait_for_creation_max_attempts: int = 300,
    env_id: str = "rlm_terminal",
    **env_kwargs: object,
) -> ComposableEnv:
    """Load RLM on Terminal-Lego through experimental composable tasksets."""

    taskset = make_terminal_lego_taskset(
        dataset_path=dataset_path,
        task_names=task_names,
        image_map_path=image_map_path
        or _package_data_path("terminal-lego-task-image-map.jsonl"),
        exclusion_path=exclusion_path
        or _package_data_path("terminal-lego-excluded-task-ids.txt"),
        hf_repo_id=hf_repo_id,
        hf_revision=hf_revision,
        image_ref_field=image_ref_field,
        filter_fn=filter_fn,
    )

    tools = _normalize_csv(rlm_tools) or ["bash", "edit"]
    harness = rlm_harness(
        workdir="/app",
        instruction_path="/task/instruction.md",
        rlm_repo_url=rlm_repo_url,
        rlm_ref=rlm_ref,
        rlm_max_turns=rlm_max_turns,
        rlm_exec_timeout=rlm_exec_timeout,
        rlm_max_depth=rlm_max_depth,
        summarize_at_tokens=summarize_at_tokens,
        include_sub_rlm_trajectories=include_sub_rlm_trajectories,
        append_to_system_prompt=append_to_system_prompt,
        local_checkout=local_checkout,
        gh_token=gh_token or os.environ.get("GH_TOKEN"),
        rlm_tools=tools,
    )

    return ComposableEnv(
        taskset=taskset,
        harness=harness,
        timeout_seconds=timeout_seconds,
        max_turns=max_turns if max_turns is not None else rlm_max_turns + 2,
        keep_sandbox_for_scoring=keep_sandbox_for_scoring,
        sandbox_wait_for_creation_max_attempts=sandbox_wait_for_creation_max_attempts,
        env_id=env_id,
        **env_kwargs,
    )


def _package_data_path(filename: str) -> Path:
    return Path(__file__).resolve().parent / "data" / filename


def _normalize_csv(value: list[str] | str | None) -> list[str] | None:
    if value is None:
        return None
    if isinstance(value, str):
        items = value.split(",")
    else:
        items = value
    normalized = [str(item).strip() for item in items]
    return [item for item in normalized if item]
