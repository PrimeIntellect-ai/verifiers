from __future__ import annotations

import hashlib
import json
import random
import shlex
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import verifiers.v1 as vf
from verifiers.envs.experimental.utils.git_checkout_cache import (
    resolve_git_checkout,
    validate_git_checkout,
)

DEFAULT_RLM_REPO_URL = "github.com/PrimeIntellect-ai/rlm-harness.git"
DEFAULT_RLM_REF = "main"
DEFAULT_RLM_MAX_TURNS = 100
DEFAULT_RLM_EXEC_TIMEOUT = 300
DEFAULT_RLM_MAX_DEPTH = 0
DEFAULT_APPEND_TO_SYSTEM_PROMPT_PATH = "/task/append_to_system_prompt.txt"
DEFAULT_RLM_CHECKOUT_PATH = "/tmp/rlm-checkout"
DEFAULT_RLM_LOCAL_CHECKOUT_CACHE_ROOT = (
    Path.home() / ".cache" / "verifiers" / "rlm-checkouts"
)
REQUIRED_RLM_CHECKOUT_FILES = ("install.sh", "pyproject.toml")


@vf.reward(weight=1.0)
async def exact_answer(task, state) -> float:
    stdout = str(state.get("command", {}).get("stdout") or "")
    return float(str(task["answer"]).lower() in stdout.lower())


@vf.metric
async def rlm_sub_llm_call_count(task, state) -> float:
    return float(rlm_metric(state, "sub_llm_call_count"))


@vf.metric
async def rlm_sub_llm_total_turns(task, state) -> float:
    return float(rlm_metric(state, "sub_llm_total_turns"))


@vf.metric
async def rlm_sub_llm_total_tool_calls(task, state) -> float:
    return float(rlm_metric(state, "sub_llm_total_tool_calls"))


def source():
    return [
        {
            "question": "Reply with exactly hello rlm.",
            "answer": "hello rlm",
        },
        {
            "question": "Reply with exactly taskset harness.",
            "answer": "taskset harness",
        },
        {
            "question": "Reply with exactly runtime boundary.",
            "answer": "runtime boundary",
        },
        {
            "question": "Reply with exactly sandbox lease.",
            "answer": "sandbox lease",
        },
        {
            "question": "Reply with exactly toolset scope.",
            "answer": "toolset scope",
        },
        {
            "question": "Reply with exactly group reward.",
            "answer": "group reward",
        },
        {
            "question": "Reply with exactly endpoint proxy.",
            "answer": "endpoint proxy",
        },
        {
            "question": "Reply with exactly cleanup signal.",
            "answer": "cleanup signal",
        },
        {
            "question": "Reply with exactly harbor task.",
            "answer": "harbor task",
        },
        {
            "question": "Reply with exactly recursive model.",
            "answer": "recursive model",
        },
    ]


def load_taskset(config: vf.TasksetConfig | None = None):
    return vf.Taskset(
        source=source,
        rewards=[exact_answer],
        config=config,
    )


def load_harness(
    config: vf.HarnessConfig | None = None,
    workdir: str = "/workspace",
    instruction_path: str = "/task/instruction.md",
    rlm_repo_url: str = DEFAULT_RLM_REPO_URL,
    rlm_ref: str = DEFAULT_RLM_REF,
    rlm_max_turns: int = DEFAULT_RLM_MAX_TURNS,
    rlm_exec_timeout: int = DEFAULT_RLM_EXEC_TIMEOUT,
    rlm_max_depth: int = DEFAULT_RLM_MAX_DEPTH,
    summarize_at_tokens: int | tuple[int, int] | list[int] | None = None,
    include_sub_rlm_trajectories: bool = False,
    append_to_system_prompt: str = "",
    local_checkout: str | Path | None = None,
    gh_token: str | None = None,
    rlm_tools: list[str] | None = None,
    rlm_env: Mapping[str, str] | None = None,
):
    harness_config = vf.HarnessConfig(config)
    if not include_sub_rlm_trajectories:
        harness_config.keep_trajectory_step = keep_only_parent_rlm_steps
    tool_names = list(rlm_tools) if rlm_tools is not None else ["ipython"]
    summarize_resolver = build_summarize_resolver(summarize_at_tokens)
    env = {
        "PATH": "/root/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
        "OPENAI_MODEL": "runtime.model",
        "RLM_MODEL": "runtime.model",
        "RLM_TOOLS": ",".join(tool_names),
        "RLM_MAX_TURNS": str(rlm_max_turns),
        "RLM_EXEC_TIMEOUT": str(rlm_exec_timeout),
        "RLM_MAX_DEPTH": str(rlm_max_depth),
        **dict(rlm_env or {}),
    }
    if summarize_resolver is not None:
        env["RLM_SUMMARIZE_AT_TOKENS"] = summarize_resolver

    return vf.Harness(
        sandbox={
            "image": "python:3.11-slim",
            "workdir": workdir,
            "cpu_cores": 1,
            "memory_gb": 2,
            "disk_size_gb": 5,
            "network_access": True,
            "timeout_minutes": 60,
            "command_timeout": max(rlm_exec_timeout + 120, 600),
        },
        program={
            "sandbox": True,
            "dirs": {
                DEFAULT_RLM_CHECKOUT_PATH: rlm_checkout_loader(
                    local_checkout=local_checkout,
                    rlm_repo_url=rlm_repo_url,
                    rlm_ref=rlm_ref,
                    gh_token=gh_token,
                )
            },
            "files": {
                instruction_path: "task.question",
                DEFAULT_APPEND_TO_SYSTEM_PROMPT_PATH: append_to_system_prompt,
            },
            "setup": [
                "apt-get update && apt-get install -y --no-install-recommends "
                "ca-certificates curl git && rm -rf /var/lib/apt/lists/*",
                build_install_command(),
            ],
            "command": ["bash", "-lc", build_run_script(instruction_path, workdir)],
            "env": env,
            "artifacts": {
                "rlm_metrics": {
                    "path": f"{workdir}/.rlm/sessions/*/meta.json",
                    "format": "json",
                    "key": "metrics",
                }
            },
        },
        metrics=[
            rlm_sub_llm_call_count,
            rlm_sub_llm_total_turns,
            rlm_sub_llm_total_tool_calls,
        ],
        config=harness_config,
    )


def load_environment(config: vf.EnvConfig | None = None):
    config = config or vf.EnvConfig()
    return vf.Env(
        taskset=load_taskset(config=config.taskset),
        harness=load_harness(config=config.harness),
    )


def build_install_command() -> str:
    script = f"""
set -eo pipefail
export RLM_CHECKOUT_PATH={shlex.quote(DEFAULT_RLM_CHECKOUT_PATH)}
test -f "$RLM_CHECKOUT_PATH/install.sh"
bash "$RLM_CHECKOUT_PATH/install.sh"
"""
    return f"bash -lc {shlex.quote(script)}"


def build_run_script(instruction_path: str, workdir: str) -> str:
    return f"""
set -eo pipefail
export PATH="$HOME/.local/bin:$PATH"
export RLM_MODEL="${{RLM_MODEL:-$OPENAI_MODEL}}"
export OPENAI_API_KEY="${{OPENAI_API_KEY:-intercepted}}"
export RLM_APPEND_TO_SYSTEM_PROMPT="$(cat {shlex.quote(DEFAULT_APPEND_TO_SYSTEM_PROMPT_PATH)} 2>/dev/null || true)"
cd "${{AGENT_WORKDIR:-{workdir}}}"
rlm "$(cat {shlex.quote(instruction_path)})"
"""


def rlm_checkout_loader(
    local_checkout: str | Path | None,
    rlm_repo_url: str,
    rlm_ref: str,
    gh_token: str | None,
) -> Callable[[], Path]:
    checkout: Path | None = None

    def load() -> Path:
        nonlocal checkout
        if checkout is not None:
            return checkout
        if local_checkout is not None:
            checkout = validate_git_checkout(
                Path(local_checkout),
                required_files=REQUIRED_RLM_CHECKOUT_FILES,
            )
        else:
            checkout = resolve_git_checkout(
                repo_url=rlm_repo_url,
                ref=rlm_ref,
                cache_root=DEFAULT_RLM_LOCAL_CHECKOUT_CACHE_ROOT,
                gh_token=gh_token,
                required_files=REQUIRED_RLM_CHECKOUT_FILES,
            )
        return checkout

    return load


def keep_only_parent_rlm_steps(step, state, headers) -> bool:
    return str(headers.get("x-rlm-depth", "0")) == "0"


def rlm_metric(state: Mapping[str, Any], key: str) -> float:
    artifacts = state.get("artifacts")
    if not isinstance(artifacts, Mapping):
        return 0.0
    metrics = artifacts.get("rlm_metrics")
    if not isinstance(metrics, Mapping):
        return 0.0
    return float(metrics.get(key, 0.0) or 0.0)


def build_summarize_resolver(
    value: int | tuple[int, int] | list[int] | None,
) -> Callable[..., str | None] | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError("summarize_at_tokens must be an int or (lo, hi) pair")
    if isinstance(value, int):
        if value <= 0:
            raise ValueError("summarize_at_tokens must be positive")

        def fixed_threshold(state):
            _ = state
            return str(value)

        return fixed_threshold
    if isinstance(value, (tuple, list)):
        if len(value) != 2:
            raise ValueError("summarize_at_tokens pair must have 2 elements")
        lo, hi = int(value[0]), int(value[1])
        if lo <= 0 or hi <= 0 or lo > hi:
            raise ValueError("summarize_at_tokens pair must satisfy 0 < lo <= hi")

        def sampled_threshold(state):
            return str(draw_threshold(state, lo, hi))

        return sampled_threshold
    raise ValueError("summarize_at_tokens must be int, (lo, hi), or None")


def draw_threshold(state: Mapping[str, Any], lo: int, hi: int) -> int:
    prompt = json.dumps(state.get("prompt"), sort_keys=True, default=str)
    digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    return random.Random(int(digest[:16], 16)).randint(lo, hi)
