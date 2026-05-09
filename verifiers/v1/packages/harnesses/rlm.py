from __future__ import annotations

import hashlib
import json
import random
import shlex
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from verifiers.envs.experimental.utils.git_checkout_cache import (
    resolve_git_checkout,
    validate_git_checkout,
)

from ...config import HarnessConfig, SandboxConfig
from ...state import State
from ...task import Task
from ...utils.prompt_utils import task_text
from .cli import CLIHarness

DEFAULT_RLM_REPO_URL = "github.com/PrimeIntellect-ai/rlm-harness.git"
DEFAULT_RLM_REF = "main"
DEFAULT_RLM_MAX_TURNS = 100
DEFAULT_RLM_EXEC_TIMEOUT = 300
DEFAULT_RLM_MAX_DEPTH = 0
DEFAULT_RLM_INSTRUCTION_PATH = "/rlm/instruction.txt"
DEFAULT_APPEND_TO_SYSTEM_PROMPT_PATH = "/rlm/append_to_system_prompt.txt"
DEFAULT_RLM_CHECKOUT_PATH = "/tmp/rlm-checkout"
DEFAULT_RLM_SKILLS_PATH = "/rlm/skills"
DEFAULT_RLM_LOCAL_CHECKOUT_CACHE_ROOT = (
    Path.home() / ".cache" / "verifiers" / "rlm-checkouts"
)
REQUIRED_RLM_CHECKOUT_FILES = ("install.sh", "pyproject.toml")


class RLM(CLIHarness):
    def __init__(
        self,
        *,
        workdir: str = "/workspace",
        instruction_path: str = DEFAULT_RLM_INSTRUCTION_PATH,
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
        rlm_env: Mapping[str, object] | None = None,
        skills: str | Path | None = None,
        sandbox: bool | Mapping[str, object] | SandboxConfig = True,
        program: Mapping[str, object] | None = None,
        config: HarnessConfig | Mapping[str, object] | None = None,
        **kwargs: Any,
    ):
        harness_config = HarnessConfig.from_config(config)
        if (
            not include_sub_rlm_trajectories
            and harness_config.keep_trajectory_step is None
        ):
            harness_config.keep_trajectory_step = keep_only_parent_rlm_steps
        tool_names = list(rlm_tools) if rlm_tools is not None else ["ipython"]
        summarize_resolver = build_summarize_resolver(summarize_at_tokens)
        env: dict[str, object] = {
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
        sandbox_config: Mapping[str, object] | SandboxConfig | bool
        sandbox_config = sandbox
        if sandbox is True:
            sandbox_config = {
                "image": "python:3.11-slim",
                "workdir": workdir,
                "cpu_cores": 1,
                "memory_gb": 2,
                "disk_size_gb": 5,
                "network_access": True,
                "timeout_minutes": 60,
                "command_timeout": max(rlm_exec_timeout + 120, 600),
            }
        elif isinstance(sandbox, Mapping):
            sandbox_config = {
                "workdir": workdir,
                "command_timeout": max(rlm_exec_timeout + 120, 600),
                **dict(sandbox),
            }
        dirs: dict[str, object] = {
            DEFAULT_RLM_CHECKOUT_PATH: rlm_checkout_loader(
                local_checkout=local_checkout,
                rlm_repo_url=rlm_repo_url,
                rlm_ref=rlm_ref,
                gh_token=gh_token,
            )
        }
        if skills is not None:
            dirs[DEFAULT_RLM_SKILLS_PATH] = Path(skills)
        super().__init__(
            command=["bash", "-lc", build_run_script(instruction_path, workdir)],
            sandbox=sandbox_config,
            files={
                instruction_path: task_instruction_text,
                DEFAULT_APPEND_TO_SYSTEM_PROMPT_PATH: append_to_system_prompt,
            },
            dirs=dirs,
            setup=[
                "apt-get -o Acquire::Retries=3 update && "
                "apt-get -o Acquire::Retries=3 install -y --no-install-recommends "
                "ca-certificates curl git && rm -rf /var/lib/apt/lists/*",
                build_install_command(),
            ],
            env=env,
            artifacts={
                "rlm_metrics": {
                    "path": f"{workdir}/.rlm/sessions/*/meta.json",
                    "format": "json",
                    "key": "metrics",
                    "optional": True,
                }
            },
            program=program,
            metrics=[
                rlm_sub_llm_call_count,
                rlm_sub_llm_total_turns,
                rlm_sub_llm_total_tool_calls,
            ],
            config=harness_config,
            **kwargs,
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


def task_instruction_text(task: Task, state: State) -> str:
    return task_text(task, state, keys=("instruction", "question"))


def keep_only_parent_rlm_steps(
    step: object, state: State, headers: Mapping[str, object]
) -> bool:
    _ = step, state
    return str(headers.get("x-rlm-depth", "0")) == "0"


def rlm_metric(state: Mapping[str, Any], key: str) -> float:
    artifacts = state.get("artifacts")
    if not isinstance(artifacts, Mapping):
        return 0.0
    metrics = artifacts.get("rlm_metrics")
    if not isinstance(metrics, Mapping):
        return 0.0
    return float(metrics.get(key, 0.0) or 0.0)


async def rlm_sub_llm_call_count(task: Task, state: State) -> float:
    _ = task
    return rlm_metric(state, "sub_llm_call_count")


async def rlm_sub_llm_total_turns(task: Task, state: State) -> float:
    _ = task
    return rlm_metric(state, "sub_llm_total_turns")


async def rlm_sub_llm_total_tool_calls(task: Task, state: State) -> float:
    _ = task
    return rlm_metric(state, "sub_llm_total_tool_calls")


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

        def fixed_threshold(state: State) -> str:
            _ = state
            return str(value)

        return fixed_threshold
    if isinstance(value, (tuple, list)):
        if len(value) != 2:
            raise ValueError("summarize_at_tokens pair must have 2 elements")
        lo, hi = int(value[0]), int(value[1])
        if lo <= 0 or hi <= 0 or lo > hi:
            raise ValueError("summarize_at_tokens pair must satisfy 0 < lo <= hi")

        def sampled_threshold(state: State) -> str:
            return str(draw_threshold(state, lo, hi))

        return sampled_threshold
    raise ValueError("summarize_at_tokens must be int, (lo, hi), or None")


def draw_threshold(state: Mapping[str, Any], lo: int, hi: int) -> int:
    prompt = json.dumps(state.get("prompt"), sort_keys=True, default=str)
    digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    return random.Random(int(digest[:16], 16)).randint(lo, hi)
