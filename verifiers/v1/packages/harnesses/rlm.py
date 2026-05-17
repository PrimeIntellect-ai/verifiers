import hashlib
import json
import random
import shlex
from collections.abc import Callable, Mapping
from importlib.abc import Traversable
from pathlib import Path
from typing import cast

from typing_extensions import Unpack

from verifiers.envs.experimental.utils.git_checkout_cache import (
    resolve_git_checkout,
    validate_git_checkout,
)

from ...config import SandboxConfig, sandbox_config_mapping
from ...harness import Harness
from ...state import State
from ...task import Task
from ...taskset import Taskset
from ...utils.prompt_utils import task_text
from .command import HarnessKwargs, command_program
from .configs import (
    RLM_DEFAULT_APPEND_TO_SYSTEM_PROMPT_PATH,
    RLM_DEFAULT_EXEC_TIMEOUT,
    RLM_DEFAULT_MAX_DEPTH,
    RLM_DEFAULT_MAX_TURNS,
    RLM_DEFAULT_REPO_REF,
    RLM_DEFAULT_REPO_URL,
    RLM_DEFAULT_INSTRUCTION_PATH,
    RLMConfig,
)
from ...types import ConfigMap, ProgramMap, ProgramOptionMap, ProgramValue

DEFAULT_RLM_REPO_URL = RLM_DEFAULT_REPO_URL
DEFAULT_RLM_REPO_REF = RLM_DEFAULT_REPO_REF
DEFAULT_RLM_MAX_TURNS = RLM_DEFAULT_MAX_TURNS
DEFAULT_RLM_EXEC_TIMEOUT = RLM_DEFAULT_EXEC_TIMEOUT
DEFAULT_RLM_MAX_DEPTH = RLM_DEFAULT_MAX_DEPTH
DEFAULT_RLM_INSTRUCTION_PATH = RLM_DEFAULT_INSTRUCTION_PATH
DEFAULT_APPEND_TO_SYSTEM_PROMPT_PATH = RLM_DEFAULT_APPEND_TO_SYSTEM_PROMPT_PATH
DEFAULT_RLM_CHECKOUT_PATH = "/tmp/rlm-checkout"
DEFAULT_RLM_SKILLS_PATH = "/rlm/skills"
DEFAULT_RLM_LOCAL_CHECKOUT_CACHE_ROOT = (
    Path.home() / ".cache" / "verifiers" / "rlm-checkouts"
)
REQUIRED_RLM_CHECKOUT_FILES = ("install.sh", "pyproject.toml")
ProgramDir = str | Path | Traversable


class RLM(Harness):
    config_type = RLMConfig

    def __init__(
        self,
        *,
        workdir: str | None = None,
        instruction_path: str | None = None,
        rlm_repo_url: str | None = None,
        rlm_repo_ref: str | None = None,
        rlm_max_turns: int | None = None,
        rlm_exec_timeout: int | None = None,
        rlm_max_depth: int | None = None,
        summarize_at_tokens: int | tuple[int, int] | list[int] | None = None,
        include_sub_rlm_trajectories: bool | None = None,
        append_to_system_prompt: str | None = None,
        local_checkout: str | Path | None = None,
        gh_token: str | None = None,
        rlm_tools: list[str] | None = None,
        env_vars: ProgramOptionMap | None = None,
        skills: str | Path | None = None,
        sandbox: bool | ConfigMap | SandboxConfig | None = None,
        program: ProgramMap | None = None,
        config: RLMConfig | None = None,
        **kwargs: Unpack[HarnessKwargs],
    ):
        harness_config = RLMConfig.from_config(
            config,
            workdir=workdir,
            instruction_path=instruction_path,
            rlm_repo_url=rlm_repo_url,
            rlm_repo_ref=rlm_repo_ref,
            rlm_max_turns=rlm_max_turns,
            rlm_exec_timeout=rlm_exec_timeout,
            rlm_max_depth=rlm_max_depth,
            summarize_at_tokens=summarize_at_tokens,
            include_sub_rlm_trajectories=include_sub_rlm_trajectories,
            append_to_system_prompt=append_to_system_prompt,
            local_checkout=local_checkout,
            gh_token=gh_token,
            rlm_tools=rlm_tools,
            env_vars=dict(env_vars) if env_vars is not None else None,
            skills=skills,
        )
        if (
            not harness_config.include_sub_rlm_trajectories
            and harness_config.keep_trajectory_step is None
        ):
            harness_config.keep_trajectory_step = keep_only_parent_rlm_steps
        summarize_resolver = build_summarize_resolver(
            harness_config.summarize_at_tokens
        )
        env: ProgramOptionMap = {
            "PATH": "/root/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
            "OPENAI_MODEL": "runtime.model",
            "RLM_MODEL": "runtime.model",
            "RLM_TOOLS": ",".join(harness_config.rlm_tools),
            "RLM_MAX_TURNS": str(harness_config.rlm_max_turns),
            "RLM_EXEC_TIMEOUT": str(harness_config.rlm_exec_timeout),
            "RLM_MAX_DEPTH": str(harness_config.rlm_max_depth),
            **harness_config.env_vars,
        }
        if summarize_resolver is not None:
            env["RLM_SUMMARIZE_AT_TOKENS"] = summarize_resolver
        sandbox_config: ConfigMap | SandboxConfig | bool
        sandbox_config = (
            harness_config.sandbox
            if sandbox is None and harness_config.sandbox is not None
            else True
        )
        if sandbox is not None:
            sandbox_config = sandbox
        if sandbox_config is True:
            sandbox_config = {
                "image": "python:3.11-slim",
                "workdir": harness_config.workdir,
                "cpu_cores": 1,
                "memory_gb": 2,
                "disk_size_gb": 5,
                "network_access": True,
                "timeout_minutes": 60,
                "command_timeout": max(harness_config.rlm_exec_timeout + 120, 600),
            }
        elif sandbox_config is not False:
            sandbox_config = {
                "workdir": harness_config.workdir,
                "command_timeout": max(harness_config.rlm_exec_timeout + 120, 600),
                **(sandbox_config_mapping(sandbox_config) or {}),
            }
        dirs: dict[str, ProgramValue] = {
            DEFAULT_RLM_CHECKOUT_PATH: rlm_checkout_loader(
                local_checkout=harness_config.local_checkout,
                rlm_repo_url=harness_config.rlm_repo_url,
                rlm_repo_ref=harness_config.rlm_repo_ref,
                gh_token=harness_config.gh_token,
            )
        }
        if harness_config.skills is not None:
            dirs[DEFAULT_RLM_SKILLS_PATH] = Path(harness_config.skills)
        self._explicit_skills = harness_config.skills is not None
        command = [
            "bash",
            "-lc",
            build_run_script(harness_config.instruction_path, harness_config.workdir),
        ]
        super().__init__(
            program=command_program(
                command=command,
                sandbox=sandbox_config,
                files={
                    harness_config.instruction_path: task_instruction_text,
                    DEFAULT_APPEND_TO_SYSTEM_PROMPT_PATH: (
                        harness_config.append_to_system_prompt
                    ),
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
                        "path": f"{harness_config.workdir}/.rlm/sessions/*/meta.json",
                        "format": "json",
                        "key": "metrics",
                        "optional": True,
                    }
                },
                program=program,
            ),
            sandbox=None if sandbox_config is False else sandbox_config,
            metrics=[
                rlm_sub_llm_call_count,
                rlm_sub_llm_total_turns,
                rlm_sub_llm_total_tool_calls,
            ],
            config=harness_config,
            **kwargs,
        )

    def attach_taskset(self, taskset: Taskset) -> None:
        if not self._explicit_skills:
            upload_dirs = taskset.get_upload_dirs()
            if not isinstance(upload_dirs, Mapping):
                raise TypeError("Taskset.get_upload_dirs() must return a mapping.")
            skills = upload_dirs.get("skills")
            self.set_program_dir(
                DEFAULT_RLM_SKILLS_PATH,
                cast(ProgramDir | None, skills),
            )
        super().attach_taskset(taskset)
        self._program = self.compile_program(self.program)

    def set_program_dir(
        self, remote_path: str, local_source: ProgramDir | None
    ) -> None:
        if not isinstance(self.program, Mapping):
            raise TypeError("RLM program must be a mapping.")
        program = dict(cast(ConfigMap, self.program))
        dirs = dict(cast(ConfigMap, program.get("dirs") or {}))
        if local_source is None:
            dirs.pop(remote_path, None)
        else:
            dirs[remote_path] = local_source
        program["dirs"] = dirs
        self.program = program


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
    rlm_repo_ref: str,
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
                ref=rlm_repo_ref,
                cache_root=DEFAULT_RLM_LOCAL_CHECKOUT_CACHE_ROOT,
                gh_token=gh_token,
                required_files=REQUIRED_RLM_CHECKOUT_FILES,
            )
        return checkout

    return load


def task_instruction_text(task: Task, state: State) -> str:
    return task_text(task, state, keys=("instruction", "question"))


def keep_only_parent_rlm_steps(step: object, state: State, headers: ConfigMap) -> bool:
    _ = step, state
    return str(headers.get("x-rlm-depth", "0")) == "0"


def rlm_metric(state: ConfigMap, key: str) -> float:
    artifacts = state.get("artifacts")
    if not isinstance(artifacts, Mapping):
        return 0.0
    artifacts = cast(ConfigMap, artifacts)
    metrics = artifacts.get("rlm_metrics")
    if not isinstance(metrics, Mapping):
        return 0.0
    metrics = cast(ConfigMap, metrics)
    value = metrics.get(key, 0.0)
    if isinstance(value, bool) or not isinstance(value, int | float | str):
        return 0.0
    return float(value or 0.0)


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


def draw_threshold(state: ConfigMap, lo: int, hi: int) -> int:
    prompt = json.dumps(state.get("prompt"), sort_keys=True, default=str)
    digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    return random.Random(int(digest[:16], 16)).randint(lo, hi)
