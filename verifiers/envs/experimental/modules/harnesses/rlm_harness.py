from __future__ import annotations

import shlex
from collections.abc import Callable, Iterable
from pathlib import Path

from verifiers.envs.experimental.channels import SandboxSpec
from verifiers.envs.experimental.modules.harnesses.cli_harness import CliHarness
from verifiers.envs.experimental.utils.git_checkout_cache import (
    resolve_git_checkout,
    validate_git_checkout,
)
from verifiers.rubrics.rubric import Rubric
from verifiers.types import ClientType

DEFAULT_RLM_REPO_URL = "github.com/PrimeIntellect-ai/rlm.git"
DEFAULT_RLM_REF = "main"
DEFAULT_RLM_MAX_TURNS = 100
DEFAULT_RLM_MAX_TURNS_IN_CONTEXT = -1
DEFAULT_RLM_EXEC_TIMEOUT = 300
DEFAULT_APPEND_TO_SYSTEM_PROMPT_PATH = "/task/append_to_system_prompt.txt"
DEFAULT_RLM_CHECKOUT_PATH = "/tmp/rlm-checkout"
DEFAULT_RLM_CHECKOUT_UPLOAD_NAME = "rlm_checkout"
DEFAULT_RLM_LOCAL_CHECKOUT_CACHE_ROOT = (
    Path.home() / ".cache" / "verifiers" / "rlm-checkouts"
)
DEFAULT_RLM_TOOLS = ["ipython", "summarize"]
REQUIRED_CHECKOUT_FILES = ("install.sh", "pyproject.toml")

GIT_SHIM_BODY = (
    "#!/bin/sh\n"
    "echo \"Bash command 'git' is not allowed. "
    'Please use a different command or tool." >&2\n'
    "exit 1\n"
)


def resolve_local_checkout(
    local_checkout: str | Path | None = None,
    rlm_repo_url: str = DEFAULT_RLM_REPO_URL,
    rlm_ref: str = DEFAULT_RLM_REF,
    gh_token: str | None = None,
) -> Path:
    if local_checkout is not None:
        return validate_git_checkout(
            Path(local_checkout),
            required_files=REQUIRED_CHECKOUT_FILES,
        )
    return resolve_git_checkout(
        repo_url=rlm_repo_url,
        ref=rlm_ref,
        cache_root=DEFAULT_RLM_LOCAL_CHECKOUT_CACHE_ROOT,
        gh_token=gh_token,
        required_files=REQUIRED_CHECKOUT_FILES,
    )


def build_rlm_install_command() -> str:
    script = f"""\
set -eo pipefail
export RLM_CHECKOUT_PATH={shlex.quote(DEFAULT_RLM_CHECKOUT_PATH)}
test -f "$RLM_CHECKOUT_PATH/install.sh"
bash "$RLM_CHECKOUT_PATH/install.sh"
"""
    return f"bash -lc {shlex.quote(script)}"


def build_rlm_command(
    instruction_path: str = "/task/instruction.md",
    workdir: str = "/testbed",
) -> str:
    script = f"""\
set -eo pipefail
export PATH="$HOME/.local/bin:$PATH"
export RLM_MODEL=$OPENAI_MODEL
export OPENAI_API_KEY="${{OPENAI_API_KEY:-intercepted}}"
export RLM_APPEND_TO_SYSTEM_PROMPT="$(cat {shlex.quote(DEFAULT_APPEND_TO_SYSTEM_PROMPT_PATH)} 2>/dev/null || true)"
cd "${{AGENT_WORKDIR:-{workdir}}}"

if [ -x .venv/bin/python3 ]; then
    PYVER=$(.venv/bin/python3 -c "import sys; print(sys.version_info[:2] >= (3,10))" 2>/dev/null || true)
    if [ "$PYVER" = "True" ]; then
        IPYKERNEL="ipykernel"
    else
        IPYKERNEL="ipykernel<7"
    fi
    if .venv/bin/python3 -m pip install -q "$IPYKERNEL" nest_asyncio 2>/dev/null; then
        export RLM_KERNEL_PYTHON="$(pwd)/.venv/bin/python3"
    fi
fi

rlm "$(cat {shlex.quote(instruction_path)})"
"""
    return f"bash -lc {shlex.quote(script)}"


class RLMHarness(CliHarness):
    """RLM sandbox CLI harness with host-side checkout caching."""

    def __init__(
        self,
        workdir: str = "/testbed",
        instruction_path: str = "/task/instruction.md",
        rlm_repo_url: str = DEFAULT_RLM_REPO_URL,
        rlm_ref: str = DEFAULT_RLM_REF,
        rlm_max_turns: int = DEFAULT_RLM_MAX_TURNS,
        rlm_max_turns_in_context: int = DEFAULT_RLM_MAX_TURNS_IN_CONTEXT,
        rlm_exec_timeout: int = DEFAULT_RLM_EXEC_TIMEOUT,
        append_to_system_prompt: str | None = None,
        local_checkout: str | Path | None = None,
        gh_token: str | None = None,
        rlm_tools: list[str] | None = None,
        allow_git: bool = False,
        sandbox: SandboxSpec | None = None,
        install_timeout: int = 300,
        timeout_seconds: float = 3600.0,
        poll_interval: float = 1.0,
        environment_vars: dict[str, str] | None = None,
        keep_sandbox_for_scoring: bool = False,
        endpoint_port: int | None = None,
        endpoint_url: str | None = None,
        endpoint_secret: str | None = None,
        api_client_type: ClientType = "openai_chat_completions",
        rubric: Rubric | None = None,
        tools: Iterable[object] | None = None,
        max_turns: int = -1,
        parallel_model_requests: bool = True,
        error_formatter: Callable[[Exception], str] = str,
        stop_errors: list[type[Exception]] | None = None,
    ):
        self.rlm_repo_url = rlm_repo_url
        self.rlm_ref = rlm_ref
        self.local_checkout = local_checkout
        self.gh_token = gh_token
        tool_names = (
            list(rlm_tools) if rlm_tools is not None else list(DEFAULT_RLM_TOOLS)
        )
        rlm_environment_vars = {
            "RLM_TOOLS": ",".join(tool_names),
            "RLM_MAX_TURNS": str(rlm_max_turns),
            "RLM_MAX_TURNS_IN_CONTEXT": str(rlm_max_turns_in_context),
            "RLM_EXEC_TIMEOUT": str(rlm_exec_timeout),
        }
        if environment_vars:
            overlap = set(environment_vars) & set(rlm_environment_vars)
            if overlap:
                raise ValueError(
                    "Configure RLM environment variables through RLMHarness "
                    f"constructor args, not environment_vars: {sorted(overlap)}"
                )
            rlm_environment_vars = {**environment_vars, **rlm_environment_vars}

        post_install_uploads: dict[str, str] | None = None
        post_install_command: str | None = None
        if not allow_git:
            post_install_uploads = {"/usr/local/bin/git": GIT_SHIM_BODY}
            post_install_command = "chmod +x /usr/local/bin/git"

        super().__init__(
            command=build_rlm_command(instruction_path, workdir),
            instruction_path=instruction_path,
            system_prompt_path=DEFAULT_APPEND_TO_SYSTEM_PROMPT_PATH,
            agent_workdir=workdir,
            system_prompt=append_to_system_prompt,
            sandbox=sandbox,
            install_command=build_rlm_install_command(),
            install_timeout=install_timeout,
            post_install_uploads=post_install_uploads,
            post_install_command=post_install_command,
            skills_path="/task/rlm-skills",
            uploads={
                DEFAULT_RLM_CHECKOUT_UPLOAD_NAME: self.resolve_checkout,
            },
            upload_mapping={
                DEFAULT_RLM_CHECKOUT_UPLOAD_NAME: DEFAULT_RLM_CHECKOUT_PATH,
            },
            metrics_path="{workdir}/.rlm/sessions/*/meta.json",
            metrics_key="metrics",
            metrics_prefix="rlm_",
            tool_names=tool_names,
            timeout_seconds=timeout_seconds,
            poll_interval=poll_interval,
            environment_vars=rlm_environment_vars,
            keep_sandbox_for_scoring=keep_sandbox_for_scoring,
            endpoint_port=endpoint_port,
            endpoint_url=endpoint_url,
            endpoint_secret=endpoint_secret,
            api_client_type=api_client_type,
            rubric=rubric,
            tools=tools,
            max_turns=max_turns,
            parallel_model_requests=parallel_model_requests,
            error_formatter=error_formatter,
            stop_errors=stop_errors,
        )

    def resolve_checkout(self) -> Path:
        return resolve_local_checkout(
            local_checkout=self.local_checkout,
            rlm_repo_url=self.rlm_repo_url,
            rlm_ref=self.rlm_ref,
            gh_token=self.gh_token,
        )
