"""RLM agent harness: install script, run command, and harness factory."""

from __future__ import annotations

import shlex

from verifiers.envs.experimental.composable import Harness

DEFAULT_RLM_REPO_URL = "github.com/PrimeIntellect-ai/rlm.git"
DEFAULT_RLM_TOOLS = "bash,edit"
DEFAULT_RLM_MAX_TURNS = 100
DEFAULT_APPEND_TO_SYSTEM_PROMPT_PATH = "/task/append_to_system_prompt.txt"


def build_install_script(rlm_repo_url: str = DEFAULT_RLM_REPO_URL) -> str:
    return (
        "set -e; "
        'command -v uv >/dev/null 2>&1 || { curl -LsSf https://astral.sh/uv/install.sh | sh; source "$HOME/.local/bin/env"; }; '
        f'uv tool install --python 3.11 "rlm @ git+https://${{GH_TOKEN}}@{rlm_repo_url}"'
    )


def build_run_command(
    instruction_path: str = "/task/instruction.md",
    workdir: str = "/testbed",
) -> str:
    script = f"""\
set -eo pipefail
export RLM_MODEL=$OPENAI_MODEL
export OPENAI_API_KEY=intercepted
export RLM_APPEND_TO_SYSTEM_PROMPT="$(cat {shlex.quote(DEFAULT_APPEND_TO_SYSTEM_PROMPT_PATH)} 2>/dev/null || true)"
cd {workdir}
rlm "$(cat {instruction_path})"
"""
    return f"bash -lc {shlex.quote(script)}"


def rlm_harness(
    workdir: str = "/testbed",
    instruction_path: str = "/task/instruction.md",
    rlm_repo_url: str = DEFAULT_RLM_REPO_URL,
    append_to_system_prompt: str | None = None,
) -> Harness:
    return Harness(
        install_script=build_install_script(rlm_repo_url),
        run_command=build_run_command(instruction_path, workdir),
        system_prompt=append_to_system_prompt,
        system_prompt_path=DEFAULT_APPEND_TO_SYSTEM_PROMPT_PATH,
        instruction_path=instruction_path,
    )
