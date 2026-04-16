"""RLM agent harness: install script, run command, and harness factory."""

from __future__ import annotations

import shlex

from verifiers.envs.experimental.composable import Harness

DEFAULT_RLM_REPO_URL = "github.com/PrimeIntellect-ai/rlm.git"
DEFAULT_RLM_BRANCH = "main"
DEFAULT_RLM_TOOLS = "bash,edit"
DEFAULT_RLM_MAX_TURNS = 100
DEFAULT_APPEND_TO_SYSTEM_PROMPT_PATH = "/task/append_to_system_prompt.txt"


def build_install_script(
    rlm_repo_url: str = DEFAULT_RLM_REPO_URL,
    rlm_branch: str = DEFAULT_RLM_BRANCH,
) -> str:
    # Clone via git protocol instead of fetching install.sh from
    # raw.githubusercontent.com which has a 60 req/hr hard cap per IP.
    # rlm_repo_url is expected to be a bare github.com/org/repo.git path;
    # GH_TOKEN is injected at shell expansion time for private repos.
    return (
        "command -v git >/dev/null 2>&1 || { apt-get update -qq && apt-get install -y -qq git; }"
        f" && git clone --depth 1 --branch {rlm_branch}"
        f' "https://${{GH_TOKEN:+${{GH_TOKEN}}@}}{rlm_repo_url}" /tmp/rlm-checkout'
        f" && RLM_REPO_URL={rlm_repo_url}"
        f" RLM_REPO_BRANCH={rlm_branch}"
        " bash /tmp/rlm-checkout/install.sh"
    )


def build_run_command(
    instruction_path: str = "/task/instruction.md",
    workdir: str = "/testbed",
) -> str:
    script = f"""\
set -eo pipefail
export PATH="$HOME/.local/bin:$PATH"
export RLM_MODEL=$OPENAI_MODEL
export OPENAI_API_KEY=intercepted
export RLM_APPEND_TO_SYSTEM_PROMPT="$(cat {shlex.quote(DEFAULT_APPEND_TO_SYSTEM_PROMPT_PATH)} 2>/dev/null || true)"
cd "${{AGENT_WORKDIR:-{workdir}}}"

# If the sandbox has a .venv, run the ipython kernel inside it so the
# agent can inline-import project packages (numpy, pandas, etc.).
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

rlm "$(cat {instruction_path})"
"""
    return f"bash -lc {shlex.quote(script)}"


def rlm_harness(
    workdir: str = "/testbed",
    instruction_path: str = "/task/instruction.md",
    rlm_repo_url: str = DEFAULT_RLM_REPO_URL,
    rlm_branch: str = DEFAULT_RLM_BRANCH,
    append_to_system_prompt: str | None = None,
) -> Harness:
    return Harness(
        install_script=build_install_script(rlm_repo_url, rlm_branch),
        run_command=build_run_command(instruction_path, workdir),
        system_prompt=append_to_system_prompt,
        system_prompt_path=DEFAULT_APPEND_TO_SYSTEM_PROMPT_PATH,
        instruction_path=instruction_path,
        skills_path="/task/rlm-skills",
        metrics_path="{workdir}/.rlm/sessions/*/meta.json",
        metrics_key="metrics",
        metrics_prefix="rlm_",
    )
