"""RLM agent harness: install script, run command, and harness factory."""

from __future__ import annotations

from importlib.abc import Traversable
from pathlib import Path
import shlex

from verifiers.envs.experimental.composable import Harness
from verifiers.envs.experimental.utils.git_checkout_cache import (
    resolve_git_checkout,
    validate_git_checkout,
)

DEFAULT_RLM_REPO_URL = "github.com/PrimeIntellect-ai/rlm.git"
DEFAULT_RLM_REF = "main"
DEFAULT_RLM_MAX_TURNS = 100
DEFAULT_APPEND_TO_SYSTEM_PROMPT_PATH = "/task/append_to_system_prompt.txt"
DEFAULT_RLM_CHECKOUT_PATH = "/tmp/rlm-checkout"
DEFAULT_RLM_CHECKOUT_UPLOAD_NAME = "rlm_checkout"
DEFAULT_RLM_LOCAL_CHECKOUT_CACHE_ROOT = (
    Path.home() / ".cache" / "verifiers" / "rlm-checkouts"
)
_REQUIRED_CHECKOUT_FILES = ("install.sh", "pyproject.toml")


def resolve_local_checkout(
    local_checkout: str | Path | None = None,
    *,
    rlm_repo_url: str = DEFAULT_RLM_REPO_URL,
    rlm_ref: str = DEFAULT_RLM_REF,
    gh_token: str | None = None,
) -> Path:
    if local_checkout is not None:
        return validate_git_checkout(
            Path(local_checkout),
            required_files=_REQUIRED_CHECKOUT_FILES,
        )
    return resolve_git_checkout(
        repo_url=rlm_repo_url,
        ref=rlm_ref,
        cache_root=DEFAULT_RLM_LOCAL_CHECKOUT_CACHE_ROOT,
        gh_token=gh_token,
        required_files=_REQUIRED_CHECKOUT_FILES,
    )


def build_install_script() -> str:
    script = f"""\
set -eo pipefail
export RLM_CHECKOUT_PATH={shlex.quote(DEFAULT_RLM_CHECKOUT_PATH)}
test -f "$RLM_CHECKOUT_PATH/install.sh"
bash "$RLM_CHECKOUT_PATH/install.sh"
"""
    return f"bash -lc {shlex.quote(script)}"


def build_run_command(
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
    rlm_ref: str = DEFAULT_RLM_REF,
    append_to_system_prompt: str | None = None,
    local_checkout: str | Path | None = None,
    gh_token: str | None = None,
    rlm_tools: list[str] | None = None,
) -> Harness:
    """Build an RLM harness.

    ``rlm_tools`` is the single source of truth for which builtin tools are
    active. The same list drives both ``Harness.tool_names`` (so
    ``ToolMonitorRubric`` tracks exactly the active tools) and
    ``Harness.environment_vars["RLM_TOOLS"]`` (so the RLM sandbox advertises
    the same set to the model). Callers do not need to — and should not —
    add ``RLM_TOOLS`` to ``ComposableEnv(environment_vars=...)`` themselves;
    the harness owns it.
    """
    upload_dir_mapping: dict[str, str] = {
        DEFAULT_RLM_CHECKOUT_UPLOAD_NAME: DEFAULT_RLM_CHECKOUT_PATH,
    }
    resolved_upload_dirs: dict[str, Traversable | Path] | None = None

    def get_upload_dirs() -> dict[str, Traversable | Path]:
        nonlocal resolved_upload_dirs
        if resolved_upload_dirs is not None:
            return resolved_upload_dirs
        upload_dirs: dict[str, Traversable | Path] = {
            DEFAULT_RLM_CHECKOUT_UPLOAD_NAME: resolve_local_checkout(
                local_checkout,
                rlm_repo_url=rlm_repo_url,
                rlm_ref=rlm_ref,
                gh_token=gh_token,
            )
        }
        resolved_upload_dirs = upload_dirs
        return resolved_upload_dirs

    tool_names = list(rlm_tools) if rlm_tools is not None else ["ipython", "summarize"]
    return Harness(
        install_script=build_install_script(),
        run_command=build_run_command(instruction_path, workdir),
        system_prompt=append_to_system_prompt,
        system_prompt_path=DEFAULT_APPEND_TO_SYSTEM_PROMPT_PATH,
        instruction_path=instruction_path,
        skills_path="/task/rlm-skills",
        get_upload_dirs=get_upload_dirs,
        upload_dir_mapping=upload_dir_mapping,
        metrics_path="{workdir}/.rlm/sessions/*/meta.json",
        metrics_key="metrics",
        metrics_prefix="rlm_",
        tool_names=tool_names,
        environment_vars={"RLM_TOOLS": ",".join(tool_names)},
    )
