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
DEFAULT_RLM_MAX_TURNS_IN_CONTEXT = -1
DEFAULT_RLM_EXEC_TIMEOUT = 300
DEFAULT_APPEND_TO_SYSTEM_PROMPT_PATH = "/task/append_to_system_prompt.txt"
DEFAULT_RLM_CHECKOUT_PATH = "/tmp/rlm-checkout"
DEFAULT_RLM_CHECKOUT_UPLOAD_NAME = "rlm_checkout"
DEFAULT_RLM_LOCAL_CHECKOUT_CACHE_ROOT = (
    Path.home() / ".cache" / "verifiers" / "rlm-checkouts"
)
_REQUIRED_CHECKOUT_FILES = ("install.sh", "pyproject.toml")

_GIT_SHIM_BODY = (
    "#!/bin/sh\n"
    "echo \"Bash command 'git' is not allowed. "
    'Please use a different command or tool." >&2\n'
    "exit 1\n"
)


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
    rlm_max_turns: int = DEFAULT_RLM_MAX_TURNS,
    rlm_max_turns_in_context: int = DEFAULT_RLM_MAX_TURNS_IN_CONTEXT,
    rlm_exec_timeout: int = DEFAULT_RLM_EXEC_TIMEOUT,
    append_to_system_prompt: str | None = None,
    local_checkout: str | Path | None = None,
    gh_token: str | None = None,
    rlm_tools: list[str] | None = None,
    allow_git: bool = False,
) -> Harness:
    """Build an RLM harness.

    The harness is the single source of truth for every ``RLM_*`` sandbox
    env var the RLM subprocess reads. Kwargs map 1:1 onto env vars written
    to ``Harness.environment_vars`` and merged into the sandbox by
    ``ComposableEnv`` (harness-wins):

    - ``rlm_tools`` → ``RLM_TOOLS`` (also drives ``Harness.tool_names`` so
      ``ToolMonitorRubric`` tracks exactly the active tools)
    - ``rlm_max_turns`` → ``RLM_MAX_TURNS``
    - ``rlm_max_turns_in_context`` → ``RLM_MAX_TURNS_IN_CONTEXT``
    - ``rlm_exec_timeout`` → ``RLM_EXEC_TIMEOUT``

    Callers do not need to — and should not — add these keys to
    ``ComposableEnv(environment_vars=...)`` themselves; pass the kwargs
    here and the harness owns the env var plumbing.

    ``allow_git`` defaults to False, mirroring opencode's bash tool. When
    False, a ``/usr/local/bin/git`` shim is uploaded that refuses on any
    invocation — this covers the RLM bash tool, the ipython tool's
    ``!cmd`` / ``%%bash`` cells, and any ``subprocess.run(["git", ...])``
    from inside ipython, since all three resolve via PATH and hit the
    shim first. Set ``allow_git=True`` for environments that genuinely
    need git.
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
            ),
        }
        resolved_upload_dirs = upload_dirs
        return resolved_upload_dirs

    tool_names = list(rlm_tools) if rlm_tools is not None else ["ipython", "summarize"]

    post_install_uploads: dict[str, str] | None = None
    post_install_script: str | None = None
    if not allow_git:
        post_install_uploads = {"/usr/local/bin/git": _GIT_SHIM_BODY}
        post_install_script = "chmod +x /usr/local/bin/git"

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
        environment_vars={
            "RLM_TOOLS": ",".join(tool_names),
            "RLM_MAX_TURNS": str(rlm_max_turns),
            "RLM_MAX_TURNS_IN_CONTEXT": str(rlm_max_turns_in_context),
            "RLM_EXEC_TIMEOUT": str(rlm_exec_timeout),
        },
        post_install_uploads=post_install_uploads,
        post_install_script=post_install_script,
    )
