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
DEFAULT_RLM_EXEC_TIMEOUT = 300
DEFAULT_APPEND_TO_SYSTEM_PROMPT_PATH = "/task/append_to_system_prompt.txt"
DEFAULT_RLM_CHECKOUT_PATH = "/tmp/rlm-checkout"
DEFAULT_RLM_CHECKOUT_UPLOAD_NAME = "rlm_checkout"
DEFAULT_RLM_LOCAL_CHECKOUT_CACHE_ROOT = (
    Path.home() / ".cache" / "verifiers" / "rlm-checkouts"
)
_REQUIRED_CHECKOUT_FILES = ("install.sh", "pyproject.toml")

_GIT_SHIM_INSTALL_SCRIPT = """set -eu
git_path="$(command -v git 2>/dev/null || true)"
[ -z "$git_path" ] && exit 0
real_path="$(readlink -f "$git_path")"
[ -z "$real_path" ] || [ ! -e "$real_path" ] && exit 0
mv "$real_path" "$(dirname "$real_path")/twit"
cat > "$real_path" <<'EOF'
#!/bin/sh
echo "Bash command 'git' is not allowed. Please use a different command or tool." >&2
exit 1
EOF
chmod 0755 "$real_path"
"""

_GIT_SHIM_UNINSTALL_SCRIPT = """set -eu
git_path="$(command -v git 2>/dev/null || true)"
[ -z "$git_path" ] && exit 0
real_path="$(readlink -f "$git_path")"
twit_path="$(dirname "$real_path")/twit"
[ -e "$twit_path" ] && mv -f "$twit_path" "$real_path"
exit 0
"""


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

rlm "$(cat {instruction_path})"
"""
    return f"bash -lc {shlex.quote(script)}"


def rlm_harness(
    workdir: str = "/testbed",
    instruction_path: str = "/task/instruction.md",
    rlm_repo_url: str = DEFAULT_RLM_REPO_URL,
    rlm_ref: str = DEFAULT_RLM_REF,
    rlm_max_turns: int = DEFAULT_RLM_MAX_TURNS,
    rlm_exec_timeout: int = DEFAULT_RLM_EXEC_TIMEOUT,
    summarize_at_tokens: int | None = None,
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
    - ``rlm_exec_timeout`` → ``RLM_EXEC_TIMEOUT``
    - ``summarize_at_tokens`` → ``RLM_SUMMARIZE_AT_TOKENS``: when set to
      a positive int, rlm auto-compacts the current branch once the
      prompt_tokens of a turn reach the threshold. ``None`` disables
      auto-compaction.

    Callers do not need to — and should not — add these keys to
    ``ComposableEnv(environment_vars=...)`` themselves; pass the kwargs
    here and the harness owns the env var plumbing.

    With ``allow_git=False`` (default), the post-install step renames
    the real ``git`` to ``twit`` in the same directory and drops a
    refusal shim at git's resolved path. The pre-score hook reverses
    the rename before scoring.
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

    tool_names = list(rlm_tools) if rlm_tools is not None else ["ipython"]

    post_install_script: str | None = None
    pre_score_script: str | None = None
    if not allow_git:
        post_install_script = _GIT_SHIM_INSTALL_SCRIPT
        pre_score_script = _GIT_SHIM_UNINSTALL_SCRIPT

    environment_vars: dict[str, str] = {
        "RLM_TOOLS": ",".join(tool_names),
        "RLM_MAX_TURNS": str(rlm_max_turns),
        "RLM_EXEC_TIMEOUT": str(rlm_exec_timeout),
    }
    summarize_env = _format_summarize_at_tokens(summarize_at_tokens)
    if summarize_env is not None:
        environment_vars["RLM_SUMMARIZE_AT_TOKENS"] = summarize_env

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
        environment_vars=environment_vars,
        post_install_script=post_install_script,
        pre_score_script=pre_score_script,
    )


def _format_summarize_at_tokens(value: int | None) -> str | None:
    """Format ``summarize_at_tokens`` as the ``RLM_SUMMARIZE_AT_TOKENS`` string.

    Returns ``None`` when auto-compaction should be disabled (matches what
    the engine expects when the env var is absent). Rejects bad shapes
    here so configuration errors surface at harness-build time rather
    than deep inside the sandbox.
    """
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(
            f"summarize_at_tokens must be an int or None (got {type(value).__name__})"
        )
    if value <= 0:
        raise ValueError(f"summarize_at_tokens must be positive (got {value})")
    return str(value)
