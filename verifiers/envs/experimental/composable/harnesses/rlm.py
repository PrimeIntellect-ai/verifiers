"""RLM agent harness: install script, run command, and harness factory."""

from __future__ import annotations

from collections.abc import Iterable
from importlib.abc import Traversable
from pathlib import Path
import shlex

from verifiers.envs.experimental.composable import Harness

DEFAULT_RLM_REPO_URL = "github.com/PrimeIntellect-ai/rlm.git"
DEFAULT_RLM_BRANCH = "main"
DEFAULT_RLM_TOOLS = "bash,edit"
DEFAULT_RLM_MAX_TURNS = 100
DEFAULT_APPEND_TO_SYSTEM_PROMPT_PATH = "/task/append_to_system_prompt.txt"
DEFAULT_RLM_CHECKOUT_PATH = "/tmp/rlm-checkout"
DEFAULT_RLM_CHECKOUT_UPLOAD_NAME = "rlm_checkout"


def _validate_local_checkout(path: Path) -> Path:
    path = path.expanduser().resolve()
    if not path.is_dir():
        raise ValueError(f"RLM local checkout is not a directory: {path}")
    required = [path / "install.sh", path / "pyproject.toml"]
    missing = [item.name for item in required if not item.is_file()]
    if missing:
        missing_list = ", ".join(missing)
        raise ValueError(
            f"RLM local checkout is missing required files ({missing_list}): {path}"
        )
    return path


def _discover_local_checkout(search_dirs: Iterable[str | Path]) -> Path | None:
    for search_dir in search_dirs:
        candidate = Path(search_dir).expanduser().resolve() / "rlm"
        if not candidate.is_dir():
            continue
        try:
            return _validate_local_checkout(candidate)
        except ValueError:
            continue
    return None


def resolve_local_checkout(
    local_checkout: str | Path | None = None,
    *,
    prefer_local_checkout: bool = True,
    local_checkout_search_dirs: Iterable[str | Path] | None = None,
) -> Path | None:
    if local_checkout is not None:
        return _validate_local_checkout(Path(local_checkout))
    if prefer_local_checkout:
        return _discover_local_checkout(local_checkout_search_dirs or ())
    return None


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
    local_checkout: str | Path | None = None,
    prefer_local_checkout: bool = True,
    local_checkout_search_dirs: Iterable[str | Path] | None = None,
) -> Harness:
    resolved_local_checkout = resolve_local_checkout(
        local_checkout,
        prefer_local_checkout=prefer_local_checkout,
        local_checkout_search_dirs=local_checkout_search_dirs,
    )
    upload_dirs: dict[str, Traversable | Path] | None = None
    upload_dir_mapping: dict[str, str] | None = None
    if resolved_local_checkout is not None:
        upload_dirs = {
            DEFAULT_RLM_CHECKOUT_UPLOAD_NAME: resolved_local_checkout,
        }
        upload_dir_mapping = {
            DEFAULT_RLM_CHECKOUT_UPLOAD_NAME: DEFAULT_RLM_CHECKOUT_PATH,
        }
    return Harness(
        install_script=build_install_script(rlm_repo_url, rlm_branch),
        run_command=build_run_command(instruction_path, workdir),
        system_prompt=append_to_system_prompt,
        system_prompt_path=DEFAULT_APPEND_TO_SYSTEM_PROMPT_PATH,
        instruction_path=instruction_path,
        skills_path="/task/rlm-skills",
        upload_dirs=upload_dirs,
        upload_dir_mapping=upload_dir_mapping,
        metrics_path="{workdir}/.rlm/sessions/*/meta.json",
        metrics_key="metrics",
        metrics_prefix="rlm_",
    )
