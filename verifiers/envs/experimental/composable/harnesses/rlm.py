"""RLM agent harness: install script, run command, and harness factory."""

from __future__ import annotations

import hashlib
import os
from collections.abc import Iterator
from contextlib import contextmanager
import fcntl
from importlib.abc import Traversable
from pathlib import Path
import shlex
import shutil
import subprocess
import tempfile
from urllib.parse import quote, urlsplit, urlunsplit

from verifiers.envs.experimental.composable import Harness

DEFAULT_RLM_REPO_URL = "github.com/PrimeIntellect-ai/rlm.git"
DEFAULT_RLM_BRANCH = "main"
DEFAULT_RLM_TOOL_NAMES = ["ipython", "summarize"]
DEFAULT_RLM_MAX_TURNS = 100
DEFAULT_APPEND_TO_SYSTEM_PROMPT_PATH = "/task/append_to_system_prompt.txt"
DEFAULT_RLM_CHECKOUT_PATH = "/tmp/rlm-checkout"
DEFAULT_RLM_CHECKOUT_UPLOAD_NAME = "rlm_checkout"
DEFAULT_RLM_LOCAL_CHECKOUT_CACHE_ROOT = (
    Path.home() / ".cache" / "verifiers" / "rlm-checkouts"
)


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
    if not (path / ".git").exists():
        raise ValueError(
            f"RLM local checkout must be a git checkout or worktree: {path}"
        )
    return path


def _slugify_cache_component(text: str) -> str:
    slug = "".join(char.lower() if char.isalnum() else "-" for char in text)
    slug = slug.strip("-")
    return slug or "rlm"


def _default_local_checkout_cache_dir(
    rlm_repo_url: str,
    rlm_branch: str,
) -> Path:
    repo_name = rlm_repo_url.rstrip("/").rsplit("/", 1)[-1].removesuffix(".git")
    fingerprint = hashlib.sha256(f"{rlm_repo_url}\n{rlm_branch}".encode()).hexdigest()[
        :12
    ]
    cache_name = (
        f"{_slugify_cache_component(repo_name)}-"
        f"{_slugify_cache_component(rlm_branch)}-{fingerprint}"
    )
    return DEFAULT_RLM_LOCAL_CHECKOUT_CACHE_ROOT / cache_name


def _build_clone_url(
    rlm_repo_url: str,
    gh_token: str | None = None,
) -> str:
    if rlm_repo_url.startswith("github.com/"):
        token_prefix = f"{quote(gh_token, safe='')}@" if gh_token else ""
        return f"https://{token_prefix}{rlm_repo_url}"

    if gh_token and rlm_repo_url.startswith(
        ("https://github.com/", "http://github.com/")
    ):
        parsed = urlsplit(rlm_repo_url)
        if "@" not in parsed.netloc:
            return urlunsplit(
                (
                    parsed.scheme,
                    f"{quote(gh_token, safe='')}@{parsed.netloc}",
                    parsed.path,
                    parsed.query,
                    parsed.fragment,
                )
            )

    return rlm_repo_url


def _redact_clone_error_detail(detail: str, gh_token: str | None = None) -> str:
    if not gh_token:
        return detail
    redacted = detail.replace(gh_token, "<redacted>")
    quoted_token = quote(gh_token, safe="")
    if quoted_token != gh_token:
        redacted = redacted.replace(quoted_token, "<redacted>")
    return redacted


@contextmanager
def _checkout_lock(lock_path: Path) -> Iterator[None]:
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("w") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        yield


def _clone_checkout(
    target_dir: Path,
    *,
    rlm_repo_url: str,
    rlm_branch: str,
    gh_token: str | None = None,
) -> Path:
    target_dir.parent.mkdir(parents=True, exist_ok=True)
    clone_url = _build_clone_url(rlm_repo_url, gh_token)
    temp_parent = target_dir.parent
    temp_dir = Path(
        tempfile.mkdtemp(prefix=f".{target_dir.name}.tmp-", dir=temp_parent)
    )
    checkout_dir = temp_dir / "checkout"
    env = os.environ.copy()
    if gh_token:
        env.setdefault("GH_TOKEN", gh_token)
    try:
        subprocess.run(
            [
                "git",
                "clone",
                "--depth",
                "1",
                "--branch",
                rlm_branch,
                clone_url,
                str(checkout_dir),
            ],
            check=True,
            capture_output=True,
            text=True,
            env=env,
        )
    except FileNotFoundError as exc:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise RuntimeError(
            "git is required to populate the local RLM checkout cache"
        ) from exc
    except subprocess.CalledProcessError as exc:
        shutil.rmtree(temp_dir, ignore_errors=True)
        detail = exc.stderr.strip() or exc.stdout.strip() or str(exc)
        detail = _redact_clone_error_detail(detail, gh_token)
        raise RuntimeError(
            f"Failed to clone RLM checkout into host cache at {target_dir}: {detail}"
        ) from exc

    if target_dir.exists():
        if target_dir.is_dir():
            shutil.rmtree(target_dir)
        else:
            target_dir.unlink()
    checkout_dir.rename(target_dir)
    shutil.rmtree(temp_dir, ignore_errors=True)
    return _validate_local_checkout(target_dir)


def ensure_local_checkout(
    *,
    rlm_repo_url: str,
    rlm_branch: str,
    local_checkout: str | Path | None = None,
    gh_token: str | None = None,
) -> Path:
    checkout_dir = (
        Path(local_checkout).expanduser()
        if local_checkout is not None
        else _default_local_checkout_cache_dir(rlm_repo_url, rlm_branch)
    ).resolve()
    lock_path = checkout_dir.parent / f".{checkout_dir.name}.lock"
    with _checkout_lock(lock_path):
        try:
            return _validate_local_checkout(checkout_dir)
        except ValueError:
            return _clone_checkout(
                checkout_dir,
                rlm_repo_url=rlm_repo_url,
                rlm_branch=rlm_branch,
                gh_token=gh_token,
            )


def resolve_local_checkout(
    local_checkout: str | Path | None = None,
    *,
    rlm_repo_url: str = DEFAULT_RLM_REPO_URL,
    rlm_branch: str = DEFAULT_RLM_BRANCH,
    gh_token: str | None = None,
) -> Path:
    return ensure_local_checkout(
        rlm_repo_url=rlm_repo_url,
        rlm_branch=rlm_branch,
        local_checkout=local_checkout,
        gh_token=gh_token,
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
    gh_token: str | None = None,
) -> Harness:
    upload_dir_mapping: dict[str, str] = {
        DEFAULT_RLM_CHECKOUT_UPLOAD_NAME: DEFAULT_RLM_CHECKOUT_PATH,
    }

    def get_upload_dirs() -> dict[str, Traversable | Path]:
        resolved_local_checkout = resolve_local_checkout(
            local_checkout,
            rlm_repo_url=rlm_repo_url,
            rlm_branch=rlm_branch,
            gh_token=gh_token,
        )
        return {
            DEFAULT_RLM_CHECKOUT_UPLOAD_NAME: resolved_local_checkout,
        }

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
        tool_names=list(DEFAULT_RLM_TOOL_NAMES),
    )
