"""Terminal-Lego taskset built on the composable Harbor task layout."""

import contextlib
import logging
import math
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Iterator

from verifiers.envs.experimental.composable import SandboxSpec
from verifiers.envs.experimental.composable.tasksets.harbor.harbor import (
    HarborDatasetTaskSet,
    _load_task_entry,
)

logger = logging.getLogger(__name__)

DEFAULT_HF_REPO_ID = "PrimeIntellect/Terminal-Lego-15k"
DEFAULT_WORKDIR = "/app"
DATASET_PATH_ENV = "TERMINAL_LEGO_DATASET_PATH"


class TerminalLegoTaskSet(HarborDatasetTaskSet):
    """Terminal-Lego tasks with prebuilt per-task images from our registry.

    Terminal-Lego ships as a Hugging Face dataset repository whose rows are
    Terminal-Bench/Harbor-style task directories. The Docker build context is
    each task's ``environment/`` directory; this taskset expects a local Git
    LFS/Xet checkout, cloned automatically if needed, and prebuilt per-task
    image refs in ``task.toml``.
    """

    def __init__(
        self,
        dataset_path: str | Path | None = None,
        task_names: list[str] | None = None,
        hf_repo_id: str = DEFAULT_HF_REPO_ID,
        hf_revision: str | None = None,
        filter_fn: str | None = None,
    ):
        self.hf_repo_id = hf_repo_id
        self.hf_revision = hf_revision

        resolved_dataset_path = _resolve_dataset_path(
            dataset_path=dataset_path,
            hf_repo_id=hf_repo_id,
            hf_revision=hf_revision,
        )
        super().__init__(
            dataset_path=resolved_dataset_path,
            task_names=task_names,
            filter_fn=filter_fn,
        )
        self.name = "terminal-lego"

    def _build_dataset(self) -> Any:
        from datasets import Dataset

        requested = set(self.task_names or [])
        seen: set[str] = set()
        tasks: list[dict] = []

        for task_dir in sorted(self.dataset_path.iterdir()):
            if not task_dir.is_dir() or not task_dir.name.startswith("task_"):
                continue
            task_name = task_dir.name
            if requested and task_name not in requested:
                continue
            seen.add(task_name)

            missing = _missing_required_files(task_dir)
            if missing:
                if requested:
                    missing_str = ", ".join(missing)
                    raise ValueError(
                        f"Terminal-Lego task {task_name!r} is incomplete; "
                        f"missing {missing_str}"
                    )
                logger.warning(
                    "Skipping %s: missing required files: %s",
                    task_name,
                    ", ".join(missing),
                )
                continue

            entry = _load_task_entry(task_dir, len(tasks))
            config = entry["info"]["config"]
            image_ref = config.get("environment", {}).get("docker_image")
            if not image_ref:
                if requested:
                    raise ValueError(
                        f"Terminal-Lego task {task_name!r} is missing "
                        "[environment].docker_image in task.toml"
                    )
                logger.warning(
                    "Skipping %s: missing [environment].docker_image in task.toml",
                    task_name,
                )
                continue
            entry["info"].update(
                {
                    "docker_image": image_ref,
                    "terminal_lego_source": {
                        "hf_repo_id": self.hf_repo_id,
                        "hf_revision": self.hf_revision,
                    },
                    "config": config,
                }
            )
            tasks.append(entry)

        if requested:
            missing_tasks = sorted(requested - seen)
            if missing_tasks:
                raise ValueError(
                    "Requested Terminal-Lego tasks were not found under "
                    f"{self.dataset_path}: {missing_tasks}"
                )

        if not tasks:
            raise ValueError(
                "No runnable Terminal-Lego tasks found. Check dataset_path and "
                "that task.toml files contain [environment].docker_image."
            )

        logger.info(
            "Loaded %s Terminal-Lego tasks from %s",
            len(tasks),
            self.dataset_path,
        )
        return Dataset.from_list(tasks)

    def get_sandbox_spec(self, info: dict) -> SandboxSpec:
        image = info.get("docker_image")
        if not image:
            task_name = info.get("task_name", "<unknown>")
            raise ValueError(f"Terminal-Lego task {task_name!r} is missing an image")

        env_config = (info.get("config") or {}).get("environment", {})
        return SandboxSpec(
            image=str(image),
            cpu_cores=_parse_int(env_config.get("cpus"), default=1),
            memory_gb=_parse_size_gb(env_config.get("memory"), default=1),
            disk_size_gb=_parse_size_gb(env_config.get("storage"), default=5),
        )

    def get_workdir(self, info: dict) -> str:
        return DEFAULT_WORKDIR

    async def setup(self, state) -> None:
        await super().setup(state)
        sandbox_client = state["sandbox_client"]
        sandbox_id = state["sandbox_id"]
        await sandbox_client.execute_command(
            sandbox_id,
            "if [ -d /app/task_file ] && [ ! -e /app/task_file/output ]; then mkdir -p /app/task_file/output; fi",
            timeout=10,
        )
        config = state.get("info", {}).get("config") or {}
        verifier_timeout = _timeout_seconds(
            config.get("verifier", {}).get("timeout_sec")
        )
        agent_timeout = _timeout_seconds(config.get("agent", {}).get("timeout_sec"))
        if verifier_timeout is not None:
            state["test_timeout"] = verifier_timeout
        if agent_timeout is not None:
            state["solution_timeout"] = agent_timeout


def make_terminal_lego_taskset(
    dataset_path: str | Path | None = None,
    task_names: list[str] | str | None = None,
    hf_repo_id: str = DEFAULT_HF_REPO_ID,
    hf_revision: str | None = None,
    filter_fn: str | None = None,
) -> TerminalLegoTaskSet:
    return TerminalLegoTaskSet(
        dataset_path=dataset_path,
        task_names=_normalize_task_names(task_names),
        hf_repo_id=hf_repo_id,
        hf_revision=hf_revision,
        filter_fn=filter_fn,
    )


def _normalize_task_names(task_names: list[str] | str | None) -> list[str] | None:
    if task_names is None:
        return None
    if isinstance(task_names, str):
        names = [name.strip() for name in task_names.split(",")]
    else:
        names = [str(name).strip() for name in task_names]
    return [name for name in names if name]


def _resolve_dataset_path(
    *,
    dataset_path: str | Path | None,
    hf_repo_id: str,
    hf_revision: str | None,
) -> Path:
    if dataset_path is not None:
        path = Path(dataset_path).expanduser()
    elif env_path := os.environ.get(DATASET_PATH_ENV):
        path = Path(env_path).expanduser()
    else:
        path = _ensure_git_checkout(hf_repo_id=hf_repo_id, hf_revision=hf_revision)

    if not path.exists():
        raise FileNotFoundError(f"Terminal-Lego dataset path not found: {path}")
    return path


def _ensure_git_checkout(*, hf_repo_id: str, hf_revision: str | None) -> Path:
    cache_root = _git_cache_root()
    cache_root.mkdir(parents=True, exist_ok=True)
    checkout_dir = cache_root / _cache_key(hf_repo_id, hf_revision)
    lock_path = cache_root / f"{checkout_dir.name}.lock"

    with _file_lock(lock_path):
        if checkout_dir.exists():
            if not (checkout_dir / ".git").exists():
                raise RuntimeError(
                    f"Terminal-Lego cache path exists but is not a Git checkout: "
                    f"{checkout_dir}"
                )
            return checkout_dir

        tmp_dir = cache_root / f".{checkout_dir.name}.tmp-{os.getpid()}"
        shutil.rmtree(tmp_dir, ignore_errors=True)
        token = _hf_token()
        repo_url = f"https://huggingface.co/datasets/{hf_repo_id}"
        try:
            _run_git(["git", "clone", repo_url, str(tmp_dir)], token=token)
            if hf_revision:
                _checkout_revision(tmp_dir, hf_revision, token=token)
            _run_git(["git", "-C", str(tmp_dir), "lfs", "version"], token=token)
            _run_git(["git", "-C", str(tmp_dir), "lfs", "pull"], token=token)
            tmp_dir.rename(checkout_dir)
        except Exception:
            shutil.rmtree(tmp_dir, ignore_errors=True)
            raise

    return checkout_dir


def _checkout_revision(checkout_dir: Path, revision: str, token: str | None) -> None:
    verify = _run_git(
        [
            "git",
            "-C",
            str(checkout_dir),
            "rev-parse",
            "--verify",
            f"{revision}^{{commit}}",
        ],
        check=False,
        token=token,
    )
    if verify.returncode != 0:
        _run_git(
            ["git", "-C", str(checkout_dir), "fetch", "origin", revision], token=token
        )
    _run_git(
        ["git", "-C", str(checkout_dir), "checkout", "--detach", revision], token=token
    )


def _git_cache_root() -> Path:
    return Path(os.environ.get("HF_HOME", "~/.cache/huggingface")).expanduser() / (
        "terminal-lego-git"
    )


def _cache_key(hf_repo_id: str, hf_revision: str | None) -> str:
    revision = hf_revision or "default"
    raw = f"{hf_repo_id}--{revision}"
    return "".join(ch if ch.isalnum() or ch in ("-", ".", "_") else "-" for ch in raw)


@contextlib.contextmanager
def _file_lock(lock_path: Path) -> Iterator[None]:
    import fcntl

    with lock_path.open("w") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def _hf_token() -> str | None:
    for env_name in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_HUB_TOKEN"):
        token = os.environ.get(env_name)
        if token:
            return token
    try:
        from huggingface_hub import get_token
    except ImportError:
        return None
    return get_token()


def _run_git(
    args: list[str],
    *,
    token: str | None,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    if shutil.which("git") is None:
        raise RuntimeError("Terminal-Lego dataset cloning requires git on PATH")

    env = os.environ.copy()
    env["GIT_TERMINAL_PROMPT"] = "0"
    askpass_path: str | None = None
    if token:
        fd, askpass_path = tempfile.mkstemp(prefix="hf-git-askpass-")
        with os.fdopen(fd, "w") as askpass:
            askpass.write(
                "#!/bin/sh\n"
                'case "$1" in\n'
                '*Username*) printf "%s\\n" "hf_user" ;;\n'
                '*) printf "%s\\n" "$HF_GIT_TOKEN" ;;\n'
                "esac\n"
            )
        os.chmod(askpass_path, 0o700)
        env["GIT_ASKPASS"] = askpass_path
        env["HF_GIT_TOKEN"] = token

    try:
        result = subprocess.run(
            args,
            check=False,
            capture_output=True,
            env=env,
            text=True,
        )
        if check and result.returncode != 0:
            command = " ".join(args)
            raise RuntimeError(
                f"Git command failed with exit code {result.returncode}: {command}\n"
                f"stdout:\n{result.stdout[-2000:]}\n"
                f"stderr:\n{result.stderr[-2000:]}"
            )
        return result
    finally:
        if askpass_path:
            Path(askpass_path).unlink(missing_ok=True)


def _missing_required_files(task_dir: Path) -> list[str]:
    required = (
        "task.toml",
        "instruction.md",
        "solution/solve.sh",
        "tests/test.sh",
        "tests/test_outputs.py",
    )
    return [relative for relative in required if not (task_dir / relative).exists()]


def _parse_int(value: Any, default: int) -> int:
    if value is None:
        return default
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _parse_size_gb(value: Any, default: int) -> int:
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return max(1, math.ceil(float(value)))

    raw = str(value).strip().lower()
    for suffix in ("ib", "b"):
        if raw.endswith(suffix):
            raw = raw[: -len(suffix)]
    if not raw:
        return default

    multipliers = {
        "k": 1 / (1024 * 1024),
        "m": 1 / 1024,
        "g": 1,
        "t": 1024,
    }
    unit = raw[-1]
    if unit in multipliers:
        number = raw[:-1]
        multiplier = multipliers[unit]
    else:
        number = raw
        multiplier = 1
    try:
        return max(1, math.ceil(float(number) * multiplier))
    except ValueError:
        return default


def _timeout_seconds(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return max(1, math.ceil(float(value)))
    except (TypeError, ValueError):
        return None
