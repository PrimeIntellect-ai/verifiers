"""Terminal-Lego taskset built on the composable Harbor task layout."""

import json
import logging
import math
import os
from pathlib import Path
from typing import Any

from verifiers.envs.experimental.composable import SandboxSpec
from verifiers.envs.experimental.composable.tasksets.harbor.harbor import (
    HarborDatasetTaskSet,
    _load_task_entry,
)

logger = logging.getLogger(__name__)

DEFAULT_HF_REPO_ID = "PrimeIntellect/Terminal-Lego-15k"
DEFAULT_IMAGE_REF_FIELD = "full_image_path"
DEFAULT_WORKDIR = "/app"
DATASET_PATH_ENV = "TERMINAL_LEGO_DATASET_PATH"
IMAGE_MAP_PATH_ENV = "TERMINAL_LEGO_IMAGE_MAP_PATH"


class TerminalLegoTaskSet(HarborDatasetTaskSet):
    """Terminal-Lego tasks with prebuilt per-task images from our registry.

    Terminal-Lego ships as a Hugging Face dataset repository whose rows are
    Terminal-Bench/Harbor-style task directories. The Docker build context is
    each task's ``environment/`` directory; this taskset expects those contexts
    to have already been pushed and supplies the corresponding image per row.
    """

    def __init__(
        self,
        dataset_path: str | Path | None = None,
        task_names: list[str] | None = None,
        image_map_path: str | Path | None = None,
        hf_repo_id: str = DEFAULT_HF_REPO_ID,
        hf_revision: str | None = None,
        image_ref_field: str = DEFAULT_IMAGE_REF_FIELD,
        filter_fn: str | None = None,
    ):
        self.hf_repo_id = hf_repo_id
        self.hf_revision = hf_revision
        self.image_ref_field = image_ref_field
        self.image_map_path = _resolve_image_map_path(image_map_path)
        self.image_map = _load_image_map(self.image_map_path, image_ref_field)

        resolved_dataset_path = _resolve_dataset_path(
            dataset_path=dataset_path,
            task_names=task_names,
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

            image_row = self.image_map.get(task_name)
            if image_row is None:
                if requested:
                    raise ValueError(
                        f"Terminal-Lego task {task_name!r} has no pushed image "
                        f"in {self.image_map_path}"
                    )
                continue

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
            image_ref = image_row[self.image_ref_field]
            entry["info"].update(
                {
                    "docker_image": image_ref,
                    "terminal_lego_image": image_row,
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
                f"image_map_path={self.image_map_path}."
            )

        logger.info(
            "Loaded %s Terminal-Lego tasks from %s with image map %s",
            len(tasks),
            self.dataset_path,
            self.image_map_path,
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
    image_map_path: str | Path | None = None,
    hf_repo_id: str = DEFAULT_HF_REPO_ID,
    hf_revision: str | None = None,
    image_ref_field: str = DEFAULT_IMAGE_REF_FIELD,
    filter_fn: str | None = None,
) -> TerminalLegoTaskSet:
    return TerminalLegoTaskSet(
        dataset_path=dataset_path,
        task_names=_normalize_task_names(task_names),
        image_map_path=image_map_path,
        hf_repo_id=hf_repo_id,
        hf_revision=hf_revision,
        image_ref_field=image_ref_field,
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
    task_names: list[str] | None,
    hf_repo_id: str,
    hf_revision: str | None,
) -> Path:
    if dataset_path is not None:
        path = Path(dataset_path).expanduser()
    elif env_path := os.environ.get(DATASET_PATH_ENV):
        path = Path(env_path).expanduser()
    else:
        try:
            from huggingface_hub import snapshot_download
        except ImportError as exc:
            raise RuntimeError(
                "huggingface_hub is required to download Terminal-Lego. "
                f"Set {DATASET_PATH_ENV} or pass dataset_path to use a local clone."
            ) from exc

        allow_patterns = None
        if task_names:
            allow_patterns = [
                pattern
                for task_name in task_names
                for pattern in (f"{task_name}/**", f"{task_name}")
            ]
        path = Path(
            snapshot_download(
                repo_id=hf_repo_id,
                repo_type="dataset",
                revision=hf_revision,
                allow_patterns=allow_patterns,
            )
        )

    if not path.exists():
        raise FileNotFoundError(f"Terminal-Lego dataset path not found: {path}")
    return path


def _resolve_image_map_path(image_map_path: str | Path | None) -> Path:
    if image_map_path is not None:
        path = Path(image_map_path).expanduser()
    elif env_path := os.environ.get(IMAGE_MAP_PATH_ENV):
        path = Path(env_path).expanduser()
    else:
        raise ValueError(
            "Terminal-Lego requires a pushed image map. Pass image_map_path or "
            f"set {IMAGE_MAP_PATH_ENV}."
        )
    if not path.exists():
        raise FileNotFoundError(f"Terminal-Lego image map not found: {path}")
    return path


def _load_image_map(path: Path, image_ref_field: str) -> dict[str, dict[str, str]]:
    image_map: dict[str, dict[str, str]] = {}
    with path.open() as f:
        for line_number, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path}:{line_number}") from exc
            task_id = row.get("item_id") or row.get("task_id")
            image_ref = row.get(image_ref_field)
            if not task_id or not image_ref:
                raise ValueError(
                    f"Image map row {path}:{line_number} must contain item_id "
                    f"and {image_ref_field}"
                )
            image_map[str(task_id)] = {
                str(key): str(value) for key, value in row.items()
            }
    if not image_map:
        raise ValueError(f"Terminal-Lego image map is empty: {path}")
    return image_map


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
    if not raw:
        return default
    for suffix in ("ib", "b"):
        if raw.endswith(suffix):
            raw = raw[: -len(suffix)]

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
