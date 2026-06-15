"""CLI-Gym taskset backed by Harbor-format task directories.

Important limitations:
- Official CLI-Gym does not provide gold/oracle solutions. Materialized Harbor
  tasks do not include ``solution/solve.sh``, so gold-patch validation is not
  supported.
- CLI-Gym scoring tests are visible inside the single task sandbox. Treat this
  taskset as an SFT/trajectory-generation source, not as a hardened RL reward
  environment.
- Use only after careful filtering. The default PrimeIntellect/CLI-Gym mirror
  removes failed image builds and rows that pass no-op debug validation; custom
  mirrors should run equivalent validation before use.
"""

import json
import logging
import math
import os
import re
import shutil
import textwrap
from pathlib import Path
from typing import Any

import yaml

from verifiers.envs.experimental.composable import SandboxSpec
from verifiers.envs.experimental.composable.tasksets.harbor.harbor import (
    HarborDatasetTaskSet,
)
from verifiers.envs.experimental.composable.tasksets.harbor.terminal_lego import (
    _normalize_task_names,
)

logger = logging.getLogger(__name__)

try:
    import tomllib
except ImportError:
    import tomli as tomllib

DEFAULT_HF_REPO_ID = "PrimeIntellect/CLI-Gym"
DEFAULT_SPLIT = "train"
DEFAULT_WORKDIR = "/testbed"
PRIME_SANDBOX_REGISTRY = (
    "us-central1-docker.pkg.dev/prime-intellect-platform/prod-sandbox"
)
_CACHE_VERSION = "v1"

_TASK_YAML_KEYS = (
    "author_email",
    "author_name",
    "difficulty",
    "category",
    "tags",
    "parser_name",
    "max_agent_timeout_sec",
    "max_test_timeout_sec",
    "run_tests_in_same_shell",
)
_PYTEST_FAILURE_INDICATORS = (
    "ERROR: usage:",
    "ERROR: file or directory not found",
    "ImportError:",
    "ModuleNotFoundError:",
    "SyntaxError:",
    "command not found",
    "No module named",
)


class CLIGymTaskSet(HarborDatasetTaskSet):
    """CLI-Gym rows materialized as Harbor-format terminal tasks.

    The default PrimeIntellect mirror adds prebuilt Prime sandbox image refs to
    the official CLI-Gym rows. The taskset writes each selected row to a local
    Harbor task directory, then delegates sandbox setup and scoring upload to
    :class:`HarborDatasetTaskSet`.

    Official CLI-Gym does not ship gold/oracle solutions. This taskset is
    intended for carefully filtered SFT trajectory generation, not gold-patch
    validation or hardened single-sandbox RL.
    """

    def __init__(
        self,
        dataset_path: str | Path | None = None,
        task_names: list[str] | None = None,
        hf_repo_id: str = DEFAULT_HF_REPO_ID,
        hf_revision: str | None = None,
        split: str = DEFAULT_SPLIT,
        cache_dir: str | Path | None = None,
        filter_fn: str | None = None,
    ):
        self.hf_repo_id = hf_repo_id
        self.hf_revision = hf_revision
        self.split = split
        self.cache_dir = Path(cache_dir).expanduser() if cache_dir else None

        resolved_dataset_path = _resolve_dataset_path(
            dataset_path=dataset_path,
            task_names=task_names,
            hf_repo_id=hf_repo_id,
            hf_revision=hf_revision,
            split=split,
            cache_dir=self.cache_dir,
        )
        super().__init__(
            dataset_path=resolved_dataset_path,
            task_names=task_names,
            filter_fn=filter_fn,
        )
        self.name = "cli-gym"

    def _build_dataset(self) -> Any:
        from datasets import Dataset

        requested = set(self.task_names or [])
        seen: set[str] = set()
        tasks: list[dict] = []

        for task_dir in sorted(self.dataset_path.iterdir()):
            if not task_dir.is_dir():
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
                        f"CLI-Gym task {task_name!r} is incomplete; "
                        f"missing {missing_str}"
                    )
                logger.warning(
                    "Skipping %s: missing required files: %s",
                    task_name,
                    ", ".join(missing),
                )
                continue

            entry = _load_cli_gym_entry(task_dir)
            image_ref = entry["info"].get("docker_image")
            if not image_ref:
                if requested:
                    raise ValueError(
                        f"CLI-Gym task {task_name!r} is missing docker_image"
                    )
                logger.warning("Skipping %s: missing docker_image", task_name)
                continue
            tasks.append(entry)

        if requested:
            missing_tasks = sorted(requested - seen)
            if missing_tasks:
                raise ValueError(
                    "Requested CLI-Gym tasks were not found under "
                    f"{self.dataset_path}: {missing_tasks}"
                )

        if not tasks:
            raise ValueError(
                "No runnable CLI-Gym tasks found. Check dataset_path or that HF "
                "rows contain docker_image."
            )

        logger.info("Loaded %s CLI-Gym tasks from %s", len(tasks), self.dataset_path)
        return Dataset.from_list(tasks)

    def get_sandbox_spec(self, info: dict) -> SandboxSpec:
        image = info.get("docker_image")
        if not image:
            task_name = info.get("task_name", "<unknown>")
            raise ValueError(f"CLI-Gym task {task_name!r} is missing an image")

        timeout = _timeout_minutes(info.get("max_agent_timeout_sec"))
        return SandboxSpec(image=str(image), timeout_minutes=timeout)

    def get_workdir(self, info: dict) -> str:
        return DEFAULT_WORKDIR

    async def setup(self, state) -> None:
        await super().setup(state)
        info = state.get("info") or {}
        if test_timeout := _timeout_seconds(info.get("max_test_timeout_sec")):
            state["test_timeout"] = test_timeout

    async def _apply_gold_patch(
        self, sandbox_client: Any, sandbox_id: str, state: dict
    ) -> None:
        raise RuntimeError(
            "CLI-Gym does not provide gold/oracle solutions. Use this taskset "
            "for carefully filtered SFT trajectory generation only; gold-patch "
            "validation is unsupported."
        )

    async def validate_instance(self, state) -> bool:
        raise RuntimeError(
            "CLI-Gym does not provide gold/oracle solutions. Use this taskset "
            "for carefully filtered SFT trajectory generation only; "
            "TaskSet.validate() gold-solution validation is unsupported."
        )


def make_cli_gym_taskset(
    dataset_path: str | Path | None = None,
    task_names: list[str] | str | None = None,
    hf_repo_id: str = DEFAULT_HF_REPO_ID,
    hf_revision: str | None = None,
    split: str = DEFAULT_SPLIT,
    cache_dir: str | Path | None = None,
    filter_fn: str | None = None,
) -> CLIGymTaskSet:
    """Create a filtered CLI-Gym Harbor taskset.

    Official CLI-Gym does not provide gold/oracle solutions. The default
    ``PrimeIntellect/CLI-Gym`` mirror is filtered for failed image builds and
    no-op debug passes, and should be used as an SFT/trajectory-generation
    source rather than a hardened RL reward environment. Custom mirrors should
    run equivalent filtering before use.
    """
    return CLIGymTaskSet(
        dataset_path=dataset_path,
        task_names=_normalize_task_names(task_names),
        hf_repo_id=hf_repo_id,
        hf_revision=hf_revision,
        split=split,
        cache_dir=cache_dir,
        filter_fn=filter_fn,
    )


def _resolve_dataset_path(
    *,
    dataset_path: str | Path | None,
    task_names: list[str] | None,
    hf_repo_id: str,
    hf_revision: str | None,
    split: str,
    cache_dir: Path | None,
) -> Path:
    if dataset_path is not None:
        path = Path(dataset_path).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"CLI-Gym dataset path not found: {path}")
        return path
    materialized_path = _materialized_dataset_path(
        hf_repo_id=hf_repo_id,
        hf_revision=hf_revision,
        split=split,
        cache_dir=cache_dir,
    )
    _materialize_hf_dataset(
        output_dir=materialized_path,
        task_names=task_names,
        hf_repo_id=hf_repo_id,
        hf_revision=hf_revision,
        split=split,
    )
    return materialized_path


def _materialized_dataset_path(
    *,
    hf_repo_id: str,
    hf_revision: str | None,
    split: str,
    cache_dir: Path | None,
) -> Path:
    root = cache_dir or (
        Path(os.environ.get("HF_HOME", "~/.cache/huggingface")).expanduser()
        / "cli-gym-harbor"
    )
    key = _cache_key(hf_repo_id, hf_revision, split)
    return root / key


def _materialize_hf_dataset(
    *,
    output_dir: Path,
    task_names: list[str] | None,
    hf_repo_id: str,
    hf_revision: str | None,
    split: str,
) -> None:
    from datasets import load_dataset

    requested = set(task_names or [])
    if output_dir.exists():
        if not requested or requested.issubset(_existing_task_names(output_dir)):
            return

    output_dir.mkdir(parents=True, exist_ok=True)
    ds = load_dataset(
        hf_repo_id,
        split=split,
        revision=hf_revision,
    )

    seen: set[str] = set()
    materialized = 0
    for row in ds:
        task_id = str(row.get("task_id") or "").strip()
        if not task_id:
            continue
        if requested and task_id not in requested:
            continue
        seen.add(task_id)

        source_image_ref = str(row.get("docker_image") or "").strip()
        image_ref = _runnable_image_ref(row, source_image_ref)
        if not image_ref:
            if requested:
                raise ValueError(f"CLI-Gym task {task_id!r} is missing docker_image")
            logger.warning("Skipping %s: missing docker_image", task_id)
            continue

        _write_task_dir(
            output_dir / task_id,
            row,
            image_ref=image_ref,
            source_image_ref=source_image_ref,
            hf_repo_id=hf_repo_id,
        )
        materialized += 1

    if requested:
        missing = sorted(requested - seen)
        if missing:
            raise ValueError(f"Requested CLI-Gym tasks were not found: {missing}")
    if materialized == 0:
        raise ValueError(
            f"No runnable CLI-Gym rows materialized from {hf_repo_id!r} split {split!r}"
        )


def _write_task_dir(
    task_dir: Path,
    row: dict,
    *,
    image_ref: str,
    source_image_ref: str,
    hf_repo_id: str,
) -> None:
    task_id = str(row["task_id"])
    task_yaml = str(row.get("task_yaml") or "")
    metadata = _parse_task_yaml(task_yaml)
    instruction = _extract_instruction(task_yaml, metadata)

    tmp_dir = task_dir.with_name(f".{task_dir.name}.tmp-{os.getpid()}")
    shutil.rmtree(tmp_dir, ignore_errors=True)
    (tmp_dir / "tests").mkdir(parents=True, exist_ok=True)

    (tmp_dir / "instruction.md").write_text(instruction + "\n", encoding="utf-8")
    (tmp_dir / "task.yaml").write_text(task_yaml, encoding="utf-8")
    (tmp_dir / "Dockerfile").write_text(
        str(row.get("dockerfile") or ""), encoding="utf-8"
    )
    (tmp_dir / "docker-compose.yaml").write_text(
        str(row.get("docker_compose") or ""), encoding="utf-8"
    )
    (tmp_dir / "task.toml").write_text(
        _task_toml(task_id, image_ref, source_image_ref, metadata, hf_repo_id),
        encoding="utf-8",
    )
    (tmp_dir / "tests" / "run_tests.sh").write_text(
        str(row.get("run_tests") or ""), encoding="utf-8"
    )
    (tmp_dir / "tests" / "test.sh").write_text(_test_wrapper_script(), encoding="utf-8")

    shutil.rmtree(task_dir, ignore_errors=True)
    tmp_dir.rename(task_dir)


def _load_cli_gym_entry(task_dir: Path) -> dict:
    task_toml = task_dir / "task.toml"
    with open(task_toml, "rb") as f:
        config = tomllib.load(f)

    instruction = (task_dir / "instruction.md").read_text().strip()
    metadata = config.get("cli_gym", {}) if isinstance(config, dict) else {}
    return {
        "question": instruction,
        "info": {
            "task_dir": str(task_dir),
            "task_name": task_dir.name,
            "instance_id": task_dir.name,
            "docker_image": config.get("environment", {}).get("docker_image"),
            "config": config,
            "cli_gym_source": metadata.get("source"),
            "max_agent_timeout_sec": metadata.get("max_agent_timeout_sec"),
            "max_test_timeout_sec": metadata.get("max_test_timeout_sec"),
        },
        "answer": "",
    }


def _parse_task_yaml(task_yaml: str) -> dict[str, Any]:
    try:
        data = yaml.safe_load(task_yaml)
    except yaml.YAMLError:
        data = None
    if not isinstance(data, dict):
        data = {}

    for key in ("max_agent_timeout_sec", "max_test_timeout_sec"):
        if key not in data:
            value = _extract_top_level_scalar(task_yaml, key)
            if value is not None:
                data[key] = value
    return data


def _extract_instruction(task_yaml: str, metadata: dict[str, Any]) -> str:
    instruction = metadata.get("instruction")
    if isinstance(instruction, str) and instruction.strip():
        return instruction.strip()

    lines = task_yaml.splitlines()
    for idx, line in enumerate(lines):
        if not line.startswith("instruction:"):
            continue
        after = line.split(":", 1)[1].strip()
        if after and after not in {"|", "|-", "|+"}:
            return after.strip("\"'")
        block: list[str] = []
        key_pattern = re.compile(rf"^({'|'.join(_TASK_YAML_KEYS)}):\\s*")
        for item in lines[idx + 1 :]:
            if key_pattern.match(item):
                break
            block.append(item)
        return textwrap.dedent("\n".join(block)).strip()
    raise ValueError("CLI-Gym task_yaml is missing instruction")


def _extract_top_level_scalar(task_yaml: str, key: str) -> str | None:
    match = re.search(rf"^{re.escape(key)}:\s*(.+?)\s*$", task_yaml, re.MULTILINE)
    if match:
        return match.group(1).strip().strip("\"'")
    return None


def _task_toml(
    task_id: str,
    image_ref: str,
    source_image_ref: str,
    metadata: dict[str, Any],
    hf_repo_id: str,
) -> str:
    agent_timeout = _timeout_seconds(metadata.get("max_agent_timeout_sec"))
    test_timeout = _timeout_seconds(metadata.get("max_test_timeout_sec"))
    lines = [
        "[metadata]",
        f"name = {_toml_string(task_id)}",
        'description = "CLI-Gym Harbor task."',
        "",
        "[environment]",
        f"docker_image = {_toml_string(image_ref)}",
        "",
        "[cli_gym]",
        f"source = {_toml_string(hf_repo_id)}",
    ]
    if source_image_ref and source_image_ref != image_ref:
        lines.append(f"source_docker_image = {_toml_string(source_image_ref)}")
    if agent_timeout is not None:
        lines.append(f"max_agent_timeout_sec = {agent_timeout}")
    if test_timeout is not None:
        lines.append(f"max_test_timeout_sec = {test_timeout}")
    return "\n".join(lines) + "\n"


def _test_wrapper_script() -> str:
    indicators = ", ".join(repr(item) for item in _PYTEST_FAILURE_INDICATORS)
    return f"""#!/usr/bin/env bash
set -uo pipefail

mkdir -p /logs/verifier
chmod +x /tests/run_tests.sh

set +e
cd /testbed
bash /tests/run_tests.sh 2>&1 | tee /logs/verifier/cligym-tests.log
test_exit=${{PIPESTATUS[0]}}
set -e

TEST_EXIT="$test_exit" python - <<'PY'
import os
import re
from pathlib import Path

combined = Path("/logs/verifier/cligym-tests.log").read_text(errors="replace")
test_log_path = Path("/test.log")
if test_log_path.exists():
    combined += "\\n" + test_log_path.read_text(errors="replace")

failure_indicators = ({indicators},)
failed = os.environ.get("TEST_EXIT") != "0"
failed = failed or any(indicator in combined for indicator in failure_indicators)
failed = failed or re.search(r"^FAILED\\s+", combined, re.MULTILINE) is not None
failed = failed or re.search(r"^ERROR\\s+", combined, re.MULTILINE) is not None
failed = failed or re.search(r"=+\\s+\\d+\\s+(failed|errors?)\\b", combined, re.IGNORECASE) is not None
passed = re.search(r"=+\\s+\\d+\\s+passed\\b", combined, re.IGNORECASE) is not None

reward = 1.0 if passed and not failed else 0.0
Path("/logs/verifier/reward.txt").write_text(str(reward))
PY

exit 0
"""


def _missing_required_files(task_dir: Path) -> list[str]:
    required = (
        "task.toml",
        "instruction.md",
        "tests/test.sh",
        "tests/run_tests.sh",
    )
    return [relative for relative in required if not (task_dir / relative).exists()]


def _existing_task_names(dataset_path: Path) -> set[str]:
    return {path.name for path in dataset_path.iterdir() if path.is_dir()}


def _cache_key(hf_repo_id: str, hf_revision: str | None, split: str) -> str:
    revision = hf_revision or "default"
    raw = f"{_CACHE_VERSION}--{hf_repo_id}--{revision}--{split}"
    return "".join(ch if ch.isalnum() or ch in ("-", ".", "_") else "-" for ch in raw)


def _runnable_image_ref(row: dict, image_ref: str) -> str:
    if not image_ref or _has_registry_host(image_ref):
        return image_ref
    if row.get("prime_image_tag") or row.get("prime_image_build_id"):
        return f"{PRIME_SANDBOX_REGISTRY}/{image_ref.lstrip('/')}"
    return image_ref


def _has_registry_host(image_ref: str) -> bool:
    head, sep, _tail = image_ref.partition("/")
    return bool(sep and ("." in head or ":" in head or head == "localhost"))


def _timeout_seconds(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return max(1, math.ceil(float(value)))
    except (TypeError, ValueError):
        return None


def _timeout_minutes(value: Any) -> int | None:
    seconds = _timeout_seconds(value)
    if seconds is None:
        return None
    return max(1, math.ceil(seconds / 60))


def _toml_string(value: str) -> str:
    return json.dumps(value)
