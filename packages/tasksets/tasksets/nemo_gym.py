import json
from copy import deepcopy
from pathlib import Path
from typing import TypeAlias, cast

import verifiers as vf
from verifiers.v1.utils.endpoint_utils import normalize_openai_responses_input

DEFAULT_NEMO_GYM_DATA_NAME = "example.jsonl"
ConfigData: TypeAlias = dict[str, object]
ConfigMap: TypeAlias = dict[str, object]
TaskRow: TypeAlias = dict[str, object]


def nemo_gym_package_root() -> Path:
    try:
        from nemo_gym import PARENT_DIR as nemo_gym_root  # ty: ignore[unresolved-import]
    except ImportError as exc:
        raise ImportError(
            "NeMo Gym integration requires nemo-gym. Install as `verifiers[nemogym]`."
        ) from exc
    return Path(nemo_gym_root)


def resolve_nemo_gym_data_path(
    nemo_env: str,
    data_name: str = DEFAULT_NEMO_GYM_DATA_NAME,
) -> Path:
    path = nemo_gym_package_root() / "resources_servers" / nemo_env / "data" / data_name
    if not path.exists():
        raise FileNotFoundError(f"NeMo Gym data file not found: {path}")
    return path


def agent_ref_name(value: object) -> str | None:
    if not isinstance(value, dict):
        return None
    name = cast(ConfigMap, value).get("name")
    return name if isinstance(name, str) and name else None


class NeMoGymTasksetConfig(vf.TasksetConfig):
    taskset_id: str | None = "nemo_gym"
    nemo_env: str | None = None
    jsonl_path: str | None = None
    data_name: str = DEFAULT_NEMO_GYM_DATA_NAME
    agent_name: str | None = None
    limit: int | None = None


class NeMoGymTaskset(vf.Taskset[NeMoGymTasksetConfig]):
    """Taskset adapter for NeMo Gym JSONL rows.

    Each task keeps the original NeMo Gym row under ``nemo_gym_row`` so the
    harness can post it to the configured NeMo Gym agent unchanged.
    """

    def jsonl_path(self) -> Path | None:
        raw_path = self.config.jsonl_path
        if raw_path:
            return Path(str(raw_path)).expanduser()
        if self.config.nemo_env:
            return resolve_nemo_gym_data_path(
                self.config.nemo_env,
                self.config.data_name,
            )
        return None

    def load_tasks(self, split: vf.TaskSplit = "train") -> vf.Tasks:
        jsonl_path = self.jsonl_path()
        if jsonl_path is None:
            raise ValueError("NeMoGymTaskset requires nemo_env=... or jsonl_path=...")
        raw_rows: list[TaskRow] = []
        with jsonl_path.open(encoding="utf-8") as f:
            for line in f:
                stripped = line.strip()
                if stripped:
                    raw_rows.append(cast(TaskRow, json.loads(stripped)))
        if self.config.limit is not None:
            raw_rows = raw_rows[: self.config.limit]
        tasks = [
            normalize_nemo_gym_task_row(row, index, agent_name=self.config.agent_name)
            for index, row in enumerate(raw_rows)
        ]
        return cast(vf.Tasks, tasks)


def normalize_nemo_gym_task_row(
    row: TaskRow,
    index: int,
    *,
    agent_name: str | None = None,
) -> ConfigData:
    nemo_row: ConfigData = deepcopy(dict(row))
    if agent_name and not agent_ref_name(nemo_row.get("agent_ref")):
        nemo_row["agent_ref"] = {
            "type": "responses_api_agents",
            "name": agent_name,
        }
    task_row: ConfigData = deepcopy(nemo_row)
    task_row["nemo_gym_row"] = nemo_row
    task_row.setdefault("example_id", index)
    prompt, system_prompt = prompt_parts_from_nemo_gym_row(nemo_row)
    task_row.setdefault("prompt", prompt)
    if system_prompt:
        task_row.setdefault("system_prompt", system_prompt)
    raw_info = task_row.get("info")
    info = dict(cast(ConfigMap, raw_info)) if isinstance(raw_info, dict) else {}
    info.setdefault(
        "nemo_gym",
        {
            "agent_name": agent_ref_name(nemo_row.get("agent_ref")) or agent_name,
        },
    )
    task_row["info"] = info
    return task_row


def prompt_parts_from_nemo_gym_row(
    row: TaskRow,
) -> tuple[list[ConfigData], list[ConfigData]]:
    create_params = row.get("responses_create_params")
    if not isinstance(create_params, dict):
        return [], []
    create_params = cast(ConfigMap, create_params)
    try:
        messages = normalize_openai_responses_input(create_params.get("input"))
    except Exception:
        return [], []
    prompt: list[ConfigData] = []
    system_prompt: list[ConfigData] = []
    for message in messages:
        dumped = cast(ConfigData, message.model_dump(exclude_none=True))
        if getattr(message, "role", None) == "system":
            system_prompt.append(dumped)
        else:
            prompt.append(dumped)
    return prompt, system_prompt
