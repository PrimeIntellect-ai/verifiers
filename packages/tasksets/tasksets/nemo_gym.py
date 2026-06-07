import json
from copy import deepcopy
from pathlib import Path

from pydantic import TypeAdapter
import verifiers.v1 as vf
from verifiers.v1.utils.json_utils import json_data, json_value

DEFAULT_NEMO_GYM_DATA_NAME = "example.jsonl"
_MESSAGES_ADAPTER = TypeAdapter(vf.Messages)


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


def agent_ref_name(value: vf.JsonValue) -> str | None:
    if not isinstance(value, dict):
        return None
    name = value.get("name")
    return name if isinstance(name, str) and name else None


class NeMoGymTasksetConfig(vf.TasksetConfig):
    id: str | None = "nemo_gym"
    nemo_env: str | None = None
    jsonl_path: str | None = None
    data_name: str = DEFAULT_NEMO_GYM_DATA_NAME
    agent_name: str | None = None
    limit: int | None = None


class NeMoGymTask(vf.Task, frozen=True):
    nemo_gym_row: vf.JsonData
    info: vf.JsonData = {}
    system_prompt: list[vf.JsonData] = []


class NeMoGymTaskset(vf.Taskset[NeMoGymTasksetConfig]):
    task_type = NeMoGymTask

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
        if split == "eval":
            return []
        jsonl_path = self.jsonl_path()
        if jsonl_path is None:
            raise ValueError("NeMoGymTaskset requires nemo_env=... or jsonl_path=...")
        raw_rows: list[vf.JsonData] = []
        with jsonl_path.open(encoding="utf-8") as f:
            for line in f:
                stripped = line.strip()
                if stripped:
                    raw_rows.append(json_data(json.loads(stripped)))
        if self.config.limit is not None:
            raw_rows = raw_rows[: self.config.limit]
        tasks = [
            normalize_nemo_gym_task_row(row, index, agent_name=self.config.agent_name)
            for index, row in enumerate(raw_rows)
        ]
        return tasks


def normalize_nemo_gym_task_row(
    row: vf.JsonData,
    index: int,
    *,
    agent_name: str | None = None,
) -> vf.JsonData:
    nemo_row: vf.JsonData = deepcopy(dict(row))
    if agent_name and not agent_ref_name(nemo_row.get("agent_ref")):
        nemo_row["agent_ref"] = {
            "type": "responses_api_agents",
            "name": agent_name,
        }
    task_row: vf.JsonData = deepcopy(nemo_row)
    task_row["nemo_gym_row"] = nemo_row
    task_row.setdefault("row_id", index)
    prompt, system_prompt = prompt_parts_from_nemo_gym_row(nemo_row)
    if "prompt" not in task_row:
        task_row["prompt"] = json_value(prompt)
    if system_prompt and "system_prompt" not in task_row:
        task_row["system_prompt"] = json_value(system_prompt)
    raw_info = task_row.get("info")
    info = dict(raw_info) if isinstance(raw_info, dict) else {}
    info.setdefault(
        "nemo_gym",
        {
            "agent_name": agent_ref_name(nemo_row.get("agent_ref")) or agent_name,
        },
    )
    task_row["info"] = info
    return task_row


def prompt_parts_from_nemo_gym_row(
    row: vf.JsonData,
) -> tuple[list[vf.JsonData], list[vf.JsonData]]:
    create_params = row.get("responses_create_params")
    if not isinstance(create_params, dict):
        return [], []
    try:
        messages = normalize_responses_input(create_params.get("input"))
    except Exception:
        return [], []
    prompt: list[vf.JsonData] = []
    system_prompt: list[vf.JsonData] = []
    for message in messages:
        dumped = json_data(message.model_dump(exclude_none=True))
        if getattr(message, "role", None) == "system":
            system_prompt.append(dumped)
        else:
            prompt.append(dumped)
    return prompt, system_prompt


def normalize_responses_input(value: vf.JsonValue) -> vf.Messages:
    if isinstance(value, str):
        return [vf.UserMessage(content=value)]
    if isinstance(value, list):
        raw_messages: list[vf.JsonData] = []
        for item in value:
            if not isinstance(item, dict):
                raise TypeError("responses_create_params.input must contain objects.")
            raw_messages.append(json_data(item))
        return _MESSAGES_ADAPTER.validate_python(raw_messages)
    raise TypeError("responses_create_params.input must be a string or message list.")
