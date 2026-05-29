import json
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import cast

from ...config import TasksetConfig
from ...taskset import Taskset
from ...types import ConfigData, ConfigMap, TaskRow
from ...utils.endpoint_utils import normalize_openai_responses_input
from ..nemo_gym import (
    DEFAULT_NEMO_GYM_DATA_NAME,
    agent_ref_name,
    resolve_nemo_gym_data_path,
)


class NeMoGymTasksetConfig(TasksetConfig):
    nemo_env: str | None = None
    jsonl_path: str | None = None
    data_name: str = DEFAULT_NEMO_GYM_DATA_NAME
    agent_name: str | None = None
    limit: int | None = None


class NeMoGymTaskset(Taskset[NeMoGymTasksetConfig]):
    """Taskset adapter for NeMo Gym JSONL rows.

    Each task keeps the original NeMo Gym row under ``nemo_gym_row`` so the
    harness can post it to the configured NeMo Gym agent unchanged.
    """

    def __init__(self, config: NeMoGymTasksetConfig | None = None):
        config = NeMoGymTasksetConfig() if config is None else config
        assert isinstance(config, NeMoGymTasksetConfig)
        super().__init__(config=config)
        self.taskset_id = self.config.taskset_id or "nemo_gym"
        raw_path = self.config.jsonl_path
        if raw_path:
            self.jsonl_path: Path | None = Path(str(raw_path)).expanduser()
        elif self.config.nemo_env:
            self.jsonl_path = resolve_nemo_gym_data_path(
                self.config.nemo_env,
                self.config.data_name,
            )
        else:
            self.jsonl_path = None

    def load_tasks(self) -> list[ConfigData]:
        if self.jsonl_path is None:
            raise ValueError("NeMoGymTaskset requires nemo_env=... or jsonl_path=...")
        raw_rows: list[TaskRow] = []
        with self.jsonl_path.open(encoding="utf-8") as f:
            for line in f:
                stripped = line.strip()
                if stripped:
                    raw_rows.append(cast(TaskRow, json.loads(stripped)))
        if self.config.limit is not None:
            raw_rows = raw_rows[: self.config.limit]
        return [
            normalize_nemo_gym_task_row(row, index, agent_name=self.config.agent_name)
            for index, row in enumerate(raw_rows)
        ]


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
    info = dict(cast(ConfigMap, raw_info)) if isinstance(raw_info, Mapping) else {}
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
    if not isinstance(create_params, Mapping):
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
