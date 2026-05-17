import json
from collections.abc import Callable, Mapping
from copy import deepcopy
from pathlib import Path
from typing import cast
from typing_extensions import Unpack

from ...config import TasksetConfig, merge_config_value
from ...taskset import Taskset, TasksetKwargs
from ...types import ConfigData, ConfigMap, TaskRow, TaskRows, TaskRowsSource
from ...utils.endpoint_utils import normalize_openai_responses_input
from ..nemo_gym import (
    DEFAULT_NEMO_GYM_DATA_NAME,
    agent_ref_name,
    resolve_nemo_gym_data_path,
)


class NeMoGymTasksetConfig(TasksetConfig):
    rows: TaskRowsSource | None = None
    nemo_env: str | None = None
    jsonl_path: str | None = None
    data_name: str = DEFAULT_NEMO_GYM_DATA_NAME
    agent_name: str | None = None
    limit: int | None = None


class NeMoGymTaskset(Taskset):
    """Taskset adapter for NeMo Gym JSONL rows.

    Each task keeps the original NeMo Gym row under ``nemo_gym_row`` so the
    harness can post it to the configured NeMo Gym agent unchanged.
    """

    config_type = NeMoGymTasksetConfig
    config: NeMoGymTasksetConfig

    def __init__(
        self,
        rows: TaskRowsSource | None = None,
        nemo_env: str | None = None,
        jsonl_path: str | Path | None = None,
        data_name: str | None = None,
        agent_name: str | None = None,
        limit: int | None = None,
        config: NeMoGymTasksetConfig | None = None,
        **kwargs: Unpack[TasksetKwargs],
    ):
        self.config = NeMoGymTasksetConfig.from_config(config)
        self._rows_source = cast(
            TaskRowsSource | None,
            merge_config_value(rows, self.config.rows),
        )
        self.nemo_env = cast(
            str | None,
            merge_config_value(nemo_env, self.config.nemo_env),
        )
        self.data_name = cast(
            str,
            merge_config_value(data_name, self.config.data_name),
        )
        raw_jsonl_path = merge_config_value(
            str(jsonl_path) if jsonl_path is not None else None,
            self.config.jsonl_path,
        )
        if raw_jsonl_path:
            self.jsonl_path = Path(str(raw_jsonl_path)).expanduser()
        elif self.nemo_env:
            self.jsonl_path = resolve_nemo_gym_data_path(
                self.nemo_env,
                self.data_name,
            )
        else:
            self.jsonl_path = None
        self.agent_name = cast(
            str | None,
            merge_config_value(agent_name, self.config.agent_name),
        )
        raw_limit = merge_config_value(limit, self.config.limit)
        if raw_limit is None:
            self.limit = None
        elif isinstance(raw_limit, int) and not isinstance(raw_limit, bool):
            self.limit = raw_limit
        elif isinstance(raw_limit, str):
            self.limit = int(raw_limit)
        else:
            raise TypeError("NeMoGymTaskset limit must be an integer.")
        super().__init__(
            source=self.load_rows,
            taskset_id="nemo_gym",
            config=self.config,
            **kwargs,
        )

    def load_rows(self) -> list[ConfigData]:
        raw_rows = self._load_raw_rows()
        if self.limit is not None:
            raw_rows = raw_rows[: self.limit]
        return [
            normalize_nemo_gym_task_row(row, index, agent_name=self.agent_name)
            for index, row in enumerate(raw_rows)
        ]

    def _load_raw_rows(self) -> list[TaskRow]:
        if self._rows_source is not None:
            source = self._rows_source
            rows = (
                cast(Callable[[], TaskRows], source)() if callable(source) else source
            )
            return [dict(row) for row in rows]
        if self.jsonl_path is None:
            raise ValueError("NeMoGymTaskset requires rows=... or jsonl_path=...")
        rows: list[TaskRow] = []
        with self.jsonl_path.open(encoding="utf-8") as f:
            for line in f:
                stripped = line.strip()
                if stripped:
                    rows.append(cast(TaskRow, json.loads(stripped)))
        return rows


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
