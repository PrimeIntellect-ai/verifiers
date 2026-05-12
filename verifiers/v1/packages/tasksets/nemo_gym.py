from __future__ import annotations

import json
from collections.abc import Callable, Iterable, Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any, cast

from ...config import TasksetConfig, merge_config_value
from ...taskset import Taskset
from ...utils.endpoint_utils import normalize_openai_responses_input
from ..nemo_gym import DEFAULT_NEMO_GYM_DATA_NAME, resolve_nemo_gym_data_path


class NeMoGymTasksetConfig(TasksetConfig):
    rows: object | None = None
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

    def __init__(
        self,
        rows: Iterable[Mapping[str, Any]]
        | Callable[[], Iterable[Mapping[str, Any]]]
        | None = None,
        nemo_env: str | None = None,
        jsonl_path: str | Path | None = None,
        data_name: str | None = None,
        agent_name: str | None = None,
        limit: int | None = None,
        config: NeMoGymTasksetConfig | Mapping[str, object] | None = None,
        **kwargs: Any,
    ):
        self.config = type(self).config_type.from_config(config)
        self._rows_source = merge_config_value(rows, self.config.rows)
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
        self.limit = int(raw_limit) if raw_limit is not None else None
        super().__init__(
            source=self.load_rows,
            taskset_id="nemo_gym",
            config=self.config,
            **kwargs,
        )

    def load_rows(self) -> list[dict[str, Any]]:
        raw_rows = self._load_raw_rows()
        if self.limit is not None:
            raw_rows = raw_rows[: self.limit]
        return [
            normalize_nemo_gym_task_row(row, index, agent_name=self.agent_name)
            for index, row in enumerate(raw_rows)
        ]

    def _load_raw_rows(self) -> list[Mapping[str, Any]]:
        if self._rows_source is not None:
            source = (
                self._rows_source()
                if callable(self._rows_source)
                else self._rows_source
            )
            return [dict(row) for row in cast(Iterable[Mapping[str, Any]], source)]
        if self.jsonl_path is None:
            raise ValueError("NeMoGymTaskset requires rows=... or jsonl_path=...")
        rows: list[Mapping[str, Any]] = []
        with self.jsonl_path.open(encoding="utf-8") as f:
            for line in f:
                stripped = line.strip()
                if stripped:
                    rows.append(cast(Mapping[str, Any], json.loads(stripped)))
        return rows


def normalize_nemo_gym_task_row(
    row: Mapping[str, Any],
    index: int,
    *,
    agent_name: str | None = None,
) -> dict[str, Any]:
    nemo_row = deepcopy(dict(row))
    if agent_name and not _agent_ref_name(nemo_row.get("agent_ref")):
        nemo_row["agent_ref"] = {
            "type": "responses_api_agents",
            "name": agent_name,
        }
    task_row = deepcopy(nemo_row)
    task_row["nemo_gym_row"] = nemo_row
    task_row.setdefault("example_id", index)
    prompt, system_prompt = prompt_parts_from_nemo_gym_row(nemo_row)
    task_row.setdefault("prompt", prompt)
    if system_prompt:
        task_row.setdefault("system_prompt", system_prompt)
    info = dict(task_row.get("info") or {})
    info.setdefault(
        "nemo_gym",
        {
            "agent_name": _agent_ref_name(nemo_row.get("agent_ref")) or agent_name,
        },
    )
    task_row["info"] = info
    return task_row


def prompt_from_nemo_gym_row(row: Mapping[str, Any]) -> list[dict[str, Any]]:
    prompt, _ = prompt_parts_from_nemo_gym_row(row)
    return prompt


def prompt_parts_from_nemo_gym_row(
    row: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    create_params = row.get("responses_create_params")
    if not isinstance(create_params, Mapping):
        return [], []
    try:
        messages = normalize_openai_responses_input(create_params.get("input"))
    except Exception:
        return [], []
    prompt: list[dict[str, Any]] = []
    system_prompt: list[dict[str, Any]] = []
    for message in messages:
        dumped = message.model_dump(exclude_none=True)
        if getattr(message, "role", None) == "system":
            system_prompt.append(dumped)
        else:
            prompt.append(dumped)
    return prompt, system_prompt


def _agent_ref_name(value: object) -> str | None:
    if not isinstance(value, Mapping):
        return None
    name = value.get("name")
    return name if isinstance(name, str) and name else None
