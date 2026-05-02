from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
import time
from typing import Any
import uuid

from verifiers.types import State as VerifiersState

from .task import Task, assert_serializable


class State(VerifiersState):
    def __init__(self, value: Mapping[str, Any] | None = None):
        super().__init__(deepcopy(dict(value or {})))

    @classmethod
    def for_task(cls, task: Task) -> State:
        state = cls(
            {
                "task": dict(task),
                "runtime": dict(task.get("runtime", {})),
                "trajectory": [],
                "trajectory_id": uuid.uuid4().hex,
                "artifacts": {},
                "metrics": {},
                "reward": 0.0,
                "is_completed": False,
                "is_truncated": False,
                "stop_condition": None,
                "completion": None,
                "error": None,
                "timing": {
                    "generation_ms": 0.0,
                    "scoring_ms": 0.0,
                    "total_ms": 0.0,
                    "start_time": time.time(),
                },
            }
        )
        for key in ("prompt", "answer", "info", "example_id"):
            if key in task:
                state[key] = deepcopy(task[key])
        return state

    def assert_serializable(self) -> None:
        assert_serializable(self)
