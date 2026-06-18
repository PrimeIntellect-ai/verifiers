"""`DB` + `Tools` base classes and the `@tool` decorator that every task's `tools.py` builds on.

A task's `tools.py` declares `TaskDB(DB)` (its world state), `TaskTools(Tools)` (the `@tool` methods
the agent calls, which mutate `self.db`) and a module-level `verify(db) -> float`. The corpus's task
files import these from `general_agent.tools`; the corpus loader installs a `sys.modules` shim so that
name resolves here (see `corpus._install_shim`), so the raw task files load unmodified.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Callable

from pydantic import BaseModel

_TOOL_ATTR = "__ga_tool__"


def tool(func: Callable) -> Callable:
    """Mark a `Tools` method as a tool exposed to the agent."""
    setattr(func, _TOOL_ATTR, True)
    return func


class DB(BaseModel):
    """Pydantic base for a task's database — its world state, loaded from `db.json`."""

    @classmethod
    def load(cls, path: str | Path) -> "DB":
        with open(path) as f:
            return cls.model_validate(json.load(f))

    def get_hash(self) -> str:
        """First 12 hex of the sha256 of the canonical JSON dump — the exact-match scoring
        primitive (the agent's final DB must hash-equal the gold solution's)."""
        return hashlib.sha256(
            self.model_dump_json(exclude_none=False).encode()
        ).hexdigest()[:12]


class Tools:
    """Base for a task's tools: owns a mutable `DB` and exposes its `@tool` methods."""

    def __init__(self, db: DB):
        self.db = db

    @property
    def tool_methods(self) -> dict[str, Callable]:
        """The `@tool`-decorated methods across the MRO, as bound methods."""
        methods: dict[str, Callable] = {}
        for name in dir(self):
            attr = getattr(type(self), name, None)
            if callable(attr) and getattr(attr, _TOOL_ATTR, False):
                methods[name] = getattr(self, name)
        return methods

    def call_tool(self, tool_name: str, **kwargs) -> Any:
        methods = self.tool_methods
        if tool_name not in methods:
            raise ValueError(f"Unknown tool: {tool_name}")
        return methods[tool_name](**kwargs)
