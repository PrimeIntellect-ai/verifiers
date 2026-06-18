"""The per-task tool server.

Unlike a normal `vf.Toolset` (one fixed set of `@vf.tool` methods), each general-agent task ships
its own `tools.py` with a different tool set. The server's lifecycle runs `setup_task` (which loads
that task's `tools.py`) before `_register`, so the tools are known by registration time: `setup_task`
builds the live `TaskTools`, and `_register` advertises each of its methods over MCP. Every call
mutates the live `TaskDB`; the mutated DB is pushed onto `self.state` so the taskset's reward can hash
it against the gold solution.
"""

from __future__ import annotations

import inspect
import json
from pathlib import Path
from typing import Callable

import verifiers.v1 as vf

from general_agent_v1.common import GeneralAgentState, GeneralAgentToolsetConfig
from general_agent_v1.corpus import load_task_attr


class GeneralAgentToolset(vf.Toolset[GeneralAgentToolsetConfig, GeneralAgentState]):
    TOOL_PREFIX = "tools"  # the model sees `tools_<name>` for each of the task's tools

    async def setup_task(self, task) -> None:
        task_dir = Path(task.dir)
        task_db = load_task_attr(task_dir, "TaskDB")
        task_tools = load_task_attr(task_dir, "TaskTools")
        if task_db is None or task_tools is None:
            raise ValueError(f"tools.py must define TaskDB and TaskTools: {task_dir}")
        self._tools = task_tools(task_db.load(task_dir / "db.json"))

    def _register(self, mcp) -> None:
        for name, method in sorted(self._tools.tool_methods.items()):
            mcp.add_tool(
                self._with_state(self._make_tool(name, method)),
                name=name,
                description=(method.__doc__ or "").strip() or None,
            )

    def _make_tool(self, name: str, method: Callable) -> Callable:
        async def call(**kwargs):
            result = method(**kwargs)  # mutates the live TaskDB
            self.state.db = self._tools.db.model_dump(mode="json")
            return (
                result if isinstance(result, str) else json.dumps(result, default=str)
            )

        call.__name__ = name
        call.__doc__ = method.__doc__
        call.__signature__ = inspect.signature(method).replace(return_annotation=str)
        return call


if __name__ == "__main__":
    GeneralAgentToolset.run()
