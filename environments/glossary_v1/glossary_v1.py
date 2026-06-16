"""glossary: a custom tool server, authored as a vf-native class.

Each task asks the model to look up an entity. The taskset declares its tool server as a
`vf.Toolset` subclass with `@vf.tool` methods (no FastMCP boilerplate, no separate server
file): the framework launches it in its own runtime and surfaces its `lookup` tool as
`facts_lookup`. The reward checks the looked-up fact reached the answer. The simplest tool
example — contrast `wikispeedia` (per-task state), `wiki_search` (shared), `deepwiki` (remote).
"""

import json
from pathlib import Path

import verifiers.v1 as vf

HERE = Path(__file__).resolve().parent
FACTS: dict[str, str] = json.loads((HERE / "facts.json").read_text())


class GlossaryToolset(vf.Toolset[vf.ToolsetConfig]):
    TOOL_PREFIX = "facts"  # the model sees `facts_lookup` (matches the instruction)

    @vf.tool
    def lookup(self, name: str) -> str:
        """Look up what a person or thing is known for."""
        return FACTS.get(name.strip().lower(), "no entry found")


class GlossaryTask(vf.Task):
    answer: str
    """The fact the `lookup` tool returns for this task's entity."""


class GlossaryConfig(vf.TasksetConfig):
    tools: vf.ToolsetConfig = vf.ToolsetConfig()


class GlossaryTaskset(vf.Taskset[GlossaryTask, GlossaryConfig]):
    def load_tasks(self) -> list[GlossaryTask]:
        return [
            GlossaryTask(
                idx=i,
                name=entity.title(),
                instruction=(
                    f'Use the `facts_lookup` tool to look up "{entity.title()}", then '
                    "reply with exactly what it returns inside <answer></answer> tags."
                ),
                answer=fact,
            )
            for i, (entity, fact) in enumerate(FACTS.items())
        ]

    def tools(self, task: GlossaryTask) -> list[vf.Toolset]:
        return [GlossaryToolset(self.config.tools)]

    @vf.reward(weight=1.0)
    async def looked_up(
        self, task: GlossaryTask, trace: vf.Trace, runtime: vf.Runtime
    ) -> float:
        # The fact only reaches the answer if the model called the MCP tool.
        last = trace.assistant_messages[-1].content if trace.assistant_messages else ""
        return float(task.answer.lower() in (last or "").lower())


def load_taskset(config: GlossaryConfig) -> GlossaryTaskset:
    return GlossaryTaskset(config)
