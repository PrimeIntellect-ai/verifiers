"""glossary: a custom tool server, authored as a vf-native class.

Each task asks the model to look up an entity. The tool server is a `vf.Toolset` subclass in
`servers/facts.py` (`@vf.tool` methods, no FastMCP boilerplate); the framework launches it in its
own runtime and surfaces its `lookup` tool as `facts_lookup`. The reward checks the looked-up fact
reached the answer. The simplest tool example — contrast `wikispeedia` (per-task state),
`wiki_search` (shared), `deepwiki` (remote).
"""

import verifiers.v1 as vf

from glossary_v1.servers.facts import FACTS, GlossaryToolset


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
                prompt=(
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
        last = trace.final_assistant_content or ""
        return float(task.answer.lower() in last.lower())
