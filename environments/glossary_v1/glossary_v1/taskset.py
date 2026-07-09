"""glossary: a custom tool server, authored as a vf-native class.

Each task asks the model to look up an entity. The tool server is a `vf.Toolset` subclass in
`servers/facts.py` (`@vf.tool` methods, no FastMCP boilerplate); the framework launches it in its
own runtime and surfaces its `lookup` tool as `facts_lookup`. The reward checks the looked-up fact
reached the answer. The simplest tool example — contrast `wikispeedia` (per-task state),
`wiki_search` (shared), `deepwiki` (remote).
"""

import verifiers.v1 as vf

from glossary_v1.servers.facts import FACTS, GlossaryToolset


class GlossaryTaskConfig(vf.TaskConfig):
    tools: vf.ToolsetConfig = vf.ToolsetConfig()


class GlossaryTask(vf.Task[vf.State, GlossaryTaskConfig]):
    answer: str
    """The fact the `lookup` tool returns for this task's entity."""

    tools = (GlossaryToolset,)
    # Built with the task config's `tools` field (placement stays CLI-tunable via
    # --taskset.task.tools.*), resolved by `Task.server_config`.

    @vf.reward(weight=1.0)
    async def looked_up(self, trace: vf.Trace) -> float:
        # The fact only reaches the answer if the model called the MCP tool.
        last = trace.last_reply
        return float(self.answer.lower() in (last or "").lower())


class GlossaryConfig(vf.TasksetConfig):
    task: GlossaryTaskConfig = GlossaryTaskConfig()


class GlossaryTaskset(vf.Taskset[GlossaryTask, GlossaryConfig]):
    def load(self) -> list[GlossaryTask]:
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
