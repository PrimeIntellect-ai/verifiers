"""The simplest locally authored, task-scoped tool example.

`GlossaryToolset` is a vf-native class with an `@vf.tool` method. Verifiers launches
one server per rollout and exposes it as `facts_lookup`; compare `deepwiki` for a remote
server and `wiki_search` for an expensive worker-shared server.
"""

import verifiers.v1 as vf

from glossary_v1.servers.facts import FACTS, GlossaryToolset


class GlossaryTaskConfig(vf.TaskConfig):
    tools: vf.ToolsetConfig = vf.ToolsetConfig()


class GlossaryTaskData(vf.TaskData):
    answer: str
    """The fact the `lookup` tool returns for this task's entity."""


class GlossaryTask(vf.Task[GlossaryTaskData, vf.State, GlossaryTaskConfig]):
    tools = (GlossaryToolset,)

    @vf.reward(weight=1.0)
    async def looked_up(self, trace: vf.Trace) -> float:
        # The fact only reaches the answer if the model called the MCP tool.
        last = trace.last_reply
        return float(self.data.answer.lower() in (last or "").lower())


class GlossaryConfig(vf.TasksetConfig):
    task: GlossaryTaskConfig = GlossaryTaskConfig()


class GlossaryTaskset(vf.Taskset[GlossaryTask, GlossaryConfig]):
    def load(self) -> list[GlossaryTask]:
        return [
            GlossaryTask(
                GlossaryTaskData(
                    idx=i,
                    name=entity.title(),
                    prompt=(
                        f'Use the `facts_lookup` tool to look up "{entity.title()}", then '
                        "reply with exactly what it returns inside <answer></answer> tags."
                    ),
                    answer=fact,
                ),
                self.config.task,
            )
            for i, (entity, fact) in enumerate(FACTS.items())
        ]
