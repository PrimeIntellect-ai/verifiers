"""The simplest locally authored, task-scoped tool example.

`GlossaryToolset` is a vf-native class with an `@vf.tool` method. Verifiers launches
one server per rollout and exposes it as `facts_lookup`; compare `deepwiki` for a remote
server and `wiki_search` for an expensive worker-shared server.
"""

import verifiers.v1 as vf
from glossary.servers.facts import FACTS, GlossaryToolset
from verifiers.v1.types import content_text


class GlossaryTaskConfig(vf.TaskConfig):
    tools: vf.ToolsetConfig = vf.ToolsetConfig()


class GlossaryTaskData(vf.TaskData):
    answer: str
    """The fact the `lookup` tool returns for this task's entity."""


def tagged_answer(reply: str) -> str:
    _, opened, rest = reply.partition("<answer>")
    answer, closed, _ = rest.partition("</answer>")
    return answer.strip() if opened and closed else ""


class GlossaryTask(vf.Task[GlossaryTaskData, vf.State, GlossaryTaskConfig]):
    @classmethod
    def toolsets(cls, config: GlossaryTaskConfig) -> list[vf.Toolset]:
        return [GlossaryToolset(config.tools)]

    @vf.reward(weight=1.0)
    async def looked_up(self, trace: vf.Trace) -> float:
        fact = self.data.answer
        returned = any(
            content_text(message.content).strip() == fact
            for message in trace.tool_messages
        )
        reported = tagged_answer(trace.last_reply).lower() == fact.lower()
        return float(returned and reported)


class GlossaryConfig(vf.TasksetConfig):
    task: GlossaryTaskConfig = GlossaryTaskConfig()


class GlossaryTaskset(vf.Taskset[GlossaryTask, GlossaryConfig]):
    def load(self) -> list[GlossaryTask]:
        return [
            GlossaryTask(
                GlossaryTaskData(
                    name=entity.title(),
                    prompt=(
                        f'Use the `facts_lookup` tool to look up "{entity.title()}", then '
                        "reply with exactly what it returns inside <answer></answer> tags."
                    ),
                    answer=fact,
                ),
                self.config.task,
            )
            for entity, fact in FACTS.items()
        ]
