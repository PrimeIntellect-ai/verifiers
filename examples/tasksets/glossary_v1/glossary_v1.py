"""glossary: a custom COLOCATED tool server, authored as a vf-native class.

Each task asks the model to look up an entity. The taskset declares its tool server as a
`vf.Toolset` subclass with `@vf.tool` methods (no FastMCP boilerplate, no separate server
file): the framework serializes it, launches it in the harness's runtime, and surfaces its
`lookup` tool as `facts_lookup`. The reward checks the looked-up fact reached the answer.

This is the colocated example (`tools.colocated=True`, the default): the server is small and
self-contained, so it runs *inside the harness's own runtime*, reached over localhost with no
tunnel. The right placement for a lightweight, per-rollout tool — contrast `wikispeedia` (its
own runtime), `wiki_search` (shared), and `deepwiki` (a remote URL).
"""

import json
from pathlib import Path

import verifiers.v1 as vf

HERE = Path(__file__).resolve().parent
FACTS: dict[str, str] = json.loads((HERE / "facts.json").read_text())


class GlossaryToolset(vf.Toolset[vf.ToolsetConfig]):
    # `FACTS` is global state (the corpus, loaded once at import) — not a config knob and not
    # per-task, so the tool reads it directly. No subclass config: it has no knobs of its own.
    @vf.tool
    def lookup(self, name: str) -> str:
        """Look up what a person or thing is known for."""
        return FACTS.get(name.strip().lower(), "no entry found")


class GlossaryTask(vf.Task):
    answer: str
    """The fact the `lookup` tool returns for this task's entity."""


class GlossaryTaskset(vf.Taskset[GlossaryTask, vf.TasksetConfig]):
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
        return [GlossaryToolset(vf.ToolsetConfig(name="facts"))]

    @vf.reward(weight=1.0)
    async def looked_up(
        self, task: GlossaryTask, trace: vf.Trace, runtime: vf.Runtime
    ) -> float:
        # The fact only reaches the answer if the model called the MCP tool.
        last = trace.assistant_messages[-1].content if trace.assistant_messages else ""
        return float(task.answer.lower() in (last or "").lower())


def load_taskset(config: vf.TasksetConfig) -> GlossaryTaskset:
    return GlossaryTaskset(config)
