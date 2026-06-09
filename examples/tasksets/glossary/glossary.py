"""glossary: a custom COLOCATED tool server.

Each task asks the model to look up an entity. The taskset ships a tiny tool server
(`server.py`, a single-file uv script) and declares it via `tool_servers`, passing the
facts in through an env var; the harness surfaces its `lookup` tool as `facts_lookup`, and
the reward checks the looked-up fact reached the answer.

This is the colocated example (`tools.colocated=True`, the default): the server is small
and self-contained (its only runtime dep is `uv`), so it runs *inside the harness's own
runtime*, reached over localhost with no tunnel. The right placement for a lightweight,
per-rollout tool — contrast `wikispeedia` (its own runtime), `wiki_search` (shared), and
`deepwiki` (a remote URL).
"""

import json
from pathlib import Path

import verifiers.v1 as vf

HERE = Path(__file__).resolve().parent
SERVER = (HERE / "server.py").read_bytes()
FACTS: dict[str, str] = json.loads((HERE / "facts.json").read_text())


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

    def tool_servers(self, task: GlossaryTask) -> list[vf.ToolServer]:
        return [
            vf.ToolServer(
                name="facts", script=SERVER, env={"FACTS_JSON": json.dumps(FACTS)}
            )
        ]

    @vf.reward(weight=1.0)
    async def looked_up(
        self, task: GlossaryTask, trace: vf.Trace, runtime: vf.Runtime
    ) -> float:
        # The fact only reaches the answer if the model called the MCP tool.
        last = trace.assistant_messages[-1].content if trace.assistant_messages else ""
        return float(task.answer.lower() in (last or "").lower())


def load_taskset(config: vf.TasksetConfig) -> GlossaryTaskset:
    return GlossaryTaskset(config)
