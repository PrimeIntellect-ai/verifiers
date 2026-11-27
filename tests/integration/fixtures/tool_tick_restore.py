"""tool-tick-restore: a task whose tool server holds a counter only in its memory. Each
`tick` call takes the next number and appends it to the rollout state. The state ends
as 1..N only if every call ran exactly once on a server that kept its memory: a server
that started over, a call that ran twice, or one whose number never reached the host
leaves another sequence.

A call first sleeps, so a save requested when a turn lands on the trace finds a call
under way between the box and the server. Place the server on a Prime VM of its own
(`taskset.task.tools.runtime`) to checkpoint it with the box.
"""

import asyncio

from pydantic import Field

import verifiers.v1 as vf

TURNS = 4
HOLD_SECONDS = 3
EXPECTED = list(range(1, TURNS + 1))


class TickState(vf.State):
    ticks: list[int] = Field(default_factory=list)


class TickerToolset(vf.Toolset[vf.ToolsetConfig, TickState]):
    TOOL_PREFIX = "ticker"
    count = 0

    @vf.tool
    async def tick(self) -> str:
        """Take the next number."""
        await asyncio.sleep(HOLD_SECONDS)
        self.count += 1
        self.state.ticks.append(self.count)
        return str(self.count)


class ToolTickTaskConfig(vf.TaskConfig):
    tools: vf.ToolsetConfig = vf.ToolsetConfig()


class ToolTickTask(vf.Task[vf.TaskData, TickState, ToolTickTaskConfig]):
    @classmethod
    def toolsets(cls, config: ToolTickTaskConfig) -> list[vf.Toolset]:
        return [TickerToolset(config.tools)]

    @vf.reward(weight=1.0)
    async def counted(self, trace: vf.Trace) -> float:
        """The state holds 1..TURNS, in order."""
        return float(trace.state.ticks == EXPECTED)


class ToolTickConfig(vf.TasksetConfig):
    task: ToolTickTaskConfig = ToolTickTaskConfig()


class ToolTickTaskset(vf.Taskset[ToolTickTask, ToolTickConfig]):
    def load(self) -> list[ToolTickTask]:
        return [
            ToolTickTask(
                vf.TaskData(
                    idx=0,
                    prompt=(
                        f"Call the `tick` tool from the `ticker` MCP server {TURNS} "
                        "times, one call per turn."
                    ),
                ),
                self.config.task,
            )
        ]


__all__ = ["ToolTickTaskset"]


if __name__ == "__main__":
    TickerToolset.run()
