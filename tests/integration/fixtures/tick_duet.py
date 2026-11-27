"""tick-duet: two seats each run the tick-restore task at once, each in a box of its own.

The fan-out shape for episode checkpointing: neither seat waits for the other, so a save
can find both running, or one finished and the other running.
"""

import asyncio

from tick_restore import TickRestoreTaskset

import verifiers.v1 as vf


class TickDuetEnvConfig(vf.EnvConfig):
    a: vf.AgentConfig = vf.AgentConfig()
    b: vf.AgentConfig = vf.AgentConfig()


class TickDuetEnv(vf.Env[TickDuetEnvConfig]):
    async def run(self, task, agents):
        await asyncio.gather(agents.a.run(task), agents.b.run(task))


class TickDuetTaskset(TickRestoreTaskset):
    pass


__all__ = ["TickDuetEnv", "TickDuetTaskset"]
