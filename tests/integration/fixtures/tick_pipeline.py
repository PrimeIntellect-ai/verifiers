"""tick-pipeline: one seat runs the tick-restore task, then a second seat runs a task built
from the first seat's result, each in a box of its own.

The sequential shape for episode checkpointing: the second seat's task depends on a value
the first seat's trace holds, so a continued episode reaches the same second task only
if the first seat's trace is replayed exactly. `note` changes that task without changing
the first seat's, which makes a replay diverge on purpose.
"""

from tick_restore import TickRestoreTaskset

import verifiers.v1 as vf


class TickPipelineEnvConfig(vf.EnvConfig):
    a: vf.AgentConfig = vf.AgentConfig()
    b: vf.AgentConfig = vf.AgentConfig()
    note: str = ""


class TickPipelineEnv(vf.Env[TickPipelineEnvConfig]):
    async def run(self, task, agents):
        first = await agents.a.run(task)
        prompt = f"{task.data.prompt_text}\n\nThe previous run ended with:\n{first.last_reply}"
        if self.config.note:
            prompt += f"\n\n{self.config.note}"
        second = type(task)(
            task.data.model_copy(update={"prompt": prompt}), task.config
        )
        await agents.b.run(second)


class TickPipelineTaskset(TickRestoreTaskset):
    pass


__all__ = ["TickPipelineEnv", "TickPipelineTaskset"]
