from contextlib import AsyncExitStack

import verifiers.v1 as vf
from verifiers.v1.tasksets.tau.harness import TauHarness


class TauEnvConfig(vf.EnvConfig):
    assistant: vf.AgentConfig = vf.AgentConfig(runtime=vf.SubprocessConfig())
    user: vf.AgentConfig = vf.AgentConfig(
        runtime=vf.SubprocessConfig(),
        model="openai/gpt-4.1-2025-04-14",
        sampling={"temperature": 0},
    )
    judge: vf.AgentConfig = vf.AgentConfig(
        runtime=vf.SubprocessConfig(),
        model="openai/gpt-4.1-2025-04-14",
        sampling={"temperature": 0},
    )


class TauEnv(vf.Env[TauEnvConfig]):
    async def setup(self, agents):
        for role in ("assistant", "user", "judge"):
            agent = getattr(agents, role)
            if not isinstance(agent.harness, TauHarness):
                raise TypeError(f"Tau requires its official harness for {role}")
            agent.trainable = role == "assistant"

    async def run(self, task, agents):
        # The native role sessions own interception and traces; Tau owns all turns.
        async with (
            agents.assistant.provision(task) as runtime,
            AsyncExitStack() as stack,
        ):
            for role in ("user", "judge"):
                await stack.enter_async_context(
                    getattr(agents, role).interaction(
                        vf.Task(vf.TaskData()), runtime=runtime
                    )
                )
            async with agents.assistant.interaction(task, runtime=runtime) as assistant:
                agents.assistant.harness.peers = {
                    role: getattr(agents, role).harness.connection
                    for role in ("user", "judge")
                }
                await assistant.turn()
