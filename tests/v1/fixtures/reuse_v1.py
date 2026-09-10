"""reuse: the smallest env that keeps a seat's box across episodes.

The runtime-pool fixture for the v1 e2e suite (resolved by id `reuse-v1`): an `Env`
whose `run()` provisions the seat's box under a reuse key and plays the task in it —
with `--env.runtimes` set, every episode of the run lands in one box; without it, each
episode provisions and tears down its own, exactly as `provision(task)` does."""

from echo_v1 import EchoTaskset

import verifiers.v1 as vf


class ReuseEnvConfig(vf.EnvConfig):
    agent: vf.AgentConfig = vf.AgentConfig()


class ReuseEnv(vf.Env[ReuseEnvConfig]):
    async def run(self, task, agents):
        async with agents.agent.provision(task, reuse="seat") as box:
            await agents.agent.run(task, runtime=box)


class ReuseTaskset(EchoTaskset):
    pass


__all__ = ["ReuseEnv", "ReuseTaskset"]
