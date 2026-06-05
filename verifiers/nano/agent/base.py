"""The agent: a program that runs in a runtime and drives the conversation.

An `Agent` provisions itself into the (already-started) runtime and runs there;
its model calls hit the interception server, which records the turns. Concrete
agents differ only in how they provision + invoke — `DefaultAgent` stages a small
script and runs `python3`; `RLMAgent` installs the rlm binary and runs it. The
runtime and the interception server are owned by the Environment.
"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from verifiers.nano.context import RolloutContext
from verifiers.nano.errors import ProgramError
from verifiers.nano.runtime import Runtime
from verifiers.nano.transcript import Transcript

if TYPE_CHECKING:
    from verifiers.nano.agent import AgentConfig


class Agent(ABC):
    def __init__(self, config: "AgentConfig") -> None:
        self.config = config

    async def run(
        self,
        ctx: RolloutContext,
        transcript: Transcript,
        runtime: Runtime,
        endpoint: str,
        secret: str,
    ) -> None:
        """Provision and run the agent in `runtime`; its model calls reach the
        interception server at `endpoint`."""
        env = {
            "OPENAI_BASE_URL": endpoint,
            "OPENAI_API_KEY": secret,
            "OPENAI_MODEL": ctx.model,
            "RLM_MODEL": ctx.model,
            **self.config.env,
        }
        command = await self.prepare(runtime, env)
        result = await runtime.run([*command, transcript.task.instruction], env)
        if transcript.stop_condition is not None:
            return  # a @stop refused a turn mid-rollout; the agent's exit is expected
        if result.exit_code != 0:
            raise ProgramError(
                f"agent exited {result.exit_code}: {result.stderr[:1000]}"
            )
        transcript.stop("agent_completed")

    @abstractmethod
    async def prepare(self, runtime: Runtime, env: dict[str, str]) -> list[str]:
        """Provision the agent in the runtime; return its run command (the argv
        prefix, before the task instruction). `env` may be mutated (e.g. PATH)."""
