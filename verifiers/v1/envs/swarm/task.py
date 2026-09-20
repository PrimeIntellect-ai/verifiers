"""Task-owned world preparation and team scoring for SwarmEnv."""

import asyncio
from abc import abstractmethod
from typing import Any

import verifiers.v1 as vf
from verifiers.v1.envs.swarm.world import WORLD_PROGRAM, WorldConnection


class SwarmTask(vf.Task):
    """Adapt a task to shared state without putting orchestration in Worlds.

    Override prepare_world, participant_prompt, capture, and evaluate. Capture
    must identify immutable submissions (for example a Git commit), not a moving
    branch. Evaluate runs on the host after participant credentials are revoked.
    A benchmark adapter can launch its independent verifier there.
    """

    NEEDS_CONTAINER = True
    connection: WorldConnection | None = None
    review_repository: str | None = None
    review_round: str | None = None
    team_stop: asyncio.Event | None = None

    @vf.stop
    def team_finished(self, trace: vf.Trace) -> bool:
        return self.team_stop is not None and self.team_stop.is_set()

    async def check_revision(self, world: WorldConnection, commit_oid: str) -> dict:
        """Run public checks on the proposed commit; return passed and evidence."""
        raise NotImplementedError("A review task must implement public revision checks")

    async def prepare_world(self, world: WorldConnection) -> None:
        """Seed this episode's shared resources before any participant runs."""

    def participant_prompt(self, index: int, accounts: list[str]) -> str:
        return self.data.prompt_text

    def participant(
        self, world: WorldConnection, index: int, accounts: list[str]
    ) -> "SwarmTask":
        task = self.defer_scoring()
        task.connection = world
        task.data = self.data.model_copy(
            update={
                "prompt": self.participant_prompt(index, accounts)
                + f"\n\nYour world account is {world.account}. Collaborators: {', '.join(accounts)}."
                "\nUse `uv run python /tmp/world.py OPERATION 'JSON'` to read the world. "
                "Append `--mutate` for writes. Credentials are already in your environment; "
                "never print them. Use the shared general channel for communication."
            }
        )
        return task

    def runtime_env(self) -> dict[str, str]:
        if self.connection is None:
            return {}
        return {
            "WORLDS_URL": self.connection.url,
            "WORLDS_ID": self.connection.world_id,
            "WORLDS_TOKEN": self.connection.token,
        }

    async def setup(self, trace: vf.Trace, runtime: vf.Runtime) -> None:
        await runtime.write("/tmp/world.py", WORLD_PROGRAM.encode())

    @abstractmethod
    async def capture(self, world: WorldConnection) -> dict[str, Any]:
        """Capture the shared submission after participants stop."""

    @abstractmethod
    async def evaluate(self, submission: dict[str, Any]) -> float:
        """Score that immutable submission once for the whole team."""
