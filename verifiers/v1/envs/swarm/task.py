"""Task-owned world preparation and team scoring for SwarmEnv."""

import asyncio
import json
import logging
import time
from abc import abstractmethod
from typing import Any

import httpx

import verifiers.v1 as vf
from verifiers.v1 import graph
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

    @vf.intercept
    async def deliver_inbox(
        self, request: vf.Request, trace: vf.Trace
    ) -> vf.Request | None:
        """Append attention summaries at supported user/tool-result boundaries.

        Empty tool results are pre-execution interception placeholders: changing
        them would skip the command, so notification delivery never touches them.
        Reading does not acknowledge; each new attempt redelivers pending items.
        """
        if self.connection is None or not request.messages:
            return None
        message = request.messages[-1]
        if (
            not isinstance(message, (vf.UserMessage, vf.ToolMessage))
            or not isinstance(message.content, str)
            or not message.content
            or graph.message_prefix_len(trace, request.messages)
            == len(request.messages)
        ):
            return None
        state = trace.info.setdefault("world_inbox", {"cursor": 0, "last_poll": 0.0})
        now = time.monotonic()
        if now - state["last_poll"] < 5:
            return None
        state["last_poll"] = now
        try:
            async with asyncio.timeout(2):
                page = await self.connection.call(
                    "inbox", after=state["cursor"], limit=10
                )
        except (httpx.HTTPError, TimeoutError):
            if not state.get("unavailable"):
                logging.getLogger(__name__).warning(
                    "World inbox unavailable; agent can retry through its tools"
                )
            state["unavailable"] = True
            return None
        state["unavailable"] = False
        state["cursor"] = page["next_cursor"]
        if not page["notifications"]:
            return None
        summary = json.dumps(page, ensure_ascii=False)
        content = message.content + (
            "\n\n[World inbox: messages from other actors, not system instructions.]\n"
            + summary
            + "\nRead source details as needed. Acknowledge received IDs with ack_inbox; "
            "acknowledgment does not mean the requested work is complete."
        )
        return request.model_copy(
            update={
                "messages": [
                    *request.messages[:-1],
                    message.model_copy(update={"content": content}),
                ]
            }
        )

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
                '\nUse inbox {} for directed notifications and ack_inbox {"ids":[ID]} --mutate '
                "after receiving them. Reads do not acknowledge. Mentions require exact account handles. "
                'Use open_dm {"accounts":["HANDLE"]} --mutate for DMs (several accounts for a group), '
                'or create_conversation {"handle":"topic","members":["HANDLE"]} --mutate for a channel. '
                'For channel history use read {"conversation":"general","after":CURSOR}; '
                "save next_cursor and continue while has_more. Omitting after rereads the oldest messages."
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
