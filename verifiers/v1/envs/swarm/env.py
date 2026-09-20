"""Run an episode's agents against a persistent Worlds server."""

import asyncio
import json
import logging
import os
import uuid
from contextlib import AsyncExitStack

import httpx
from pydantic import Field

import verifiers.v1 as vf
from verifiers.v1.configs.env import _declared_agent_configs
from verifiers.v1.envs.swarm.task import SwarmTask
from verifiers.v1.envs.swarm.world import WorldConnection
from verifiers.v1.utils.aio import run_shielded

logger = logging.getLogger(__name__)


class WorldConfig(vf.BaseConfig):
    url: str = "http://127.0.0.1:8787"
    agent_url: str | None = None
    """Sandbox-reachable URL; defaults to url. Both must name the same server."""
    admin_token_env: str = "WORLDS_ADMIN_TOKEN"
    world_id: str | None = None
    """Attach to an existing world, or create an independent world per episode."""


class SwarmEnvConfig(vf.EnvConfig):
    agent: vf.AgentConfig = vf.AgentConfig()
    participants: dict[str, int] = Field(default_factory=lambda: {"agent": 2})
    """Copies per declared agent role. Subclasses can declare additional roles."""
    world: WorldConfig = WorldConfig()
    max_concurrent_agents: int | None = Field(2, ge=1)
    review_timeout: float = Field(600, gt=0)
    """Maximum seconds for a coordinated submission, including agent startup."""


class SwarmEnv(vf.Env[SwarmEnvConfig]):
    async def run(self, task: vf.Task, agents: vf.Agents) -> None:
        if not isinstance(task, SwarmTask):
            raise TypeError("swarm requires a SwarmTask with shared submission scoring")
        roles = _declared_agent_configs(self.config)
        for name, count in self.config.participants.items():
            if name not in roles or isinstance(count, bool) or count < 1:
                raise ValueError(
                    f"Invalid swarm participant role/count: {name}={count}"
                )
        if not self.config.participants:
            raise ValueError("A swarm needs at least one participant")
        token = os.environ.get(self.config.world.admin_token_env)
        if not token:
            raise ValueError(
                f"Set {self.config.world.admin_token_env} on the controller"
            )
        run_id = uuid.uuid4().hex[:12]
        seats = [
            (role, f"{role.replace('_', '-')[:24]}-{run_id}-{i}")
            for role, count in self.config.participants.items()
            for i in range(count)
        ]
        async with httpx.AsyncClient(
            base_url=self.config.world.url.rstrip("/"),
            headers={"Authorization": f"Bearer {token}"},
            timeout=120,
        ) as control:

            async def request(method, path, payload=None):
                response = await control.request(
                    method, "/control" + path, json=payload
                )
                response.raise_for_status()
                return response.json()

            async def deactivate(world_id, account):
                await run_shielded(
                    request(
                        "PATCH",
                        f"/worlds/{world_id}/accounts/{account}",
                        {"active": False},
                    )
                )

            # Each callback runs even if another cleanup fails. Retain accounts
            # and history, but disable every episode credential on all exit paths.
            async with AsyncExitStack() as cleanup:
                world_id = self.config.world.world_id
                controller = f"controller-{run_id}"
                if world_id is None:
                    created = await request(
                        "POST",
                        "/worlds",
                        {
                            "name": f"Swarm: {task.data.name or task.data.idx or 'episode'} — {run_id}"[
                                :128
                            ],
                            "accounts": [{"handle": controller, "role": "owner"}],
                            "services": ["chat", "forge", "decisions"],
                            "conversations": [{"handle": "general"}],
                        },
                    )
                    world_id = created["id"]
                    controller_token = created["accounts"][controller]
                else:
                    created = await request(
                        "POST",
                        f"/worlds/{world_id}/accounts",
                        {
                            "handle": controller,
                            "role": "owner",
                        },
                    )
                    controller_token = created["token"]

                def retire(account):
                    cleanup.push_async_callback(
                        deactivate,
                        world_id,
                        account,
                    )

                retire(controller)
                world = WorldConnection(
                    self.config.world.url, world_id, controller, controller_token
                )
                logger.info(
                    "Swarm world %s — viewer %s", world_id, self.config.world.url
                )
                connections = []
                for _, account in seats:
                    created = await request(
                        "POST",
                        f"/worlds/{world_id}/accounts",
                        {
                            "handle": account,
                            "role": "member",
                        },
                    )
                    retire(account)
                    connections.append(
                        WorldConnection(
                            self.config.world.agent_url or self.config.world.url,
                            world_id,
                            account,
                            created["token"],
                        )
                    )
                await task.prepare_world(world)
                accounts = [account for _, account in seats]
                review = None
                team_stop = asyncio.Event()
                review_result = {}
                if task.review_repository:
                    coordinators = [
                        account for role, account in seats if role == "coordinator"
                    ]
                    if len(coordinators) != 1:
                        raise ValueError(
                            "Revision review needs exactly one coordinator"
                        )
                    review = await world.mutate(
                        "create_review_round",
                        repository=task.review_repository,
                        electorate=accounts,
                        coordinator=coordinators[0],
                    )

                if review:
                    task.review_round = review["id"]

                async def review_progress():
                    assert review is not None
                    deadline = (
                        asyncio.get_running_loop().time() + self.config.review_timeout
                    )
                    while True:
                        state = await world.call("review_round", round_id=review["id"])
                        remaining = deadline - asyncio.get_running_loop().time()
                        if remaining <= 0:
                            review_result.update(state, termination="budget")
                            team_stop.set()
                            return
                        if state["status"] == "requested":
                            try:
                                async with asyncio.timeout(remaining):
                                    checks = await task.check_revision(
                                        world, state["candidate"]["commit_oid"]
                                    )
                            except TimeoutError:
                                review_result.update(state, termination="budget")
                                team_stop.set()
                                return
                            if not isinstance(checks.get("passed"), bool):
                                raise TypeError(
                                    "Public revision checks must return a boolean passed field"
                                )
                            try:
                                state = await world.mutate(
                                    "accept_revision",
                                    round_id=review["id"],
                                    proposal=state["proposal"],
                                    checks_passed=checks["passed"],
                                    evidence=json.dumps(checks),
                                )
                            except httpx.HTTPStatusError as exc:
                                if exc.response.status_code != 409:
                                    raise
                                # A changed candidate or withdrawn vote invalidates checks.
                                await asyncio.sleep(1)
                                continue
                        if state["status"] == "accepted":
                            review_result.update(state)
                            team_stop.set()
                            return
                        if all(run.done() for run in running):
                            review_result.update(
                                state, termination="participants_finished"
                            )
                            team_stop.set()
                            return
                        await asyncio.sleep(1)

                running = []
                async with asyncio.TaskGroup() as group:
                    for index, ((role, account), connection) in enumerate(
                        zip(seats, connections)
                    ):

                        def identify(trace, account=account):
                            trace.info["swarm"] = {
                                "world_id": world_id,
                                "account": account,
                                "run_id": run_id,
                            }

                        participant = task.participant(connection, index, accounts)
                        participant.team_stop = team_stop
                        if review:
                            participant.review_round = review["id"]
                            participant.data = participant.data.model_copy(
                                update={
                                    "prompt": participant.data.prompt_text
                                    + f"\nYour role is {role}. Review round: {review['id']}. "
                                    f"Submission coordinator: {review['coordinator']}.\n"
                                    "Structured operations (use --mutate for writes):\n"
                                    "review_round {round_id}\n"
                                    "propose_revision {round_id, commit_oid, reason}\n"
                                    "vote_revision {round_id, proposal, verdict: approve|object|withdraw, reason}\n"
                                    "request_acceptance {round_id, proposal} (coordinator only)\n"
                                    "Read repository {repository} to obtain main's exact commit. "
                                    "Any participant can propose. Read the proposal, independently check it, "
                                    "then vote on its exact ID. Everyone including the coordinator must approve. "
                                    "Only the coordinator requests acceptance. If checks fail, read evidence "
                                    "and address the problem. Remain available until status is accepted; "
                                    "wait with bash sleep 3 between polls. The controller will run public checks."
                                }
                            )
                        running.append(
                            group.create_task(
                                getattr(agents, role).run(
                                    participant,
                                    on_trace=identify,
                                )
                            )
                        )
                    if review:
                        group.create_task(review_progress())
                # Seal participant access before capturing a shared submission.
                for account in accounts:
                    await request(
                        "PATCH",
                        f"/worlds/{world_id}/accounts/{account}",
                        {"active": False},
                    )
                submission = await task.capture(world)
                if review:
                    submission["review"] = review_result
                for result in running:
                    result.result().info["swarm"]["submission"] = submission

    async def finalize(self, task: vf.Task, episode: vf.Episode) -> None:
        if not isinstance(task, SwarmTask):
            raise TypeError("swarm requires a SwarmTask")
        submission = episode.traces[0].info["swarm"]["submission"]
        reward = await task.evaluate(submission)
        for trace in episode.traces:
            trace.record_reward("team", reward)
