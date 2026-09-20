"""Run an episode's agents against a persistent Worlds server."""

import asyncio
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

                        running.append(
                            group.create_task(
                                getattr(agents, role).run(
                                    task.participant(connection, index, accounts),
                                    on_trace=identify,
                                )
                            )
                        )
                # Seal participant access before capturing a shared submission.
                for account in accounts:
                    await request(
                        "PATCH",
                        f"/worlds/{world_id}/accounts/{account}",
                        {"active": False},
                    )
                submission = await task.capture(world)
                for result in running:
                    result.result().info["swarm"]["submission"] = submission

    async def finalize(self, task: vf.Task, episode: vf.Episode) -> None:
        if not isinstance(task, SwarmTask):
            raise TypeError("swarm requires a SwarmTask")
        submission = episode.traces[0].info["swarm"]["submission"]
        reward = await task.evaluate(submission)
        for trace in episode.traces:
            trace.record_reward("team", reward)
