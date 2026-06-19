"""Run a prebuilt OpenEnv image in the rollout container."""

import asyncio
import json
from pathlib import Path
from typing import Literal

import aiohttp

import verifiers.v1 as vf

OPENENV_SOCKET = "/tmp/openenv.sock"
OPENENV_URL = "http://openenv"


class OpenEnvTask(vf.Task):
    seed: int


class OpenEnvState(vf.State):
    reward: float = 0.0
    done: bool = False


class OpenEnvConfig(vf.TasksetConfig):
    image: str
    """Prebuilt OpenEnv image."""
    contract: Literal["gym", "mcp"]
    """OpenEnv interaction contract exposed by the image."""
    num_tasks: int = 100
    """How many seeded episodes to expose."""
    seed: int = 0
    """First episode seed."""
    prompt: str = (
        "Use the available OpenEnv tools to interact with the environment, then answer."
    )
    """Opening prompt for MCP environments."""
    system_prompt: str | None = None


class OpenEnvUser(vf.User[vf.UserConfig, OpenEnvState]):
    """Drive an OpenEnv gym episode through the v1 user-simulator interface."""

    async def setup_task(self, task: OpenEnvTask) -> None:
        self.task = task
        async with asyncio.timeout(60):
            while not Path(OPENENV_SOCKET).exists():
                await asyncio.sleep(0.1)
        connector = aiohttp.UnixConnector(path=OPENENV_SOCKET)
        async with aiohttp.ClientSession(connector=connector) as session:
            async with session.get(f"{OPENENV_URL}/schema") as response:
                response.raise_for_status()
                self.action_schema = (await response.json())["action"]
        self.socket = None

    async def respond(self, message: str) -> vf.Messages:
        if self.socket is None:
            connector = aiohttp.UnixConnector(path=OPENENV_SOCKET)
            self.session = aiohttp.ClientSession(connector=connector)
            self.socket = await self.session.ws_connect(f"{OPENENV_URL}/ws")
            payload = {"type": "reset", "data": {"seed": self.task.seed}}
        else:
            cleaned = message.strip()
            if cleaned.startswith("```") and cleaned.endswith("```"):
                cleaned = "\n".join(cleaned.splitlines()[1:-1]).strip()
            try:
                action = json.loads(cleaned)
            except json.JSONDecodeError:
                action = cleaned
            if not isinstance(action, dict):
                fields = self.action_schema.get("required") or list(
                    self.action_schema.get("properties", {})
                )
                if len(fields) != 1:
                    raise ValueError(
                        "Return a JSON object matching the OpenEnv action schema."
                    )
                action = {fields[0]: action}
            payload = {"type": "step", "data": action}

        await self.socket.send_json(payload)
        response = await self.socket.receive_json()
        if response.get("type") == "error":
            raise RuntimeError(response["data"]["message"])
        result = response["data"]
        self.state.reward += result.get("reward") or 0.0
        self.state.done = result.get("done", False)

        observation = result.get("observation")
        content = observation
        if isinstance(observation, dict):
            messages = observation.get("messages") or []
            content = (
                messages[-1]
                if messages
                else observation.get("prompt") or json.dumps(observation)
            )
            if isinstance(content, dict):
                content = content.get("content")

        if self.state.done:
            await self.socket.close()
            await self.session.close()
        return [{"role": "user", "content": str(content)}]


class OpenEnvTaskset(vf.Taskset[OpenEnvTask, OpenEnvConfig, OpenEnvState]):
    NEEDS_CONTAINER = True

    def load_tasks(self) -> list[OpenEnvTask]:
        return [
            OpenEnvTask(
                idx=index,
                name=f"openenv-{self.config.contract}-{self.config.seed + index}",
                prompt=None if self.config.contract == "gym" else self.config.prompt,
                system_prompt=self.config.system_prompt,
                image=self.config.image,
                workdir="/app/env",
                seed=self.config.seed + index,
            )
            for index in range(self.config.num_tasks)
        ]

    async def setup(self, task: OpenEnvTask, runtime: vf.Runtime) -> None:
        await runtime.run_background(
            [
                "/app/.venv/bin/uvicorn",
                "server.app:app",
                "--uds",
                OPENENV_SOCKET,
            ],
            {"ENABLE_WEB_INTERFACE": "false"},
            "openenv.log",
        )

    def tools(self, task: OpenEnvTask) -> list[vf.Toolset]:
        if self.config.contract == "mcp":
            return [
                vf.JSONRPCToolset(
                    vf.JSONRPCToolsetConfig(
                        colocated=True,
                        endpoint=f"{OPENENV_URL}/mcp",
                        uds=OPENENV_SOCKET,
                    )
                )
            ]
        return []

    def user(self, task: OpenEnvTask) -> vf.User | None:
        if self.config.contract == "gym":
            return OpenEnvUser(vf.UserConfig(colocated=True))
        return None

    @vf.stop
    async def openenv_done(self, trace: vf.Trace) -> bool:
        return trace.state.done

    @vf.reward(weight=1.0)
    async def openenv_reward(self, trace: vf.Trace) -> float:
        return trace.state.reward


if __name__ == "__main__":
    OpenEnvUser.run()
