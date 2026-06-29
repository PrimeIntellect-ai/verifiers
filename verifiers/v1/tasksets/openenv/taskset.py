"""Run a prebuilt OpenEnv image as a Verifiers taskset."""

from __future__ import annotations

import json
from typing import Any, Literal

import aiohttp
from pydantic import Field

from verifiers.v1.decorators import reward, stop
from verifiers.v1.mcp import (
    JSONRPCToolset,
    JSONRPCToolsetConfig,
    Toolset,
    User,
    UserConfig,
)
from verifiers.v1.runtimes import Runtime
from verifiers.v1.state import State
from verifiers.v1.task import Task, TaskResources
from verifiers.v1.taskset import Taskset, TasksetConfig
from verifiers.v1.types import Messages

OPENENV_SOCKET = "/tmp/openenv.sock"
OPENENV_URL = "http://openenv"
ACTION_INSTRUCTIONS = (
    "Respond with only a JSON object matching this OpenEnv action schema. "
    "If the schema has one required string field, you may reply with just that string value."
)


class OpenEnvState(State):
    done: bool = False
    reward: float = 0.0
    action_schema: dict[str, Any] = Field(default_factory=dict)


class OpenEnvConfig(TasksetConfig):
    image: str | None = None
    """Prebuilt OpenEnv image."""
    contract: Literal["mcp", "gym"] = "mcp"
    """OpenEnv runtime contract exposed by the image."""
    prompt: str | None = None
    """Opening prompt for MCP environments. Gym environments open from reset()."""
    workdir: str = "/app/env"
    """Environment project directory inside the image."""
    app: str = "server.app:app"
    """ASGI app from the image's OpenEnv manifest."""
    system_prompt: str | None = None
    resources: TaskResources = TaskResources()
    reset: dict[str, Any] = Field(default_factory=dict)
    """Extra reset kwargs for gym environments."""


class OpenEnvUserConfig(UserConfig):
    colocated: bool = True
    base_url: str = OPENENV_URL
    uds: str = OPENENV_SOCKET
    reset: dict[str, Any] = Field(default_factory=dict)
    timeout: float = 60.0


class OpenEnvUser(User[OpenEnvUserConfig, OpenEnvState]):
    async def setup_task(self, task) -> None:
        self.session: aiohttp.ClientSession | None = None
        self.ws: aiohttp.ClientWebSocketResponse | None = None
        self.action_schema: dict[str, Any] = {}
        self.initial: dict[str, Any] | None = None

    async def respond(self, message: str) -> Messages:
        await self._connect()
        if message:
            result = await self._send(
                {"type": "step", "data": self._parse_action(message)}
            )
        else:
            if self.initial is None:
                self.initial = await self._send(
                    {"type": "reset", "data": self.config.reset}
                )
            result = self.initial

        data = result.get("data", {})
        self.state.done = bool(data.get("done", False))
        self.state.reward += float(data.get("reward") or 0.0)
        self.state.action_schema = self.action_schema
        if self.state.done:
            await self._close()
        return self._render(data.get("observation", {}))

    async def _connect(self) -> None:
        if self.ws is not None:
            return
        connector = aiohttp.UnixConnector(path=self.config.uds)
        timeout = aiohttp.ClientTimeout(total=self.config.timeout)
        self.session = aiohttp.ClientSession(connector=connector, timeout=timeout)
        self.ws = await self.session.ws_connect(f"{self.config.base_url}/ws")
        self.action_schema = await self._schema()

    async def _close(self) -> None:
        if self.ws is not None:
            await self.ws.close()
            self.ws = None
        if self.session is not None:
            await self.session.close()
            self.session = None

    async def _schema(self) -> dict[str, Any]:
        if self.session is None:
            raise RuntimeError("OpenEnv gym client is not connected.")
        async with self.session.get(f"{self.config.base_url}/schema") as response:
            response.raise_for_status()
            payload = await response.json()
        schema = payload.get("action", {})
        return schema if isinstance(schema, dict) else {}

    async def _send(self, payload: dict[str, Any]) -> dict[str, Any]:
        if self.ws is None:
            raise RuntimeError("OpenEnv gym client is not connected.")
        await self.ws.send_json(payload)
        message = await self.ws.receive_json()
        if message.get("type") == "error":
            raise RuntimeError(message.get("data", {}).get("message", str(message)))
        return message

    def _parse_action(self, message: str) -> dict[str, Any]:
        text = message.strip()
        if text.startswith("```"):
            text = text.strip("`").removeprefix("json").strip()
        try:
            action = json.loads(text)
            if isinstance(action, dict):
                return action
        except json.JSONDecodeError:
            pass

        properties = self.action_schema.get("properties", {})
        required = self.action_schema.get("required", [])
        if (
            isinstance(properties, dict)
            and isinstance(required, list)
            and len(required) == 1
            and properties.get(required[0], {}).get("type") == "string"
        ):
            return {required[0]: text}
        raise ValueError(
            "OpenEnv gym actions must be JSON objects matching the action schema."
        )

    def _render(self, observation: object) -> Messages:
        schema = json.dumps(self.action_schema, ensure_ascii=True)
        suffix = f"\n\n{ACTION_INSTRUCTIONS}\n{schema}"
        if isinstance(observation, dict):
            messages = observation.get("messages")
            if isinstance(messages, list) and messages:
                rendered = [
                    {"role": "user", "content": str(item.get("content", item))}
                    if isinstance(item, dict)
                    else {"role": "user", "content": str(item)}
                    for item in messages
                ]
                rendered[-1]["content"] += suffix
                return rendered
            for key in ("prompt", "question", "instruction", "content", "text"):
                value = observation.get(key)
                if isinstance(value, str) and value.strip():
                    return [{"role": "user", "content": f"{value.strip()}{suffix}"}]
        content = json.dumps(observation, ensure_ascii=True, default=str)
        return [{"role": "user", "content": f"{content}{suffix}"}]


class OpenEnvTaskset(Taskset[Task, OpenEnvConfig, OpenEnvState]):
    NEEDS_CONTAINER = True

    def load_tasks(self) -> list[Task]:
        if self.config.image is None:
            raise ValueError("OpenEnv tasksets require `image`.")
        if self.config.contract == "mcp" and self.config.prompt is None:
            raise ValueError("OpenEnv MCP tasksets require `prompt`.")
        return [
            Task(
                idx=0,
                name=self.config.name,
                prompt=self.config.prompt if self.config.contract == "mcp" else None,
                system_prompt=self.config.system_prompt,
                image=self.config.image,
                workdir=self.config.workdir,
                resources=self.config.resources,
            )
        ]

    async def setup(self, task: Task, runtime: Runtime) -> None:
        await runtime.run_background(
            [
                "/app/.venv/bin/uvicorn",
                self.config.app,
                "--uds",
                OPENENV_SOCKET,
            ],
            {"ENABLE_WEB_INTERFACE": "false"},
            "/tmp/openenv.log",
        )

    def has_tools(self) -> bool:
        return self.config.contract == "mcp"

    def tools(self, task: Task) -> list[Toolset]:
        if self.config.contract != "mcp":
            return []
        return [
            JSONRPCToolset(
                JSONRPCToolsetConfig(
                    colocated=True,
                    endpoint=f"{OPENENV_URL}/mcp",
                    uds=OPENENV_SOCKET,
                )
            )
        ]

    def has_user(self) -> bool:
        return self.config.contract == "gym"

    def user(self, task: Task) -> User | None:
        if self.config.contract != "gym":
            return None
        return OpenEnvUser(OpenEnvUserConfig(reset=self.config.reset))

    @stop
    async def openenv_done(self, trace) -> bool:
        return trace.state.done

    @reward(weight=1.0)
    async def openenv_reward(self, trace) -> float:
        return trace.state.reward


if __name__ == "__main__":
    OpenEnvUser.run()
