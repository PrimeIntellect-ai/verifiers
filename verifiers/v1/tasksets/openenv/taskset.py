"""Run OpenEnv environments with UV by default, or Docker when requested.

The engine plays the user: the env's `run()` opens the model's seat as a chat
session and steps the OpenEnv client host-side — each assistant action advances the
environment, the next observation comes back as the user turn, and a `done` result
ends the exchange. OpenEnv's per-step rewards are summed onto the seat's trace
(`openenv_reward`)."""

import json
from collections.abc import Iterator
from typing import Any, Self

import httpx
from pydantic import model_validator

import verifiers.v1 as vf


class OpenEnvData(vf.TaskData):
    env: str | None
    base_url: str | None
    use_docker: bool
    provider_kwargs: dict[str, Any]
    reset: dict[str, Any]


class OpenEnvConfig(vf.TasksetConfig):
    env: str | None = None
    """Environment id passed to OpenEnv. Required unless `base_url` is set."""
    base_url: str | None = None
    """Connect to an existing OpenEnv server instead of starting `env`."""
    use_docker: bool = False
    """Use OpenEnv's Docker provider instead of the default UV provider. The engine
    runs host-side (in the eval process), so this needs Docker on the host."""
    provider_kwargs: dict[str, Any] = {}
    """Extra arguments for OpenEnv's provider (`GenericEnvClient.from_env`)."""
    resets: list[dict[str, Any]] = [{}]
    """One finite task per set of arguments passed to OpenEnv's `reset`."""

    @model_validator(mode="after")
    def validate_config(self) -> Self:
        if not self.env and not self.base_url:
            raise ValueError("pass `env` or `base_url`")
        return self


class OpenEnvTask(vf.Task[OpenEnvData]):
    pass


def parse_action(message: str, action_schema: dict[str, Any]) -> dict[str, Any]:
    """The model's reply as an OpenEnv action dict (JSON, fenced JSON, or — for a
    single-required-field schema such as Wordle's — the raw field value)."""
    message = message.strip()
    if message.startswith("```") and message.endswith("```"):
        message = "\n".join(message.splitlines()[1:-1]).strip()
    try:
        action = json.loads(message)
    except json.JSONDecodeError:
        action = message
    if isinstance(action, dict):
        return action
    required = action_schema.get("required", [])
    if len(required) != 1:
        raise ValueError("non-object actions require exactly one required field")
    return {required[0]: action}


class OpenEnvEnvConfig(vf.EnvConfig):
    player: vf.AgentConfig = vf.AgentConfig()


class OpenEnvEnv(vf.Env[OpenEnvEnvConfig]):
    async def run(self, task, agents):
        from openenv import GenericEnvClient
        from openenv.core import CallToolAction

        data = task.data
        if data.base_url:
            client = GenericEnvClient(base_url=data.base_url)
        else:
            assert data.env is not None
            client = await GenericEnvClient.from_env(
                data.env, use_docker=data.use_docker, **data.provider_kwargs
            )
        total = 0.0
        async with client:
            # OpenEnv exposes schemas over HTTP but not through GenericEnvClient.
            base_url = client._base_url.replace("ws://", "http://", 1).replace(
                "wss://", "https://", 1
            )
            async with httpx.AsyncClient(timeout=10) as http:
                response = await http.get(f"{base_url}/schema")
                response.raise_for_status()
            action_schema = response.json()["action"]
            if action_schema.get("title") in {
                "Action",
                "CallToolAction",
                "ListToolsAction",
            }:
                # Generic MCP Action omits how to call the tools it advertises.
                result = await client.step({"type": "list_tools"})
                action_schema = CallToolAction.model_json_schema() | {
                    "available_tools": result.observation["tools"]
                }
            result = await client.reset(**data.reset)

            def payload() -> str:
                return json.dumps(
                    {
                        "observation": result.observation,
                        "action_schema": action_schema,
                    },
                    ensure_ascii=False,
                )

            async with agents.player.chat(task) as session:
                reply = await session.turn(payload())
                while not reply.stopped and not result.done:
                    if reply.text.strip():
                        result = await client.step(
                            parse_action(reply.text, action_schema)
                        )
                        # OpenEnv reports per-step rewards; v1 scores their total.
                        total += result.reward or 0.0
                        if result.done:
                            break
                    reply = await session.turn(payload())
        trace = session.trace
        trace.record_reward("openenv_reward", total)


class OpenEnvTaskset(vf.Taskset[OpenEnvTask, OpenEnvConfig]):
    def load(self) -> Iterator[OpenEnvTask]:
        config = self.config
        source = config.base_url or config.env
        for idx, reset in enumerate(config.resets):
            yield OpenEnvTask(
                OpenEnvData(
                    idx=idx,
                    name=f"{source}#{idx}",
                    prompt=None,
                    env=config.env,
                    base_url=config.base_url,
                    use_docker=config.use_docker,
                    provider_kwargs=config.provider_kwargs,
                    reset=reset,
                ),
                config.task,
            )
