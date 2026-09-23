"""Run OpenEnv environments with UV by default, or Docker when requested.

The engine plays the user: the env's `run()` opens the model's interaction and
steps the OpenEnv client host-side — each assistant action advances the
environment, the next observation comes back as the user turn, and a `done` result
ends the exchange. OpenEnv's per-step rewards are summed onto the seat's trace
(`openenv_reward`)."""

import json
from collections.abc import Iterator
from copy import deepcopy
from typing import Any

import httpx
from pydantic import Field

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
    provider_kwargs: dict[str, Any] = Field(default_factory=dict)
    """Extra arguments for OpenEnv's provider (`GenericEnvClient.from_env`)."""
    resets: list[dict[str, Any]] = Field(default_factory=lambda: [{}])
    """One finite task per set of arguments passed to OpenEnv's `reset`."""


class OpenEnvTask(vf.Task[OpenEnvData]):
    pass


def parse_action(message: str, action_schema: dict[str, Any]) -> dict[str, Any]:
    """The model's reply as an OpenEnv action dict (JSON, fenced JSON, or — for a
    single-field schema such as Wordle's — the raw field value)."""
    message = message.strip()
    if message.startswith("```") and message.endswith("```"):
        message = "\n".join(message.splitlines()[1:-1]).strip()
    try:
        action = json.loads(message)
    except json.JSONDecodeError:
        action = message
    if isinstance(action, dict):
        return action
    fields = action_schema.get("required") or [
        name for name in action_schema.get("properties", {}) if name != "metadata"
    ]
    if len(fields) != 1:
        raise ValueError("non-object actions require exactly one action field")
    field = fields[0]
    schema = action_schema.get("properties", {}).get(field, {})
    if any(
        "string" in (option.get("type") or [])
        for option in [schema, *schema.get("anyOf", []), *schema.get("oneOf", [])]
    ):
        action = action if isinstance(action, str) else message
    return {field: action}


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
        scored = False
        async with client:
            # OpenEnv exposes schemas over HTTP but not through GenericEnvClient.
            base_url = client._base_url.replace("ws://", "http://", 1).replace(
                "wss://", "https://", 1
            )
            async with httpx.AsyncClient(timeout=60) as http:
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
            if result.reward is not None:
                total += result.reward
                scored = True

            def payload() -> str | vf.Messages:
                observation = deepcopy(result.observation)
                images: list[tuple[str, str]] = []
                pending = [(None, None, observation)]
                while pending:
                    parent, key, value = pending.pop()
                    if isinstance(value, list):
                        pending.extend((None, None, item) for item in reversed(value))
                    elif isinstance(value, dict):
                        if value.get("type") == "image" and isinstance(
                            value.get("data"), str
                        ):
                            images.append(
                                (value.get("mimeType", "image/png"), value.pop("data"))
                            )
                            value["data"] = "<image>"
                        pending.extend(
                            (value, name, item)
                            for name, item in reversed(list(value.items()))
                        )
                    elif (
                        isinstance(parent, dict)
                        and key is not None
                        and key.endswith("_base64")
                        and isinstance(value, str)
                        and value
                    ):
                        fmt = parent.get("image_format") or (
                            "png"
                            if key.endswith("png_base64")
                            or parent.get("image_kind") == "map"
                            else "jpeg"
                        )
                        images.append((f"image/{fmt}", value))
                        parent[key] = "<image>"
                message = json.dumps(
                    {"observation": observation, "action_schema": action_schema},
                    ensure_ascii=False,
                )
                if not images:
                    return message
                return [
                    vf.UserMessage(
                        content=[
                            vf.TextContentPart(text=message),
                            *[
                                vf.ImageUrlContentPart(
                                    image_url=vf.ImageUrlSource(
                                        url=f"data:{mime};base64,{data}"
                                    )
                                )
                                for mime, data in images
                            ],
                        ]
                    )
                ]

            async with agents.player.interaction(task) as interaction:
                segment = await interaction.turn(payload()) if not result.done else None
                while (
                    segment is not None and not segment.terminated and not result.done
                ):
                    action = segment.last_reply.strip()
                    if not action:
                        # No action can advance OpenEnv. End this run explicitly
                        # instead of replaying the same observation forever when
                        # the agent has no turn or episode cap.
                        interaction.trace.stop("empty_action")
                        break
                    result = await client.step(parse_action(action, action_schema))
                    # OpenEnv reports per-step rewards; v1 scores their total.
                    if result.reward is not None:
                        total += result.reward
                        scored = True
                    if result.done:
                        break
                    segment = await interaction.turn(payload())
        trace = interaction.trace
        if scored:
            trace.record_reward("openenv_reward", total)


class OpenEnvTaskset(vf.Taskset[OpenEnvTask, OpenEnvConfig]):
    def load(self) -> Iterator[OpenEnvTask]:
        config = self.config
        source = config.base_url or config.env
        if not source:
            raise ValueError("pass `env` or `base_url`")
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
