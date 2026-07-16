"""Run OpenEnv environments with UV by default, or Docker when requested."""

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
    reset: dict[str, Any]


class OpenEnvState(vf.State):
    reward: float = 0.0
    done: bool = False


class OpenEnvUserConfig(vf.UserConfig):
    provider_kwargs: dict[str, Any] = {}


class OpenEnvTaskConfig(vf.TaskConfig):
    user: OpenEnvUserConfig = OpenEnvUserConfig()


class OpenEnvConfig(vf.TasksetConfig):
    env: str | None = None
    """Environment id passed to OpenEnv. Required unless `base_url` is set."""
    base_url: str | None = None
    """Connect to an existing OpenEnv server instead of starting `env`."""
    use_docker: bool = False
    """Use OpenEnv's Docker provider instead of the default UV provider."""
    resets: list[dict[str, Any]] = [{}]
    """One finite task per set of arguments passed to OpenEnv's `reset`."""
    task: OpenEnvTaskConfig = OpenEnvTaskConfig()

    @model_validator(mode="after")
    def validate_config(self) -> Self:
        if not self.env and not self.base_url:
            raise ValueError("pass `env` or `base_url`")
        if not self.base_url and self.use_docker:
            # Docker runs inside a VM rather than nesting inside another container.
            self.task.user.colocated = False
            if isinstance(self.task.user.runtime, vf.PrimeConfig):
                self.task.user.runtime.vm = True
            else:
                self.task.user.runtime = vf.PrimeConfig(vm=True)
        return self


class OpenEnvUser(vf.User[OpenEnvUserConfig, OpenEnvState]):
    async def setup_task(self, task: OpenEnvData) -> None:
        from openenv import GenericEnvClient
        from openenv.core import CallToolAction

        if task.base_url:
            client = GenericEnvClient(base_url=task.base_url)
        else:
            assert task.env is not None
            client = await GenericEnvClient.from_env(
                task.env,
                use_docker=task.use_docker,
                **self.config.provider_kwargs,
            )
        self.client = await self._exit_stack.enter_async_context(client)
        # OpenEnv exposes schemas over HTTP but not through GenericEnvClient.
        base_url = self.client._base_url.replace("ws://", "http://", 1).replace(
            "wss://", "https://", 1
        )
        async with httpx.AsyncClient(timeout=10) as http:
            response = await http.get(f"{base_url}/schema")
        self.action_schema = response.json()["action"]
        if self.action_schema.get("title") in {"Action", "CallToolAction"}:
            result = await self.client.step({"type": "list_tools"})
            # Generic MCP Action omits how to call the tools it advertises.
            self.action_schema = CallToolAction.model_json_schema() | {
                "available_tools": result.observation["tools"]
            }
        self.result = await self.client.reset(**task.reset)

    def parse_action(self, message: str) -> dict[str, Any]:
        message = message.strip()
        if message.startswith("```") and message.endswith("```"):
            message = "\n".join(message.splitlines()[1:-1]).strip()
        try:
            action = json.loads(message)
        except json.JSONDecodeError:
            action = message
        if isinstance(action, dict):
            return action
        # Single-field environments such as Wordle also accept the raw field value.
        required = self.action_schema.get("required", [])
        if len(required) != 1:
            raise ValueError("non-object actions require exactly one required field")
        return {required[0]: action}

    async def respond(self, message: str) -> vf.Messages:
        if message.strip():
            self.result = await self.client.step(self.parse_action(message))
            # OpenEnv reports per-step rewards; v1 scores their total over the trace.
            self.state.reward += self.result.reward or 0.0
        self.state.done = self.result.done
        payload = {
            "observation": self.result.observation,
            "action_schema": self.action_schema,
        }
        return [vf.UserMessage(content=json.dumps(payload, ensure_ascii=False))]


class OpenEnvTask(vf.Task[OpenEnvData, OpenEnvState, OpenEnvTaskConfig]):
    user = OpenEnvUser

    @vf.stop
    async def openenv_done(self, trace: vf.Trace) -> bool:
        return trace.state.done

    @vf.reward
    async def openenv_reward(self, trace: vf.Trace) -> float:
        return trace.state.reward


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
                    reset=reset,
                ),
                config.task,
            )


if __name__ == "__main__":
    OpenEnvUser.run()
