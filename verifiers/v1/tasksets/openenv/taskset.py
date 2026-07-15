"""Run OpenEnv environments as seeded v1 tasksets.

With ``base_url``, the user connects to an existing server. Otherwise OpenEnv
starts ``env`` with its UV provider by default, or its Docker provider when
``use_docker`` is set. Prime defaults Docker-backed runs to a VM so Docker can
start inside the sandbox.

Each reset seed becomes a v1 task. The user exposes OpenEnv's observation and
action schema, accumulates its step rewards, and maps its ``done`` signal to v1.
"""

import itertools
import json
from collections.abc import Iterator
from typing import Any, Self

import requests
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
    """Connect to an existing OpenEnv server instead of starting `env`. Docker and
    provider options only apply when starting `env`."""
    use_docker: bool = False
    """Use OpenEnv's Docker provider instead of its UV provider. Under Prime this
    defaults the outer runtime to a VM, allowing Docker to start inside it. Docker-backed
    users cannot be colocated because the taskset does not own the harness runtime."""
    provider_kwargs: dict[str, Any] = {}
    """Keyword arguments passed to `GenericEnvClient.from_env`, such as
    `app`, `env_vars`, `tag`, or `project_path`."""
    reset: dict[str, Any] = {}
    """Arguments passed to OpenEnv's `reset`; the generated seed takes precedence."""
    seed: int = 0
    """First reset seed. V1 task selection, normally `-n`, bounds the lazy sequence."""
    task: OpenEnvTaskConfig = OpenEnvTaskConfig()

    @model_validator(mode="after")
    def validate_config(self) -> Self:
        if not self.env and not self.base_url:
            raise ValueError("pass `env` or `base_url`")
        # Thin wrappers set launch defaults; base_url only conflicts with overrides.
        overrides = self.model_dump(exclude_defaults=True)
        launch_options = {"env", "use_docker", "provider_kwargs"}
        user_options = overrides.get("task", {}).get("user", {})
        if self.base_url and (
            launch_options & overrides.keys() or "provider_kwargs" in user_options
        ):
            raise ValueError("`base_url` cannot be combined with local launch options")
        if not self.base_url and self.use_docker and self.task.user.colocated:
            raise ValueError("OpenEnv Docker requires its own user runtime")

        runtime = self.task.user.runtime
        # OpenEnv Docker starts inside the user runtime. Prime therefore defaults the
        # outer sandbox to a VM while preserving an explicit `vm=True` or `vm=False`.
        if (
            not self.base_url
            and self.use_docker
            and isinstance(runtime, vf.PrimeConfig)
            and "vm" not in runtime.model_fields_set
        ):
            runtime.vm = True
        return self


class OpenEnvUser(vf.User[OpenEnvUserConfig, OpenEnvState]):
    PYTHON_DEPENDENCIES = ("openenv",)

    async def setup_task(self, task: OpenEnvData) -> None:
        self.client = None
        from openenv import GenericEnvClient

        if task.base_url:
            self.client = GenericEnvClient(base_url=task.base_url)
            await self.client.connect()
        else:
            assert task.env is not None
            self.client = await GenericEnvClient.from_env(
                task.env,
                use_docker=task.use_docker,
                **self.config.provider_kwargs,
            )

        # OpenEnv exposes schemas over HTTP but not through GenericEnvClient.
        base_url = self.client._base_url.replace("ws://", "http://", 1).replace(
            "wss://", "https://", 1
        )
        self.action_schema = requests.get(f"{base_url}/schema", timeout=10).json()[
            "action"
        ]
        action_type = self.action_schema.get("properties", {}).get("type", {})
        # OpenEnv's generic MCP Action omits the type discriminator.
        if (
            self.action_schema.get("title") == "Action"
            or action_type.get("const") == "call_tool"
            or "call_tool" in action_type.get("enum", [])
        ):
            result = await self.client.step({"type": "list_tools"})
            self.action_schema["available_tools"] = result.observation["tools"]
        self.result = await self.client.reset(**task.reset)

    async def teardown(self) -> None:
        if self.client is not None:
            await self.client.close()

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
        return [{"role": "user", "content": json.dumps(payload, ensure_ascii=False)}]


class OpenEnvTask(vf.Task[OpenEnvData, OpenEnvState, OpenEnvTaskConfig]):
    user = OpenEnvUser

    @vf.stop
    async def openenv_done(self, trace: vf.Trace) -> bool:
        return trace.state.done

    @vf.reward
    async def openenv_reward(self, trace: vf.Trace) -> float:
        return trace.state.reward


class OpenEnvTaskset(vf.Taskset[OpenEnvTask, OpenEnvConfig]):
    INFINITE = True

    def load(self) -> Iterator[OpenEnvTask]:
        config = self.config
        # Keep provider options off TaskData; top-level values override task defaults.
        task_config = config.task.model_copy(deep=True)
        task_config.user.provider_kwargs.update(config.provider_kwargs)
        source = config.base_url or config.env
        for idx, seed in enumerate(itertools.count(config.seed)):
            yield OpenEnvTask(
                OpenEnvData(
                    idx=idx,
                    name=f"{source}#{seed}",
                    prompt=None,
                    env=config.env,
                    base_url=config.base_url,
                    use_docker=config.use_docker,
                    reset=config.reset | {"seed": seed},
                ),
                task_config,
            )


if __name__ == "__main__":
    OpenEnvUser.run()
