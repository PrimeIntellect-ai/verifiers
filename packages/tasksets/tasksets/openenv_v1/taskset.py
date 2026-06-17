"""Run an OpenEnv project in the rollout container."""

import importlib
import json
from pathlib import Path
from typing import Literal

import verifiers.v1 as vf

OpenEnvContract = Literal["gym", "mcp"]


class OpenEnvTask(vf.Task):
    contract: OpenEnvContract
    port: int
    start_command: str
    seed: int


class OpenEnvState(vf.State):
    reward: float = 0.0
    done: bool = False


class OpenEnvConfig(vf.TasksetConfig):
    project: str = "proj"
    """OpenEnv project directory, relative to the concrete config module."""
    num_tasks: int = 100
    """How many seeded episodes to expose."""
    seed: int = 0
    """First episode seed."""
    instruction: str = (
        "Use the available OpenEnv tools to interact with the environment, then answer."
    )
    """Static opening instruction for MCP environments."""
    system_prompt: str | None = None


class OpenEnvTaskset(vf.Taskset[OpenEnvTask, OpenEnvConfig, OpenEnvState]):
    NEEDS_CONTAINER = True

    def load_tasks(self) -> list[OpenEnvTask]:
        project = Path(self.config.project).expanduser()
        if not project.is_absolute():
            module = importlib.import_module(type(self.config).__module__)
            project = Path(module.__file__).resolve().parent / project
        build = json.loads((project.resolve() / ".build.json").read_text())
        if build["contract"] == "gym" and build["port"] == 8000:
            raise ValueError(
                "gym OpenEnv projects must use a port other than 8000; v1 reserves 8000 "
                "for the colocated user simulator."
            )
        return [
            OpenEnvTask(
                idx=index,
                name=f"openenv-{build['contract']}-{self.config.seed + index}",
                instruction=None
                if build["contract"] == "gym"
                else self.config.instruction,
                system_prompt=self.config.system_prompt,
                image=build["image"],
                workdir="/app/env",
                contract=build["contract"],
                port=build["port"],
                start_command=build["start_command"],
                seed=self.config.seed + index,
            )
            for index in range(self.config.num_tasks)
        ]

    async def setup(self, task: OpenEnvTask, runtime: vf.Runtime) -> None:
        await runtime.run_background(
            ["sh", "-lc", task.start_command],
            {"ENABLE_WEB_INTERFACE": "false"},
            "openenv.log",
        )

    def tools(self, task: OpenEnvTask) -> list[vf.Toolset]:
        if task.contract == "mcp":
            from tasksets.openenv_v1.servers.toolset import OpenEnvToolset

            return [OpenEnvToolset(vf.ToolsetConfig(colocated=True))]
        return []

    def user(self, task: OpenEnvTask) -> vf.User | None:
        if task.contract == "gym":
            from tasksets.openenv_v1.servers.user import OpenEnvUser

            return OpenEnvUser(vf.UserConfig(colocated=True))
        return None

    @vf.stop
    async def openenv_done(self, trace: vf.Trace) -> bool:
        return trace.task.contract == "gym" and trace.state.done

    @vf.reward(weight=1.0)
    async def openenv_reward(self, trace: vf.Trace) -> float:
        return trace.state.reward
