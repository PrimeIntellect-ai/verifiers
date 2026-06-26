"""Run a prebuilt OpenEnv MCP image as a Verifiers taskset."""

from verifiers.v1.mcp import JSONRPCToolset, JSONRPCToolsetConfig, Toolset
from verifiers.v1.runtimes import Runtime
from verifiers.v1.task import Task, TaskResources
from verifiers.v1.taskset import Taskset, TasksetConfig

OPENENV_SOCKET = "/tmp/openenv.sock"
OPENENV_URL = "http://openenv"


class OpenEnvConfig(TasksetConfig):
    image: str
    """Prebuilt OpenEnv MCP image."""
    prompt: str
    """Opening prompt for the agent."""
    workdir: str = "/app/env"
    """Environment project directory inside the image."""
    app: str = "server.app:app"
    """ASGI app from the image's OpenEnv manifest."""
    system_prompt: str | None = None
    resources: TaskResources = TaskResources()


class OpenEnvTaskset(Taskset[Task, OpenEnvConfig]):
    NEEDS_CONTAINER = True

    def load_tasks(self) -> list[Task]:
        return [
            Task(
                idx=0,
                name=self.config.name,
                prompt=self.config.prompt,
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

    def tools(self, task: Task) -> list[Toolset]:
        return [
            JSONRPCToolset(
                JSONRPCToolsetConfig(
                    colocated=True,
                    endpoint=f"{OPENENV_URL}/mcp",
                    uds=OPENENV_SOCKET,
                )
            )
        ]
