"""A tool server fetches the rollout's task only when it sets itself up per task, so a
task-agnostic server (a Harbor MCP proxy) never imports a task class its runtime
lacks."""

from pydantic_config import BaseConfig

from verifiers.v1.mcp.server import ServerBase


class Agnostic(ServerBase[BaseConfig]):
    pass


class PerTask(ServerBase[BaseConfig]):
    async def setup_task(self, task) -> None:
        self.task = task


async def test_only_a_per_task_server_fetches_the_task():
    fetched = []

    async def fetch(self, url, secret):
        fetched.append(type(self).__name__)
        return "task"

    for cls in (Agnostic, PerTask):
        server = cls(BaseConfig())
        server._fetch_task = fetch.__get__(server)
        await server._setup_task_from_channel("http://host/state", "s")
    assert fetched == ["PerTask"]
