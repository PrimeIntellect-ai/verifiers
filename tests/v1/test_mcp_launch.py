import asyncio
from types import SimpleNamespace

import pytest

from verifiers.v1.mcp import launch
from verifiers.v1.runtimes.base import ProgramResult, Runtime


class FakeRuntime(Runtime):
    def __init__(self, *, exit_code=0, published_port=1234):
        super().__init__("test")
        self.config = SimpleNamespace(type="fake")
        self.info = SimpleNamespace(id="test")
        self.exit_code = exit_code
        self._published_port = published_port
        self.writes = []
        self.commands = []
        self.background = []

    async def start(self):
        return None

    async def run(self, argv, env):
        self.commands.append((argv, env))
        return ProgramResult(self.exit_code, "", "failed" if self.exit_code else "")

    async def run_background(self, argv, env, log):
        self.background.append((argv, env, log))

    async def read(self, path):
        return b""

    async def write(self, path, data):
        self.writes.append((path, data))

    @property
    def published_port(self):
        return self._published_port


class FakeServer:
    server_name = "tool"
    EXTRAS = ()

    class config:
        @staticmethod
        def model_dump_json():
            return "{}"


@pytest.mark.asyncio
async def test_launches_use_distinct_roots(monkeypatch):
    roots = []

    async def install(server, runtime, root):
        roots.append(root)
        return "/tmp/python"

    monkeypatch.setattr(launch, "_install_in_sandbox", install)
    runtimes = [FakeRuntime(), FakeRuntime()]
    await asyncio.gather(
        *(
            launch.serve_in_runtime(FakeServer(), runtime, exposed=True)
            for runtime in runtimes
        )
    )
    assert len(set(roots)) == 2
    assert len({runtime.background[0][2] for runtime in runtimes}) == 2
    assert all(
        path.startswith(root)
        for root, runtime in zip(roots, runtimes)
        for path in [runtime.background[0][2]]
    )


@pytest.mark.asyncio
async def test_install_failure_preserves_error(monkeypatch):
    async def install(server, runtime, root):
        raise launch.ToolsetError("install failed")

    monkeypatch.setattr(launch, "_install_in_sandbox", install)
    with pytest.raises(launch.ToolsetError, match="install failed"):
        await launch.serve_in_runtime(FakeServer(), FakeRuntime(), exposed=False)


@pytest.mark.asyncio
async def test_install_cancellation_propagates(monkeypatch):
    started = asyncio.Event()

    async def install(server, runtime, root):
        started.set()
        await asyncio.sleep(60)
        return "/tmp/python"

    monkeypatch.setattr(launch, "_install_in_sandbox", install)
    task = asyncio.create_task(
        launch.serve_in_runtime(FakeServer(), FakeRuntime(), exposed=False)
    )
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


@pytest.mark.asyncio
async def test_install_command_uses_unique_source_and_venv(monkeypatch):
    monkeypatch.setattr(launch, "_source_dir", lambda cls: "/tmp/env")
    monkeypatch.setattr(
        launch, "_verifiers_root", lambda: launch.Path("/tmp/verifiers")
    )
    monkeypatch.setattr(launch, "_tar_source", lambda *args: b"tar")
    monkeypatch.setattr(launch.importlib.metadata, "version", lambda name: "0")
    runtime = FakeRuntime()
    monkeypatch.setattr(launch, "VF_BUILD_INPUTS", ())
    await launch._install_in_sandbox(FakeServer(), runtime, "/tmp/vf-mcp-abc")
    assert {path for path, _ in runtime.writes} == {
        "/tmp/vf-mcp-abc/src/verifiers.tar.gz",
        "/tmp/vf-mcp-abc/src/env.tar.gz",
    }
    setup = runtime.commands[-1][0][2]
    assert "/tmp/vf-mcp-abc/src" in setup
    assert "/tmp/vf-mcp-abc/venv" in setup
