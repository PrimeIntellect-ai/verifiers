"""Runtimes: every supported runtime runs commands, streams processes, runs background
processes, and moves files; provides each capability it declares; and is refused, before
anything is provisioned, each capability it does not. A custom runtime plugin loads from
its import path and runs through the same API.

Cells provision real runtimes: Docker cells need a reachable daemon and Prime cells need
`PRIME_API_KEY`; otherwise they skip."""

import asyncio
import contextlib
import os
import shutil
import socket
import subprocess
import sys
import tempfile
from collections.abc import AsyncIterator
from functools import cache
from pathlib import Path
from types import SimpleNamespace

import aiohttp
import pytest
from pydantic import TypeAdapter, ValidationError

import verifiers.v1 as vf
from verifiers.v1.errors import SandboxError, ToolsetError
from verifiers.v1.mcp import launch
from verifiers.v1.mcp.toolset import ToolsetConfig
from verifiers.v1.runtimes import (
    CustomRuntimeConfig,
    CustomRuntimeInfo,
    Runtime,
    RuntimeBackend,
    RuntimeConfig,
    RuntimeProcess,
    backend_cls,
    provision_runtime,
)
from verifiers.v1.runtimes.compose import INSTALL_DOCKER, ComposeSpec
from verifiers.v1.runtimes.strategies import _deployment_strategy
from verifiers.v1.utils.compile import resolve_runtime_config, validate_pairing
from verifiers.v1.utils.loaders import harness_class, harness_config_type

RUNTIMES = ["subprocess", "docker", "prime"]
SUPPORTED = {
    "task_images": {"docker", "prime"},
    "user_switching": {"docker", "prime"},
    "port_exposure": {"subprocess", "docker"},
    "nested_containers": {"prime"},
    "network_policies": {"prime"},
}


@cache
def docker_available() -> bool:
    if shutil.which("docker") is None:
        return False
    return (
        subprocess.run(["docker", "info"], capture_output=True, check=False).returncode
        == 0
    )


def config(runtime: str, **fields) -> RuntimeConfig:
    if runtime == "prime":
        fields.setdefault("labels", ["vf-ci"])
    return TypeAdapter(RuntimeConfig).validate_python({"type": runtime, **fields})


@contextlib.asynccontextmanager
async def provisioned(runtime: str, **fields) -> AsyncIterator[Runtime]:
    if runtime == "docker" and not docker_available():
        pytest.skip("needs a reachable Docker daemon")
    if runtime == "prime" and not os.environ.get("PRIME_API_KEY"):
        pytest.skip("needs PRIME_API_KEY")
    async with provision_runtime(config(runtime, **fields)) as view:
        yield view


def task(**data) -> vf.Task:
    return vf.Task(data=vf.TaskData(idx=0, prompt=None, **data))


def refused(runtime: str, capability: str, error: type[Exception], message: str):
    """The framework's refusal for a capability the runtime does not declare."""
    if runtime in SUPPORTED[capability]:
        return contextlib.nullcontext()
    return pytest.raises(error, match=f"the {runtime} runtime {message}")


async def wait_for(predicate, timeout: float = 60) -> None:
    async with asyncio.timeout(timeout):
        while not await predicate():
            await asyncio.sleep(0.5)


@pytest.mark.parametrize("runtime", RUNTIMES)
async def test_runtime_runs_commands_and_moves_files(runtime) -> None:
    async with provisioned(runtime) as view:
        result = await view.run(
            ["sh", "-c", 'echo "$GREETING"; exit 3'], {"GREETING": "hi"}
        )
        assert (result.exit_code, result.stdout.strip()) == (3, "hi")
        await view.write("data/note.bin", b"\x00plugin\xff")
        assert await view.read("data/note.bin") == b"\x00plugin\xff"
        with pytest.raises(SandboxError, match="byte limit"):
            await view.read("data/note.bin", max_bytes=3)


@pytest.mark.parametrize("runtime", RUNTIMES)
async def test_runtime_streams_a_process(runtime) -> None:
    async with provisioned(runtime) as view:
        process = await view.exec(
            ["sh", "-c", 'read line; echo "got $line"; echo oops >&2; exit 4'], {}
        )
        await process.write(b"ping\n")
        stdout = b"".join([chunk async for chunk in process.stdout])
        stderr = b"".join([chunk async for chunk in process.stderr])
        assert await process.wait() == 4
        assert (stdout.strip(), stderr.strip()) == (b"got ping", b"oops")


@pytest.mark.parametrize("runtime", RUNTIMES)
async def test_runtime_runs_a_background_process(runtime) -> None:
    async with provisioned(runtime) as view:
        await view.run_background(["sh", "-c", "echo up; sleep 600"], {}, "bg.log")

        async def started() -> bool:
            with contextlib.suppress(SandboxError, OSError):
                return (await view.read("bg.log")).strip() == b"up"
            return False

        await wait_for(started)


@pytest.mark.parametrize("runtime", RUNTIMES)
async def test_task_images(runtime) -> None:
    image, version = {
        "docker": ("python:3.12-alpine", "3.12"),
        "prime": ("python:3.11-slim", "3.11"),
    }.get(runtime, ("python:3.12-alpine", ""))
    with refused(runtime, "task_images", ValueError, "cannot run images"):
        resolved = resolve_runtime_config(config(runtime), task(image=image))
        validate_pairing(
            harness_class("codex")(harness_config_type("codex")(id="codex")),
            vf.Task,
            resolved,
        )
    if runtime in SUPPORTED["task_images"]:
        async with provisioned(runtime, image=image) as view:
            result = await view.run(["python3", "--version"], {})
            assert version in result.stdout


@pytest.mark.parametrize("runtime", RUNTIMES)
async def test_user_switching(runtime) -> None:
    with refused(runtime, "user_switching", ValueError, "cannot run as another user"):
        resolve_runtime_config(config(runtime), task(agent_user="nobody"))
    async with provisioned(runtime) as view:
        with refused(
            runtime, "user_switching", ValueError, "cannot run as another user"
        ):
            nobody = view.with_user("nobody")
            assert (await nobody.run(["id", "-un"], {})).stdout.strip() == "nobody"


@pytest.mark.parametrize("runtime", RUNTIMES)
async def test_port_exposure(runtime, monkeypatch) -> None:
    if runtime not in SUPPORTED["port_exposure"]:
        server = SimpleNamespace(
            config=ToolsetConfig(runtime=config(runtime)), server_name="tools"
        )
        with refused(runtime, "port_exposure", ToolsetError, "cannot expose a port"):
            async with launch.serve(server):
                pass
        return
    async with provisioned(runtime) as view:
        port = view.published_port
        if port is None:
            with socket.socket() as probe:
                probe.bind(("127.0.0.1", 0))
                port = probe.getsockname()[1]
        python = sys.executable if runtime == "subprocess" else "python3"
        await view.write("www/index.html", b"exposed")
        await view.run_background(
            [python, "-m", "http.server", str(port), "-d", "www"], {}, "http.log"
        )
        url = await view.expose(port)

        async def serving() -> bool:
            with contextlib.suppress(aiohttp.ClientError):
                async with (
                    aiohttp.ClientSession() as session,
                    session.get(url) as response,
                ):
                    return await response.text() == "exposed"
            return False

        await wait_for(serving)


@pytest.mark.parametrize("runtime", RUNTIMES)
async def test_nested_containers(runtime) -> None:
    backend = SimpleNamespace(
        type=runtime,
        SUPPORTS_NESTED_CONTAINERS=backend_cls(
            config(runtime)
        ).SUPPORTS_NESTED_CONTAINERS,
    )
    with refused(
        runtime, "nested_containers", ValueError, "cannot run nested containers"
    ):
        _deployment_strategy(None, backend, ComposeSpec(directory="."))
    if runtime not in SUPPORTED["nested_containers"]:
        return
    async with provisioned(runtime, cpu=2.0, memory=4.0) as view:
        installed = await view.run(["sh", "-c", INSTALL_DOCKER], {})
        assert installed.exit_code == 0, installed.stdout + installed.stderr
        await view.run_background(
            ["dockerd", "--host=unix:///var/run/docker.sock"], {}, "/tmp/dockerd.log"
        )

        async def daemon_up() -> bool:
            return (await view.run(["docker", "info"], {})).exit_code == 0

        await wait_for(daemon_up)
        result = await view.run(
            ["docker", "run", "--rm", "alpine:3.20", "echo", "nested"], {}
        )
        assert result.stdout.strip() == "nested", result.stderr


@pytest.mark.parametrize("runtime", RUNTIMES)
async def test_network_policies(runtime) -> None:
    with refused(
        runtime, "network_policies", ValueError, "does not enforce network policies"
    ):
        resolve_runtime_config(config(runtime), task(network_allow=["pypi.org"]))
    if runtime not in SUPPORTED["network_policies"]:
        return
    probe = (
        "import sys, urllib.request\n"
        "try:\n"
        "    urllib.request.urlopen(sys.argv[1], timeout=10)\n"
        "    print('ok')\n"
        "except Exception:\n"
        "    print('denied')\n"
    )

    async def reach(view: Runtime, url: str) -> str:
        result = await view.run(["python3", "-c", probe, url], {})
        return result.stdout.strip()

    async with provisioned(runtime, allow=["pypi.org"]) as view:
        assert await reach(view, "https://example.com") == "ok"
        await view.backend.prepare_execution([])
        assert await reach(view, "https://pypi.org/simple/") == "ok"
        assert await reach(view, "https://example.com") == "denied"


def test_runtimes_declare_their_capabilities() -> None:
    flags = {
        "task_images": "SUPPORTS_TASK_IMAGES",
        "user_switching": "SUPPORTS_USER_SWITCHING",
        "port_exposure": "SUPPORTS_PORT_EXPOSURE",
        "nested_containers": "SUPPORTS_NESTED_CONTAINERS",
    }
    for runtime in RUNTIMES:
        cls = backend_cls(config(runtime))
        declared = {
            capability for capability, flag in flags.items() if getattr(cls, flag)
        }
        if isinstance(config(runtime), vf.NetworkPolicyConfig):
            declared.add("network_policies")
        expected = {c for c, runtimes in SUPPORTED.items() if runtime in runtimes}
        assert declared == expected, runtime


class HostProcess(RuntimeProcess):
    def __init__(self, process: asyncio.subprocess.Process) -> None:
        assert process.stdin and process.stdout and process.stderr
        self._process = process
        self.stdout = self._chunks(process.stdout)
        self.stderr = self._chunks(process.stderr)

    @staticmethod
    async def _chunks(reader: asyncio.StreamReader) -> AsyncIterator[bytes]:
        while chunk := await reader.read(1 << 16):
            yield chunk

    async def write(self, data: bytes) -> None:
        assert self._process.stdin is not None
        self._process.stdin.write(data)
        await self._process.stdin.drain()

    async def wait(self) -> int:
        return await self._process.wait()

    async def poll(self) -> int | None:
        return self._process.returncode

    async def terminate(self) -> None:
        if self._process.returncode is None:
            self._process.terminate()

    async def kill(self) -> None:
        if self._process.returncode is None:
            self._process.kill()


class HostConfig(CustomRuntimeConfig):
    greeting: str = "hello"


class HostBackend(RuntimeBackend):
    """A minimal plugin: `start`, `exec`, `write`, and `cleanup`. `run` and `read`
    come from the base class."""

    config_class = HostConfig

    def __init__(self, config: HostConfig, name: str | None = None) -> None:
        super().__init__(name)
        self.config = config
        self.info = CustomRuntimeInfo(**config.model_dump())
        self.root: Path | None = None

    async def start(self) -> None:
        self.root = Path(tempfile.mkdtemp(prefix=f"{self.name}-"))
        self.info.id = str(self.root)

    async def exec(
        self, runtime: Runtime, argv: list[str], env: dict[str, str]
    ) -> RuntimeProcess:
        process = await asyncio.create_subprocess_exec(
            *argv,
            env={
                **os.environ,
                "GREETING": self.config.greeting,
                **runtime.process_env(env),
            },
            cwd=self.root,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        return HostProcess(process)

    async def write(self, runtime: Runtime, path: str, data: bytes) -> None:
        assert self.root is not None
        target = self.root / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)

    def cleanup(self) -> None:
        if self.root is not None:
            shutil.rmtree(self.root, ignore_errors=True)


class UntypedBackend(HostBackend):
    config_class = dict


NOT_A_BACKEND = object()


def plugin(backend: str = "HostBackend", **fields) -> RuntimeConfig:
    return TypeAdapter(RuntimeConfig).validate_python(
        {"type": "custom", "import_path": f"{__name__}.{backend}", **fields}
    )


async def test_custom_runtime_runs_the_plugin() -> None:
    config = plugin(greeting="hi")
    assert isinstance(config, HostConfig)
    assert backend_cls(config) is HostBackend
    async with provision_runtime(config) as view:
        assert view.type == "custom"
        await view.write("note.txt", b"plugin")
        result = await view.run(["sh", "-c", 'echo "$GREETING $(cat note.txt)"'], {})
        assert result.stdout.strip() == "hi plugin"
        assert await view.read("note.txt") == b"plugin"
        assert view.info.greeting == "hi"
        root = Path(view.info.id)
    assert not root.exists()


def test_custom_runtime_validates_its_typed_config() -> None:
    with pytest.raises(ValidationError, match="greting"):
        plugin(greting="hi")
    agent = vf.AgentConfig(
        model="m",
        runtime={
            "type": "custom",
            "import_path": f"{__name__}.HostBackend",
            "greeting": "hi",
        },
    )
    loaded = vf.AgentConfig.model_validate_json(agent.model_dump_json())
    assert isinstance(loaded.runtime, HostConfig) and loaded.runtime.greeting == "hi"


@pytest.mark.parametrize(
    "import_path,error,match",
    [
        ("HostBackend", ValueError, "must be a dotted path"),
        ("no_such_package.Backend", ValueError, "cannot import custom runtime"),
        (f"{__name__}.Missing", ValueError, "cannot import custom runtime"),
        (f"{__name__}.NOT_A_BACKEND", TypeError, "must be a RuntimeBackend subclass"),
        (f"{__name__}.UntypedBackend", TypeError, "must set config_class"),
    ],
)
def test_custom_runtime_rejects_a_bad_import_path(import_path, error, match) -> None:
    with pytest.raises(error, match=match):
        TypeAdapter(RuntimeConfig).validate_python(
            {"type": "custom", "import_path": import_path}
        )
