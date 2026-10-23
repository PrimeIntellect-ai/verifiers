"""Runtimes: the backend contract and every runtime feature, against each runtime. A
runtime that declares a feature must run it. A runtime that does not must refuse it
before anything is provisioned, with a message that names the runtime. A custom runtime
plugin loads from its import path and passes the same tests.

To check a custom runtime plugin, or any other runtime config, pass it with `--runtime`:

    uv run pytest tests/unit/test_runtimes.py \\
        --runtime '{"type": "custom", "import_path": "my_package.runtime.MyBackend"}'

The tests provision real runtimes and expect `sh` and Python 3 in the default image.
Docker needs a reachable daemon and Prime needs `PRIME_API_KEY`; otherwise those cells
skip."""

import asyncio
import contextlib
import json
import os
import re
import shutil
import socket
import subprocess
import tempfile
from collections.abc import AsyncIterator, Awaitable, Callable, Iterator
from functools import cache
from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar

import aiohttp
import pytest
from pydantic import TypeAdapter, ValidationError

import verifiers.v1 as vf
from verifiers.v1.configs.runtime import NetworkPolicyConfig
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
from verifiers.v1.runtimes.compose import ComposeSpec, compose_services
from verifiers.v1.runtimes.strategies import _deployment_strategy
from verifiers.v1.utils.compile import resolve_runtime_config

RUNTIMES = ["subprocess", "docker", "prime", "plugin"]
SUPPORTED = {
    "task_images": {"docker", "prime"},
    "user_switching": {"docker", "prime"},
    "port_exposure": {"subprocess", "docker", "prime"},
    "compose": {"docker", "prime"},
    "network_policies": {"docker", "prime"},
}
PYTHON = ["sh", "-c", 'exec "$(command -v python3 || command -v python)" "$@"', "py"]
IMAGE = "python:3.12-alpine"
TIMEOUT = 600


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    if "runtime" in metafunc.fixturenames:
        option = metafunc.config.getoption("runtime")
        metafunc.parametrize(
            "runtime",
            [option] if option else RUNTIMES,
            ids=["option"] if option else RUNTIMES,
            indirect=True,
        )


@pytest.fixture
def runtime(request) -> RuntimeConfig:
    spec = request.param
    if spec == "plugin":
        data = {"type": "custom", "import_path": f"{__name__}.HostBackend"}
    elif spec.lstrip().startswith("{"):
        data = json.loads(spec)
    else:
        data = {"type": spec, **({"labels": ["vf-ci"]} if spec == "prime" else {})}
    return TypeAdapter(RuntimeConfig).validate_python(data)


@cache
def docker_available() -> bool:
    if shutil.which("docker") is None:
        return False
    return (
        subprocess.run(["docker", "info"], capture_output=True, check=False).returncode
        == 0
    )


def skip_unavailable(config: RuntimeConfig) -> None:
    if config.type == "docker" and not docker_available():
        pytest.skip("needs a reachable Docker daemon")
    if config.type == "prime" and not os.environ.get("PRIME_API_KEY"):
        pytest.skip("needs PRIME_API_KEY")


@contextlib.asynccontextmanager
async def provisioned(config: RuntimeConfig, **updates) -> AsyncIterator[Runtime]:
    skip_unavailable(config)
    if updates:
        config = type(config).model_validate({**config.model_dump(), **updates})
    async with asyncio.timeout(TIMEOUT), provision_runtime(config) as view:
        yield view


@contextlib.contextmanager
def refused(config: RuntimeConfig, error: type[Exception], message: str) -> Iterator:
    with pytest.raises(error, match=f"the {config.type} runtime {message}"):
        yield


async def wait_for(
    predicate: Callable[[], Awaitable[bool]], timeout: float = 120
) -> None:
    async with asyncio.timeout(timeout):
        while not await predicate():
            await asyncio.sleep(0.5)


RESOLVE = "import socket, sys; socket.getaddrinfo(sys.argv[1], 443)"
FETCH = "import sys, urllib.request; print(urllib.request.urlopen(sys.argv[1]).read().decode())"


@contextlib.asynccontextmanager
async def host_server(body: bytes) -> AsyncIterator[str]:
    """An HTTP server on host loopback that answers every request with `body`."""

    async def answer(
        reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        await reader.readuntil(b"\r\n\r\n")
        writer.write(
            b"HTTP/1.1 200 OK\r\nContent-Length: %d\r\nConnection: close\r\n\r\n%s"
            % (len(body), body)
        )
        await writer.drain()
        writer.close()

    server = await asyncio.start_server(answer, "127.0.0.1", 0)
    async with server:
        yield f"http://127.0.0.1:{server.sockets[0].getsockname()[1]}/"


def task(**data) -> vf.Task:
    return vf.Task(data=vf.TaskData(idx=0, prompt=None, **data))


@pytest.mark.parametrize("name", ["subprocess", "docker", "prime"])
def test_built_in_runtime_declares_its_features(name) -> None:
    config = TypeAdapter(RuntimeConfig).validate_python({"type": name})
    cls = backend_cls(config)
    declared = {
        "task_images": cls.SUPPORTS_TASK_IMAGES,
        "user_switching": cls.SUPPORTS_USER_SWITCHING,
        "port_exposure": cls.SUPPORTS_PORT_EXPOSURE,
        "compose": cls.SUPPORTS_COMPOSE,
        "network_policies": isinstance(config, NetworkPolicyConfig),
    }
    assert declared == {f: name in runtimes for f, runtimes in SUPPORTED.items()}


async def test_run(runtime) -> None:
    async with provisioned(runtime) as view:
        result = await view.run(
            ["sh", "-c", 'echo "$GREETING"; exit 3'], {"GREETING": "hi"}
        )
        assert result.exit_code == 3, result
        assert result.stdout.strip() == "hi", result
        stdin = await view.run(["cat"], {})
        assert stdin.exit_code == 0 and stdin.stdout == "", "run must close stdin"


async def test_exec(runtime) -> None:
    async with provisioned(runtime) as view:
        process = await view.exec(
            ["sh", "-c", 'read line; echo "got $line"; echo oops >&2; exit 4'], {}
        )
        await process.write(b"ping\n")
        stdout = b"".join([chunk async for chunk in process.stdout])
        stderr = b"".join([chunk async for chunk in process.stderr])
        assert await process.wait() == 4
        assert stdout.strip() == b"got ping", stdout
        assert stderr.strip() == b"oops", stderr
        assert await process.poll() == 4
        sleeper = await view.exec(["sleep", "600"], {})
        assert await sleeper.poll() is None
        await sleeper.kill()
        async with asyncio.timeout(30):
            assert await sleeper.wait() != 0


async def test_run_background(runtime) -> None:
    async with provisioned(runtime) as view:
        await view.run_background(["sh", "-c", "echo up; sleep 600"], {}, "bg.log")

        async def started() -> bool:
            with contextlib.suppress(SandboxError, OSError):
                return (await view.read("bg.log")).strip() == b"up"
            return False

        await wait_for(started)


async def test_files(runtime) -> None:
    async with provisioned(runtime) as view:
        data = bytes(range(256)) * 64
        await view.write("vf-check/nested/data.bin", data)
        assert await view.read("vf-check/nested/data.bin") == data
        assert await view.read("vf-check/nested/data.bin", max_bytes=len(data)) == data
        with pytest.raises(SandboxError):
            await view.read("vf-check/nested/data.bin", max_bytes=10)
        listed = await view.run(["sh", "-c", "wc -c < vf-check/nested/data.bin"], {})
        assert listed.stdout.strip() == str(len(data)), "write must land in the workdir"


async def test_task_images(runtime) -> None:
    if not backend_cls(runtime).SUPPORTS_TASK_IMAGES:
        with refused(runtime, ValueError, "cannot run images"):
            resolve_runtime_config(runtime, task(image=IMAGE))
        return
    # Prime builds a VM image on an image's first use; use one it has cached.
    image = "python:3.11-slim" if runtime.type == "prime" else IMAGE
    async with provisioned(runtime, image=image) as view:
        result = await view.run([*PYTHON, "--version"], {})
        version = re.match(r"python:(\d+\.\d+)", image)
        assert version is not None and version.group(1) in result.stdout, result


async def test_user_switching(runtime) -> None:
    if not backend_cls(runtime).SUPPORTS_USER_SWITCHING:
        with refused(runtime, ValueError, "cannot run as another user"):
            resolve_runtime_config(runtime, task(agent_user="nobody"))
        return
    async with provisioned(runtime) as view:
        result = await view.with_user("nobody").run(["id", "-un"], {})
        assert result.stdout.strip() == "nobody", result


async def test_port_exposure(runtime, monkeypatch) -> None:
    if not backend_cls(runtime).SUPPORTS_PORT_EXPOSURE:

        def provision(*args, **kwargs):
            raise AssertionError("an undeclared feature must be refused first")

        monkeypatch.setattr(launch, "provision_backend", provision)
        server = SimpleNamespace(
            config=ToolsetConfig(runtime=runtime), server_name="tools"
        )
        with refused(runtime, ToolsetError, "cannot expose a port"):
            async with launch.serve(server):
                pass
        return
    async with provisioned(runtime) as view:
        port = view.published_port
        if port is None:
            with socket.socket() as probe:
                probe.bind(("127.0.0.1", 0))
                port = probe.getsockname()[1]
        await view.write("vf-www/index.html", b"exposed")
        await view.run_background(
            [*PYTHON, "-m", "http.server", str(port), "-d", "vf-www"], {}, "http.log"
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


async def test_compose(runtime, tmp_path) -> None:
    if not backend_cls(runtime).SUPPORTS_COMPOSE:
        backend = SimpleNamespace(type=runtime.type, SUPPORTS_COMPOSE=False)
        with refused(runtime, ValueError, "cannot run Compose services"):
            _deployment_strategy(None, backend, ComposeSpec(directory="."))
        return
    skip_unavailable(runtime)
    (tmp_path / "docker-compose.yaml").write_text(
        "services:\n"
        "  main: {}\n"
        "  sidecar:\n"
        "    image: alpine:3.20\n"
        "    command: [sleep, infinity]\n"
    )
    spec = ComposeSpec(directory=str(tmp_path), file="docker-compose.yaml")
    async with (
        asyncio.timeout(TIMEOUT),
        compose_services(runtime, spec, {}, trust_compose=True) as (services, _, _, _),
    ):
        assert set(services) == {"main", "sidecar"}
        resolved = await services["main"].bind().run(["getent", "hosts", "sidecar"], {})
        assert resolved.exit_code == 0, resolved
        sidecar = services["sidecar"].bind()
        result = await sidecar.run(["cat", "/etc/alpine-release"], {})
        assert result.stdout.startswith("3.20"), result
        if services["main"].is_local:
            # Local services reach host loopback, as the interception server needs.
            async with host_server(b"callback") as url:
                workspace = services["main"].bind()
                fetched = await workspace.run(
                    [*PYTHON, "-c", FETCH, workspace.host_url(url)], {}
                )
                assert fetched.stdout.strip() == "callback", fetched


async def test_local_compose_needs_trust() -> None:
    spec = ComposeSpec(directory=".")
    with pytest.raises(ValueError, match="trust-compose"):
        async with compose_services(vf.DockerConfig(), spec, {}):
            pass
    restricted = vf.DockerConfig(allow=["pypi.org"])
    with pytest.raises(ValueError, match="does not enforce network policies"):
        async with compose_services(restricted, spec, {}, trust_compose=True):
            pass


async def test_network_policies(runtime) -> None:
    if not isinstance(runtime, NetworkPolicyConfig):
        with refused(runtime, ValueError, "does not enforce network policies"):
            resolve_runtime_config(
                runtime, task(network=NetworkPolicyConfig(allow=["pypi.org"]))
            )
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
        return (await view.run([*PYTHON, "-c", probe, url], {})).stdout.strip()

    async with provisioned(runtime, allow=["pypi.org"], block=[]) as view:
        assert await reach(view, "https://example.com") == "ok", "setup must be open"
        await view.backend.prepare_execution([])
        assert await reach(view, "https://pypi.org/simple/") == "ok"
        assert await reach(view, "https://example.com") == "denied"
        # A trusted phase after the agent, such as shared grading, gets the network
        # back, raw DNS included.
        await view.backend.prepare_execution(None)
        assert await reach(view, "https://example.com") == "ok"
        resolved = await view.run([*PYTHON, "-c", RESOLVE, "example.com"], {})
        assert resolved.exit_code == 0, resolved


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


class ImagelessBackend(HostBackend):
    SUPPORTS_TASK_IMAGES: ClassVar[bool] = True


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
        result = await view.run(["sh", "-c", 'echo "$GREETING"'], {})
        assert result.stdout.strip() == "hi"
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
        (
            f"{__name__}.ImagelessBackend",
            TypeError,
            r"must declare \['image', 'workdir'\]",
        ),
    ],
)
def test_custom_runtime_rejects_a_bad_import_path(import_path, error, match) -> None:
    with pytest.raises(error, match=match):
        TypeAdapter(RuntimeConfig).validate_python(
            {"type": "custom", "import_path": import_path}
        )


async def test_concurrent_background_starts_keep_every_handle() -> None:
    """A start that lands while another prunes finished handles must not be dropped:
    teardown needs every live handle."""

    class LiveProcess:
        async def poll(self) -> None:
            await asyncio.sleep(0)

    class StubBackend:
        def __init__(self) -> None:
            self.env: dict[str, str] = {}
            self._background: list = []

        async def exec(self, runtime, argv, env):
            await asyncio.sleep(0)
            return LiveProcess()

    backend = StubBackend()
    view = Runtime(backend)  # ty: ignore[invalid-argument-type]
    # The base backend's default, which keeps an `exec` handle per process.
    await asyncio.gather(
        *(
            RuntimeBackend.run_background(
                backend,  # ty: ignore[invalid-argument-type]
                view,
                ["sleep", "600"],
                {},
                f"{i}.log",
            )
            for i in range(8)
        )
    )
    assert len(backend._background) == 8
