"""Runtimes: every supported runtime passes the conformance checks in
`scripts/check_runtime.py` for the backend contract and each feature it declares, and
is refused, before anything is provisioned, each feature it does not declare. A custom runtime plugin loads from its
import path and passes the same checks.

Conformance provisions real runtimes: Docker needs a reachable daemon and Prime needs
`PRIME_API_KEY`; otherwise those cells skip."""

import asyncio
import os
import shutil
import subprocess
import tempfile
from collections.abc import AsyncIterator
from functools import cache
from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar

import pytest
from check_runtime import CORE, IMAGE, check_runtime
from pydantic import TypeAdapter, ValidationError

import verifiers.v1 as vf
from verifiers.v1.errors import ToolsetError
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
from verifiers.v1.runtimes.compose import ComposeSpec
from verifiers.v1.runtimes.strategies import _deployment_strategy
from verifiers.v1.utils.compile import resolve_runtime_config

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


def config(runtime: str) -> RuntimeConfig:
    fields = {"labels": ["vf-ci"]} if runtime == "prime" else {}
    return TypeAdapter(RuntimeConfig).validate_python({"type": runtime, **fields})


def task(**data) -> vf.Task:
    return vf.Task(data=vf.TaskData(idx=0, prompt=None, **data))


@pytest.mark.parametrize("runtime", RUNTIMES)
async def test_runtime_conformance(runtime) -> None:
    if runtime == "docker" and not docker_available():
        pytest.skip("needs a reachable Docker daemon")
    if runtime == "prime" and not os.environ.get("PRIME_API_KEY"):
        pytest.skip("needs PRIME_API_KEY")
    # Prime builds a VM image on an image's first use; use one it has cached.
    image = "python:3.11-slim" if runtime == "prime" else IMAGE
    results = await check_runtime(config(runtime), image=image)
    assert not [result for result in results if result.passed is False], results
    expected = {*CORE, *(c for c, runtimes in SUPPORTED.items() if runtime in runtimes)}
    assert {result.name for result in results if result.passed} == expected


async def refuse_task_images(runtime: str, monkeypatch) -> None:
    resolve_runtime_config(config(runtime), task(image=IMAGE))


async def refuse_user_switching(runtime: str, monkeypatch) -> None:
    resolve_runtime_config(config(runtime), task(agent_user="nobody"))


async def refuse_network_policies(runtime: str, monkeypatch) -> None:
    resolve_runtime_config(config(runtime), task(network_allow=["pypi.org"]))


async def refuse_nested_containers(runtime: str, monkeypatch) -> None:
    cls = backend_cls(config(runtime))
    backend = SimpleNamespace(
        type=runtime, SUPPORTS_NESTED_CONTAINERS=cls.SUPPORTS_NESTED_CONTAINERS
    )
    _deployment_strategy(None, backend, ComposeSpec(directory="."))


async def refuse_port_exposure(runtime: str, monkeypatch) -> None:
    def provision(*args, **kwargs):
        raise AssertionError(
            "an undeclared feature must be refused before provisioning"
        )

    monkeypatch.setattr(launch, "provision_backend", provision)
    server = SimpleNamespace(
        config=ToolsetConfig(runtime=config(runtime)), server_name="tools"
    )
    async with launch.serve(server):
        pass


REFUSALS = {
    "task_images": (refuse_task_images, ValueError, "cannot run images"),
    "user_switching": (
        refuse_user_switching,
        ValueError,
        "cannot run as another user",
    ),
    "port_exposure": (refuse_port_exposure, ToolsetError, "cannot expose a port"),
    "nested_containers": (
        refuse_nested_containers,
        ValueError,
        "cannot run nested containers",
    ),
    "network_policies": (
        refuse_network_policies,
        ValueError,
        "does not enforce network policies",
    ),
}


@pytest.mark.parametrize(
    "capability,runtime",
    [
        (c, r)
        for c, runtimes in SUPPORTED.items()
        for r in RUNTIMES
        if r not in runtimes
    ],
)
async def test_runtime_refuses_undeclared_features(
    capability, runtime, monkeypatch
) -> None:
    refuse, error, message = REFUSALS[capability]
    with pytest.raises(error, match=f"the {runtime} runtime {message}"):
        await refuse(runtime, monkeypatch)


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


async def test_custom_runtime_conformance() -> None:
    results = await check_runtime(plugin())
    assert not [result for result in results if result.passed is False], results
    assert {result.name for result in results if result.passed} == set(CORE)


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
