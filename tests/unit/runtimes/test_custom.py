"""Custom runtimes: a `RuntimeBackend` subclass loads from its import path, validates
its typed config, runs through the framework like a built-in backend, and gets refused
features its config does not declare."""

import asyncio
import contextlib
import shutil
import tempfile
from pathlib import Path

import pytest
from pydantic import TypeAdapter, ValidationError

import verifiers.v1 as vf
from verifiers.v1.runtimes import (
    CustomRuntimeConfig,
    CustomRuntimeInfo,
    ImageConfig,
    ProgramResult,
    Runtime,
    RuntimeBackend,
    RuntimeConfig,
    backend_cls,
    provision_runtime,
)
from verifiers.v1.utils.compile import resolve_runtime_config


class HostConfig(CustomRuntimeConfig):
    greeting: str = "hello"


class HostBackend(RuntimeBackend):
    """Runs commands on the host in a scratch directory."""

    config_class = HostConfig

    def __init__(self, config: HostConfig, name: str | None = None) -> None:
        super().__init__(name)
        self.config = config
        self.info = CustomRuntimeInfo(**config.model_dump())
        self.root: Path | None = None

    async def start(self) -> None:
        self.root = Path(tempfile.mkdtemp(prefix=f"{self.name}-"))
        self.info.id = str(self.root)

    async def _run(
        self, runtime: Runtime, argv: list[str], env: dict[str, str]
    ) -> ProgramResult:
        process = await asyncio.create_subprocess_exec(
            *argv,
            env={"GREETING": self.config.greeting, **runtime.process_env(env)},
            cwd=self.root,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await process.communicate()
        return ProgramResult(process.returncode or 0, stdout.decode(), stderr.decode())

    async def _read(
        self, runtime: Runtime, path: str, max_bytes: int | None = None
    ) -> bytes:
        assert self.root is not None
        return (self.root / path).read_bytes()[:max_bytes]

    async def _write(self, runtime: Runtime, path: str, data: bytes) -> None:
        assert self.root is not None
        (self.root / path).write_bytes(data)

    def cleanup(self) -> None:
        if self.root is not None:
            shutil.rmtree(self.root, ignore_errors=True)


class ImageHostConfig(CustomRuntimeConfig, ImageConfig):
    pass


class ImageHostBackend(HostBackend):
    config_class = ImageHostConfig


class UntypedBackend(HostBackend):
    config_class = dict


NOT_A_BACKEND = object()


def config(backend: str, **fields) -> RuntimeConfig:
    return TypeAdapter(RuntimeConfig).validate_python(
        {"type": "custom", "import_path": f"{__name__}.{backend}", **fields}
    )


async def test_custom_runtime_runs_the_plugin() -> None:
    plugin = config("HostBackend", greeting="hi")
    assert isinstance(plugin, HostConfig)
    assert backend_cls(plugin) is HostBackend
    async with provision_runtime(plugin) as runtime:
        assert runtime.type == "custom"
        await runtime.write("note.txt", b"plugin")
        result = await runtime.run(["sh", "-c", 'echo "$GREETING $(cat note.txt)"'], {})
        assert result.stdout.strip() == "hi plugin"
        assert await runtime.read("note.txt") == b"plugin"
        assert runtime.info.greeting == "hi"
        root = Path(runtime.info.id)
    assert not root.exists()


def test_custom_runtime_validates_its_typed_config() -> None:
    with pytest.raises(ValidationError, match="greting"):
        config("HostBackend", greting="hi")
    with pytest.raises(ValidationError, match="greeting"):
        config("HostBackend", greeting=["hi"])


def test_custom_runtime_config_round_trips() -> None:
    agent = vf.AgentConfig(
        model="m",
        runtime={
            "type": "custom",
            "import_path": f"{__name__}.HostBackend",
            "greeting": "hi",
        },
    )
    loaded = vf.AgentConfig.model_validate_json(agent.model_dump_json())
    assert isinstance(loaded.runtime, HostConfig)
    assert loaded.runtime.greeting == "hi"


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


@pytest.mark.parametrize("backend", ["HostBackend", "ImageHostBackend"])
def test_custom_runtime_config_declares_image_support(backend) -> None:
    task = vf.Task(data=vf.TaskData(idx=0, prompt=None, image="alpine"))
    supported = backend == "ImageHostBackend"
    with (
        contextlib.nullcontext()
        if supported
        else pytest.raises(ValueError, match="custom runtime cannot run images")
    ):
        assert resolve_runtime_config(config(backend), task).image == "alpine"


def test_custom_runtime_refuses_undeclared_network_policy() -> None:
    task = vf.Task(data=vf.TaskData(idx=0, prompt=None, network_allow=["pypi.org"]))
    with pytest.raises(ValueError, match="does not enforce network policies"):
        resolve_runtime_config(config("ImageHostBackend"), task)
