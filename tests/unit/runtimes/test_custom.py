"""Custom runtimes: a `RuntimeBackend` subclass loads from its import path, runs through
the framework like a built-in backend, and gets refused features it does not declare."""

import asyncio
import shutil
import tempfile
from pathlib import Path

import pytest

import verifiers.v1 as vf
from verifiers.v1.runtimes import (
    CustomRuntimeConfig,
    CustomRuntimeInfo,
    ProgramResult,
    Runtime,
    RuntimeBackend,
    backend_cls,
    provision_runtime,
)
from verifiers.v1.utils.compile import resolve_runtime_config


class HostBackend(RuntimeBackend):
    """Runs commands on the host in a scratch directory."""

    def __init__(self, config: CustomRuntimeConfig, name: str | None = None) -> None:
        super().__init__(name)
        self.config = config
        self.info = CustomRuntimeInfo(**config.model_dump())
        self.root: Path | None = None

    async def start(self) -> None:
        self.root = Path(tempfile.mkdtemp(prefix=f"{self.name}-"))
        self.config = self.config.model_copy(update={"workdir": str(self.root)})
        self.info.id = str(self.root)

    async def _run(
        self, runtime: Runtime, argv: list[str], env: dict[str, str]
    ) -> ProgramResult:
        process = await asyncio.create_subprocess_exec(
            *argv,
            env=runtime.process_env(env),
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


NOT_A_BACKEND = object()


def plugin(**fields) -> CustomRuntimeConfig:
    return CustomRuntimeConfig(import_path=f"{__name__}.HostBackend", **fields)


async def test_custom_runtime_runs_the_plugin() -> None:
    config = plugin(kwargs={"region": "us"})
    assert backend_cls(config) is HostBackend
    async with provision_runtime(config, env={"GREETING": "hi"}) as runtime:
        assert runtime.type == "custom"
        assert runtime.config.kwargs == {"region": "us"}
        await runtime.write("note.txt", b"plugin")
        result = await runtime.run(["sh", "-c", 'echo "$GREETING $(cat note.txt)"'], {})
        assert result.stdout.strip() == "hi plugin"
        assert await runtime.read("note.txt") == b"plugin"
        root = Path(runtime.info.id)
    assert not root.exists()


@pytest.mark.parametrize(
    "import_path,error,match",
    [
        ("HostBackend", ValueError, "must be a dotted path"),
        ("no_such_package.Backend", ValueError, "cannot import custom runtime"),
        (f"{__name__}.Missing", ValueError, "cannot import custom runtime"),
        (f"{__name__}.NOT_A_BACKEND", TypeError, "must be a RuntimeBackend subclass"),
    ],
)
def test_custom_runtime_rejects_a_bad_import_path(import_path, error, match) -> None:
    with pytest.raises(error, match=match):
        backend_cls(CustomRuntimeConfig(import_path=import_path))


def test_custom_runtime_refuses_undeclared_features() -> None:
    task = vf.Task(data=vf.TaskData(idx=0, prompt=None, image="alpine"))
    with pytest.raises(ValueError, match="custom runtime cannot run images"):
        resolve_runtime_config(plugin(), task)
