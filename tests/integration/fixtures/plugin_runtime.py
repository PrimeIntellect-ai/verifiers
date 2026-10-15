"""A runtime plugin outside the verifiers package, selected with
`{"type": "custom", "import_path": "plugin_runtime.HostBackend"}`.

It runs commands on the host in a scratch directory and implements the full
`RuntimeBackend` interface through public imports only. It declares no image, user,
or network-policy support, so the framework refuses those features up front."""

import asyncio
import contextlib
import os
import shutil
import signal
import tempfile
from collections.abc import AsyncIterator
from pathlib import Path

from verifiers.v1.runtimes import (
    CustomRuntimeConfig,
    CustomRuntimeInfo,
    ProgramResult,
    Runtime,
    RuntimeBackend,
    RuntimeProcess,
)


async def _chunks(reader: asyncio.StreamReader) -> AsyncIterator[bytes]:
    while chunk := await reader.read(64 * 1024):
        yield chunk


class HostProcess(RuntimeProcess):
    def __init__(self, process: asyncio.subprocess.Process) -> None:
        assert process.stdin and process.stdout and process.stderr
        self._process = process
        self.stdout = _chunks(process.stdout)
        self.stderr = _chunks(process.stderr)

    async def write(self, data: bytes) -> None:
        assert self._process.stdin is not None
        self._process.stdin.write(data)
        await self._process.stdin.drain()

    async def wait(self) -> int:
        return await self._process.wait()

    async def poll(self) -> int | None:
        return self._process.returncode

    def _signal(self, sig: signal.Signals) -> None:
        if self._process.returncode is None:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(self._process.pid, sig)

    async def terminate(self) -> None:
        self._signal(signal.SIGTERM)

    async def kill(self) -> None:
        self._signal(signal.SIGKILL)


class HostBackend(RuntimeBackend):
    def __init__(self, config: CustomRuntimeConfig, name: str | None = None) -> None:
        super().__init__(name)
        self.config = config
        self.info = CustomRuntimeInfo(**config.model_dump())
        self.root: Path | None = None
        self._processes: list[asyncio.subprocess.Process] = []

    async def start(self) -> None:
        self.root = Path(tempfile.mkdtemp(prefix=f"{self.name}-"))
        self.config = self.config.model_copy(update={"workdir": str(self.root)})
        self.info.id = self.info.workdir = str(self.root)

    def _env(self, runtime: Runtime, env: dict[str, str]) -> dict[str, str]:
        return {**os.environ, **runtime.process_env(env)}

    async def _spawn(
        self, runtime: Runtime, argv: list[str], env: dict[str, str], **io
    ):
        return await asyncio.create_subprocess_exec(
            *argv,
            env=self._env(runtime, env),
            cwd=self.root,
            start_new_session=True,
            **io,
        )

    async def _run(
        self, runtime: Runtime, argv: list[str], env: dict[str, str]
    ) -> ProgramResult:
        process = await self._spawn(
            runtime,
            argv,
            env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, stderr = await process.communicate()
        finally:
            if process.returncode is None:
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(process.pid, signal.SIGKILL)
        return ProgramResult(
            process.returncode or 0,
            stdout.decode(errors="replace"),
            stderr.decode(errors="replace"),
        )

    async def _open_process(
        self, runtime: Runtime, argv: list[str], env: dict[str, str]
    ) -> RuntimeProcess:
        process = await self._spawn(
            runtime,
            argv,
            env,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        self._processes.append(process)
        return HostProcess(process)

    async def _run_background(
        self, runtime: Runtime, argv: list[str], env: dict[str, str], log: str
    ) -> None:
        assert self.root is not None
        with (self.root / log).open("wb") as output:
            process = await self._spawn(
                runtime, argv, env, stdout=output, stderr=asyncio.subprocess.STDOUT
            )
        self._processes.append(process)

    async def _read(
        self, runtime: Runtime, path: str, max_bytes: int | None = None
    ) -> bytes:
        assert self.root is not None
        with (self.root / path).open("rb") as file:
            return file.read(-1 if max_bytes is None else max_bytes)

    async def _write(self, runtime: Runtime, path: str, data: bytes) -> None:
        assert self.root is not None
        target = self.root / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)

    def cleanup(self) -> None:
        for process in self._processes:
            if process.returncode is None:
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(process.pid, signal.SIGKILL)
        self._processes = []
        if self.root is not None:
            shutil.rmtree(self.root, ignore_errors=True)
