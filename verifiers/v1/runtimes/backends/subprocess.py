"""Local subprocess runtime."""

from __future__ import annotations

import asyncio
import contextlib
import os
import shutil
import signal
import stat
from collections.abc import AsyncIterator
from pathlib import Path
from typing import ClassVar

from verifiers.v1.configs.runtime import SubprocessConfig
from verifiers.v1.errors import SandboxError
from verifiers.v1.runtimes.base import (
    BaseRuntimeInfo,
    ProgramResult,
    RuntimeBackend,
    RuntimeProcess,
)
from verifiers.v1.runtimes.runtime import Runtime
from verifiers.v1.utils.paths import CACHE_DIR

_BACKGROUND_STOP_TIMEOUT = 5

# Implicit host inheritance removes every name containing "API_KEY" while keeping
# harmless settings such as PATH, HOME, and cache locations. The explicit `env`
# argument is merged afterward, so callers can deliberately pass credentials and
# child processes inherit them. Containers and sandboxes inherit no host environment.


class SubprocessRuntimeInfo(SubprocessConfig, BaseRuntimeInfo):
    pass


async def read_stream(reader: asyncio.StreamReader) -> AsyncIterator[bytes]:
    while chunk := await reader.read(64 * 1024):
        yield chunk


def signal_process(
    process: asyncio.subprocess.Process, signal_: signal.Signals
) -> None:
    if process.returncode is not None:
        return
    # exec() creates a new session, so pgid == pid and signalling the
    # group reaps the complete process tree.
    with contextlib.suppress(ProcessLookupError, PermissionError):
        os.killpg(os.getpgid(process.pid), signal_)


class SubprocessProcess(RuntimeProcess):
    def __init__(self, process: asyncio.subprocess.Process) -> None:
        self._process = process
        assert process.stdin is not None
        assert process.stdout is not None
        assert process.stderr is not None
        self._stdin = process.stdin
        self.stdout = read_stream(process.stdout)
        self.stderr = read_stream(process.stderr)

    async def write(self, data: bytes) -> None:
        self._stdin.write(data)
        await self._stdin.drain()

    async def wait(self) -> int:
        return await self._process.wait()

    async def poll(self) -> int | None:
        return self._process.returncode

    async def terminate(self) -> None:
        signal_process(self._process, signal.SIGTERM)

    async def kill(self) -> None:
        signal_process(self._process, signal.SIGKILL)


# Shared across the worker's per-rollout runtimes, which use one host script cache.
_UV_INTERPRETERS: dict[str, str] = {}
_UV_LOCKS: dict[str, asyncio.Lock] = {}


class SubprocessBackend(RuntimeBackend):
    is_local: ClassVar[bool] = True
    SUPPORTS_PORT_EXPOSURE: ClassVar[bool] = True
    # Share prepared script environments across the worker's per-rollout runtimes.
    scripts_dir: ClassVar[str] = str(CACHE_DIR / "runtimes" / "scripts")

    def __init__(self, config: SubprocessConfig, name: str | None = None) -> None:
        super().__init__(name)
        self.cache.uv_interpreters = _UV_INTERPRETERS
        self.cache.uv_locks = _UV_LOCKS
        self.config = config
        self.info = SubprocessRuntimeInfo(**config.model_dump())
        self.workdir: Path | None = None
        self._children: list[asyncio.subprocess.Process] = []

    async def start(self) -> None:
        self.workdir = CACHE_DIR / "runtimes" / "subprocess" / self.name
        self.workdir.mkdir(parents=True)
        self.info.id = str(self.workdir)

    async def run(
        self, runtime: Runtime, argv: list[str], env: dict[str, str]
    ) -> ProgramResult:
        full_env = {k: v for k, v in os.environ.items() if "API_KEY" not in k.upper()}
        full_env.update(runtime.process_env(env))
        proc = await asyncio.create_subprocess_exec(
            *argv,
            env=full_env,
            cwd=self.workdir,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=True,  # own process group, so we can reap the whole tree
        )
        try:
            stdout, stderr = await proc.communicate()
        finally:
            # If the await didn't finish, the caller cancelled it (e.g. the rollout's
            # scoring_timeout / agent_timeout fired): communicate() leaves the process
            # running, so SIGKILL its whole group (start_new_session => pgid == pid) — otherwise
            # a hung child (a wedged uv/sympy verify) outlives the rollout and leaks CPU. A
            # no-op once it has exited on its own.
            if proc.returncode is None:
                with contextlib.suppress(ProcessLookupError, PermissionError):
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        return ProgramResult(
            exit_code=proc.returncode or 0,
            stdout=stdout.decode(errors="replace"),
            stderr=stderr.decode(errors="replace"),
        )

    async def exec(
        self, runtime: Runtime, argv: list[str], env: dict[str, str]
    ) -> RuntimeProcess:
        full_env = {k: v for k, v in os.environ.items() if "API_KEY" not in k.upper()}
        full_env.update(runtime.process_env(env))
        proc = await asyncio.create_subprocess_exec(
            *argv,
            env=full_env,
            cwd=self.workdir,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=True,
        )
        self._children = [p for p in self._children if p.returncode is None]
        self._children.append(proc)
        return SubprocessProcess(proc)

    async def read(
        self, runtime: Runtime, path: str, max_bytes: int | None = None
    ) -> bytes:
        if max_bytes is None:
            return await asyncio.to_thread((self.workdir / path).read_bytes)

        # Leave special files to the cancellable shell path without opening a
        # FIFO and waking its writer. A nonblocking open and descriptor check
        # also cover a regular path being replaced by a FIFO after the stat.
        def read() -> bytes | None:
            target = self.workdir / path
            if not stat.S_ISREG(target.stat().st_mode):
                return None
            fd = os.open(target, os.O_RDONLY | os.O_NONBLOCK)
            with os.fdopen(fd, "rb") as file:
                if not stat.S_ISREG(os.fstat(file.fileno()).st_mode):
                    return None
                return file.read(max_bytes)

        try:
            data = await asyncio.to_thread(read)
        except OSError as exc:
            raise SandboxError(f"read {path!r}: {exc}") from exc
        return (
            data if data is not None else await super().read(runtime, path, max_bytes)
        )

    async def write(self, runtime: Runtime, path: str, data: bytes) -> None:
        target = self.workdir / path
        target.parent.mkdir(parents=True, exist_ok=True)
        await asyncio.to_thread(target.write_bytes, data)

    async def teardown(self) -> None:
        """Stop and reap background servers before their event loop closes."""
        background = list(self._children)
        for proc in background:
            signal_process(proc, signal.SIGTERM)
        if background:
            try:
                await asyncio.wait_for(
                    asyncio.gather(
                        *(proc.wait() for proc in background), return_exceptions=True
                    ),
                    timeout=_BACKGROUND_STOP_TIMEOUT,
                )
            except TimeoutError:
                for proc in background:
                    signal_process(proc, signal.SIGKILL)
                with contextlib.suppress(TimeoutError):
                    await asyncio.wait_for(
                        asyncio.gather(
                            *(proc.wait() for proc in background),
                            return_exceptions=True,
                        ),
                        timeout=_BACKGROUND_STOP_TIMEOUT,
                    )
        self._children = []
        if self.workdir is not None:
            await asyncio.to_thread(shutil.rmtree, self.workdir, True)

    def cleanup(self) -> None:
        for proc in self._children:
            signal_process(proc, signal.SIGTERM)
        self._children = []
        if self.workdir is not None:
            shutil.rmtree(self.workdir, ignore_errors=True)
