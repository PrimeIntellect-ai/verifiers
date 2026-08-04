"""Local subprocess runtime."""

import asyncio
import contextlib
import os
import shutil
import signal
from collections.abc import AsyncIterator
from pathlib import Path
from typing import ClassVar, Literal

from pydantic_config import BaseConfig

from verifiers.v1.runtimes.base import (
    BaseRuntimeInfo,
    ProgramResult,
    Runtime,
    RuntimeProcess,
)

_BACKGROUND_STOP_TIMEOUT = 5

# Implicit host inheritance removes every name containing "API_KEY" while keeping
# harmless settings such as PATH, HOME, and cache locations. The explicit `env`
# argument is merged afterward, so callers can deliberately pass credentials and
# child processes inherit them. Containers and sandboxes inherit no host environment.


class SubprocessConfig(BaseConfig):
    type: Literal["subprocess"] = "subprocess"


class SubprocessRuntimeInfo(SubprocessConfig, BaseRuntimeInfo):
    pass


async def _read_stream(reader: asyncio.StreamReader) -> AsyncIterator[bytes]:
    while chunk := await reader.read(64 * 1024):
        yield chunk


def _signal_process(
    process: asyncio.subprocess.Process, signal_: signal.Signals
) -> None:
    if process.returncode is not None:
        return
    # open_process() creates a new session, so pgid == pid and signalling the
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
        self.stdout = _read_stream(process.stdout)
        self.stderr = _read_stream(process.stderr)

    async def write(self, data: bytes) -> None:
        self._stdin.write(data)
        await self._stdin.drain()

    async def wait(self) -> int:
        return await self._process.wait()

    async def terminate(self) -> None:
        _signal_process(self._process, signal.SIGTERM)

    async def kill(self) -> None:
        _signal_process(self._process, signal.SIGKILL)


class SubprocessRuntime(Runtime):
    # Share prepared script environments across the worker's per-rollout runtimes.
    _interpreters: ClassVar[dict[str, str]] = {}
    _locks: ClassVar[dict[str, asyncio.Lock]] = {}

    def __init__(self, config: SubprocessConfig, name: str | None = None) -> None:
        super().__init__(name)
        self._uv_interpreters = self._interpreters
        self._uv_script_locks = self._locks
        self.config = config
        self.info = SubprocessRuntimeInfo(**config.model_dump())
        self.workdir: Path | None = None
        self._background: list[asyncio.subprocess.Process] = []

    async def start(self) -> None:
        self.workdir = Path("/tmp") / self.name
        self.workdir.mkdir()
        self.info.id = str(self.workdir)

    async def run(self, argv: list[str], env: dict[str, str]) -> ProgramResult:
        full_env = {k: v for k, v in os.environ.items() if "API_KEY" not in k.upper()}
        full_env.update(env)
        proc = await asyncio.create_subprocess_exec(
            *argv,
            env=full_env,
            cwd=self.workdir,
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

    async def open_process(
        self, argv: list[str], env: dict[str, str]
    ) -> RuntimeProcess:
        full_env = {k: v for k, v in os.environ.items() if "API_KEY" not in k.upper()}
        full_env.update(env)
        proc = await asyncio.create_subprocess_exec(
            *argv,
            env=full_env,
            cwd=self.workdir,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=True,
        )
        self._background.append(proc)
        return SubprocessProcess(proc)

    async def run_background(
        self, argv: list[str], env: dict[str, str], log: str
    ) -> None:
        full_env = {k: v for k, v in os.environ.items() if "API_KEY" not in k.upper()}
        full_env.update(env)
        logfile = self.workdir / log
        with logfile.open(
            "wb"
        ) as f:  # child dups the fd; safe to close ours after spawn
            proc = await asyncio.create_subprocess_exec(
                *argv,
                env=full_env,
                cwd=self.workdir,
                stdout=f,
                stderr=asyncio.subprocess.STDOUT,
                start_new_session=True,  # own process group, so cleanup() reaps the whole tree
            )
        self._background.append(
            proc
        )  # killed in stop() — a host process won't die on its own

    async def _read(self, path: str) -> bytes:
        return await asyncio.to_thread((self.workdir / path).read_bytes)

    async def write(self, path: str, data: bytes) -> None:
        target = self.workdir / path
        target.parent.mkdir(parents=True, exist_ok=True)
        await asyncio.to_thread(target.write_bytes, data)

    async def teardown(self) -> None:
        """Stop and reap background servers before their event loop closes."""
        background = list(self._background)
        for proc in background:
            _signal_process(proc, signal.SIGTERM)
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
                    _signal_process(proc, signal.SIGKILL)
                with contextlib.suppress(TimeoutError):
                    await asyncio.wait_for(
                        asyncio.gather(
                            *(proc.wait() for proc in background),
                            return_exceptions=True,
                        ),
                        timeout=_BACKGROUND_STOP_TIMEOUT,
                    )
        self._background = []
        if self.workdir is not None:
            await asyncio.to_thread(shutil.rmtree, self.workdir, True)

    def cleanup(self) -> None:
        for proc in self._background:
            _signal_process(proc, signal.SIGTERM)
        self._background = []
        if self.workdir is not None:
            shutil.rmtree(self.workdir, ignore_errors=True)
