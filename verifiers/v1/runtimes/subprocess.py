"""Local subprocess runtime."""

import asyncio
import contextlib
import os
import shutil
import signal
from pathlib import Path
from typing import Literal

from pydantic_config import BaseConfig

from verifiers.v1.runtimes.base import BaseRuntimeInfo, ProgramResult, Runtime

_BACKGROUND_STOP_TIMEOUT = 5

# Implicit host inheritance removes every name containing "API_KEY" while keeping
# harmless settings such as PATH, HOME, and cache locations. The explicit `env`
# argument is merged afterward, so callers can deliberately pass credentials and
# child processes inherit them. Containers and sandboxes inherit no host environment.


class SubprocessConfig(BaseConfig):
    type: Literal["subprocess"] = "subprocess"


class SubprocessRuntimeInfo(SubprocessConfig, BaseRuntimeInfo):
    pass


class SubprocessRuntime(Runtime):
    # Share prepared script environments across the worker's per-rollout runtimes.
    _interpreters: dict[str, str] = {}
    _locks: dict[str, asyncio.Lock] = {}

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
            # scoring_timeout / harness_timeout fired): communicate() leaves the process
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

    async def read(self, path: str) -> bytes:
        return await asyncio.to_thread((self.workdir / path).read_bytes)

    async def write(self, path: str, data: bytes) -> None:
        target = self.workdir / path
        target.parent.mkdir(parents=True, exist_ok=True)
        await asyncio.to_thread(target.write_bytes, data)

    @staticmethod
    def _signal(proc: asyncio.subprocess.Process, sig: signal.Signals) -> None:
        if proc.returncode is not None:
            return
        # Signal the whole group (start_new_session => pgid == pid), not just proc.pid,
        # so a background server's children (sh -> uv -> python) stop with it.
        with contextlib.suppress(ProcessLookupError, PermissionError):
            os.killpg(os.getpgid(proc.pid), sig)

    async def teardown(self) -> None:
        """Stop and reap background servers before their event loop closes."""
        background = list(self._background)
        for proc in background:
            self._signal(proc, signal.SIGTERM)
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
                    self._signal(proc, signal.SIGKILL)
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
            self._signal(proc, signal.SIGTERM)
        self._background = []
        if self.workdir is not None:
            shutil.rmtree(self.workdir, ignore_errors=True)
