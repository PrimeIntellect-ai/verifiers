"""Local subprocess runtime: run the program on the host; server on localhost.

Each rollout gets a fresh `/tmp/<name>` workspace (created on `start`, removed on
`stop`/`cleanup`) used as the program's cwd, so concurrent local rollouts are isolated
and trivially cleaned up. Relative `read`/`write` paths resolve against it.
"""

import asyncio
import contextlib
import os
import shutil
import signal
from pathlib import Path
from typing import Literal

from pydantic_config import BaseConfig

from verifiers.v1.runtimes.base import ProgramResult, Runtime

# A local subprocess inherits the host environment EXCEPT any var whose name
# contains "API_KEY" — so it can never reach a real provider with the host's API
# key, while still inheriting harmless config (PATH, HOME, UV_CACHE_DIR, HF_HOME,
# ...). The harness injects its own interception endpoint over OPENAI_* on top
# (see `run`). A container/sandbox is isolated and inherits nothing, so this
# allow-by-default model is subprocess-only. NOTE: this strip applies to EVERY
# program run here, including a task's tool/user server — so a tool server that
# genuinely needs an API key won't get one on subprocess placement; give it its
# own runtime (docker/prime) or have it fetch the key itself.


class SubprocessConfig(BaseConfig):
    type: Literal["subprocess"] = "subprocess"


class SubprocessRuntime(Runtime):
    """Runs the program as a local subprocess in a unique /tmp workspace."""

    # Share prepared script environments across the worker's per-rollout runtimes.
    _interpreters: dict[str, str] = {}
    _locks: dict[str, asyncio.Lock] = {}

    def __init__(self, config: SubprocessConfig, name: str | None = None) -> None:
        super().__init__(name)
        self._uv_interpreters = self._interpreters
        self._uv_script_locks = self._locks
        self.config = config
        self.workdir: Path | None = None
        self._background: list[asyncio.subprocess.Process] = []

    @property
    def descriptor(self) -> str | None:
        return self.workdir.name if self.workdir else None

    async def start(self) -> None:
        self.workdir = Path("/tmp") / self.name
        self.workdir.mkdir()

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

    def cleanup(self) -> None:
        for proc in self._background:
            # Kill the whole group (start_new_session => pgid == pid), not just proc.pid, so a
            # background server's children (sh -> uv -> python) are reaped too.
            with contextlib.suppress(ProcessLookupError, PermissionError):
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        self._background = []
        if self.workdir is not None:
            shutil.rmtree(self.workdir, ignore_errors=True)
