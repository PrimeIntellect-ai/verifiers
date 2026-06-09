"""Local subprocess runtime: run the program on the host; server on localhost.

Each rollout gets a fresh, unique `/tmp/v1-<pid>-*` workspace (created on `start`,
removed on `stop`) used as the program's cwd, so concurrent local rollouts are
isolated and trivially cleaned up. Relative `read`/`write` paths resolve against it.
`stop()` removes it; a startup sweep reclaims any left behind by a process that was
killed before it could (the pid in the name says whose workspace it was).
"""

import asyncio
import contextlib
import glob
import os
import shutil
import tempfile
from pathlib import Path
from typing import Literal

from pydantic_config import BaseConfig

from verifiers.v1.runtimes.base import ProgramResult, Runtime

_WORKDIR_PREFIX = f"v1-{os.getpid()}-"
_swept_orphans = False


def _sweep_orphan_workdirs() -> None:
    """Remove `/tmp/v1-<pid>-*` workspaces whose owning process is gone — leftovers from
    a rollout killed before `stop()` ran. PID-keyed, so a concurrent live process's
    workspaces are never touched; best-effort, so a racing peer's `rmtree` can't fail us."""
    for path in glob.glob("/tmp/v1-*-*"):
        pid = os.path.basename(path).split("-")[1]
        if not pid.isdigit():
            continue
        try:
            os.kill(int(pid), 0)
        except ProcessLookupError:  # owner is gone → orphan
            shutil.rmtree(path, ignore_errors=True)
        except OSError:  # alive (or not ours to signal) → keep
            pass

# A local subprocess inherits the host environment EXCEPT any var whose name
# contains "API_KEY" — so it can never reach a real provider with the host's API
# key, while still inheriting harmless config (PATH, HOME, UV_CACHE_DIR, HF_HOME,
# ...). The harness injects its own interception endpoint over OPENAI_* on top
# (see `run`). A container/sandbox is isolated and inherits nothing, so this
# allow-by-default model is subprocess-only.


class SubprocessConfig(BaseConfig):
    type: Literal["subprocess"] = "subprocess"


class SubprocessRuntime(Runtime):
    """Runs the program as a local subprocess in a unique /tmp workspace."""

    def __init__(self, config: SubprocessConfig) -> None:
        self.config = config
        self.workdir: Path | None = None
        self._background: list[asyncio.subprocess.Process] = []

    @property
    def descriptor(self) -> str | None:
        return self.workdir.name if self.workdir else None

    async def start(self) -> None:
        global _swept_orphans
        if not _swept_orphans:  # once per process: reclaim dead processes' leftovers
            _swept_orphans = True
            await asyncio.to_thread(_sweep_orphan_workdirs)
        self.workdir = Path(tempfile.mkdtemp(prefix=_WORKDIR_PREFIX, dir="/tmp"))

    async def run(self, argv: list[str], env: dict[str, str]) -> ProgramResult:
        full_env = {k: v for k, v in os.environ.items() if "API_KEY" not in k.upper()}
        full_env.update(env)
        proc = await asyncio.create_subprocess_exec(
            *argv,
            env=full_env,
            cwd=self.workdir,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()
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

    async def stop(self) -> None:
        for proc in self._background:
            with contextlib.suppress(ProcessLookupError):
                proc.terminate()
        self._background = []
        if self.workdir is not None:
            await asyncio.to_thread(shutil.rmtree, self.workdir, True)
