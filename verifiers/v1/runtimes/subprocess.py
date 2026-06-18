"""Local subprocess runtime: run the program on the host; server on localhost.

Each rollout gets a fresh `/tmp/<name>` workspace (created on `start`, removed on
`stop`/`cleanup`) used as the program's cwd, so concurrent local rollouts are isolated
and trivially cleaned up. Relative `read`/`write` paths resolve against it.
"""

import asyncio
import contextlib
import hashlib
import os
import shlex
import shutil
import signal
from pathlib import Path
from typing import Literal

from pydantic_config import BaseConfig

from verifiers.v1.runtimes.base import _ENSURE_UV, ProgramResult, Runtime

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

    # script-sha -> the interpreter for that script's deps, resolved once and shared across
    # the worker's per-rollout runtimes (+ a per-sha lock so first-callers provision once).
    _interpreters: dict[str, str] = {}
    _locks: dict[str, asyncio.Lock] = {}
    # Process-wide cap on concurrently-running `cpu_bound` subprocesses (shared across all
    # per-rollout runtimes), created lazily on the eval's loop.
    _cpu_semaphore: asyncio.Semaphore | None = None

    @classmethod
    def _cpu_limiter(cls) -> asyncio.Semaphore:
        """Cap concurrent `cpu_bound` subprocesses at the cores available to this process (override
        with VF_SUBPROCESS_CPU_SLOTS). At an eval's start the whole in-flight batch finishes
        generation together and fires its scoring scripts at once; uncapped they oversubscribe the
        cores N:1, each cold-import (sympy) thrashes, and the per-score wall-clock — which the
        scoring timeout bounds — balloons. Capping at the core count keeps each on its own core."""
        if cls._cpu_semaphore is None:
            override = os.environ.get("VF_SUBPROCESS_CPU_SLOTS")
            try:
                cores = len(os.sched_getaffinity(0))
            except AttributeError:  # not Linux
                cores = os.cpu_count() or 1
            cls._cpu_semaphore = asyncio.Semaphore(
                int(override) if override else max(1, cores)
            )
        return cls._cpu_semaphore

    def __init__(self, config: SubprocessConfig, name: str | None = None) -> None:
        super().__init__(name)
        self.config = config
        self.workdir: Path | None = None
        self._background: list[asyncio.subprocess.Process] = []

    @property
    def descriptor(self) -> str | None:
        return self.workdir.name if self.workdir else None

    async def start(self) -> None:
        self.workdir = Path("/tmp") / self.name
        self.workdir.mkdir()

    async def run(
        self, argv: list[str], env: dict[str, str], cpu_bound: bool = False
    ) -> ProgramResult:
        # A `cpu_bound` command holds a core slot for its whole run, so a herd of scores at an
        # eval's start drains through the cores instead of thrashing all at once (see _cpu_limiter).
        limiter = self._cpu_limiter() if cpu_bound else contextlib.nullcontext()
        async with limiter:
            full_env = {
                k: v for k, v in os.environ.items() if "API_KEY" not in k.upper()
            }
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
                # running, so SIGKILL its whole group (start_new_session => pgid == pid) —
                # otherwise a hung child (a wedged uv/sympy verify) outlives the rollout and leaks
                # CPU. A no-op once it has exited on its own.
                if proc.returncode is None:
                    with contextlib.suppress(ProcessLookupError, PermissionError):
                        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            return ProgramResult(
                exit_code=proc.returncode or 0,
                stdout=stdout.decode(errors="replace"),
                stderr=stderr.decode(errors="replace"),
            )

    async def run_uv_script(
        self,
        script: str | bytes,
        args: list[str] | None = None,
        env: dict[str, str] | None = None,
        cpu_bound: bool = False,
    ) -> ProgramResult:
        """Run a PEP 723 script via a pre-provisioned interpreter instead of `uv run` per call.

        uv keeps a persistent env per script under its cache; we resolve that env's interpreter
        ONCE and exec it directly, keeping uv's per-invocation resolve, cache-lock, ephemeral-env
        materialization and `command -v uv` probe out of the per-rollout hot path (the subprocess
        runtime launches two uv-scripts per rollout — the harness program and, for math, verify).
        docker/prime keep the base `uv run` — their deps live in the sandbox, not on the host."""
        data = script.encode() if isinstance(script, str) else script
        sha = hashlib.sha256(data).hexdigest()
        path = Path("/tmp/vf-scripts") / f"{sha}.py"
        if sha not in self._interpreters:
            async with self._locks.setdefault(sha, asyncio.Lock()):
                if sha not in self._interpreters:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(data)
                    # `uv sync` locks its own cache, so concurrent workers building the same
                    # script are safe; `uv python find` prints the resolved env's interpreter.
                    s = shlex.quote(str(path))
                    cmd = f"{_ENSURE_UV}; uv sync --script {s} -q && uv python find --script {s}"
                    provision = await self.run(["sh", "-c", cmd], {})
                    if provision.exit_code != 0:
                        raise RuntimeError(
                            f"failed to provision env for {path}: {provision.stderr}"
                        )
                    self._interpreters[sha] = provision.stdout.strip().splitlines()[-1]
        return await self.run(
            [self._interpreters[sha], str(path), *(args or [])], env or {}, cpu_bound
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
