"""Local subprocess runtime: run the program on the host; server on localhost.

Each rollout gets a fresh `/tmp/<name>` workspace (created on `start`, removed on
`stop`/`cleanup`) used as the program's cwd, so concurrent local rollouts are isolated
and trivially cleaned up. Relative `read`/`write` paths resolve against it.

`run_uv_script` is specialized here: instead of `uv run` per call (which re-resolves,
re-locks the cache, re-materializes the ephemeral env and re-probes for `uv` on EVERY
invocation — overhead that dominates and serializes at high concurrency), it pre-resolves
one venv per distinct PEP 723 dependency set ONCE and execs that interpreter directly. The
hot path becomes a bare `python <script>` with no `uv` in it. docker/prime keep the base
`uv run` (their deps live in the sandbox, not on the host).
"""

import asyncio
import contextlib
import fcntl
import hashlib
import os
import re
import shutil
import signal
import subprocess
import sys
import tomllib
import uuid
from pathlib import Path
from typing import Literal

from pydantic_config import BaseConfig

from verifiers.v1.runtimes.base import ProgramResult, Runtime

# A local subprocess inherits the host environment EXCEPT any var whose name
# contains "API_KEY" — so it can never reach a real provider with the host's API
# key, while still inheriting harmless config (PATH, HOME, UV_CACHE_DIR, HF_HOME,
# ...). The harness injects its own interception endpoint over OPENAI_* on top
# (see `run`). A container/sandbox is isolated and inherits nothing, so this
# allow-by-default model is subprocess-only.

# One venv per distinct PEP 723 dependency set, built once and reused by every rollout, so
# the per-rollout hot path is a bare interpreter exec (no `uv run`). Local + persistent.
_SCRIPT_VENV_ROOT = Path(
    os.environ.get("VF_SCRIPT_ENV_ROOT", Path.home() / ".cache" / "vf-script-venvs")
)
# PEP 723 inline-metadata block (the spec's reference regex).
_PEP723_RE = r"(?m)^# /// (?P<type>[a-zA-Z0-9-]+)$\s(?P<content>(^#(| .*)$\s)+)^# ///$"


def _pep723_dependencies(source: str) -> list[str]:
    """The `dependencies` of a PEP 723 `script` block (empty list if there's no block)."""
    blocks = [m for m in re.finditer(_PEP723_RE, source) if m.group("type") == "script"]
    if not blocks:
        return []
    content = "".join(
        line[2:] if line.startswith("# ") else line[1:]
        for line in blocks[0].group("content").splitlines(keepends=True)
    )
    return list(tomllib.loads(content).get("dependencies", []))


def _build_script_venv(venv: Path, deps: list[str]) -> None:
    """Create `venv` with `deps` installed — once, race-safe across processes. An exclusive
    flock serializes concurrent first-callers; a `.ready` sentinel (written only after a clean
    install) makes a crash-interrupted build rebuild instead of being reused half-finished.
    uv hardlinks wheels from its warm cache, so this is a one-time, near-instant cost."""
    ready = venv / ".ready"
    venv.parent.mkdir(parents=True, exist_ok=True)
    with open(venv.parent / f"{venv.name}.lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if ready.exists():
            return
        env = {
            **os.environ,
            "PATH": f"{Path.home() / '.local' / 'bin'}:{os.environ.get('PATH', '')}",
        }
        subprocess.run(["uv", "venv", "--quiet", str(venv)], check=True, env=env)
        if deps:
            subprocess.run(
                [
                    "uv",
                    "pip",
                    "install",
                    "--quiet",
                    "--python",
                    str(venv / "bin" / "python"),
                    *deps,
                ],
                check=True,
                env=env,
            )
        ready.write_text("ok")


class SubprocessConfig(BaseConfig):
    type: Literal["subprocess"] = "subprocess"


class SubprocessRuntime(Runtime):
    """Runs the program as a local subprocess in a unique /tmp workspace."""

    # Shared across the per-rollout runtime instances on a worker: dep-set hash -> interpreter
    # path, and a per-hash build lock (so concurrent first-callers build once, in-process).
    _venv_cache: dict[str, str] = {}
    _venv_locks: dict[str, asyncio.Lock] = {}

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

    async def run_uv_script(
        self,
        script: str | bytes,
        args: list[str] | None = None,
        env: dict[str, str] | None = None,
    ) -> ProgramResult:
        """Run a PEP 723 script via a pre-resolved cached interpreter instead of `uv run`.

        uv keys an environment by dependency set; we build that env ONCE (`_build_script_venv`)
        and exec its python directly. This drops uv's per-invocation resolve, cache-lock,
        ephemeral-env materialization, and `command -v uv` probe from the per-rollout hot path —
        the overhead that dominates and serializes (cache-lock contention) at high concurrency.
        The script is written once to a stable, content-addressed path (so its absolute path is
        constant) and execed with the cached interpreter as a bare subprocess (isolated cwd,
        own process group, killable on timeout — exactly like `run`)."""
        data = script.encode() if isinstance(script, str) else script
        path = Path("/tmp/vf-scripts") / f"{hashlib.sha256(data).hexdigest()}.py"
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_name(f"{path.name}.{uuid.uuid4().hex}.tmp")
            tmp.write_bytes(data)
            os.replace(tmp, path)  # atomic publish; concurrent writers are idempotent
        python = await self._script_python(data.decode(errors="replace"))
        return await self.run([python, str(path), *(args or [])], env or {})

    async def _script_python(self, source: str) -> str:
        """The cached interpreter for a script's PEP 723 deps, building its venv on first use.
        Stdlib-only scripts (no deps) just use the host interpreter."""
        deps = _pep723_dependencies(source)
        if not deps:
            return sys.executable
        key = hashlib.sha256("\x00".join(sorted(deps)).encode()).hexdigest()
        if key not in self._venv_cache:
            lock = self._venv_locks.setdefault(key, asyncio.Lock())
            async with lock:
                if key not in self._venv_cache:
                    venv = _SCRIPT_VENV_ROOT / key
                    await asyncio.to_thread(_build_script_venv, venv, deps)
                    self._venv_cache[key] = str(venv / "bin" / "python")
        return self._venv_cache[key]

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
