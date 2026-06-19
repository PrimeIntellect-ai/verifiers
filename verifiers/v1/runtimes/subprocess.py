"""Local subprocess runtime: run the program on the host; server on localhost.

Each rollout gets a fresh `/tmp/<name>` workspace (created on `start`, removed on
`stop`/`cleanup`) used as the program's cwd, so concurrent local rollouts are isolated
and trivially cleaned up. Relative `read`/`write` paths resolve against it.
"""

import asyncio
import contextlib
import hashlib
import json
import os
import shlex
import shutil
import signal
import tempfile
from pathlib import Path
from typing import Literal

from verifiers.v1.runtimes.base import (
    _ENSURE_UV,
    BaseRuntimeConfig,
    ProgramResult,
    Runtime,
)

# A local subprocess inherits the host environment EXCEPT any var whose name
# contains "API_KEY" — so it can never reach a real provider with the host's API
# key, while still inheriting harmless config (PATH, HOME, UV_CACHE_DIR, HF_HOME,
# ...). Harnesses hand their interception endpoint + per-rollout secret to their own
# client via argv/CLI (not OPENAI_*), so nothing they spawn inherits the endpoint.
# A container/sandbox is isolated and inherits nothing, so this
# allow-by-default model is subprocess-only. NOTE: this strip applies to EVERY
# program run here, including a task's tool/user server — so a tool server that
# genuinely needs an API key won't get one on subprocess placement; give it its
# own runtime (docker/prime) or have it fetch the key itself.


class SubprocessConfig(BaseRuntimeConfig):
    type: Literal["subprocess"] = "subprocess"


# Wrapper that turns a `main(argv)`-exposing uv script into a long-lived warm worker: it loads
# the script as a module ONCE (its top-level deps — e.g. math_verify — imported here, once), then
# answers one request per stdin line (JSON args list) with one JSON line (`main`'s return value).
_WARM_WORKER = """\
import importlib.util, json, sys
spec = importlib.util.spec_from_file_location("warm_user_script", sys.argv[1])
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
sys.stdout.write("READY\\n"); sys.stdout.flush()
for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    try:
        out = mod.main(json.loads(line))
    except Exception as e:
        out = "__ERR__ " + repr(e)
    sys.stdout.write(json.dumps(out) + "\\n"); sys.stdout.flush()
"""


class _WarmWorker:
    """A long-lived `_WARM_WORKER` process for one script. Its stdio is a serial request channel,
    so `call` is guarded by a lock (one in-flight request at a time)."""

    def __init__(self, proc: asyncio.subprocess.Process, log: Path) -> None:
        self.proc = proc
        self.log = log
        self.lock = asyncio.Lock()

    async def call(self, args: list[str]) -> ProgramResult:
        async with self.lock:
            assert self.proc.stdin is not None and self.proc.stdout is not None
            self.proc.stdin.write((json.dumps(args) + "\n").encode())
            await self.proc.stdin.drain()
            line = await self.proc.stdout.readline()
            if not line:  # worker died mid-request
                detail = self.log.read_text(errors="replace").strip()[-500:]
                raise RuntimeError(f"warm worker died: {detail or '<no output>'}")
            out = json.loads(line.decode())
            if isinstance(out, str) and out.startswith("__ERR__"):
                return ProgramResult(
                    exit_code=1, stdout="", stderr=out[len("__ERR__ ") :]
                )
            return ProgramResult(exit_code=0, stdout=str(out), stderr="")


class SubprocessRuntime(Runtime):
    """Runs the program as a local subprocess in a unique /tmp workspace."""

    # script-sha -> the interpreter for that script's deps, resolved once and shared across
    # the worker's per-rollout runtimes (+ a per-sha lock so first-callers provision once).
    _interpreters: dict[str, str] = {}
    _locks: dict[str, asyncio.Lock] = {}

    def __init__(self, config: SubprocessConfig, name: str | None = None) -> None:
        super().__init__(name)
        self.config = config
        self.workdir: Path | None = None
        self._background: list[asyncio.subprocess.Process] = []
        # script-sha -> warm worker (one per distinct script); persists across this runtime's
        # rollouts, so a persistent runtime imports each warm script's deps once. Separate from
        # `_background`/the workspace: untouched by `reset`, reaped in `cleanup`.
        self._warm: dict[str, _WarmWorker] = {}
        self._warm_locks: dict[str, asyncio.Lock] = {}

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

    async def _resolve_interpreter(self, data: bytes) -> tuple[str, Path]:
        """Resolve (and cache) the uv env interpreter for a script's deps, returning
        `(interpreter, script_path)`. uv keeps a persistent env per script under its cache; we
        resolve that env's interpreter ONCE (per sha, shared across this worker's runtimes) and
        exec it directly, keeping uv's per-invocation resolve, cache-lock, ephemeral-env
        materialization and `command -v uv` probe out of the per-rollout hot path."""
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
        return self._interpreters[sha], path

    async def run_uv_script(
        self,
        script: str | bytes,
        args: list[str] | None = None,
        env: dict[str, str] | None = None,
        warm: bool = False,
    ) -> ProgramResult:
        """Run a PEP 723 script via a pre-provisioned interpreter instead of `uv run` per call
        (the subprocess runtime launches two uv-scripts per rollout — the harness program and,
        for math, verify). docker/prime keep the base `uv run` — their deps live in the sandbox.

        `warm=True` on a `persistent` runtime routes to a long-lived worker that imports the
        script's deps once (see `_run_warm`); otherwise the script is exec'd fresh each call (the
        warm worker would die with the rollout that spawned it, so it only pays off when reused)."""
        data = script.encode() if isinstance(script, str) else script
        interpreter, path = await self._resolve_interpreter(data)
        if warm and self.config.persistent:
            return await self._run_warm(data, interpreter, path, args or [])
        return await self.run([interpreter, str(path), *(args or [])], env or {})

    async def _run_warm(
        self, data: bytes, interpreter: str, path: Path, args: list[str]
    ) -> ProgramResult:
        """Route a call to this runtime's warm worker for `path`, spawning it on first use. The
        worker outlives the rollout (reaped in `cleanup`), so a persistent runtime imports the
        script's deps once across all its rollouts."""
        sha = hashlib.sha256(data).hexdigest()
        worker = self._warm.get(sha)
        if worker is None:
            async with self._warm_locks.setdefault(sha, asyncio.Lock()):
                worker = self._warm.get(sha)
                if worker is None:
                    worker = await self._spawn_warm(interpreter, path, sha)
                    self._warm[sha] = worker
        return await worker.call(args)

    async def _spawn_warm(self, interpreter: str, path: Path, sha: str) -> _WarmWorker:
        wrapper = Path("/tmp/vf-scripts") / "_warm_worker.py"
        if not wrapper.exists():
            wrapper.write_text(_WARM_WORKER)
        log = Path("/tmp/vf-scripts") / f"{sha}.warm.log"
        # cwd is /tmp (not the resettable per-rollout workspace), so `reset` never pulls it out
        # from under a running worker; stderr -> a log so a crash is diagnosable on death.
        with log.open("wb") as f:
            proc = await asyncio.create_subprocess_exec(
                interpreter, str(wrapper), str(path),
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=f,
                cwd=tempfile.gettempdir(),
                start_new_session=True,  # own group, so cleanup reaps the whole tree
            )  # fmt: skip
        assert proc.stdout is not None
        ready = await proc.stdout.readline()
        if ready.strip() != b"READY":
            detail = log.read_text(errors="replace").strip()[-500:]
            raise RuntimeError(
                f"warm worker failed to start: {detail or '<no output>'}"
            )
        return _WarmWorker(proc, log)

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

    async def reset(self) -> None:
        """Recreate the per-rollout workspace for the next rollout. Warm workers (cwd /tmp, in
        `_warm`) and the interpreter cache are untouched, so they carry over to the next rollout."""
        if self.workdir is not None:
            await asyncio.to_thread(shutil.rmtree, self.workdir, ignore_errors=True)
            await asyncio.to_thread(self.workdir.mkdir)

    def cleanup(self) -> None:
        procs = self._background + [w.proc for w in self._warm.values()]
        for proc in procs:
            # Kill the whole group (start_new_session => pgid == pid), not just proc.pid, so a
            # background server's children (sh -> uv -> python) are reaped too.
            with contextlib.suppress(ProcessLookupError, PermissionError):
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        self._background = []
        self._warm = {}
        if self.workdir is not None:
            shutil.rmtree(self.workdir, ignore_errors=True)
