"""Immutable process and file access bindings for a provisioned runtime."""

from __future__ import annotations

import asyncio
import hashlib
import shlex
import uuid
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from pathlib import PurePosixPath
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Self

from verifiers.v1.configs.runtime import NetworkPolicyConfig
from verifiers.v1.errors import SandboxError
from verifiers.v1.runtimes.base import (
    _ENSURE_UV,
    ProgramResult,
    RuntimeBackend,
    RuntimeProcess,
)

if TYPE_CHECKING:
    from verifiers.v1.runtimes.bridge import ExecutionBridge


@dataclass(frozen=True, eq=False)
class Runtime:
    """Command and file access with immutable user and environment bindings.

    Derived views share one backend, including provider state and lifecycle.
    Creating a view neither provisions a resource nor creates isolation:
    the backend enforces the selected user's permissions.
    """

    backend: RuntimeBackend
    env: Mapping[str, str] = field(default_factory=dict)
    user: str | None = None
    home: str | None = None
    workdir: str | None = None
    _execution_bridge: ExecutionBridge | None = None
    _lease: Any = None

    def __post_init__(self) -> None:
        if self.user is not None and not self.backend.SUPPORTS_USER_SWITCHING:
            raise ValueError(
                f"the {self.backend.type} runtime cannot run as another user"
            )
        if self.user is None and self.home is not None:
            raise ValueError("home requires a user binding")
        object.__setattr__(self, "env", MappingProxyType(dict(self.env)))

    def with_env(self, env: Mapping[str, str]) -> Self:
        return replace(self, env=env)

    def with_user(self, user: str | None, home: str | None = None) -> Self:
        return replace(self, user=user, home=home if user is not None else None)

    def with_workdir(self, workdir: str | None) -> Self:
        return replace(self, workdir=workdir)

    async def wait_lan_ready(self) -> None:
        if self._lease is not None:
            await self._lease.wait_lan_ready()
            return
        if wait_lan := getattr(self.backend, "wait_lan_ready", None):
            await wait_lan()

    def process_env(self, env: dict[str, str]) -> dict[str, str]:
        home = {} if self.home is None else {"HOME": self.home}
        return {**self.env, **home, **env}

    @property
    def config(self):
        return self.backend.config

    @property
    def info(self):
        info = self.backend.info
        if (
            self.workdir is not None
            and hasattr(info, "workdir")
            and info.workdir != self.workdir
        ):
            return info.model_copy(update={"workdir": self.workdir})
        return info

    @property
    def name(self) -> str:
        return self.backend.name

    @property
    def type(self) -> str:
        return self.backend.type

    @property
    def is_local(self) -> bool:
        return self.backend.is_local

    @property
    def network_restricted(self) -> bool:
        return self.backend.network_restricted

    @property
    def published_port(self) -> int | None:
        return self.backend.published_port

    def host_url(self, url: str) -> str:
        return self.backend.host_url(url)

    async def expose(self, port: int) -> str:
        return await self.backend.expose(port)

    async def set_network(self, network: NetworkPolicyConfig) -> None:
        """Enforce `network` as given, in place of the policy the runtime started with,
        on the whole runtime rather than only this view; return once the provider
        enforces it. Framework routes are not added. A runtime that started without a
        policy can only stay open."""
        if not network.network_restricted:
            config = self.backend.config
            control_hosts = getattr(self.backend, "_control_blocked_hosts", frozenset())
            if (
                control_hosts
                and isinstance(config, NetworkPolicyConfig)
                and config.allow == ["*"]
                and set(config.block) <= control_hosts
            ):
                return
            await self.backend.prepare_execution(None)
            return
        config = self.backend.config
        if not self.backend.network_restricted or not isinstance(
            config, NetworkPolicyConfig
        ):
            raise SandboxError(
                f"runtime {self.name!r} started without a network policy, so it cannot "
                "take one"
            )
        policy = type(config).model_validate(
            {**config.model_dump(), "allow": network.allow, "block": network.block}
        )
        await self.backend.prepare_execution([], policy)

    async def run(self, argv: list[str], env: dict[str, str]) -> ProgramResult:
        """Run `argv` to completion and collect its output."""
        return await self.backend.run(self, argv, env)

    async def exec(self, argv: list[str], env: dict[str, str]) -> RuntimeProcess:
        """Start `argv` and return a handle that streams its stdio."""
        return await self.backend.exec(self, argv, env)

    async def run_background(
        self, argv: list[str], env: dict[str, str], log: str
    ) -> None:
        """Start `argv` with its combined output in `log`, a path in the working
        directory, and return. The process runs until the runtime stops."""
        await self.backend.run_background(self, argv, env, log)

    def resolve_path(self, path: str) -> str:
        if path.startswith("/"):
            return path
        if self.type == "subprocess" and self.backend.info.id:
            base = self.backend.info.id
        else:
            base = (
                self.workdir
                or getattr(self.backend.config, "workdir", "/app")
                or "/app"
            )
        return f"{base.rstrip('/')}/{path}"

    async def write(self, path: str, data: bytes) -> None:
        if (
            self.user is None
            and len(data) <= 512 * 1024
            and (daemon := getattr(self.backend, "_daemon", None)) is not None
            and daemon.alive
        ):
            await daemon.write_file(self.resolve_path(path), data)
            return
        await self.backend.write(self, path, data)

    async def alive(self) -> bool:
        """Whether the box still executes anything. Not every runtime raises when
        the box is gone — some surface it as `exec`'s own non-zero result,
        indistinguishable from the command failing. One probe on the failure path
        tells the two apart before we blame anyone."""
        try:
            return (await self.run(["true"], {})).exit_code == 0
        except Exception:  # noqa: BLE001 - failing to exec at all means the box is gone
            return False

    def temp_path(
        self,
        *parts: str,
        env: Mapping[str, str] | None = None,
        default: str | None = None,
    ) -> str:
        """Framework scratch path under the process's explicit TMPDIR, or /tmp.

        Without TMPDIR, an established ``default`` path is retained. Callers choose
        it from task configuration so setup, execution and cleanup agree across views.
        The selected directory still needs to exist and be writable in the view.
        """
        directory = (env or {}).get("TMPDIR") or self.env.get("TMPDIR")
        if not directory and default is not None:
            return default
        if (
            not directory
            and getattr(self, "type", None) == "subprocess"
            and getattr(getattr(self, "backend", None), "info", None) is not None
            and self.backend.info.id
        ):
            directory = self.backend.info.id
        return str(PurePosixPath(directory or "/tmp").joinpath(*parts))

    async def resolve_python(self) -> str:
        """Locate a Python 3 interpreter in this runtime without bootstrapping `uv`."""
        if self.backend.cache.system_python:
            return self.backend.cache.system_python
        async with self.backend.cache.setup_lock:
            if self.backend.cache.system_python:
                return self.backend.cache.system_python
            probe = await self.with_user(None).run(
                [
                    "sh",
                    "-c",
                    (
                        'printf "%s\\n%s\\n" "$(uname -m)" '
                        '"$(command -v python3 || command -v python || true)"'
                    ),
                ],
                {},
            )
            lines = probe.stdout.splitlines()
            if lines and lines[0].strip() and not self.backend.cache.arch:
                self.backend.cache.arch = lines[0].strip()
            python = lines[1].strip() if len(lines) > 1 else ""
            if not python:
                raise SandboxError(
                    f"no Python 3 interpreter available in runtime {self.name!r}"
                )
            self.backend.cache.system_python = python
            return python

    def _script_directory(self) -> str:
        directory = self.backend.scripts_dir
        if self.user is not None:
            directory += f"-{self.user}"
        if self.home is not None:
            directory += "-" + hashlib.sha256(self.home.encode()).hexdigest()[:16]
        return directory

    async def prepare_uv_script(
        self,
        script: str | bytes,
        env: dict[str, str] | None = None,
        *,
        activate: bool = True,
    ) -> list[str]:
        """Prepare a PEP 723 script and return an argv that runs it.

        ``activate=False`` invokes the resolved script interpreter directly, keeping
        its environment variables out of child processes spawned by the script.
        """
        data = script.encode() if isinstance(script, str) else script
        digest = hashlib.sha256(data).hexdigest()
        # Each user needs its own writable script dir and uv environment.
        directory = self._script_directory()
        path = f"{directory}/{digest}.py"
        ready_path = f"{path}.ready"
        if path not in self.backend.cache.uv_interpreters:
            async with (
                self.backend.cache.setup_lock,
                self.backend.cache.uv_locks.setdefault(path, asyncio.Lock()),
            ):
                if path not in self.backend.cache.uv_interpreters:
                    path_q = shlex.quote(path)
                    ready_q = shlex.quote(ready_path)
                    # Check whether a prepared image or checkpoint already synced this script.
                    probe = await self.run(
                        [
                            "sh",
                            "-c",
                            (
                                f"[ -f {path_q} ] && [ -r {ready_q} ] "
                                f'&& py=$(cat {ready_q}) && [ -x "$py" ] '
                                '&& printf "%s\\n" "$py"'
                            ),
                        ],
                        {},
                    )
                    if probe.exit_code == 0 and probe.stdout.strip():
                        self.backend.cache.uv_interpreters[path] = (
                            probe.stdout.strip().splitlines()[-1]
                        )
                    else:
                        tmp = f"{path}.{uuid.uuid4().hex}.tmp"
                        await self.write(tmp, data)
                        share_py = (
                            {
                                "UV_PYTHON_INSTALL_DIR": f"{directory}/python",
                                "UV_CACHE_DIR": f"{directory}/uv-cache",
                            }
                            if self.user is None
                            else {}
                        )
                        chmod_cmd = (
                            f" && chmod -R a+rX {shlex.quote(directory)}"
                            if self.user is None
                            else ""
                        )
                        command = (
                            f"mv -f {shlex.quote(tmp)} {path_q} "
                            f"&& {{ {_ENSURE_UV}; }} "
                            f"&& uv sync --script {path_q} -q --no-config "
                            f"&& python=$(uv python find --script {path_q} --no-config) "
                            f'&& venv=$(dirname "$(dirname "$python")") '
                            '&& chmod -R a+rX "$venv"'
                            f"{chmod_cmd} "
                            f'&& printf "%s" "$python" > {ready_q} '
                            '&& printf "%s\\n" "$python"'
                        )
                        # Bootstrap dependencies before task networking is restricted;
                        # keep task offline settings on the returned program's processes.
                        result = await self.run(
                            ["sh", "-c", command],
                            {**share_py, **(env or {}), "UV_OFFLINE": "false"},
                        )
                        if result.exit_code != 0:
                            raise RuntimeError(
                                "failed to prepare uv script: "
                                f"{result.stderr.strip()[-2000:]}"
                            )
                        self.backend.cache.uv_interpreters[path] = (
                            result.stdout.strip().splitlines()[-1]
                        )
        interpreter = self.backend.cache.uv_interpreters[path]
        if not activate:
            return [interpreter, path]
        venv = str(PurePosixPath(interpreter).parent.parent)
        command = (
            'export VIRTUAL_ENV="$1" PATH="${1}/bin:$HOME/.local/bin:$PATH" '
            'UV_INSTALL_DIR="$HOME/.local/bin" UV_RUN_RECURSION_DEPTH=1; '
            'shift; exec "$@"'
        )
        return [
            "sh",
            "-c",
            command,
            "uv-script",
            venv,
            interpreter,
            path,
        ]

    async def run_uv_script(
        self,
        script: str | bytes,
        args: list[str] | None = None,
        env: dict[str, str] | None = None,
    ) -> ProgramResult:
        """The script is written to a stable, content-addressed path rather than the per-rollout
        workspace: uv keys its per-script environment by the script's full path, so a
        unique path per call would mint a fresh env every rollout. A path derived from the
        content means identical scripts share one path → uv reuses one env, bounded by the
        number of distinct scripts. Published via a unique temp + atomic `mv`, so
        concurrent rollouts writing the same content never race a half-written read."""
        argv = await self.prepare_uv_script(script, env)
        return await self.run([*argv, *(args or [])], env or {})

    async def read(self, path: str, max_bytes: int | None = None) -> bytes:
        """Read `path` into host memory. `max_bytes` caps the transfer, raising past
        the cap — for a file written by something we don't control, whose size we
        can't assume. The cap is enforced inside the box rather than after the
        transfer. Backends implement `RuntimeBackend.read`."""
        if max_bytes is None:
            return await self.backend.read(self, path)
        if max_bytes < 0:
            raise ValueError("max_bytes must be non-negative")
        # One extra byte distinguishes an exact fit from a truncated file.
        data = await self.backend.read(self, path, max_bytes=max_bytes + 1)
        if len(data) > max_bytes:
            raise SandboxError(f"read {path!r}: over the {max_bytes} byte limit")
        return data
