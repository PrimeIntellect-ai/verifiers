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
from typing import Self

from verifiers.v1.configs.runtime import NetworkPolicyConfig
from verifiers.v1.errors import SandboxError
from verifiers.v1.runtimes.base import (
    _ENSURE_UV,
    _INSTALL_CURL,
    ProgramResult,
    RuntimeBackend,
    RuntimeProcess,
)


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

    def process_env(self, env: dict[str, str]) -> dict[str, str]:
        home = {} if self.home is None else {"HOME": self.home}
        return {**self.env, **home, **env}

    @property
    def config(self):
        return self.backend.config

    @property
    def info(self):
        return self.backend.info

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

    async def write(self, path: str, data: bytes) -> None:
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
        directory = (
            self.backend.scripts_dir
            if self.user is None
            else f"{self.backend.scripts_dir}-{self.user}"
        )
        path = f"{directory}/{digest}.py"
        if path not in self.backend.cache.uv_interpreters:
            async with self.backend.cache.uv_locks.setdefault(path, asyncio.Lock()):
                if self.user is not None and not self.backend.cache.curl_ready:
                    # Only root can install a downloader for the user's uv bootstrap.
                    curl = await self.with_user(None).run(
                        ["sh", "-c", _INSTALL_CURL], {}
                    )
                    self.backend.cache.curl_ready = curl.exit_code == 0
                if path not in self.backend.cache.uv_interpreters:
                    tmp = f"{path}.{uuid.uuid4().hex}.tmp"
                    await self.write(tmp, data)
                    command = (
                        f"mv -f {shlex.quote(tmp)} {shlex.quote(path)} "
                        f"&& {{ {_ENSURE_UV}; }} "
                        f"&& uv sync --script {shlex.quote(path)} -q --no-config "
                        f"&& uv python find --script {shlex.quote(path)} --no-config"
                    )
                    # Bootstrap dependencies before task networking is restricted;
                    # keep task offline settings on the returned program's processes.
                    result = await self.run(
                        ["sh", "-c", command], {**(env or {}), "UV_OFFLINE": "false"}
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
