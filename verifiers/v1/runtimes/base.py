"""The runtime contract: provision execution, run the program, tear down.

A runtime decides WHERE the program runs and HOW it reaches the host interception
server. Concrete runtimes live alongside this base; harnesses and the Environment
depend only on this contract, so they stay runtime-agnostic.
"""

import hashlib
import shlex
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass

# Ensure `uv` is available to run our PEP 723 scripts (the harness + tool servers): use it
# if present, else bootstrap it — via pip; else via the standalone installer (curl/wget),
# first installing curl + CA certs from the distro package manager when the image has no
# downloader at all (bare task images, e.g. Harbor's). It installs to ~/.local/bin, which
# we prepend to PATH so the next `uv run` finds it; uv then resolves each script's inline
# deps into its own cache, isolated from the eval process. (Needs network + one of
# uv / pip / curl / wget / apt-get / apk.)
_INSTALL_CURL = (  # only when the image has no downloader; needs a known package manager
    "{ command -v curl >/dev/null 2>&1 || command -v wget >/dev/null 2>&1; } "
    "|| { apt-get update -qq && apt-get install -y -qq curl ca-certificates; } "
    "|| apk add --no-cache curl ca-certificates"
)
_DOWNLOAD_UV = (
    "{ command -v curl >/dev/null 2>&1 && curl -LsSf https://astral.sh/uv/install.sh | sh; } "
    "|| { command -v wget >/dev/null 2>&1 && wget -qO- https://astral.sh/uv/install.sh | sh; }"
)
_ENSURE_UV = (
    'export PATH="$HOME/.local/bin:$PATH" UV_INSTALL_DIR="$HOME/.local/bin"; '
    "command -v uv >/dev/null 2>&1 "
    "|| pip install -q uv 2>/dev/null "
    f"|| {{ {_INSTALL_CURL}; {_DOWNLOAD_UV}; }}"
)


@dataclass(frozen=True)
class ProgramResult:
    exit_code: int
    stdout: str
    stderr: str


class Runtime(ABC):
    @abstractmethod
    async def start(self) -> None:
        """Provision execution (workspace / container / sandbox). Use `expose` to turn a
        host port into a URL the program can reach."""

    async def expose(self, port: int) -> str:
        """A base URL the program (inside this runtime) can use to reach a host service
        on localhost `port` — the interception endpoint and host-side tool servers both
        go through this. Default: localhost, which works when the runtime shares the host
        network (subprocess, docker --network host). Remote runtimes (prime) override to
        tunnel the port."""
        return f"http://127.0.0.1:{port}"

    async def stop(self) -> None:
        """Tear down any provisioned resources. Default no-op."""

    @abstractmethod
    async def run(self, argv: list[str], env: dict[str, str]) -> ProgramResult:
        """Run `argv` (with the interception env vars `env`) to completion."""

    async def run_background(
        self, argv: list[str], env: dict[str, str], log: str
    ) -> None:
        """Start `argv` as a background process in the runtime (combined output to
        `log`, a path in the workspace) and return immediately. It runs until `stop()`
        tears the runtime down. Used to host a tool server colocated with the harness."""
        raise NotImplementedError(
            f"{type(self).__name__} does not support run_background"
        )

    @property
    def descriptor(self) -> str | None:
        """A short resolved id for display (None until provisioned). Overridden per
        runtime: subprocess workdir, docker image, prime sandbox id."""
        return None

    async def public_url(self, port: int) -> str | None:
        """A URL anyone can use to reach `port` running *inside this runtime*, or None if
        this runtime can't self-publish (it's on the host network, so the caller reaches
        it via the harness runtime's `expose`). A remote runtime overrides this to publish
        the port (e.g. a prime sandbox exposes it natively). Cleaned up by `stop()`."""
        return None

    @abstractmethod
    async def read(self, path: str) -> bytes:
        """Read a file from the runtime's workspace. The caller need not know
        whether that's the host fs or across a container/sandbox boundary."""

    @abstractmethod
    async def write(self, path: str, data: bytes) -> None:
        """Write a file into the runtime's workspace, creating parent dirs."""

    async def run_uv_script(
        self,
        script: str | bytes,
        args: list[str] | None = None,
        env: dict[str, str] | None = None,
    ) -> ProgramResult:
        """Run a self-contained uv script (PEP 723 inline deps) in this runtime, with
        `args` as its positional arguments (the script's `sys.argv[1:]`).

        Writes `script`, ensures `uv` is present, and runs `uv run` — so the script's
        dependencies resolve into uv's cache inside the runtime, never the eval process.
        Built on `write`/`run`, so it works the same on every runtime. `args` are
        forwarded via the shell's `"$@"` (never interpolated), so spaces / quotes /
        newlines in them are safe; pass structured data as a JSON string if you need to.

        The script is written to a stable, content-addressed path (NOT the per-rollout
        workspace): uv keys its per-script environment by the script's full path, so a
        unique path per call would mint a fresh env every rollout. A path derived from the
        content means identical scripts share one path → uv reuses one env, bounded by the
        number of distinct scripts. Published via a unique temp + atomic `mv`, so
        concurrent rollouts writing the same content never race a half-written read."""
        data = script.encode() if isinstance(script, str) else script
        path = f"/tmp/v1-scripts/{hashlib.sha256(data).hexdigest()}.py"
        tmp = f"{path}.{uuid.uuid4().hex}.tmp"
        await self.write(tmp, data)
        await self.run(
            ["sh", "-c", f"mv -f {shlex.quote(tmp)} {shlex.quote(path)}"], {}
        )
        command = f'{_ENSURE_UV}; exec uv run {shlex.quote(path)} "$@"'
        return await self.run(["sh", "-c", command, path, *(args or [])], env or {})
