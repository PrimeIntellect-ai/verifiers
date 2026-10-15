"""Task-declared tool bundles: mount, seed, start and dump a bundle's tool server.

A task declares bundles in `task.toml`:

    [[tools]]
    bundle = "crm-suite"                  # tools/crm-suite/ in the task dir
    seed = "tools/seed/crm-suite.json"    # default: tools/seed/<bundle>.json
    placement = "box"                     # or "sidecar"

A bundle dir holds `bundle.toml` (name, version, owner, interface, kind) and a
`tools.py` written against the stdlib runtime `vf_tools.py`, which ships into the box.
The server speaks MCP over streamable HTTP; the task reaches it as a Harbor MCP
server. After the agent stops, a collect hook dumps the bundle's state to
`/logs/tools/<bundle>.json`, a declared artifact that travels to the grader.

- `box` runs the server as root in the solver's box; its code and state dirs are 0700
  root, so a non-root agent changes state only through the tools.
- `sidecar` runs it in a Compose service of its own (`ComposeSpec.services`), which
  the agent reaches only over the network, whatever user it runs as. A task without a
  Compose file becomes a Compose deployment for it: on Prime, a VM per rollout.

Either way the solver's box gets the `vf-tools` CLI and the endpoint map, so scripts
there (an oracle) call the tools as the agent does.
"""

from __future__ import annotations

import hashlib
import io
import json
import shlex
import tarfile
import tomllib
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

if TYPE_CHECKING:
    from verifiers.v1.runtimes import Runtime

RUNTIME_SOURCE = Path(__file__).with_name("vf_tools.py")
RUNTIME_DIR = "/usr/local/lib/vf-tools"
"""World-readable: the runtime and the endpoint map, so `vf-tools call` works for
any user in the box."""
RUNTIME = f"{RUNTIME_DIR}/vf_tools.py"
ENDPOINTS = f"{RUNTIME_DIR}/endpoints.json"
CLI = "/usr/local/bin/vf-tools"
BUNDLES_DIR = "/opt/vf-tools"
"""0700 root: bundle code."""
STATE_DIR = "/var/lib/vf-tools"
"""0700 root: bundle state."""
DUMP_DIR = "/logs/tools"
SIDECAR_IMAGE = "python:3.12-slim"
IDLE = ["sh", "-c", "trap 'exit 0' TERM; while :; do sleep 3600 & wait $!; done"]
BASE_PORT = 18400
NAME_PATTERN = r"^[a-z][a-z0-9-]{0,39}$"


class BundleManifest(BaseModel):
    """A bundle's `bundle.toml`."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(pattern=NAME_PATTERN)
    version: int = Field(ge=1)
    owner: Literal["registry", "task"] = "registry"
    """`registry`: a versioned shared environment. `task`: a one-off owned by one task."""
    kind: Literal["state", "environment"] = "state"
    """`state`: pure tools over one JSON state document, served as root. `environment`:
    tools over the task's own services, served as an unprivileged user of their own and
    confined to `endpoints`; the task's services hold the state and `dump` observes it."""
    interface: Literal["direct", "search_execute"] = "direct"
    description: str = ""
    endpoints: list[str] = Field(default_factory=list)
    """An environment bundle's reach: `host:port` sockets and absolute path prefixes it
    may write under (declare /tmp for scratch space)."""
    commands: list[str] = Field(default_factory=list)
    """Executables an environment bundle may run."""
    dump: str | None = None
    """An environment bundle's observation: a shell command whose stdout is its dump,
    run as the bundle's user after the agent stops."""

    @model_validator(mode="after")
    def _kind_fields(self) -> BundleManifest:
        if self.kind == "state" and (self.endpoints or self.commands or self.dump):
            raise ValueError(
                "endpoints, commands and dump belong to environment bundles"
            )
        if self.kind == "environment" and not self.dump:
            raise ValueError("an environment bundle declares its dump command")
        for entry in self.endpoints:
            if entry.startswith("/"):
                continue
            host, _, port = entry.rpartition(":")
            if not host or not port.isdigit():
                raise ValueError(f"endpoint {entry!r} is neither host:port nor a path")
        return self

    @classmethod
    def read(cls, directory: Path) -> BundleManifest:
        path = Path(directory) / "bundle.toml"
        if not path.is_file():
            raise ValueError(f"{directory}: no bundle.toml")
        if not (Path(directory) / "tools.py").is_file():
            raise ValueError(f"{directory}: no tools.py")
        return cls.model_validate(tomllib.loads(path.read_text()))

    @property
    def ref(self) -> str:
        return f"{self.name}@{self.version}"


class ToolMount(BaseModel):
    """One `[[tools]]` entry, resolved against the task dir."""

    model_config = ConfigDict(frozen=True)

    bundle: str = Field(pattern=NAME_PATTERN)
    directory: str
    """Host path to the bundle dir."""
    seed: str | None = None
    """Host path to the seed JSON; None starts from `{}`."""
    placement: Literal["box", "sidecar"] = "box"
    port: int
    interface: Literal["direct", "search_execute"] = "direct"
    version: int = 1
    owner: Literal["registry", "task"] = "registry"
    kind: Literal["state", "environment"] = "state"
    endpoints: list[str] = Field(default_factory=list)
    commands: list[str] = Field(default_factory=list)
    dump: str | None = None

    @property
    def user(self) -> str | None:
        """The unprivileged user an environment bundle's server runs as."""
        return f"vf-tools-{self.bundle}"[:32] if self.kind == "environment" else None

    @property
    def service(self) -> str:
        """The Compose service the server runs in."""
        return "main" if self.placement == "box" else self.bundle

    @property
    def url(self) -> str:
        """Where the solver's box reaches the server."""
        host = "127.0.0.1" if self.placement == "box" else self.bundle
        return f"http://{host}:{self.port}/mcp"

    @property
    def local_url(self) -> str:
        """Where the server's own box reaches it."""
        return f"http://127.0.0.1:{self.port}/mcp"

    @property
    def state_file(self) -> str:
        return f"{STATE_DIR}/{self.bundle}/state.json"

    @property
    def dump_file(self) -> str:
        suffix = "json" if self.kind == "state" else "out"
        return f"{DUMP_DIR}/{self.bundle}.{suffix}"

    def mcp_server(self) -> dict:
        """The Harbor `mcp_servers` entry the harness connects through."""
        return {"name": self.bundle, "transport": "streamable-http", "url": self.url}

    def dump_command(self) -> str:
        if self.kind == "environment":
            assert self.dump is not None and self.user is not None
            return (
                f"python3 {RUNTIME} observe --user {shlex.quote(self.user)} "
                f"--out {shlex.quote(self.dump_file)} {shlex.quote(self.dump)}"
            )
        return (
            f"python3 {RUNTIME} dump --state {shlex.quote(self.state_file)} "
            f"--out {shlex.quote(self.dump_file)}"
        )

    def sidecar_service(self) -> dict:
        """The Compose service a sidecar-placed server runs in: an idle Python box that
        setup fills, seeds and starts as it does the solver's."""
        return {"image": SIDECAR_IMAGE, "command": IDLE}

    def serve_command(self) -> list[str]:
        confined = (
            [
                "--run-as",
                self.user,
                "--endpoints",
                json.dumps(self.endpoints),
                "--commands",
                json.dumps(self.commands),
            ]
            if self.user is not None
            else ["--state", self.state_file]
        )
        return [
            "python3",
            RUNTIME,
            "serve",
            "--bundle",
            f"{BUNDLES_DIR}/{self.bundle}",
            "--name",
            self.bundle,
            *confined,
            "--host",
            "127.0.0.1" if self.placement == "box" else "0.0.0.0",
            "--port",
            str(self.port),
            "--interface",
            self.interface,
        ]


def parse_mounts(task_dir: Path, entries: object) -> list[ToolMount]:
    """The task's `[[tools]]` entries, each checked against its bundle dir."""
    if entries is None:
        return []
    if not isinstance(entries, list) or not all(isinstance(e, dict) for e in entries):
        raise TypeError(f"{task_dir.name}: [[tools]] must be an array of tables")
    task_dir = Path(task_dir)
    mounts: list[ToolMount] = []
    for index, entry in enumerate(entries):
        unknown = set(entry) - {"bundle", "seed", "placement"}
        if unknown:
            raise ValueError(
                f"{task_dir.name}: [[tools]] unknown keys {sorted(unknown)}"
            )
        name = entry.get("bundle")
        if not isinstance(name, str):
            raise TypeError(f"{task_dir.name}: [[tools]] entry {index} names no bundle")
        directory = _inside(task_dir, f"tools/{name}")
        manifest = BundleManifest.read(directory)
        if manifest.name != name:
            raise ValueError(
                f"{task_dir.name}: tools/{name}/bundle.toml names {manifest.name!r}"
            )
        seed_rel = entry.get("seed", f"tools/seed/{name}.json")
        seed = _inside(task_dir, seed_rel)
        if "seed" in entry and not seed.is_file():
            raise ValueError(f"{task_dir.name}: seed {seed_rel} does not exist")
        if manifest.kind == "environment" and seed.is_file():
            raise ValueError(
                f"{task_dir.name}: environment bundle {name} takes no seed: the task's "
                "own services hold its state"
            )
        mounts.append(
            ToolMount(
                bundle=name,
                directory=str(directory),
                seed=str(seed) if seed.is_file() else None,
                placement=entry.get("placement", "box"),
                port=BASE_PORT + index,
                interface=manifest.interface,
                version=manifest.version,
                owner=manifest.owner,
                kind=manifest.kind,
                endpoints=manifest.endpoints,
                commands=manifest.commands,
                dump=manifest.dump,
            )
        )
    names = [m.bundle for m in mounts]
    if len(set(names)) != len(names):
        raise ValueError(f"{task_dir.name}: a bundle is mounted twice")
    return mounts


def _inside(task_dir: Path, relative: str) -> Path:
    path = (task_dir / relative).resolve()
    if not path.is_relative_to(task_dir.resolve()):
        raise ValueError(f"{task_dir.name}: {relative} leaves the task dir")
    return path


def bundle_digest(directory: Path) -> str:
    """Content digest of a bundle dir: relative paths and bytes, caches excluded."""
    digest = hashlib.sha256()
    root = Path(directory)
    for path in sorted(root.rglob("*")):
        if not path.is_file() or "__pycache__" in path.parts or path.suffix == ".pyc":
            continue
        digest.update(path.relative_to(root).as_posix().encode() + b"\0")
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


def _tar(files: dict[str, bytes | Path]) -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
        for name, source in files.items():
            if isinstance(source, Path) and source.is_dir():
                for path in sorted(source.rglob("*")):
                    if path.is_file() and "__pycache__" not in path.parts:
                        tar.add(
                            path,
                            arcname=f"{name}/{path.relative_to(source).as_posix()}",
                        )
                continue
            data = source.read_bytes() if isinstance(source, Path) else source
            info = tarfile.TarInfo(name)
            info.size = len(data)
            info.mode = 0o644
            tar.addfile(info, io.BytesIO(data))
    return buffer.getvalue()


_PYTHON_CHECK = (
    "command -v python3 >/dev/null && "
    "python3 -c 'import sys; sys.exit(sys.version_info < (3, 9))'"
)
_CREATE_USER = (
    'id -u "$1" >/dev/null 2>&1 || useradd -m -s /bin/sh "$1" 2>/dev/null '
    '|| adduser -D -s /bin/sh "$1"'
)
_CREATE_SYSTEM_USER = (
    "id -u {user} >/dev/null 2>&1 || useradd -r -M -s /bin/false {user} 2>/dev/null "
    "|| adduser -S -D -H {user}"
)
_READY = (
    "for i in $(seq 1 300); do python3 {runtime} ping --url {url} 2>/dev/null && exit 0; "
    "sleep 0.1; done; exit 1"
)


async def _sh(runtime: Runtime, script: str, what: str, *args: str) -> None:
    result = await runtime.run(["sh", "-c", script, "sh", *args], {})
    if result.exit_code != 0:
        detail = (result.stderr or result.stdout or "").strip()[-1500:]
        raise RuntimeError(f"{what} (exit {result.exit_code}): {detail}")


async def mount(
    runtime: Runtime,
    mounts: list[ToolMount],
    *,
    endpoints: list[ToolMount] | None = None,
    agent_user: str | None = None,
) -> None:
    """Install the runtime, the `vf-tools` CLI and the endpoint map of ``endpoints``
    (default: ``mounts``), then the bundles of ``mounts``: seed their state and start
    their servers. Runs as the box's default user (root), before the agent starts.
    The solver's box gets every task's endpoint and its box-placed bundles; a sidecar
    gets its own bundle."""
    endpoints = mounts if endpoints is None else endpoints
    await _sh(runtime, _PYTHON_CHECK, "tool bundles need python3 >= 3.9 in the image")
    if agent_user is not None:
        await _sh(
            runtime, _CREATE_USER, f"creating agent user {agent_user!r}", agent_user
        )
    files: dict[str, bytes | Path] = {"runtime/vf_tools.py": RUNTIME_SOURCE}
    files["runtime/endpoints.json"] = json.dumps(
        {m.bundle: m.url for m in endpoints}, sort_keys=True
    ).encode()
    for m in mounts:
        files[f"bundles/{m.bundle}"] = Path(m.directory)
        files[f"seeds/{m.bundle}.json"] = Path(m.seed).read_bytes() if m.seed else b"{}"
    archive = "/var/tmp/vf-tools.tgz"
    await runtime.write(archive, _tar(files))
    seeds = " ".join(
        f"python3 {RUNTIME} seed --state {shlex.quote(m.state_file)} "
        f"--seed /var/tmp/vf-tools/seeds/{m.bundle}.json;"
        for m in mounts
        if m.kind == "state"
    )
    # An environment bundle's server runs as a user of its own, which reads its code
    # (group) and nothing else of the tools: the bundles' root is traversable only.
    users = " ".join(
        f"{_CREATE_SYSTEM_USER.format(user=shlex.quote(m.user))}; "
        f"chgrp -R {shlex.quote(m.user)} {BUNDLES_DIR}/{m.bundle}; "
        f"chmod -R g+rX {BUNDLES_DIR}/{m.bundle};"
        + "".join(
            # A declared path is the bundle's to write: its group gets write access.
            f" mkdir -p {shlex.quote(path)}; chgrp -R {shlex.quote(m.user)} "
            f"{shlex.quote(path)}; chmod -R g+rwX {shlex.quote(path)};"
            for path in m.endpoints
            if path.startswith("/")
        )
        for m in mounts
        if m.user is not None
    )
    await _sh(
        runtime,
        "set -e; rm -rf /var/tmp/vf-tools; mkdir -p /var/tmp/vf-tools; "
        f"tar --no-same-owner -xzf {archive} -C /var/tmp/vf-tools; rm -f {archive}; "
        f"mkdir -p {RUNTIME_DIR} {BUNDLES_DIR} {STATE_DIR} {DUMP_DIR}; "
        f"cp /var/tmp/vf-tools/runtime/* {RUNTIME_DIR}/; chmod 755 {RUNTIME_DIR}; "
        f"chmod 644 {RUNTIME_DIR}/*; "
        f"printf '#!/bin/sh\\nexec python3 {RUNTIME} \"$@\"\\n' > {CLI}; chmod 755 {CLI}; "
        + " ".join(
            f"rm -rf {BUNDLES_DIR}/{m.bundle} {STATE_DIR}/{m.bundle}; "
            f"cp -r /var/tmp/vf-tools/bundles/{m.bundle} {BUNDLES_DIR}/{m.bundle}; "
            f"mkdir -p {STATE_DIR}/{m.bundle};"
            for m in mounts
        )
        + f" {seeds} chmod -R go-rwx {BUNDLES_DIR} {STATE_DIR}; chmod 711 {BUNDLES_DIR}; "
        + f"{users} rm -rf /var/tmp/vf-tools",
        "installing tool bundles",
    )
    await start(runtime, mounts)


async def start(runtime: Runtime, mounts: list[ToolMount]) -> None:
    """Start the servers of ``mounts`` in ``runtime`` over their current state and wait
    until each lists its tools. `mount` calls this; a box restored from a checkpoint
    calls it alone, so the run continues from the state on disk."""
    for m in mounts:
        log = f"/var/log/vf-tools-{m.bundle}.log"
        await runtime.run_background(m.serve_command(), {}, log)
        result = await runtime.run(
            ["sh", "-c", _READY.format(runtime=RUNTIME, url=shlex.quote(m.local_url))],
            {},
        )
        if result.exit_code != 0:
            tail = await runtime.run(["tail", "-c", "2000", log], {})
            raise RuntimeError(
                f"tool bundle {m.bundle!r} did not start: {(tail.stdout or '').strip()}"
            )


__all__ = [
    "BundleManifest",
    "ToolMount",
    "bundle_digest",
    "mount",
    "parse_mounts",
    "start",
]
