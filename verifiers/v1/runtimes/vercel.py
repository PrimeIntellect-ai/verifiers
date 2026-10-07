"""Remote Vercel Sandbox runtime.

Sandboxes boot only from Vercel Container Registry (VCR) images. A Docker image ref is
mirrored once into the project's `verifiers` VCR repository by a builder sandbox running
`crane copy`; the registry credential is injected by the sandbox firewall, so it never
enters the VM. Refs under `vercel/sandbox/` (managed images) or `vcr.vercel.com/` boot
as is.

The SDK's process log streams are text-decoded (lossy for bytes) and can replay output,
so durable programs and live processes are file-backed: output is redirected to files
read back losslessly, and live stdin is a FIFO written through short commands.
"""

import asyncio
import base64
import contextlib
import fcntl
import hashlib
import ipaddress
import logging
import math
import os
import re
import shlex
import time
import uuid
import weakref
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import PurePosixPath
from typing import ClassVar, Literal
from urllib.parse import urlsplit

import httpx
from pydantic import Field, model_validator

from verifiers.v1.configs.runtime import NetworkPolicyConfig
from verifiers.v1.errors import SandboxError
from verifiers.v1.runtimes.base import (
    SERVICE_PORT,
    BaseRuntimeInfo,
    ProgramResult,
    Runtime,
    RuntimeProcess,
)
from verifiers.v1.runtimes.limiters import creation_limiter
from verifiers.v1.utils.aio import run_shielded
from verifiers.v1.utils.paths import CACHE_DIR
from verifiers.v1.utils.scope import run_scope

logger = logging.getLogger(__name__)

_VCR_HOST = "vcr.vercel.com"
_VCR_REPOSITORY = "verifiers"
_PROJECT_NAME = "verifiers-sandbox"
_BUILDER_IMAGE = "vercel/sandbox/python:3.14"
_CRANE_URL = (
    "https://github.com/google/go-containerregistry/releases/download/"
    "v0.20.6/go-containerregistry_Linux_x86_64.tar.gz"
)
_MIRROR_SCRIPT = (
    "set -eu; cd /tmp; "
    'python3 -c "import sys, urllib.request; urllib.request.urlretrieve(*sys.argv[1:])" '
    '"$1" crane.tgz; tar -xzf crane.tgz crane; ./crane copy --platform linux/amd64 "$2" "$3"'
)
_MANIFEST_TYPES = (
    "application/vnd.oci.image.index.v1+json, "
    "application/vnd.oci.image.manifest.v1+json, "
    "application/vnd.docker.distribution.manifest.list.v2+json, "
    "application/vnd.docker.distribution.manifest.v2+json"
)
# Idempotent: a replay after the move only re-applies ownership.
_MOVE_STAGED = (
    '{ [ ! -e "$1" ] || { mkdir -p "$(dirname "$2")" && mv -f "$1" "$2"; }; } '
    '&& chown root:root "$2"'
)
_MIRROR_LOCK_DIR = CACHE_DIR / "vercel-image-locks"
_MIRRORED: set[str] = set()
_REGISTRY_TOKENS: dict[str, str] = {}

_LIFETIME = 24 * 60 * 60
_OIDC_MARGIN = 10 * 60
_START_TIMEOUT = 30
_IMAGE_READY_TIMEOUT = 600
_DESTROY_TIMEOUT = 60
_MEMORY_MB_PER_VCPU = 2048
_MAX_VCPUS = 32
_WAIT_RECONNECTS = 4
_EXEC_ATTEMPTS = 3
_STDIN_CHUNK = 48 * 1024
_POLL_MIN, _POLL_MAX = 0.05, 1.0
_UNBOUNDED = 1 << 62

_HOSTNAME = re.compile(
    r"(?:\*\.)?(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)*"
    r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?",
    re.IGNORECASE,
)


def _address(rule: str) -> ipaddress.IPv4Network | ipaddress.IPv6Network | None:
    try:
        return ipaddress.ip_network(rule, strict=False)
    except ValueError:
        return None


def _validate_egress_rules(allow: list[str], block: list[str]) -> None:
    for rule in block:
        if _address(rule) is None:
            raise ValueError(
                f"Vercel block rules must be IP addresses or CIDR blocks, got {rule!r}"
            )
    for rule in allow:
        if _address(rule) is None and not _HOSTNAME.fullmatch(rule):
            raise ValueError(
                "Vercel allow rules must be hostnames, wildcard hostnames, IP addresses, "
                f"or CIDR blocks (no schemes, ports, or paths), got {rule!r}"
            )


def _egress_policy(config: "VercelSandboxConfig", routes: list[str] | None):
    """The complete firewall policy; `routes` (interception and MCP endpoints) stay
    reachable alongside the allowlist. None restores unrestricted egress."""
    from vercel.sandbox import NetworkPolicy, NetworkPolicyRule, NetworkPolicySubnets

    if routes is None:
        return NetworkPolicy.allow_all()
    if config.allow == ["*"]:
        return NetworkPolicy.custom(
            allow={"*": (NetworkPolicyRule(),)},
            subnets=NetworkPolicySubnets(deny=config.block),
        )
    domains: list[str] = []
    subnets: list[str] = []
    for route in routes:
        url = urlsplit(route)
        host = url.hostname or ""
        if host == "localhost" or (
            (ip := _address(host)) and ip.network_address.is_loopback
        ):
            continue
        if ip is not None:
            subnets.append(str(ip))
        elif url.scheme == "https":
            domains.append(host)
        else:
            raise SandboxError(
                f"Vercel filters plaintext HTTP by IP only; framework route {route!r} "
                "needs HTTPS or an IP address"
            )
    for rule in config.allow:
        (subnets if _address(rule) else domains).append(rule)
    if not domains and not subnets:
        return NetworkPolicy.deny_all()
    return NetworkPolicy.custom(
        allow={domain: (NetworkPolicyRule(),) for domain in dict.fromkeys(domains)},
        subnets=NetworkPolicySubnets(allow=list(dict.fromkeys(subnets)))
        if subnets
        else None,
    )


def _vcpus(cpu: float, memory: float) -> int:
    """Vercel allocates 1 or an even number of vCPUs with 2 GB of memory each."""
    vcpus = max(1, math.ceil(cpu), math.ceil(memory * 1024 / _MEMORY_MB_PER_VCPU))
    if vcpus > 1 and vcpus % 2:
        vcpus += 1
    if vcpus > _MAX_VCPUS:
        raise ValueError(
            f"Vercel sandboxes support at most {_MAX_VCPUS} vCPUs, got cpu={cpu:g} "
            f"memory={memory:g} GB"
        )
    return vcpus


class VercelSandboxConfig(NetworkPolicyConfig):
    """Authenticates from a `vercel link`ed directory, the .env.local written by
    `vercel env pull`, or VERCEL_TOKEN (team and project resolved automatically)."""

    type: Literal["vercel"] = "vercel"
    image: str = "python:3.11-slim"
    """Docker image ref, mirrored once into the project's VCR; refs under
    `vercel/sandbox/` or `vcr.vercel.com/` boot as is."""
    workdir: str | None = None
    """Working directory override; None uses the task's workdir, or /app."""
    # TaskData.resources uses these units; non-default runtime config values take precedence.
    cpu: float = Field(default=1.0, gt=0)
    """CPU cores, rounded up to 1 or an even number of vCPUs."""
    memory: float = Field(default=2.0, gt=0)
    """Memory in GB. Vercel allocates 2 GB per vCPU, so this can raise the vCPU count."""
    disk: float = 5.0
    """Advisory disk request in GB. Vercel sandboxes have fixed storage."""
    creates_per_sec: float | None = 10.0
    """Pace sandbox creation to this many per second, enforced run-wide across every
    env-server worker process (None/<= 0 disables it)."""

    @model_validator(mode="after")
    def _validate_egress(self) -> "VercelSandboxConfig":
        if self.network_restricted:
            _validate_egress_rules(
                [rule for rule in self.allow if rule != "*"],
                [rule for rule in self.block if rule != "*"],
            )
        return self


class VercelSandboxRuntimeInfo(VercelSandboxConfig, BaseRuntimeInfo):
    boot_image: str | None = None


_AUTH_HELP = (
    "Vercel Sandbox requires authentication: run `vercel link` here (after `vercel "
    "login`), or `vercel env pull` (verifiers loads the resulting .env.local), or set "
    "VERCEL_TOKEN (the team and project are resolved automatically)."
)
_auth_locks: "weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, asyncio.Lock]" = (
    weakref.WeakKeyDictionary()
)


def _oidc_claims(token: str) -> dict:
    from vercel.oidc import decode_oidc_payload

    try:
        return decode_oidc_payload(token)
    except Exception:  # noqa: BLE001 - not an OIDC token (e.g. an API token)
        return {}


def _oidc_expired(token: str) -> bool:
    exp = _oidc_claims(token).get("exp")
    return exp is not None and exp <= time.time() + _OIDC_MARGIN


async def _refresh_auth() -> None:
    """Keep the ambient OIDC token valid for long rollouts: the SDK re-reads it from
    the environment on every request but never refreshes it."""
    if (oidc := os.environ.get("VERCEL_OIDC_TOKEN")) and _oidc_expired(oidc):
        await _ensure_auth()


async def _team_id(token: str) -> str:
    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.get(
            "https://api.vercel.com/v2/teams",
            params={"limit": 100},
            headers={"Authorization": f"Bearer {token}"},
        )
    response.raise_for_status()
    teams = response.json().get("teams") or []
    if len(teams) == 1:
        return teams[0]["id"]
    choices = "".join(f"\n  {team['slug']} ({team['id']})" for team in teams)
    raise SystemExit(
        f"VERCEL_TOKEN can access {len(teams)} teams; set VERCEL_TEAM_ID to one of:"
        f"{choices}"
    )


async def _project_id(token: str, team_id: str) -> str:
    from vercel import projects

    async def find() -> str | None:
        found = await asyncio.to_thread(
            projects.get_projects,
            token=token,
            team_id=team_id,
            query={"search": _PROJECT_NAME},
        )
        return next(
            (
                project["id"]
                for project in found.get("projects") or []
                if project.get("name") == _PROJECT_NAME
            ),
            None,
        )

    if project_id := await find():
        return project_id
    try:
        created = await asyncio.to_thread(
            projects.create_project,
            body={"name": _PROJECT_NAME},
            token=token,
            team_id=team_id,
        )
    except Exception:
        # Another worker may have won the creation race; search is eventually consistent.
        for _ in range(10):
            await asyncio.sleep(1)
            if project_id := await find():
                return project_id
        raise
    logger.info("vercel: created project %r for sandboxes", _PROJECT_NAME)
    return created["id"]


async def _ensure_auth():
    """Resolve credentials the way Vercel tooling does: an OIDC token (from the
    environment or the .env.local `vercel env pull` writes, minted or refreshed from
    the CLI login in a `vercel link`ed directory), else VERCEL_TOKEN with its team and
    project filled in."""
    from vercel.oidc import get_credentials, get_vercel_oidc_token

    lock = _auth_locks.setdefault(asyncio.get_running_loop(), asyncio.Lock())
    async with lock:
        if not os.environ.get("VERCEL_OIDC_TOKEN") and os.path.exists(".env.local"):
            from dotenv import dotenv_values

            for key, value in dotenv_values(".env.local").items():
                if key.startswith("VERCEL_") and value:
                    os.environ.setdefault(key, value)
        oidc = os.environ.get("VERCEL_OIDC_TOKEN")
        token = os.environ.get("VERCEL_TOKEN")
        if (oidc and _oidc_expired(oidc)) or not (oidc or token):
            try:
                await asyncio.to_thread(get_vercel_oidc_token)
            except Exception as e:
                if oidc and not token:
                    raise SystemExit(
                        "VERCEL_OIDC_TOKEN has expired; re-run `vercel env pull`."
                    ) from e
                os.environ.pop("VERCEL_OIDC_TOKEN", None)
        if not os.environ.get("VERCEL_OIDC_TOKEN") and token:
            if not os.environ.get("VERCEL_TEAM_ID"):
                os.environ["VERCEL_TEAM_ID"] = await _team_id(token)
            if not os.environ.get("VERCEL_PROJECT_ID"):
                os.environ["VERCEL_PROJECT_ID"] = await _project_id(
                    token, os.environ["VERCEL_TEAM_ID"]
                )
        try:
            return get_credentials()
        except Exception as e:
            raise SystemExit(_AUTH_HELP) from e


async def _registry_token(credentials) -> str:
    """A project OIDC token for VCR: the ambient one as is, or one exchanged for the
    API token (short-lived and project-scoped, so the API token never reaches the
    builder's firewall rule)."""
    if _oidc_claims(credentials.token).get("project"):
        return credentials.token
    cached = _REGISTRY_TOKENS.get(credentials.project_id)
    if cached and not _oidc_expired(cached):
        return cached
    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.post(
            f"https://api.vercel.com/projects/{credentials.project_id}/token",
            params={"teamId": credentials.team_id},
            headers={"Authorization": f"Bearer {credentials.token}"},
            json={"source": "vercel-cli"},
        )
    response.raise_for_status()
    token = response.json()["token"]
    _REGISTRY_TOKENS[credentials.project_id] = token
    return token


@asynccontextmanager
async def _mirror_lock(name: str):
    """Serialize one image mirror across this user's worker processes."""
    _MIRROR_LOCK_DIR.mkdir(parents=True, exist_ok=True)
    fd = os.open(_MIRROR_LOCK_DIR / f"{name}.lock", os.O_CREAT | os.O_RDWR, 0o600)
    try:
        while True:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                await asyncio.sleep(0.1)
        yield
    finally:
        os.close(fd)


async def _manifest_exists(repository: str, tag: str, auth: str) -> bool:
    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.head(
            f"https://{_VCR_HOST}/v2/{repository}/manifests/{tag}",
            headers={
                "Authorization": f"Basic {auth}",
                "Accept": _MANIFEST_TYPES,
            },
        )
    if response.status_code == 404:
        return False
    response.raise_for_status()
    return True


async def _mirror(source: str, target: str, auth: str) -> None:
    from vercel.sandbox import (
        NetworkPolicy,
        NetworkPolicyRule,
        NetworkPolicyTransform,
        create_sandbox,
    )

    inject = NetworkPolicyTransform(headers={"Authorization": f"Basic {auth}"})
    policy = NetworkPolicy.custom(
        allow={
            "*": (NetworkPolicyRule(),),
            _VCR_HOST: (NetworkPolicyRule(transform=(inject,)),),
        }
    )
    logger.info("vercel: mirroring image %s to %s", source, target)
    async with create_sandbox(
        name=f"vf-mirror-{uuid.uuid4().hex[:12]}",
        image=_BUILDER_IMAGE,
        execution_time_limit=_IMAGE_READY_TIMEOUT,
        persistent=False,
        network_policy=policy,
    ) as builder:
        result = await builder.run_process(
            "sh",
            ["-c", _MIRROR_SCRIPT, "sh", _CRANE_URL, source, target],
            capture_output=True,
        )
    if result.returncode:
        raise SandboxError(
            f"mirroring {source!r} to VCR failed: {(result.stderr or '')[-1500:]}"
        )


async def _boot_image(image: str, credentials) -> str:
    """The VCR ref a sandbox boots for `image`, mirroring a Docker image on first use."""
    if image.startswith("vercel/sandbox/"):
        return image
    if image.startswith(f"{_VCR_HOST}/"):
        return image.removeprefix(f"{_VCR_HOST}/")
    token = await _registry_token(credentials)
    claims = _oidc_claims(token)
    repository = f"{claims['owner']}/{claims['project']}/{_VCR_REPOSITORY}"
    tag = f"vf-{hashlib.sha256(image.encode()).hexdigest()[:24]}"
    ref = f"{repository}:{tag}"
    if ref in _MIRRORED:
        return ref
    async with _mirror_lock(tag):
        auth = base64.b64encode(f"oidc:{token}".encode()).decode()
        if not await _manifest_exists(repository, tag, auth):
            await _mirror(image, f"{_VCR_HOST}/{ref}", auth)
    _MIRRORED.add(ref)
    return ref


async def _queue_stream(
    queue: asyncio.Queue[bytes | SandboxError | None],
) -> AsyncIterator[bytes]:
    while (chunk := await queue.get()) is not None:
        if isinstance(chunk, SandboxError):
            raise chunk
        yield chunk


# The wrapper holds the stdin FIFO open (fd 3) for the target's lifetime, so writes
# made before the target reads are kept; it records the PID only after that. `setsid`
# gives the target its own process group (it execs in place: a background child is
# never a group leader), so signals reach its descendants.
_PROCESS_WRAPPER = (
    'd=$1; shift; exec 3<>"$d/in"; s=; command -v setsid >/dev/null && s=setsid; '
    '$s "$@" <&3 >"$d/out" 2>"$d/err" 3<&- & echo $! >"$d/pid"; '
    'wait $!; echo $? >"$d/status.tmp"; mv "$d/status.tmp" "$d/status"'
)
# Signal the target's group; succeeds when the target is already gone.
_PROCESS_SIGNAL = (
    'p=$(cat "$2/pid") && { kill -s "$1" -- "-$p" 2>/dev/null '
    '|| kill -s "$1" "$p" 2>/dev/null || ! kill -0 "$p" 2>/dev/null; }'
)
# Status first: output read after a recorded status is complete.
_PROCESS_POLL = (
    'd=$1; s=$(cat "$d/status" 2>/dev/null); '
    'tail -c +$(($2 + 1)) "$d/out" | head -c 1048576 | base64 | tr -d "\\n"; echo; '
    'tail -c +$(($3 + 1)) "$d/err" | head -c 1048576 | base64 | tr -d "\\n"; echo; '
    'echo "$s"'
)


class VercelSandboxProcess(RuntimeProcess):
    """A live process with stdin on a FIFO and file-backed stdout/stderr, relayed by
    one poller over short commands."""

    def __init__(self, runtime: "VercelSandboxRuntime", directory: str) -> None:
        self._runtime = runtime
        self._dir = directory
        self._stdout_queue: asyncio.Queue[bytes | SandboxError | None] = asyncio.Queue()
        self._stderr_queue: asyncio.Queue[bytes | SandboxError | None] = asyncio.Queue()
        self.stdout = _queue_stream(self._stdout_queue)
        self.stderr = _queue_stream(self._stderr_queue)
        self._exit_code: int | None = None
        self._poller = asyncio.create_task(self._poll_loop())

    async def _poll_loop(self) -> int:
        offsets = [0, 0]
        delay = _POLL_MIN
        try:
            while True:
                result = await self._runtime._exec(
                    ["sh", "-c", _PROCESS_POLL, "sh", self._dir, *map(str, offsets)],
                    {},
                    idempotent=True,
                )
                lines = result.stdout.split("\n")
                if result.exit_code or len(lines) < 3:
                    raise SandboxError(f"poll failed: {result.stderr.strip()[-500:]}")
                out, err = (base64.b64decode(line) for line in lines[:2])
                for i, (data, queue) in enumerate(
                    ((out, self._stdout_queue), (err, self._stderr_queue))
                ):
                    if data:
                        offsets[i] += len(data)
                        queue.put_nowait(data)
                if lines[2].strip() and not out and not err:
                    self._exit_code = int(lines[2])
                    return self._exit_code
                delay = _POLL_MIN if out or err else min(delay * 2, _POLL_MAX)
                await asyncio.sleep(delay)
        except Exception as e:
            error = SandboxError(f"vercel live process connection failed: {e}")
            self._stdout_queue.put_nowait(error)
            self._stderr_queue.put_nowait(error)
            raise error from e
        finally:
            self._stdout_queue.put_nowait(None)
            self._stderr_queue.put_nowait(None)

    async def write(self, data: bytes) -> None:
        if self._exit_code is not None:
            raise SandboxError("vercel live process stdin is closed")
        for start in range(0, len(data), _STDIN_CHUNK):
            chunk = base64.b64encode(data[start : start + _STDIN_CHUNK]).decode()
            result = await self._runtime._exec(
                [
                    "sh",
                    "-c",
                    'printf %s "$1" | base64 -d 1<>"$2/in"',
                    "sh",
                    chunk,
                    self._dir,
                ],
                {},
                idempotent=False,
            )
            if result.exit_code:
                raise SandboxError(
                    f"vercel live process stdin failed: {result.stderr.strip()}"
                )

    async def wait(self) -> int:
        return await asyncio.shield(self._poller)

    async def poll(self) -> int | None:
        return self._exit_code

    async def terminate(self) -> None:
        await self._signal("TERM")

    async def kill(self) -> None:
        await self._signal("KILL")

    async def _signal(self, signal: str) -> None:
        if self._exit_code is not None:
            return
        result = await self._runtime._exec(
            [
                "sh",
                "-c",
                _PROCESS_SIGNAL,
                "sh",
                signal,
                self._dir,
            ],
            {},
            idempotent=True,
        )
        if result.exit_code and self._exit_code is None and not self._poller.done():
            raise SandboxError(
                f"vercel live process signal failed: {result.stderr.strip()}"
            )


class VercelSandboxRuntime(Runtime):
    is_local: ClassVar[bool] = False

    def __init__(self, config: VercelSandboxConfig, name: str | None = None) -> None:
        super().__init__(name)
        self.config = config.model_copy(update={"workdir": config.workdir or "/app"})
        self.info = VercelSandboxRuntimeInfo(**self.config.model_dump())
        self._sandbox = None
        self._sudo = False

    @property
    def published_port(self) -> int | None:
        return SERVICE_PORT

    async def start(self) -> None:
        try:
            from vercel.sandbox import SandboxResources, create_sandbox
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError(
                "VercelSandboxRuntime requires the Vercel SDK; install `vercel>=0.11` "
                "(e.g. `uv add vercel`)."
            ) from e

        credentials = await _ensure_auth()
        try:
            vcpus = _vcpus(self.config.cpu, self.config.memory)
            image = await _boot_image(self.config.image, credentials)
            self.info.boot_image = image

            async def _create() -> None:
                loop = asyncio.get_running_loop()
                deadline = loop.time() + _IMAGE_READY_TIMEOUT
                async with (
                    creation_limiter(
                        self.config.creates_per_sec, "vercel-sandbox", run_scope()
                    )
                    or contextlib.nullcontext()
                ):
                    while True:
                        try:
                            # Created unrestricted: `prepare_execution` locks egress down.
                            self._sandbox = await create_sandbox(
                                name=self.name,
                                image=image,
                                ports=[SERVICE_PORT],
                                execution_time_limit=_LIFETIME,
                                resources=SandboxResources(
                                    vcpus=vcpus, memory=vcpus * _MEMORY_MB_PER_VCPU
                                ),
                                persistent=False,
                            )
                            break
                        except Exception as e:
                            # VCR prepares a pushed image asynchronously.
                            if (
                                getattr(e, "code", None) != "image_not_ready"
                                or loop.time() > deadline
                            ):
                                raise
                        await asyncio.sleep(2)
                    self.info.id = self._sandbox.name

            await run_shielded(_create())
            logger.info(
                "vercel: sandbox %s up (image=%s, vcpus=%d)", self.info.id, image, vcpus
            )
            identity = await self._sandbox.run_process(
                "id", ["-u"], capture_output=True
            )
            self._sudo = identity.stdout.strip() != "0"
            created = await self._sandbox.run_process(
                "mkdir",
                ["-p", self.config.workdir],
                cwd="/",
                sudo=self._sudo,
                capture_output=True,
            )
            if created.returncode:
                raise SandboxError(created.stderr.strip())
        except Exception as e:
            raise SandboxError(f"vercel sandbox provisioning failed: {e}") from e

    async def prepare_execution(self, routes: list[str] | None) -> None:
        if not self.network_restricted:
            return
        try:
            await _refresh_auth()
            await self._sandbox.update_network_policy(
                _egress_policy(self.config, routes)
            )
        except Exception as e:
            raise SandboxError(f"vercel egress policy failed: {e}") from e
        logger.info("vercel: egress policy applied on sandbox %s", self.info.id)

    def _command(
        self, argv: list[str], env: dict[str, str]
    ) -> tuple[str, list[str], dict[str, str]]:
        """Run as root like the sibling runtimes: elevate when the image's default user
        isn't root, restoring the caller's PATH that sudo resets."""
        command_env = self.process_env(env)
        if self._sudo and "PATH" in command_env:
            command_env["VF_RUNTIME_PATH"] = command_env.pop("PATH")
            argv = [
                "sh",
                "-c",
                'export PATH="$VF_RUNTIME_PATH"; exec "$@"',
                "sh",
                *argv,
            ]
        return argv[0], argv[1:], command_env

    async def run(self, argv: list[str], env: dict[str, str]) -> ProgramResult:
        return await self._exec(argv, env, idempotent=False)

    async def _exec(
        self, argv: list[str], env: dict[str, str], *, idempotent: bool
    ) -> ProgramResult:
        """One command. A connection failure never reached the sandbox, so it is always
        retried; a dropped response doesn't tell whether the command ran, so only this
        runtime's own idempotent commands retry it."""
        import httpx2
        from vercel.sandbox import SandboxResponseError, SandboxStreamError

        retryable: tuple[type[Exception], ...] = (
            httpx2.ConnectError,
            httpx2.ConnectTimeout,
        )
        if idempotent:
            retryable += (
                httpx2.TransportError,
                SandboxResponseError,
                SandboxStreamError,
            )
        command, args, command_env = self._command(argv, env)
        for attempt in range(1, _EXEC_ATTEMPTS + 1):
            try:
                await _refresh_auth()
                result = await self._sandbox.run_process(
                    command,
                    args,
                    cwd=self.config.workdir,
                    env=command_env,
                    sudo=self._sudo,
                    capture_output=True,
                )
                return ProgramResult(
                    result.returncode, result.stdout or "", result.stderr or ""
                )
            except retryable as e:
                if attempt == _EXEC_ATTEMPTS:
                    raise SandboxError(f"vercel exec failed: {e}") from e
                await asyncio.sleep(0.5 * attempt)
            except Exception as e:
                raise SandboxError(f"vercel exec failed: {e}") from e
        raise AssertionError("unreachable")

    async def _wait(self, process) -> int:
        """Wait on a process; the wait is idempotent, so a dropped request is reissued.
        Only consecutive fast failures count, so a long rollout survives any number of
        transient drops."""
        loop = asyncio.get_running_loop()
        failures = 0
        while True:
            started = loop.time()
            try:
                await _refresh_auth()
                return await process.wait()
            except Exception:
                failures = 1 if loop.time() - started > 60 else failures + 1
                if failures > _WAIT_RECONNECTS:
                    raise
                await asyncio.sleep(0.25 * 2**failures)

    async def run_program(self, argv: list[str], env: dict[str, str]) -> ProgramResult:
        """Run the rollout through a durable process with file-backed output, without
        ever replaying it."""
        prefix = f"/tmp/vf-program-{uuid.uuid4().hex}"
        wrapper = (
            'p=$1; shift; "$@" >"$p.stdout" 2>"$p.stderr"; '
            'printf "%s\\n" "$?" >"$p.status"'
        )
        command, args, command_env = self._command(
            ["sh", "-c", wrapper, "vf-program", prefix, *argv], env
        )
        try:
            await _refresh_auth()
            process = await self._sandbox.create_process(
                command,
                args,
                cwd=self.config.workdir,
                env=command_env,
                sudo=self._sudo,
            )
            await self._wait(process)
            status, stdout, stderr = await asyncio.gather(
                *(
                    self.read(f"{prefix}.{name}")
                    for name in ("status", "stdout", "stderr")
                )
            )
            return ProgramResult(
                int(status.decode().strip()),
                stdout.decode(errors="replace"),
                stderr.decode(errors="replace"),
            )
        except SandboxError:
            raise
        except Exception as e:
            raise SandboxError(f"vercel durable program failed: {e}") from e
        finally:
            with contextlib.suppress(Exception):
                await self._exec(
                    [
                        "rm",
                        "-f",
                        *(f"{prefix}.{n}" for n in ("status", "stdout", "stderr")),
                    ],
                    {},
                    idempotent=True,
                )

    async def open_process(
        self, argv: list[str], env: dict[str, str]
    ) -> RuntimeProcess:
        directory = f"/tmp/vf-process-{uuid.uuid4().hex}"
        command, args, command_env = self._command(
            ["sh", "-c", _PROCESS_WRAPPER, "vf-process", directory, *argv], env
        )
        try:
            made = await self._exec(
                [
                    "sh",
                    "-c",
                    'mkdir -p "$1" && { [ -p "$1/in" ] || mkfifo "$1/in"; }',
                    "sh",
                    directory,
                ],
                {},
                idempotent=True,
            )
            if made.exit_code:
                raise SandboxError(made.stderr.strip())
            await _refresh_auth()
            wrapper = await self._sandbox.create_process(
                command,
                args,
                cwd=self.config.workdir,
                env=command_env,
                sudo=self._sudo,
            )
            # Writes are safe once the wrapper holds the FIFO, which the PID marks.
            loop = asyncio.get_running_loop()
            deadline = loop.time() + _START_TIMEOUT
            while (
                await self._exec(
                    ["test", "-s", f"{directory}/pid"], {}, idempotent=True
                )
            ).exit_code:
                if loop.time() > deadline:
                    with contextlib.suppress(Exception):
                        await wrapper.kill()
                    raise SandboxError("PID unavailable")
                await asyncio.sleep(0.05)
        except Exception as e:
            raise SandboxError(f"vercel live process failed to start: {e}") from e
        return VercelSandboxProcess(self, directory)

    async def run_background(
        self, argv: list[str], env: dict[str, str], log: str
    ) -> None:
        inner = f"nohup {shlex.join(argv)} > {shlex.quote(log)} 2>&1 &"
        result = await self.run(["sh", "-c", inner], env)
        if result.exit_code != 0:
            raise SandboxError(
                f"vercel background launch failed: {result.stderr.strip()}"
            )

    def _abs(self, path: str) -> str:
        if path.startswith("/"):
            return path
        return f"{self.config.workdir.rstrip('/')}/{path}"

    async def _read(self, path: str, max_bytes: int | None = None) -> bytes:
        # The filesystem API acts as the image's default user; elevated reads go via exec.
        if max_bytes is not None or self._sudo:
            return await super()._read(
                self._abs(path), _UNBOUNDED if max_bytes is None else max_bytes
            )
        try:
            await _refresh_auth()
            return await self._sandbox.fs.read_bytes(self._abs(path))
        except Exception as e:
            raise SandboxError(f"read {path!r}: {e}") from e

    async def write(self, path: str, data: bytes) -> None:
        target = self._abs(path)
        try:
            await _refresh_auth()
            if not self._sudo:
                await self._sandbox.fs.mkdir(str(PurePosixPath(target).parent))
                await self._sandbox.fs.write_bytes(target, data)
                return
            staged = f"/tmp/vf-write-{uuid.uuid4().hex}"
            await self._sandbox.fs.write_bytes(staged, data)
            moved = await self._exec(
                [
                    "sh",
                    "-c",
                    _MOVE_STAGED,
                    "sh",
                    staged,
                    target,
                ],
                {},
                idempotent=True,
            )
            if moved.exit_code:
                raise SandboxError(moved.stderr.strip())
        except Exception as e:
            raise SandboxError(f"write {path!r}: {e}") from e

    async def expose(self, port: int) -> str | None:
        if self._sandbox is None:
            return None
        return next(
            (
                route.url.rstrip("/")
                for route in self._sandbox.routes
                if route.port == port
            ),
            None,
        )

    def cleanup(self) -> None:
        # Synchronous atexit backstop: destroy by name through the sync API.
        if self._sandbox is None or self.info.id is None:
            return
        from vercel.oidc import get_vercel_oidc_token
        from vercel.sandbox.sync import get_sandbox

        try:
            if (oidc := os.environ.get("VERCEL_OIDC_TOKEN")) and _oidc_expired(oidc):
                get_vercel_oidc_token()
            get_sandbox(name=self.info.id).destroy()
        except Exception:  # noqa: BLE001 - a later cleanup call may retry
            return
        self._sandbox = None

    async def teardown(self) -> None:
        sandbox = self._sandbox
        if sandbox is None:
            return
        try:
            await _refresh_auth()
            async with asyncio.timeout(_DESTROY_TIMEOUT):
                await sandbox.destroy()
        except Exception as e:  # noqa: BLE001 - provider teardown is best-effort
            logger.warning("vercel: failed to destroy sandbox %s: %s", self.info.id, e)
            return
        self._sandbox = None
