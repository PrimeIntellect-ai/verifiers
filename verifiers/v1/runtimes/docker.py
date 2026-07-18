"""Local Docker runtime: host networking by default, host-only networking on demand.

With `DockerConfig.network_access=False` the container keeps internet through setup
(package installs, provisioning), then — right before the agent starts
(`prepare_execution`) — loses it: on Linux it is moved onto a shared `--internal`
network where only the gateway (the host) is reachable; on macOS the Docker VM gives
internal networks no route to the host, so a one-shot helper pins a /32 route to the
host and drops the default route instead. Either way, afterwards the container reaches
host services (the interception server, host MCP servers) and nothing else."""

import asyncio
import contextlib
import logging
import shlex
import subprocess
import sys
from pathlib import PurePosixPath
from typing import Literal
from urllib.parse import urlsplit

from pydantic_config import BaseConfig

from verifiers.v1.errors import SandboxError
from verifiers.v1.runtimes.base import (
    BaseRuntimeInfo,
    ProgramResult,
    Runtime,
    parse_gpu,
)

logger = logging.getLogger(__name__)


class DockerConfig(BaseConfig):
    type: Literal["docker"] = "docker"
    image: str = "python:3.11-slim"
    workdir: str = "/app"
    # TaskData.resources uses these units; non-default runtime config values take precedence.
    cpu: float | None = None
    """Pin the container to this many CPU cores (docker `--cpus`). None = unlimited."""
    memory: float | None = None
    """Hard memory limit in GB (docker `--memory`). None = unlimited."""
    gpu: str | None = None
    """GPU spec, e.g. "A100" or "2" (docker `--gpus` uses the count; needs the nvidia
    container toolkit). None = none."""
    disk: float | None = None
    """Advisory disk request in GB. Docker has no portable per-container size limit, so
    this is accepted (so a task can declare it without a warning) but not enforced."""
    network_access: bool = True
    """False = keep internet only through setup (package installs, provisioning), then
    cut it right before the agent starts (`prepare_execution`): the container then
    reaches host services (the interception server, host MCP servers) and nothing
    else."""


class DockerRuntimeInfo(DockerConfig, BaseRuntimeInfo):
    pass


async def docker(*args: str) -> ProgramResult:
    proc = await asyncio.create_subprocess_exec(
        "docker",
        *args,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await proc.communicate()
    return ProgramResult(
        exit_code=proc.returncode or 0,
        stdout=stdout.decode(errors="replace"),
        stderr=stderr.decode(errors="replace"),
    )


async def docker_checked(*args: str) -> ProgramResult:
    result = await docker(*args)
    if result.exit_code != 0:
        raise SandboxError(f"docker {' '.join(args)} failed: {result.stderr.strip()}")
    return result


_OFFLINE_NETWORK = "verifiers-offline"
"""The one internal network every offline container shares (created on first use, kept
around afterwards; `docker network rm` it to clean up). `--internal` = no route off the
network — except the bridge gateway, a host-side interface, which stays reachable:
that's where host services (interception, host MCP servers) bind for offline containers.
Inter-container communication is disabled so concurrent rollouts can't reach each other
(the gateway is the host, not a peer — unaffected)."""


async def ensure_offline_network() -> str:
    """Create the shared offline network if needed and return its gateway IP — the
    address host services bind and offline containers reach them at."""
    inspect = await docker(
        "network",
        "inspect",
        _OFFLINE_NETWORK,
        "--format",
        "{{(index .IPAM.Config 0).Gateway}}",
    )
    if inspect.exit_code == 0 and inspect.stdout.strip():
        return inspect.stdout.strip()
    create = await docker(
        "network",
        "create",
        "--internal",
        "-o",
        "com.docker.network.bridge.enable_icc=false",
        "--label",
        "verifiers.offline=true",
        _OFFLINE_NETWORK,
    )
    # Concurrent first uses race the create; the loser re-inspects below.
    if create.exit_code != 0 and "already exists" not in create.stderr:
        raise SandboxError(f"docker network create failed: {create.stderr.strip()}")
    inspect = await docker(
        "network",
        "inspect",
        _OFFLINE_NETWORK,
        "--format",
        "{{(index .IPAM.Config 0).Gateway}}",
    )
    gateway = inspect.stdout.strip()
    if inspect.exit_code != 0 or not gateway:
        raise SandboxError(
            f"could not determine the {_OFFLINE_NETWORK} gateway: "
            f"{inspect.stderr.strip()}"
        )
    return gateway


class DockerRuntime(Runtime):
    def __init__(self, config: DockerConfig, name: str | None = None) -> None:
        super().__init__(name)
        self.config = config
        self.info = DockerRuntimeInfo(**config.model_dump())
        self._container: str | None = None  # our `--name` (used for exec/rm)
        self._stopped = False
        self._offline_gateway: str | None = None  # set on start when offline (Linux)

    async def start(self) -> None:
        try:
            version = await docker("version", "--format", "{{.Server.Version}}")
        except FileNotFoundError as e:
            raise RuntimeError(
                "docker runtime selected but the `docker` CLI is not installed"
            ) from e
        if version.exit_code != 0:
            detail = (version.stderr or version.stdout).strip()
            hint = ""
            if "permission denied" in detail.lower():
                hint = (
                    "\nYour user isn't in the `docker` group. Either run the command "
                    'under `sg docker -c "..."`, or add yourself with '
                    "`sudo usermod -aG docker $USER` and start a new login shell."
                )
            raise RuntimeError(
                f"docker runtime selected but the Docker daemon is not reachable: {detail}{hint}"
            )
        self._container = self.name
        limits: list[str] = []
        if self.config.cpu is not None:
            limits += ["--cpus", str(self.config.cpu)]
        if self.config.memory is not None:
            limits += ["--memory", f"{self.config.memory}g"]
        _, gpu_count = parse_gpu(self.config.gpu)
        if gpu_count:
            limits += ["--gpus", str(gpu_count)]
        offline = not self.config.network_access
        if offline and sys.platform == "linux":
            self._offline_gateway = await ensure_offline_network()
        if offline:
            # Offline containers set up on the default bridge (internet), then lose it
            # in `prepare_execution`. They never hold NET_ADMIN, so the agent inside
            # can't restore it.
            network = [
                "--network",
                "bridge",
                "--cap-drop",
                "NET_ADMIN",
                "--cap-drop",
                "NET_RAW",
                "--security-opt",
                "no-new-privileges",
                "--sysctl",
                "net.ipv6.conf.all.disable_ipv6=1",
            ]
        else:
            network = ["--network", "host"]
        run = await docker(
            "run",
            "--detach",
            *network,
            *limits,
            "--workdir",
            self.config.workdir,
            "--entrypoint",
            "sleep",
            "--name",
            self._container,
            self.config.image,
            "infinity",
        )
        if run.exit_code != 0:
            raise SandboxError(f"docker run failed: {run.stderr.strip()}")
        self.info.id = run.stdout.strip()[
            :12
        ]  # `docker run -d` prints the container id
        if offline and sys.platform == "linux":
            await docker_checked(
                "network", "connect", _OFFLINE_NETWORK, self._container
            )
        logger.info(
            "docker: started container %s (image=%s)",
            self._container,
            self.config.image,
        )

    def host_url(self, url: str) -> str:
        host = urlsplit(url).hostname
        # Offline on Linux: the container sits on the internal network and reaches
        # host services at that network's gateway IP.
        if self._offline_gateway is not None and host in ("127.0.0.1", "localhost"):
            return url.replace(host, self._offline_gateway, 1)
        # Docker Desktop (macOS/Windows) runs containers in a VM, so `--network host`
        # doesn't reach the host's loopback; `host.docker.internal` does.
        if sys.platform != "linux" and host in ("127.0.0.1", "localhost"):
            return url.replace(host, "host.docker.internal", 1)
        return url

    @property
    def host_service_host(self) -> str | None:
        return self._offline_gateway

    @property
    def network_isolated(self) -> bool:
        return not self.config.network_access

    async def prepare_execution(self, routes: dict[str, str]) -> dict[str, str]:
        """With `network_access=False`: cut the container's internet now that setup is
        done, refusing routes it couldn't reach afterwards (e.g. an external MCP URL).
        Linux moves the container onto the internal network (gateway/host stays
        reachable); macOS pins a host route instead (see `_pin_host_route`)."""
        if self.config.network_access:
            return routes
        local = {
            "127.0.0.1",
            "localhost",
            "host.docker.internal",
            self._offline_gateway,
        } - {None}
        for name, url in routes.items():
            parsed = urlsplit(url)
            if parsed.scheme != "http" or parsed.hostname not in local:
                raise SandboxError(
                    f"offline docker (network_access=False) reaches host-local "
                    f"services only, but route {name!r} is {url!r}; use a colocated "
                    f"or host-local server, or drop it"
                )
        if sys.platform == "linux":
            # Internal-network only from here: no external route remains; the gateway
            # (host services) stays reachable.
            await docker_checked("network", "disconnect", "bridge", self._container)
        else:
            await self._pin_host_route()
        await self._probe(routes["model"])
        return routes

    async def _probe(self, url: str) -> None:
        """Confirm the model route is reachable after the cut. Host firewalls that
        default-drop container→host traffic (e.g. ufw on servers) would otherwise
        surface as a confusing agent-side connection timeout."""
        script = (
            "if command -v python3 >/dev/null 2>&1; then "
            'python3 -c "import sys, urllib.error, urllib.request\n'
            "try:\n"
            "    urllib.request.urlopen(sys.argv[1], timeout=3)\n"
            "except urllib.error.HTTPError:\n"
            "    sys.exit(0)\n"
            "except Exception:\n"
            '    sys.exit(1)" "$1"; '
            "elif command -v wget >/dev/null 2>&1; then "
            'out=$(wget -q -O /dev/null -T 3 "$1" 2>&1) && exit 0; '
            'echo "$out" | grep -q "HTTP/"; '
            "else exit 0; "  # no probe tool in the image — proceed unprobed
            "fi"
        )
        probe = await self.run(["sh", "-c", script, "probe", url], {})
        if probe.exit_code != 0:
            raise SandboxError(
                f"offline docker: {url} is unreachable from the container after the "
                "network cut — the host firewall likely drops container-to-host "
                "traffic (ufw's default INPUT policy does). Allow the "
                f"{_OFFLINE_NETWORK} subnet to reach the host, e.g. `ufw allow from "
                "<subnet>` (see `docker network inspect verifiers-offline`)."
            )

    async def _pin_host_route(self) -> None:
        """The macOS cut: Docker Desktop / OrbStack VMs give internal networks no route
        to the host, so instead of switching networks we pin a /32 to the host's NAT IP
        and drop the default route (and blackhole DNS). Done from a one-shot privileged
        helper sharing the container's netns — the container itself never holds
        NET_ADMIN, so the agent can't undo it."""
        script = (
            "set -e; "
            "IP=$(getent hosts host.docker.internal | awk 'NR==1{print $1}'); "
            "GW=$(ip route show default | awk '/^default via/{print $3; exit}'); "
            'ip route add "$IP/32" via "$GW"; '
            "ip route del default; "
            "ip route add blackhole 127.0.0.11/32 table local; "
            'printf %s "$IP"'
        )
        run = await docker(
            "run",
            "--rm",
            "--network",
            f"container:{self._container}",
            "--cap-drop",
            "ALL",
            "--cap-add",
            "NET_ADMIN",
            "alpine:3.22",
            "sh",
            "-c",
            script,
        )
        if run.exit_code != 0:
            raise SandboxError(f"offline network cut failed: {run.stderr.strip()}")
        # DNS is dead after the cut; pin the name host URLs already use.
        await docker_checked(
            "exec",
            self._container,
            "sh",
            "-c",
            f"echo '{run.stdout.strip()} host.docker.internal' >> /etc/hosts",
        )

    async def run(self, argv: list[str], env: dict[str, str]) -> ProgramResult:
        env_args = [arg for k, v in env.items() for arg in ("--env", f"{k}={v}")]
        return await docker(
            "exec", *env_args, "--workdir", self.config.workdir, self._container, *argv
        )

    async def run_background(
        self, argv: list[str], env: dict[str, str], log: str
    ) -> None:
        env_args = [arg for k, v in env.items() for arg in ("--env", f"{k}={v}")]
        inner = f"{' '.join(shlex.quote(a) for a in argv)} > {shlex.quote(log)} 2>&1"
        run = await docker(
            "exec",
            "--detach",
            *env_args,
            "--workdir",
            self.config.workdir,
            self._container,
            "sh",
            "-c",
            inner,
        )  # detached → lives in the container until it's removed in stop()
        if run.exit_code != 0:
            raise SandboxError(f"docker exec -d failed: {run.stderr.strip()}")

    async def read(self, path: str) -> bytes:
        proc = await asyncio.create_subprocess_exec(
            "docker",
            "exec",
            "--workdir",
            self.config.workdir,
            self._container,
            "cat",
            path,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()
        if proc.returncode != 0:
            raise SandboxError(
                f"read {path!r}: {stderr.decode(errors='replace').strip()}"
            )
        return stdout

    async def write(self, path: str, data: bytes) -> None:
        parent = shlex.quote(str(PurePosixPath(path).parent))
        proc = await asyncio.create_subprocess_exec(
            "docker",
            "exec",
            "-i",
            "--workdir",
            self.config.workdir,
            self._container,
            "sh",
            "-c",
            f"mkdir -p {parent} && cat > {shlex.quote(path)}",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        _, stderr = await proc.communicate(input=data)
        if proc.returncode != 0:
            raise SandboxError(
                f"write {path!r}: {stderr.decode(errors='replace').strip()}"
            )

    def cleanup(self) -> None:
        if self._container is None or self._stopped:
            return
        self._stopped = (
            True  # idempotency guard; keep `_container` so the name still shows
        )
        logger.debug("docker: removing container %s", self._container)
        with contextlib.suppress(Exception):
            subprocess.run(
                ["docker", "rm", "--force", self._container],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=30,
            )
