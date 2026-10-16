"""Compose services on Prime VMs of their own, joined to the workspace over a LAN.

A Prime VM's egress policy covers the whole VM, so a service whose egress differs from
the workspace's runs on a VM of its own (`ComposeSpec.own_host`). A LAN network per
rollout lets the workspace's VM reach those VMs, and nothing reach the workspace; each
service keeps its VM's unrestricted egress. The workspace resolves services by their
Compose names through its hosts file.
"""

import asyncio
import contextlib
import logging
import re
import shlex
import uuid
from collections.abc import AsyncIterator, Mapping
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import dataclass
from pathlib import Path

from verifiers.v1.errors import SandboxError
from verifiers.v1.runtimes import PrimeConfig, RuntimeBackend, provision_backend
from verifiers.v1.utils.aio import run_shielded

logger = logging.getLogger(__name__)

WORKSPACE_TAG = "vf-workspace"
# Compose keys a service on its own VM can honor; anything else fails before provisioning.
SUPPORTED = {
    "image",
    "command",
    "entrypoint",
    "environment",
    "working_dir",
    "healthcheck",
    "expose",
    "cpus",
    "mem_limit",
    "deploy",
}
# The workspace image may lack getent; Python is the fallback.
RESOLVE = (
    'address=$(getent hosts "$1" 2>/dev/null | cut -d" " -f1); '
    '[ -n "$address" ] || address=$(python3 -c "import socket, sys; '
    'print(socket.gethostbyname(sys.argv[1]))" "$1" 2>/dev/null); echo "$address"'
)
# Activity that keeps a service VM from reaching its idle timeout mid-rollout.
KEEPALIVE_SECONDS = 600
_VARIABLE = re.compile(
    r"\$(?:(\$)|\{([A-Za-z_][A-Za-z0-9_]*)(?::?-([^}]*))?\}|([A-Za-z_][A-Za-z0-9_]*))"
)
_DURATION = re.compile(r"(\d+(?:\.\d+)?)(ns|us|ms|s|m|h)")
_SECONDS = {"ns": 1e-9, "us": 1e-6, "ms": 1e-3, "s": 1, "m": 60, "h": 3600}


@dataclass(frozen=True)
class ServiceProgram:
    """One Compose service as a program on its own VM."""

    image: str
    argv: list[str]
    environment: dict[str, str]
    healthcheck: list[str] | None
    interval: float
    timeout: float
    attempts: int
    cpu: float | None
    memory: float | None


def interpolate(value: str, environment: Mapping[str, str]) -> str:
    """Compose's `$$`, `$VAR`, `${VAR}` and `${VAR:-default}` substitution."""

    def substitute(match: re.Match) -> str:
        if match.group(1):
            return "$"
        name = match.group(2) or match.group(4)
        found = environment.get(name)
        return found if found else (match.group(3) or "")

    return _VARIABLE.sub(substitute, value)


def seconds(value, default: float) -> float:
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return float(value)
    parts = _DURATION.findall(str(value))
    if not parts or "".join(a + b for a, b in parts) != str(value):
        raise ValueError(f"invalid Compose duration {value!r}")
    return sum(float(amount) * _SECONDS[unit] for amount, unit in parts)


def service_program(
    name: str, definition: dict, environment: Mapping[str, str]
) -> ServiceProgram:
    """Validate an authored Compose service and read it as a VM program."""
    unsupported = sorted(set(definition) - SUPPORTED)
    if unsupported:
        raise ValueError(
            f"Compose service {name!r} runs on its own VM, which does not support "
            f"{', '.join(unsupported)}"
        )
    if not definition.get("image"):
        raise ValueError(f"Compose service {name!r} needs an image for its own VM")

    def words(value) -> list[str]:
        if value is None:
            return []
        items = shlex.split(value) if isinstance(value, str) else list(value)
        return [interpolate(str(item), environment) for item in items]

    # A VM does not run the image's entrypoint, so the service spells out its program.
    argv = words(definition.get("entrypoint")) + words(definition.get("command"))
    if not argv:
        raise ValueError(
            f"Compose service {name!r} needs a command (or entrypoint) for its own VM"
        )
    if workdir := definition.get("working_dir"):
        argv = [
            "sh",
            "-c",
            f"cd {shlex.quote(interpolate(workdir, environment))} && exec {shlex.join(argv)}",
        ]
    variables = definition.get("environment") or {}
    if isinstance(variables, list):
        variables = dict(entry.partition("=")[::2] for entry in variables)
    check = definition.get("healthcheck") or {}
    test = check.get("test")
    if check.get("disable") or test in (None, "NONE", ["NONE"]):
        healthcheck = None
    elif isinstance(test, str):
        healthcheck = ["sh", "-c", interpolate(test, environment)]
    elif test[0] == "CMD-SHELL":
        healthcheck = ["sh", "-c", interpolate(" ".join(test[1:]), environment)]
    else:
        healthcheck = words(test[1:] if test[0] == "CMD" else test)
    interval = seconds(check.get("interval"), 30)
    limits = definition.get("deploy", {}).get("resources", {}).get("limits", {})
    cpus = definition.get("cpus", limits.get("cpus"))
    memory = definition.get("mem_limit", limits.get("memory"))
    return ServiceProgram(
        image=interpolate(definition["image"], environment),
        argv=argv,
        environment={
            key: interpolate(str(value), environment)
            for key, value in variables.items()
            if value is not None
        },
        healthcheck=healthcheck,
        interval=interval,
        timeout=seconds(check.get("timeout"), 30),
        attempts=max(1, int(check.get("retries", 3)))
        + int(seconds(check.get("start_period"), 0) // interval),
        cpu=float(cpus) if cpus is not None else None,
        memory=_gigabytes(memory) if memory is not None else None,
    )


def _gigabytes(value) -> float:
    if isinstance(value, (int, float)):
        return value / 1024**3
    match = re.fullmatch(r"(\d+(?:\.\d+)?)\s*([kmgt]?)i?b?", str(value).lower())
    if match is None:
        raise ValueError(f"invalid Compose memory {value!r}")
    power = " kmgt".index(match.group(2) or " ")
    return float(match.group(1)) * 1024**power / 1024**3


def own_host_programs(
    spec, environment: Mapping[str, str]
) -> dict[str, ServiceProgram]:
    """The task's own-host services, if the workspace can be the agent's own VM."""
    import yaml

    authored = yaml.safe_load((Path(spec.directory) / spec.file).read_text()) or {}
    services = authored.get("services") or {}
    missing = sorted(set(spec.own_host) - set(services))
    if missing:
        raise ValueError(f"own-host services {missing} are not in the Compose file")
    others = sorted(set(services) - set(spec.own_host) - {spec.workspace})
    if others or services.get(spec.workspace):
        raise ValueError(
            "own-host Compose services currently need a task whose other services "
            f"also run on their own VMs and whose workspace {spec.workspace!r} is "
            "not authored; found " + ", ".join(others or [spec.workspace])
        )
    return {
        name: service_program(name, services[name], environment)
        for name in spec.own_host
    }


def _tag(name: str) -> str:
    return "vf-service-" + re.sub(r"[^a-z0-9-]", "-", name.lower())


async def _start(runtime, name: str, program: ServiceProgram) -> None:
    log = f"/tmp/vf-service-{name}.log"
    status = f"/tmp/vf-service-{name}.exit"
    # Record the exit status, so a crashed service fails setup instead of its health wait.
    script = f"{shlex.join(program.argv)}; echo $? > {shlex.quote(status)}"
    await runtime.run_background(["sh", "-c", script], program.environment, log)
    if program.healthcheck is None:
        return
    for attempt in range(program.attempts + 1):
        with contextlib.suppress(TimeoutError):
            async with asyncio.timeout(program.timeout + 10):
                if (await runtime.run(program.healthcheck, {})).exit_code == 0:
                    return
        exited = await runtime.run(["cat", status], {})
        if exited.exit_code == 0 or attempt == program.attempts:
            tail = await runtime.run(["tail", "-c", "2000", log], {})
            state = (
                f"exited with {exited.stdout.strip()}"
                if exited.exit_code == 0
                else "never became healthy"
            )
            raise SandboxError(f"Compose service {name!r} {state}: {tail.stdout}")
        await asyncio.sleep(program.interval)


async def _keepalive(runtime) -> None:
    while True:
        await asyncio.sleep(KEEPALIVE_SECONDS)
        with contextlib.suppress(Exception):
            await runtime.run(["true"], {})


@asynccontextmanager
async def own_host_services(
    workspace: RuntimeBackend,
    programs: Mapping[str, ServiceProgram],
) -> AsyncIterator[dict[str, RuntimeBackend]]:
    """Run each program on its own VM and let the workspace's VM reach them by name."""
    config = workspace.config
    if not isinstance(config, PrimeConfig):
        raise TypeError(
            f"own-host Compose services run on Prime VMs, not {config.type}"
        )
    from prime_sandboxes import LanRule

    client = workspace._client  # the workspace's leased Prime client
    async with AsyncExitStack() as stack:
        backends: dict[str, RuntimeBackend] = {}

        async def provision(name: str, program: ServiceProgram) -> None:
            vm = config.model_copy(
                update={
                    "image": program.image,
                    "workdir": "/",
                    "checkpoint": None,
                    "allow": ["*"],
                    "block": [],
                    "cpu": program.cpu or config.cpu,
                    "memory": program.memory or config.memory,
                }
            )
            backend = await stack.enter_async_context(
                provision_backend(vm, name=f"{workspace.name}-{name}")
            )
            backends[name] = backend
            await _start(backend.bind(), name, program)

        try:
            async with asyncio.TaskGroup() as group:
                for name, program in programs.items():
                    group.create_task(provision(name, program))
        except ExceptionGroup as failed:
            raise failed.exceptions[0]  # one service's failure, typed for retries

        async def stop(task: asyncio.Task) -> None:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

        for backend in backends.values():
            stack.push_async_callback(
                stop, asyncio.create_task(_keepalive(backend.bind()))
            )

        network = await client.create_lan_network(f"vf-{uuid.uuid4().hex[:16]}")

        async def delete_network() -> None:
            try:
                await client.delete_lan_network(network.id)
            except Exception:
                logger.warning(
                    "LAN network %s deletion failed", network.id, exc_info=True
                )

        stack.push_async_callback(lambda: run_shielded(delete_network()))
        await client.add_lan_member(network.id, workspace.info.id, tags=[WORKSPACE_TAG])
        for name, backend in backends.items():
            await client.add_lan_member(network.id, backend.info.id, tags=[_tag(name)])
        await client.set_lan_rules(
            network.id,
            [
                LanRule(from_=f"tag:{WORKSPACE_TAG}", to=f"tag:{_tag(name)}")
                for name in backends
            ],
        )
        try:
            async with asyncio.timeout(120):
                while True:
                    state = await client.get_lan_network(network.id)
                    if state.synced and all(m.synced for m in state.members or []):
                        break
                    await asyncio.sleep(1)
        except TimeoutError as e:
            raise SandboxError(
                f"LAN network {network.id} did not sync within 120s"
            ) from e
        runtime = workspace.bind()
        entries = []
        for name, backend in backends.items():
            found = await runtime.run(
                [
                    "sh",
                    "-c",
                    RESOLVE,
                    "vf-resolve",
                    f"{backend.info.id}.sandbox.internal",
                ],
                {},
            )
            address = found.stdout.strip()
            if not address:
                raise SandboxError(f"Compose service {name!r} has no LAN address")
            entries.append(f"{address} {name}\n")
        added = await runtime.run(
            [
                "sh",
                "-c",
                'printf "%s" "$1" >> /etc/hosts',
                "vf-hosts",
                "".join(entries),
            ],
            {},
        )
        if added.exit_code:
            raise SandboxError(f"cannot name own-host services: {added.stderr}")
        logger.info(
            "LAN network %s joins %s to services %s",
            network.id,
            workspace.info.id,
            {name: backend.info.id for name, backend in backends.items()},
        )
        yield backends
