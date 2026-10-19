"""Provider-independent Compose deployment on local Docker or a provider VM."""

import asyncio
import atexit
import io
import json
import logging
import os
import shlex
import shutil
import subprocess
import sys
import tarfile
import tempfile
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path

from verifiers.v1.errors import SandboxError
from verifiers.v1.runtimes import (
    DockerBackend,
    DockerConfig,
    ModalConfig,
    PrimeConfig,
    RuntimeBackend,
    RuntimeConfig,
    provision_backend,
)
from verifiers.v1.runtimes.base import SERVICE_PORT, ProgramResult
from verifiers.v1.runtimes.container import cli
from verifiers.v1.utils.aio import run_shielded

logger = logging.getLogger(__name__)

# Docker routing and credential-helper lookup need these; local Compose must not see
# unrelated evaluator secrets through shell interpolation.
DOCKER_ENV = (
    "PATH",
    "HOME",
    "DOCKER_HOST",
    "DOCKER_CONTEXT",
    "DOCKER_CONFIG",
    "DOCKER_TLS_VERIFY",
    "DOCKER_CERT_PATH",
)
# One provider VM hosts the Docker daemon and every service.
VM_HOST = {
    PrimeConfig: {"image": "python:3.11-slim-trixie", "workdir": "/"},
    ModalConfig: {"image": "docker:28.3.3-dind", "workdir": "/", "vm": True},
}
INSTALL_DOCKER = (
    "command -v dockerd >/dev/null || { export DEBIAN_FRONTEND=noninteractive; "
    "apt-get update -qq && apt-get install -y -qq --no-install-recommends docker.io "
    "docker-cli docker-compose iptables > /tmp/docker-install.log 2>&1 "
    "|| { tail -40 /tmp/docker-install.log; exit 1; }; }"
)
# A Prime VM is lost (the sandbox terminates) under sustained disk writes of a few hundred
# MB/s, which pulling and building multi-GB service images reaches. The daemon and every
# container it starts write through one cgroup capped at this rate, so they also cannot
# fill the page cache faster than PRIME_CACHE_DROP empties it.
PRIME_WRITE_BPS = 100 * 2**20
PRIME_DOCKERD = (
    "dev=$(awk '$5==\"/\" {print $3; exit}' /proc/self/mountinfo) && "
    'echo "+io +memory +cpu +pids" > /sys/fs/cgroup/cgroup.subtree_control && '
    "mkdir -p /sys/fs/cgroup/vf-docker/daemon /sys/fs/cgroup/vf-docker/containers && "
    'echo "+io +memory +cpu +pids" > /sys/fs/cgroup/vf-docker/cgroup.subtree_control && '
    'echo "+io +memory +cpu +pids" > /sys/fs/cgroup/vf-docker/containers/cgroup.subtree_control && '
    f'echo "$dev wbps={PRIME_WRITE_BPS}" > /sys/fs/cgroup/vf-docker/io.max && '
    "echo $$ > /sys/fs/cgroup/vf-docker/daemon/cgroup.procs && "
    "exec dockerd --host=unix:///var/run/docker.sock --cgroup-parent=/vf-docker/containers"
)
# A Prime VM is also lost when its page cache fills the guest's memory: cached file pages
# stay backed by the host, which only reclaims free pages, and pulling and building multi-GB
# images fills the cache in minutes. The host keeps dropping clean cache while the VM lives.
PRIME_CACHE_DROP = "while sleep 5; do sync; echo 1 > /proc/sys/vm/drop_caches; done"
# The project's name on its host VM, so a host restored from a checkpoint brings the same
# containers (and the files they held) back up instead of creating new ones.
PROJECT_FILE = "/var/lib/vf-compose/project"
# A host image can ship `docker save` archives here so services need no registry.
# Each archive is removed once loaded, returning its disk space to the services.
COMPOSE_IMAGES = "/opt/verifiers/compose-images"
LOAD_IMAGES = (
    f'for f in {COMPOSE_IMAGES}/*.tar; do [ -e "$f" ] || continue; '
    'docker load -q -i "$f" && rm -f "$f" || exit 1; done'
)


@dataclass(frozen=True)
class ComposeSpec:
    """Task requirements described by a Docker Compose file and its build inputs.

    Services are the named containers in that file; `workspace` selects the one
    used for task commands and files. Deployment manages their running lifetimes.
    """

    directory: str
    file: str | None = "environment/docker-compose.yaml"
    """The task's Compose file, relative to `directory`; None when the task authors
    none and only `services` join its workspace."""
    workspace: str = "main"
    host_image: str | None = None
    image_override: bool = False
    services: Mapping[str, dict] = field(default_factory=dict)
    """Services the framework adds beside the authored ones (a task's tool sidecars),
    as Compose service definitions. A name the task's file also declares is refused."""
    registry_mirrors: tuple[str, ...] = ()
    """Registry mirrors for the provider VM's Docker daemon; local Docker uses its
    own daemon configuration."""


@asynccontextmanager
async def compose_services(
    config: RuntimeConfig,
    spec: ComposeSpec,
    environment: dict[str, str],
    *,
    trust_compose: bool = False,
    setup_timeout: float | None = None,
) -> AsyncIterator[
    tuple[dict[str, DockerBackend], Callable[[], Awaitable[str]], RuntimeBackend | None]
]:
    """Own one Compose attempt and lend its services until the context exits."""
    if not isinstance(config, (DockerConfig, PrimeConfig, ModalConfig)):
        raise TypeError("Compose requires Docker, Prime VM or Modal VM")
    local = isinstance(config, DockerConfig)
    if local and spec.registry_mirrors:
        raise ValueError(
            "Compose registry mirrors require a provider VM; configure the local Docker daemon directly"
        )
    if local and not trust_compose:
        raise ValueError(
            "Local Compose tasks can access host files and Docker privileges; "
            "only run trusted tasks with --env.agent.deployment.trust-compose"
        )
    if config.gpu:
        raise ValueError("Compose currently supports CPU tasks")
    if local and config.network_restricted:
        raise ValueError("Compose on local Docker requires public networking")
    if isinstance(config, ModalConfig) and not config.network_access:
        raise ValueError("Compose on Modal requires network_access=True")
    project = ["docker", "compose", "--project-name", f"vf-{uuid.uuid4().hex}"]
    compose_argv = list(project)
    compose_env = {
        key: os.environ[key] for key in DOCKER_ENV if local and key in os.environ
    }
    host_backend = None
    prepared_images: dict[str, str] | None = None
    stack = AsyncExitStack()

    async def run_host(*argv: str) -> ProgramResult:
        if host_backend is None:
            return await cli(*argv, env=compose_env)
        return await host_backend.bind().run(list(argv), compose_env)

    async def compose(*args: str, timeout: float | None = None) -> str:
        async with asyncio.timeout(timeout):
            result = await run_host(*compose_argv, *args)
        if result.exit_code:
            raise SandboxError(
                f"Compose {args[0]} failed: {result.stderr or result.stdout}"
            )
        return result.stdout

    def down() -> None:
        # Compose finds the project by its labels; the atexit backstop stays until
        # removal succeeds.
        subprocess.run(
            [*project, "down", "--volumes", "--remove-orphans"],
            env=compose_env,
            capture_output=True,
            timeout=60,
            check=True,
        )
        atexit.unregister(down)

    async def release() -> None:
        # Like a provider sandbox's deletion, a failed removal must not fail the
        # rollout or mask its error; the atexit backstop retries it.
        try:
            await asyncio.to_thread(down)
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
            detail = e.stderr.decode(errors="replace").strip() if e.stderr else ""
            logger.warning(
                "Compose project %s removal failed: %s %s", project[-1], e, detail
            )

    try:
        async with asyncio.timeout(setup_timeout):
            import yaml

            directory = Path(
                stack.enter_context(tempfile.TemporaryDirectory(prefix="vf-compose-"))
            )
            # Relative binds and builds write into this rollout's copy of the task.
            await run_shielded(
                asyncio.to_thread(
                    shutil.copytree,
                    Path(spec.directory).resolve(),
                    directory / "task",
                )
            )
            authored = (
                yaml.safe_load((directory / "task" / spec.file).read_text())
                if spec.file is not None
                else {"services": {}}
            )
            if clash := sorted(spec.services.keys() & authored["services"].keys()):
                raise SandboxError(f"Compose services {clash} are declared twice")
            for kind in ("volumes", "networks"):
                if any(
                    value and (value.get("name") or value.get("external"))
                    for value in authored.get(kind, {}).values()
                ):
                    raise SandboxError(f"Compose {kind} must use project-scoped names")
            if any(
                service.get("container_name")
                for service in authored["services"].values()
            ):
                raise SandboxError(
                    "Remove container_name so Compose can name each rollout's services"
                )
            base = {
                "services": {
                    spec.workspace: {
                        "image": config.image,
                        "command": [
                            "sh",
                            "-c",
                            "trap 'exit 0' TERM; while :; do sleep 3600 & wait $!; done",
                        ],
                    }
                }
            }
            if {"image", "build"} & authored["services"].get(spec.workspace, {}).keys():
                # A template default must not replace an authored image or skip its build.
                del base["services"][spec.workspace]["image"]
            (directory / "base.json").write_text(json.dumps(base))
            (directory / "env.json").write_text(
                json.dumps({"services": {spec.workspace: {"environment": environment}}})
            )
            (directory / "services.json").write_text(
                json.dumps({"services": dict(spec.services)})
            )

            root = str(directory)
            if not local:
                host_config = VM_HOST[type(config)]
                if spec.host_image is not None:
                    host_config = {**host_config, "image": spec.host_image}
                host_backend = await stack.enter_async_context(
                    provision_backend(config.model_copy(update=host_config))
                )
                if isinstance(config, PrimeConfig):
                    install = await host_backend.bind().run(
                        ["sh", "-c", INSTALL_DOCKER], {}
                    )
                    if install.exit_code:
                        raise SandboxError(
                            f"Docker bootstrap failed: {install.stderr} {install.stdout}"
                        )
                mirrors = [
                    arg
                    for mirror in spec.registry_mirrors
                    for arg in ("--registry-mirror", mirror)
                ]
                await host_backend.bind().run_background(
                    [
                        "sh",
                        "-c",
                        PRIME_DOCKERD + (" " + shlex.join(mirrors) if mirrors else ""),
                    ]
                    if isinstance(config, PrimeConfig)
                    else ["dockerd", "--host=unix:///var/run/docker.sock", *mirrors],
                    {},
                    "/tmp/dockerd.log",
                )
                if isinstance(config, PrimeConfig):
                    await host_backend.bind().run_background(
                        ["sh", "-c", PRIME_CACHE_DROP], {}, "/tmp/cache-drop.log"
                    )
                if host_backend.restored:
                    saved = await run_host("cat", PROJECT_FILE)
                    if saved.exit_code or not saved.stdout.strip():
                        raise SandboxError("restored Compose host has no project name")
                    project[-1] = compose_argv[project.index("--project-name") + 1] = (
                        saved.stdout.strip()
                    )
                else:
                    saved = await run_host(
                        "sh",
                        "-c",
                        f"mkdir -p {shlex.quote(str(Path(PROJECT_FILE).parent))} && "
                        f"printf %s {shlex.quote(project[-1])} > {shlex.quote(PROJECT_FILE)}",
                    )
                    if saved.exit_code:
                        raise SandboxError(
                            f"Recording the Compose project failed: {saved.stderr}"
                        )
                # VM daemon readiness shares the caller's provisioning/setup budget.
                # A cold or restored host can legitimately need more than one minute.
                while (await run_host("docker", "info")).exit_code:
                    await asyncio.sleep(1)
                loaded = await run_host("sh", "-c", LOAD_IMAGES)
                if loaded.exit_code:
                    raise SandboxError(
                        f"Loading host images failed: {loaded.stderr or loaded.stdout}"
                    )
                manifest = await run_host(
                    "sh",
                    "-c",
                    f"if [ -f {COMPOSE_IMAGES}/manifest.json ]; then cat {COMPOSE_IMAGES}/manifest.json; fi",
                )
                if manifest.exit_code:
                    raise SandboxError("Reading prepared Compose image manifest failed")
                if manifest.stdout.strip():
                    prepared = json.loads(manifest.stdout)
                    if prepared.get("version") != 1 or not isinstance(
                        prepared.get("services"), dict
                    ):
                        raise SandboxError("Invalid prepared Compose image manifest")
                    prepared_images = prepared["services"]
                    for service, image in prepared_images.items():
                        if not isinstance(image, str) or not image.startswith(
                            "vf-prepared/"
                        ):
                            raise SandboxError(
                                f"Invalid prepared image for {service!r}"
                            )
                        inspected = await run_host("docker", "image", "inspect", image)
                        if inspected.exit_code:
                            raise SandboxError(
                                f"Missing prepared Compose image for {service!r}: {image}"
                            )
                root = "/vf-deployment"
                await host_backend.bind().write(
                    "/tmp/vf-deployment.tar.gz",
                    pack_directory(directory),
                )
                staged = await run_host(
                    "sh",
                    "-c",
                    f"mkdir -p {shlex.quote(root)} && tar -xzf /tmp/vf-deployment.tar.gz -C {shlex.quote(root)}",
                )
                if staged.exit_code:
                    raise SandboxError(
                        f"Compose environment staging failed: {staged.stderr}"
                    )
            probe = await run_host("docker", "compose", "version")
            if probe.exit_code:
                raise SandboxError(
                    "Docker Compose v2 (`docker compose`) is required on the Docker "
                    "host; install the Compose v2 plugin or use a host image that "
                    "includes it. Standalone `docker-compose` v1 is unsupported."
                )
            project_dir = (
                f"{root}/task/{Path(spec.file).parent}"
                if spec.file is not None
                else f"{root}/task"
            )
            compose_argv += ["--project-directory", project_dir]
            for file in (
                "base.json",
                *([f"task/{spec.file}"] if spec.file is not None else []),
                "services.json",
                "env.json",
            ):
                compose_argv += ["-f", f"{root}/{file}"]
            compose_env.update(
                (key, value)
                for key, value in environment.items()
                if key not in DOCKER_ENV and not key.startswith(("DOCKER_", "COMPOSE_"))
            )
            compose_env.update(
                {
                    "MAIN_IMAGE_NAME": project[-1],
                    "CONTEXT_DIR": project_dir,
                    "PREBUILT_IMAGE_NAME": config.image,
                }
            )

            # Validate authored services after interpolation, before adding our own.
            resolved = json.loads(await compose("config", "--format", "json"))
            services = resolved["services"]
            for service in services.values():
                if service.get("network_mode") == "host":
                    raise SandboxError("Compose requires an isolated service network")
                if service.get("gpus") or (
                    service.get("deploy", {})
                    .get("resources", {})
                    .get("reservations", {})
                    .get("devices")
                ):
                    raise SandboxError("Compose currently supports CPU tasks")
                if service.get("scale", 1) != 1 or (
                    service.get("deploy", {}).get("replicas", 1) != 1
                ):
                    raise SandboxError("Compose requires one container per service")
                if any(
                    port.get("published") not in (None, "", 0, "0")
                    for port in service.get("ports", [])
                ):
                    raise SandboxError(
                        "Compose ports must use dynamically assigned host ports"
                    )
            # Compose rejects cycles, so this ends at the owner of main's network.
            owner = spec.workspace
            while (mode := services[owner].get("network_mode", "")).startswith(
                "service:"
            ):
                owner = mode.removeprefix("service:")

            main = config.model_dump(
                include={"image", "workdir"}, exclude_defaults=True, exclude_none=True
            )
            if spec.image_override:
                main["image"] = config.image
            if "workdir" in main:
                main["working_dir"] = main.pop("workdir")
            overlay = {"services": {spec.workspace: main}}
            # Port publication belongs to the service that owns main's network.
            publish = overlay["services"].setdefault(owner, {})
            if local:
                if config.cpu is not None:
                    main["cpus"] = config.cpu
                if config.memory is not None:
                    main["mem_limit"] = f"{config.memory}g"
                publish["ports"] = [f"127.0.0.1::{SERVICE_PORT}"]
                if sys.platform != "linux":
                    publish["extra_hosts"] = {"host.docker.internal": "host-gateway"}
            elif isinstance(config, ModalConfig):
                publish["ports"] = [f"{SERVICE_PORT}:{SERVICE_PORT}"]
            if prepared_images is not None:
                if prepared_images.keys() != services.keys():
                    raise SandboxError(
                        "Prepared Compose image services do not match this deployment"
                    )
                for service, image in prepared_images.items():
                    overlay["services"].setdefault(service, {}).update(
                        image=image, pull_policy="never"
                    )
            if host_backend is None:
                (directory / "overlay.json").write_text(json.dumps(overlay))
            else:
                await host_backend.bind().write(
                    f"{root}/overlay.json", json.dumps(overlay).encode()
                )
            compose_argv += ["-f", f"{root}/overlay.json"]

            if local:
                atexit.register(down)
                stack.push_async_callback(release)
            # Cancellation kills the local CLI; the stack removes the project or VM.
            await compose(
                "up",
                "--detach",
                "--wait",
                *(
                    ["--no-build", "--pull", "never"]
                    if prepared_images is not None
                    else []
                ),
            )
            containers = await compose(
                "ps", "--all", "--format", "{{.ID}} {{.Service}}"
            )
            service_url = None
            if local:
                published = await compose("port", owner, str(SERVICE_PORT))
                service_url = "http://" + next(
                    address
                    for address in published.splitlines()
                    if address.startswith("127.0.0.1:")
                )
            backends: dict[str, DockerBackend] = {}
            for line in containers.splitlines():
                container, service = line.split()
                backends[service] = await stack.enter_async_context(
                    DockerBackend.attach(
                        config,
                        container,
                        host_backend=host_backend,
                        service_url=service_url if service == spec.workspace else None,
                    )
                )
            if spec.workspace not in backends:
                raise SandboxError(
                    f"Compose requires workspace service {spec.workspace!r}"
                )
        yield (
            backends,
            partial(compose, "stop", spec.workspace, timeout=60),
            host_backend,
        )
    finally:
        await run_shielded(stack.aclose())


def pack_directory(directory: Path) -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        for item in directory.iterdir():
            archive.add(item, arcname=item.name)
    return buffer.getvalue()
