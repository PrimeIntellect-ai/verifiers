"""Task services on local Docker Compose or a Prime VM, from Linux or macOS."""

import asyncio
import atexit
import io
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import dataclass
from functools import partial
from pathlib import Path

from verifiers.v1.errors import SandboxError
from verifiers.v1.runtimes import (
    DockerBackend,
    DockerConfig,
    PrimeBackend,
    PrimeConfig,
    provision_backend,
)
from verifiers.v1.runtimes.backends.docker import cli
from verifiers.v1.runtimes.base import SERVICE_PORT
from verifiers.v1.utils.aio import run_shielded
from verifiers.v1.utils.logging import get_logger

logger = get_logger(__name__)

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
VM_HOST_IMAGE = "python:3.11-slim-trixie"
INSTALL_DOCKER = (
    "command -v dockerd >/dev/null && docker compose version >/dev/null 2>&1 || "
    "{ export DEBIAN_FRONTEND=noninteractive; apt-get update -qq && "
    "apt-get install -y -qq --no-install-recommends docker.io docker-cli "
    "docker-compose iptables > /tmp/docker-install.log 2>&1 "
    "|| { tail -40 /tmp/docker-install.log; exit 1; }; }"
)
# A host image can ship docker-save archives, avoiding registry access during setup.
COMPOSE_IMAGES = "/opt/verifiers/compose-images"
LOAD_IMAGES = (
    f'for f in {COMPOSE_IMAGES}/*.tar; do [ -e "$f" ] || continue; '
    'docker load -q -i "$f" && rm -f "$f" || exit 1; done'
)
# Harbor's Compose templates mount its log directories as `${HOST_<NAME>_PATH}:
# ${ENV_<NAME>_PATH}`. Harbor binds one host directory per name, so every service
# that mounts a name shares it; a project volume per name keeps that sharing.
HARBOR_LOG_MOUNTS = {
    "VERIFIER_LOGS": "/logs/verifier",
    "AGENT_LOGS": "/logs/agent",
    "ARTIFACTS": "/logs/artifacts",
}


@dataclass(frozen=True)
class ComposeSpec:
    """Task requirements described by a Docker Compose file and its build inputs.

    Services are the named containers in that file; `workspace` selects the one
    used for task commands and files. Deployment manages their running lifetimes.
    """

    directory: str
    file: str | None = "environment/docker-compose.yaml"
    """None builds directory/Dockerfile as a single workspace service."""
    workspace: str = "main"
    image_override: bool = False
    build_dockerfile: bool = False
    host_image: str | None = None
    """Compose VM image, optionally with Docker and baked-in service archives."""


def _harbor_volume(name: str) -> str:
    return "harbor-" + name.lower().replace("_", "-")


def _archive_directory(directory: Path) -> bytes:
    with io.BytesIO() as buffer:
        with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
            archive.add(directory, arcname=".")
        return buffer.getvalue()


@dataclass(frozen=True)
class ComposeModel:
    """Solver topology and build inputs captured before workspace execution."""

    definition: dict
    implicit_build: bool


def _compose_environment_sources(
    entry: Path, task_dir: Path
) -> dict[Path, tuple[bytes, bytes]]:
    """Keep Compose's source layout while removing service topology."""
    import yaml

    loader = yaml.SafeLoader("")

    def mapping(node):
        if not isinstance(node, yaml.MappingNode):
            return {}
        loader.flatten_mapping(node)
        return {key.value: value for key, value in node.value}

    sources = {}
    pending = [(entry, entry.parent)]
    while pending:
        path, project_dir = pending.pop()
        path = path.resolve()
        if path in sources or not path.is_relative_to(task_dir) or not path.is_file():
            continue
        original = path.read_bytes()
        document = yaml.compose(original)
        if not isinstance(document, yaml.MappingNode):
            continue
        definition = mapping(document)
        # Leave interpolation, env-file precedence and YAML tags to Compose.
        includes = definition.get("include")
        if isinstance(includes, yaml.SequenceNode):
            for item in includes.value:
                options = mapping(item)
                files = options.get("path", item)
                paths = files.value if isinstance(files, yaml.SequenceNode) else [files]
                included = [project_dir / file.value for file in paths]
                if included:
                    scope = options.get("project_directory")
                    include_dir = (
                        project_dir / scope.value if scope else included[0].parent
                    )
                    pending.extend((file, include_dir) for file in included)
        for service in mapping(definition.get("services")).values():
            if not isinstance(service, yaml.MappingNode):
                continue
            fields = mapping(service)
            extends = mapping(fields.get("extends"))
            if file := extends.get("file"):
                pending.append((project_dir / file.value, project_dir))
            service.value = [
                (key, value)
                for key, value in service.value
                if key.value in {"environment", "env_file", "extends"}
            ]
        document.value = [
            (key, value)
            for key, value in document.value
            if key.value in {"services", "include"}
        ]
        sources[path] = original, yaml.serialize(document).encode()
    return sources


@asynccontextmanager
async def compose_project(
    config: DockerConfig | PrimeConfig,
    spec: ComposeSpec,
    environment: dict[str, str],
    *,
    host_backend: PrimeBackend | None = None,
    model: ComposeModel | None = None,
) -> AsyncIterator["ComposeProject"]:
    """Resolve a private copy of a task and own any local preparation containers."""
    if config.mounts:
        raise ValueError("Runtime mounts are not supported for Compose tasks")
    project = ["docker", "compose", "--project-name", f"vf-{uuid.uuid4().hex}"]
    compose_argv = list(project)
    compose_env = {
        key: os.environ[key]
        for key in DOCKER_ENV
        if host_backend is None and key in os.environ
    }
    stack = AsyncExitStack()
    started = False

    def down() -> None:
        # Find the project by its labels without reading the evaluator's .env.
        # The atexit backstop stays until removal succeeds.
        subprocess.run(
            [
                *project,
                "--env-file",
                "/dev/null",
                "down",
                "--volumes",
                "--remove-orphans",
            ],
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

    async def compose(*args: str, timeout: float | None = None) -> str:
        nonlocal started
        if host_backend is None and args[0] in {"create", "up"} and not started:
            started = True
            atexit.register(down)
            stack.push_async_callback(release)
        async with asyncio.timeout(timeout):
            result = (
                await cli(*compose_argv, *args, env=compose_env)
                if host_backend is None
                else await host_backend.run(
                    host_backend.bind(),
                    [*compose_argv, *args],
                    compose_env,
                    max_output_bytes=None,
                )
            )
        if result.exit_code:
            raise SandboxError(
                f"Compose {args[0]} failed: {result.stderr or result.stdout}"
            )
        return result.stdout

    try:
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
                symlinks=isinstance(config, PrimeConfig),
            )
        )
        task_dir = (directory / "task").resolve()
        compose_file = spec.file or "Dockerfile"
        if isinstance(config, PrimeConfig) and not (
            task_dir / compose_file
        ).resolve().is_relative_to(task_dir):
            raise SandboxError("Native Compose files must be inside the task")
        if spec.file is not None:
            authored = yaml.safe_load((task_dir / spec.file).read_text())
        else:
            authored = {"services": {spec.workspace: {}}}
        if (
            isinstance(config, PrimeConfig)
            and host_backend is None
            and (
                authored.get("include")
                or any(
                    service.get("extends")
                    for service in authored.get("services", {}).values()
                )
            )
        ):
            raise SandboxError("Native Compose does not support include or extends")
        for kind in ("volumes", "networks"):
            if host_backend is None and any(
                value and (value.get("name") or value.get("external"))
                for value in authored.get(kind, {}).values()
            ):
                raise SandboxError(f"Compose {kind} must use project-scoped names")
        # Native Compose renames containers on the build host and keeps the name as an alias.
        if isinstance(config, DockerConfig) and any(
            service.get("container_name")
            for service in authored.get("services", {}).values()
        ):
            raise SandboxError(
                "Remove container_name so Compose can name each rollout's services"
            )
        root = str(directory)
        if host_backend is not None:
            root = "/vf-deployment"
            archive = await asyncio.to_thread(_archive_directory, directory)
            await host_backend.bind().write("/vf-deployment.tar.gz", archive)
            staged = await host_backend.bind().run(
                [
                    "sh",
                    "-c",
                    f"mkdir -p {root} && tar -xzf /vf-deployment.tar.gz -C {root} && rm /vf-deployment.tar.gz",
                ],
                {},
            )
            if staged.exit_code:
                raise SandboxError(
                    f"Compose environment staging failed: {staged.stderr}"
                )
        probe = (
            await cli("docker", "compose", "version", env=compose_env)
            if host_backend is None
            else await host_backend.bind().run(["docker", "compose", "version"], {})
        )
        if probe.exit_code:
            raise SandboxError(
                "Docker Compose v2 (`docker compose`) is required on the Docker "
                "host; install the Compose v2 plugin. "
                "Standalone `docker-compose` v1 is unsupported."
            )
        project_dir = f"{root}/task/{Path(compose_file).parent}"
        compose_argv += ["--project-directory", project_dir]
        if isinstance(config, PrimeConfig):
            env_file = directory / "task" / Path(compose_file).parent / ".env"
            if not env_file.resolve().is_relative_to(task_dir):
                raise SandboxError("Native Compose .env must be inside the task")
            compose_argv += [
                "--env-file",
                f"{project_dir}/.env" if env_file.exists() else "/dev/null",
            ]
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
                "TEST_DIR": "/tests",
            }
        )
        for name, path in HARBOR_LOG_MOUNTS.items():
            compose_env[f"HOST_{name}_PATH"] = _harbor_volume(name)
            compose_env[f"ENV_{name}_PATH"] = path
        # Harbor's templates size services from these; Compose rejects them empty.
        if config.cpu is not None:
            compose_env["CPUS"] = f"{config.cpu:g}"
        if config.memory is not None:
            compose_env["MEMORY"] = f"{round(config.memory * 1024)}M"

        authored_file = f"task/{spec.file}" if spec.file else None
        if host_backend is not None and model is None and authored.get("include"):
            # Includes own their resources: resolve them before merging workspace
            # defaults and overrides, or Compose rejects duplicate service names.
            compose_argv += ["-f", f"{root}/{authored_file}"]
            authored = json.loads(
                await compose("config", "--no-consistency", "--format", "json")
            )
            del compose_argv[-2:]
            # Config emits null for an absent command; retain our workspace default
            # unless the included service declares a concrete command.
            main = authored.get("services", {}).get(spec.workspace, {})
            if main.get("command") is None:
                main.pop("command", None)
            authored_file = "authored.json"
            # `compose config` already escapes literal dollars for reuse as input.
            # Preserve that escaping when merging the flattened include model.
            await host_backend.bind().write(
                f"{root}/{authored_file}",
                json.dumps(authored).encode(),
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
            },
            "volumes": {_harbor_volume(name): {} for name in HARBOR_LOG_MOUNTS},
        }
        if {"image", "build"} & authored.get("services", {}).get(
            spec.workspace, {}
        ).keys():
            # A template default must not replace an authored image or skip its build.
            del base["services"][spec.workspace]["image"]
        elif (
            (spec.build_dockerfile or spec.file is None)
            and not spec.image_override
            and (task_dir / Path(compose_file).parent / "Dockerfile").is_file()
        ):
            del base["services"][spec.workspace]["image"]
            base["services"][spec.workspace]["build"] = {
                "context": ".",
                "dockerfile": "Dockerfile",
            }
        elif spec.file is None and not spec.image_override:
            raise SandboxError(f"No Dockerfile in {spec.directory}")
        # env.json is a Compose file: escape `$` so values reach the service verbatim.
        escaped = {key: value.replace("$", "$$") for key, value in environment.items()}
        definitions = {
            "base.json": base,
            "env.json": {"services": {spec.workspace: {"environment": escaped}}},
        }
        for file, definition in definitions.items():
            if host_backend is None:
                (directory / file).write_text(json.dumps(definition))
            else:
                await host_backend.bind().write(
                    f"{root}/{file}", json.dumps(definition).encode()
                )
        if model is None or authored_file is None:
            compose_argv += ["-f", f"{root}/base.json"]
        if authored_file is not None:
            compose_argv += ["-f", f"{root}/{authored_file}"]
        if host_backend is None:
            compose_argv += ["-f", f"{root}/env.json"]

        if isinstance(config, PrimeConfig) and host_backend is None:
            # Resolve filenames without loading them: task env files cannot read host secrets.
            unresolved = json.loads(
                await compose("config", "--no-env-resolution", "--format", "json")
            )
            for service in unresolved["services"].values():
                for env_file in service.get("env_file", []):
                    if not Path(env_file["path"]).resolve().is_relative_to(task_dir):
                        raise SandboxError(
                            "Native Compose env_file must be inside the task"
                        )
        implicit_build = "build" in base["services"][spec.workspace]
        if model is None:
            resolved = json.loads(await compose("config", "--format", "json"))
        else:
            assert host_backend is not None
            # The authored project may require solver inputs to resolve fields such
            # as ports or build arguments. Only its service env is needed here.
            try:
                phase_services = json.loads(
                    await compose("config", "--no-consistency", "--format", "json")
                )["services"]
            except SandboxError as exc:
                if authored_file is None or "error while interpolating " not in str(
                    exc
                ):
                    raise
                sources = await asyncio.to_thread(
                    _compose_environment_sources, directory / authored_file, task_dir
                )
                # Keep includes, extends and their .env scopes at the original
                # paths. Restore the files before Compose builds or starts anything.
                runtime = host_backend.bind()
                async with AsyncExitStack() as restore:
                    for path, (original, projection) in sources.items():
                        target = f"{root}/task/{path.relative_to(task_dir)}"
                        restore.push_async_callback(runtime.write, target, original)
                        await runtime.write(target, projection)
                    phase_services = json.loads(
                        await compose("config", "--no-consistency", "--format", "json")
                    ).get("services", {})
            resolved = {
                **model.definition,
                "services": {
                    name: {
                        **service,
                        "environment": phase_services.get(name, {}).get(
                            "environment", {}
                        ),
                    }
                    for name, service in model.definition["services"].items()
                },
            }
            implicit_build = model.implicit_build
            await host_backend.bind().write(
                f"{root}/model.json", json.dumps(resolved).encode()
            )
            compose_argv[compose_argv.index("-f") :] = ["-f", f"{root}/model.json"]
        if host_backend is not None:
            # The grader reuses the solver's resolved topology but receives its own env.
            compose_argv += ["-f", f"{root}/env.json"]
        yield ComposeProject(
            directory,
            compose_argv,
            compose_env,
            resolved,
            compose,
            implicit_build,
            root,
        )
    finally:
        await run_shielded(stack.aclose())


@dataclass
class ComposeProject:
    directory: Path
    argv: list[str]
    env: dict[str, str]
    config: dict
    run: Callable[..., Awaitable[str]]
    implicit_build: bool
    root: str


@asynccontextmanager
async def compose_services(
    config: DockerConfig | PrimeConfig,
    spec: ComposeSpec,
    environment: dict[str, str],
    *,
    trust_compose: bool = False,
    setup_timeout: float | None = None,
    model: ComposeModel | None = None,
) -> AsyncIterator[
    tuple[dict[str, DockerBackend], Callable[[], Awaitable[str]], ComposeModel | None]
]:
    """Own one Compose attempt and lend its services until the context exits."""
    local = isinstance(config, DockerConfig)
    if config.mounts:
        raise ValueError("Runtime mounts are not supported for Compose tasks")
    if not local and config.checkpoint:
        raise ValueError("Compose VMs require a host image, not a checkpoint")
    if local and not trust_compose:
        raise ValueError(
            "Local Compose tasks can access host files and Docker privileges; "
            "only run trusted tasks with --env.agent.deployment.trust-compose"
        )
    if local and config.network_restricted:
        raise ValueError(
            "Local Compose does not enforce network policies; use ServicesSpec on Prime"
        )
    if config.gpu:
        raise ValueError("Compose currently supports CPU tasks")
    stack = AsyncExitStack()
    try:
        async with asyncio.timeout(setup_timeout):
            host = (
                None
                if local
                else await stack.enter_async_context(provision_vm_host(config, spec))
            )
            project = await stack.enter_async_context(
                compose_project(
                    config, spec, environment, host_backend=host, model=model
                )
            )
            compose = project.run
            directory, compose_argv = project.directory, project.argv
            root = project.root
            # Validate authored services after interpolation, before adding our own.
            services = project.config["services"]
            for service in services.values():
                if local and service.get("network_mode") == "host":
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
                if local and any(
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

            # The VM project already resolved the authored image or build.
            main = config.model_dump(
                include={"image", "workdir"} if local else {"workdir"},
                exclude_defaults=True,
                exclude_none=True,
            )
            if spec.image_override:
                main["image"] = config.image
            elif project.implicit_build:
                main.pop("image", None)
            if "workdir" in main:
                main["working_dir"] = main.pop("workdir")
            overlay = {"services": {spec.workspace: main}}
            # Port publication belongs to the service that owns main's network.
            publish = overlay["services"].setdefault(owner, {})
            if local and config.cpu is not None:
                main["cpus"] = config.cpu
            if local and config.memory is not None:
                main["mem_limit"] = f"{config.memory}g"
            limits = {
                key: main[field]
                for key, field in (("cpus", "cpus"), ("memory", "mem_limit"))
                if field in main
            }
            if limits:
                main["deploy"] = {"resources": {"limits": limits}}
            if services[owner].get("network_mode") != "host":
                publish["ports"] = [
                    f"127.0.0.1::{SERVICE_PORT}" if local else str(SERVICE_PORT)
                ]
            if local and sys.platform != "linux":
                publish["extra_hosts"] = {"host.docker.internal": "host-gateway"}
            if host is None:
                (directory / "overlay.json").write_text(json.dumps(overlay))
            else:
                await host.bind().write(
                    f"{root}/overlay.json", json.dumps(overlay).encode()
                )
            compose_argv += ["-f", f"{root}/overlay.json"]

            # Cancellation stops the CLI; the stack removes the project.
            await compose("up", "--detach", "--wait")
            containers = await compose(
                "ps", "--all", "--format", "{{.ID}} {{.Service}}"
            )
            service_url = None
            host_service_port = SERVICE_PORT
            if services[owner].get("network_mode") != "host":
                published = await compose("port", owner, str(SERVICE_PORT))
                # Prefer IPv4 when both families have mappings, but accept an
                # IPv6-only Docker host. Local publication stays on IPv4 loopback.
                # Compose v2 emits :::port; newer releases bracket the host IP.
                prefixes = ("127.0.0.1:",) if local else ("0.0.0.0:", "[::]:", ":::")
                address = next(
                    (
                        address
                        for prefix in prefixes
                        for address in published.splitlines()
                        if address.startswith(prefix)
                    ),
                    None,
                )
                if address is None:
                    raise SandboxError(
                        f"Compose did not publish workspace port {SERVICE_PORT}"
                    )
                if local:
                    service_url = "http://" + address
                else:
                    host_service_port = int(address.rsplit(":", 1)[1])
            backends: dict[str, DockerBackend] = {}
            for line in containers.splitlines():
                container, service = line.split()
                backends[service] = await stack.enter_async_context(
                    DockerBackend.attach(
                        config,
                        container,
                        host_backend=host,
                        host_service_port=host_service_port
                        if service == spec.workspace
                        else None,
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
            ComposeModel(project.config, project.implicit_build)
            if host is not None
            else None,
        )
    finally:
        await run_shielded(stack.aclose())


@asynccontextmanager
async def provision_vm_host(
    config: PrimeConfig, spec: ComposeSpec
) -> AsyncIterator[PrimeBackend]:
    """Own a fresh Docker host, including partial bootstrap and cancelled setup."""
    host_config = config.model_copy(
        update={"image": spec.host_image or VM_HOST_IMAGE, "workdir": "/"}
    )
    async with provision_backend(host_config) as host:
        assert isinstance(host, PrimeBackend)
        runtime = host.bind()
        install = await runtime.run(["sh", "-c", INSTALL_DOCKER], {})
        if install.exit_code:
            raise SandboxError(
                f"Docker bootstrap failed: {install.stderr} {install.stdout}"
            )
        if (await runtime.run(["docker", "info"], {})).exit_code:
            await runtime.run_background(
                ["dockerd", "--host=unix:///var/run/docker.sock"],
                {},
                "/tmp/dockerd.log",
            )
        async with asyncio.timeout(60):
            while (await runtime.run(["docker", "info"], {})).exit_code:
                await asyncio.sleep(1)
        loaded = await runtime.run(["sh", "-c", LOAD_IMAGES], {})
        if loaded.exit_code:
            raise SandboxError(
                f"Loading host images failed: {loaded.stderr or loaded.stdout}"
            )
        yield host
