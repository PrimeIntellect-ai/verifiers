"""Resolve Compose on the build host; run its supported services as native Prime VMs."""

import asyncio
import fcntl
import hashlib
import io
import ipaddress
import json
import logging
import math
import re
import shlex
import tarfile
import time
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath

import httpx
from prime_sandboxes import AsyncImageClient
from prime_sandboxes.models import BuildImageRequest

from verifiers.v1.errors import SandboxError
from verifiers.v1.runtimes import PrimeConfig
from verifiers.v1.runtimes.backends.docker import cli
from verifiers.v1.runtimes.base import RuntimeBackend, RuntimeProcess
from verifiers.v1.runtimes.compose import ComposeSpec, compose_project
from verifiers.v1.runtimes.private_network import lan_host
from verifiers.v1.utils.aio import run_shielded
from verifiers.v1.utils.paths import CACHE_DIR

logger = logging.getLogger(__name__)

# Reject unsupported Compose fields before building images or provisioning VMs.
_SERVICE_FIELDS = {
    "build",
    "image",
    "pull_policy",
    "platform",
    "command",
    "entrypoint",
    "environment",
    "working_dir",
    "user",
    "healthcheck",
    "depends_on",
    "expose",
    "networks",
    "cpus",
    "mem_limit",
    "deploy",
    "shm_size",
    "cap_add",
    "volumes",
    "network_mode",
    "container_name",
    "hostname",
    # No effect: peers reach any port, and processes run without a terminal.
    "ports",
    "tty",
    "stdin_open",
}
_BUILD_FIELDS = {"context", "dockerfile", "args", "target", "no_cache", "pull"}
# Linux capability names in bit order (linux/capability.h).
_CAPABILITIES = (
    "CHOWN",
    "DAC_OVERRIDE",
    "DAC_READ_SEARCH",
    "FOWNER",
    "FSETID",
    "KILL",
    "SETGID",
    "SETUID",
    "SETPCAP",
    "LINUX_IMMUTABLE",
    "NET_BIND_SERVICE",
    "NET_BROADCAST",
    "NET_ADMIN",
    "NET_RAW",
    "IPC_LOCK",
    "IPC_OWNER",
    "SYS_MODULE",
    "SYS_RAWIO",
    "SYS_CHROOT",
    "SYS_PTRACE",
    "SYS_PACCT",
    "SYS_ADMIN",
    "SYS_BOOT",
    "SYS_NICE",
    "SYS_RESOURCE",
    "SYS_TIME",
    "SYS_TTY_CONFIG",
    "MKNOD",
    "LEASE",
    "AUDIT_WRITE",
    "AUDIT_CONTROL",
    "SETFCAP",
    "MAC_OVERRIDE",
    "MAC_ADMIN",
    "SYSLOG",
    "WAKE_ALARM",
    "BLOCK_SUSPEND",
    "AUDIT_READ",
    "PERFMON",
    "BPF",
    "CHECKPOINT_RESTORE",
)


# Bound task files travel with each rollout; larger inputs belong in an image.
_BIND_MAX_BYTES = 256 * 1024**2
# Mount sources stage here; no Compose target can contain it, as `/` is never one.
_STAGING = PurePosixPath("/.vf-compose")


@dataclass
class NativeMount:
    """A bind of task files, or a service-local volume seeded from the image."""

    target: str
    read_only: bool = False
    archive: bytes | None = None
    file: bool = False


@dataclass
class NativeService:
    config: PrimeConfig
    argv: list[str]
    env: dict[str, str]
    health: dict
    shell: list[str]
    definition: dict
    user: str | None = None
    stop_signal: str = "SIGTERM"
    internal: bool = False
    mounts: list[NativeMount] = field(default_factory=list)
    image_env: dict[str, str] = field(default_factory=dict)


def validate_native(project) -> None:
    config = project.config
    # Compose keeps top-level `x-*` extension blocks (YAML anchors); they configure nothing.
    if {key for key in config if not key.startswith("x-")} - {
        "name",
        "services",
        "networks",
        "volumes",
    }:
        raise SandboxError("Native Compose does not support secrets or configs")
    if any(set(volume) - {"name"} for volume in config.get("volumes", {}).values()):
        raise SandboxError("Native Compose supports project volumes without drivers")
    networks = config.get("networks", {})
    if any(
        set(network) - {"name", "internal", "ipam", "driver"}
        or network.get("ipam")
        or network.get("driver", "bridge") != "bridge"
        for network in networks.values()
    ):
        raise SandboxError(
            "Native Compose supports bridge project networks with optional internal isolation"
        )
    aliases = {}
    mounts: dict[str, set[tuple[str, str]]] = {}
    binds: list[tuple[str, str, Path, bool]] = []
    task = (project.directory / "task").resolve()
    for name, service in config["services"].items():
        unsupported = set(service) - _SERVICE_FIELDS
        if unsupported:
            raise SandboxError(
                f"Native Compose service {name!r} does not support: "
                + ", ".join(sorted(unsupported))
            )
        if service.get("platform", "linux/amd64") != "linux/amd64":
            raise SandboxError(f"Native Compose service {name!r} requires linux/amd64")
        build = service.get("build", {})
        if set(build) - _BUILD_FIELDS:
            raise SandboxError(
                f"Native Compose service {name!r} has unsupported build options"
            )
        if build and not Path(build["context"]).resolve().is_relative_to(
            (project.directory / "task").resolve()
        ):
            raise SandboxError(
                f"Native Compose build context for {name!r} must be inside the task"
            )
        if build and not (
            Path(build["context"]) / build.get("dockerfile", "Dockerfile")
        ).resolve().is_relative_to((project.directory / "task").resolve()):
            raise SandboxError(
                f"Native Compose Dockerfile for {name!r} must be inside the task"
            )
        deploy = service.get("deploy", {})
        resources = deploy.get("resources", {})
        # Compose adds empty sections, such as `placement: {}`, to every deploy block.
        if (
            {key for key, value in deploy.items() if value} - {"resources", "replicas"}
            or deploy.get("replicas", 1) != 1
            or {key for key, value in resources.items() if value} - {"limits"}
            or set(resources.get("limits", {})) - {"cpus", "memory"}
        ):
            raise SandboxError(
                f"Native Compose service {name!r} has unsupported deploy options"
            )
        for dependency, options in service.get("depends_on", {}).items():
            if (
                dependency not in config["services"]
                or options.get("restart")
                or not options.get("required", True)
                or options["condition"]
                not in {
                    "service_started",
                    "service_healthy",
                    "service_completed_successfully",
                }
            ):
                raise SandboxError(
                    f"Native Compose dependency {name!r} -> {dependency!r} is unsupported"
                )
        for volume in service.get("volumes", []):
            if PurePosixPath(volume["target"]).is_relative_to(_STAGING):
                raise SandboxError(
                    f"Native Compose service {name!r} cannot mount at {_STAGING}"
                )
            if volume["type"] == "bind":
                source = Path(volume["source"]).resolve()
                options = {
                    key for key, value in volume.get("bind", {}).items() if value
                }
                if not source.is_relative_to(task) or options - {
                    "create_host_path",
                    "selinux",
                }:
                    raise SandboxError(
                        f"Native Compose service {name!r} can bind only task files"
                    )
                binds.append(
                    (name, volume["target"], source, bool(volume.get("read_only")))
                )
                continue
            # Volumes copy image contents: no read-only, nocopy or subpath mount.
            if (
                volume["type"] != "volume"
                or volume.get("read_only")
                or any(volume.get("volume", {}).values())
            ):
                raise SandboxError(
                    f"Native Compose service {name!r} supports only task binds and plain volumes"
                )
            if volume.get("source"):
                mounts.setdefault(volume["source"], set()).add((name, volume["target"]))
        # Docker's default bridge network: no project network, so no service names.
        mode = service.get("network_mode")
        if mode not in {None, "bridge", "default"}:
            raise SandboxError(
                f"Native Compose service {name!r} does not support network_mode {mode!r}"
            )
        attached = service.get("networks", {})
        if (
            not attached
            and mode is None
            or set(attached) - set(networks)
            or any(set(options or {}) - {"aliases"} for options in attached.values())
        ):
            raise SandboxError(
                f"Native Compose service {name!r} requires project networks with optional aliases"
            )
        if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9.-]*", service.get("hostname", "x")):
            raise SandboxError(
                f"Native Compose service {name!r} has an invalid hostname"
            )
        for alias in [
            name,
            *([service["container_name"]] if "container_name" in service else []),
            *(
                alias
                for options in attached.values()
                for alias in (options or {}).get("aliases", [])
            ),
        ]:
            if (
                not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_.-]*", alias)
                or alias in {"localhost", "host.docker.internal"}
                or alias in aliases
                and aliases[alias] != name
            ):
                raise SandboxError(
                    f"Native Compose has an invalid or ambiguous alias: {alias!r}"
                )
            aliases[alias] = name
    # A volume one service mounts at one path is a directory on that service's VM disk.
    for volume, targets in mounts.items():
        names = sorted({name for name, _ in targets})
        if len(names) > 1:
            raise SandboxError(
                f"Native Compose services {', '.join(names)} share volume "
                f"{volume!r}; each service VM has its own disk"
            )
        if len(targets) > 1:
            raise SandboxError(
                f"Native Compose service {names[0]!r} mounts volume {volume!r} at "
                "more than one path"
            )
    # Each mount gets its own copy of bound files, so only read-only copies may overlap.
    for index, (name, target, source, read_only) in enumerate(binds):
        for other, other_target, other_source, other_read_only in binds[index + 1 :]:
            if not (read_only and other_read_only) and (
                source.is_relative_to(other_source)
                or other_source.is_relative_to(source)
            ):
                raise SandboxError(
                    f"Native Compose binds overlapping task files writable at "
                    f"{name}:{target} and {other}:{other_target}"
                )


async def prepare_native(config: PrimeConfig, spec: ComposeSpec, env: dict[str, str]):
    """Inspect stopped containers so Docker resolves image defaults and overrides."""
    if spec.workspace != "main" or config.checkpoint or config.gpu:
        raise SandboxError("Native Compose requires a CPU image workspace named 'main'")
    async with compose_project(config, spec, env) as project:
        validate_native(project)
        prepared = {
            **project.config,
            "networks": {},
            "services": {
                name: {
                    **{
                        key: value
                        for key, value in service.items()
                        if key != "networks"
                    },
                    "platform": "linux/amd64",
                    "network_mode": "none",
                }
                for name, service in project.config["services"].items()
            },
        }
        for name, service in project.config["services"].items():
            if "container_name" in service:
                # Container names are global on the build host; rollouts must not collide.
                prepared["services"][name]["container_name"] = (
                    f"{project.config['name']}-{name}"
                )
        main = prepared["services"]["main"]
        if spec.image_override:
            # The task image replaces an authored build; Compose pulls it only when missing.
            main["image"] = config.image
            main.pop("build", None)
            main.pop("pull_policy", None)
        if config.workdir is not None:
            main["working_dir"] = config.workdir
        prepared_path = project.directory / "native.json"
        prepared_path.write_text(json.dumps(prepared))
        project.argv[project.argv.index("-f") :] = ["-f", str(prepared_path)]
        # Stopped preparation containers need no networks; preserve the VM topology in project.config.
        await project.run("create")
        containers = await project.run(
            "ps", "--all", "--format", "{{.ID}} {{.Service}}"
        )
        ids = {
            name: container
            for container, name in map(str.split, containers.splitlines())
        }
        if set(ids) != set(project.config["services"]):
            raise SandboxError(
                "Compose did not create exactly one container per service"
            )
        result = await cli("docker", "inspect", *ids.values(), env=project.env)
        if result.exit_code:
            raise SandboxError(f"Compose container inspection failed: {result.stderr}")
        inspected = {
            item["Config"]["Labels"]["com.docker.compose.service"]: item
            for item in json.loads(result.stdout)
        }
        result = await cli(
            "docker",
            "image",
            "inspect",
            *dict.fromkeys(item["Image"] for item in inspected.values()),
            env=project.env,
        )
        if result.exit_code:
            raise SandboxError(f"Compose image inspection failed: {result.stderr}")
        images = json.loads(result.stdout)
        image_envs = {
            image["Id"]: dict(
                entry.split("=", 1) for entry in image["Config"].get("Env") or []
            )
            for image in images
        }
        image_sizes = {image["Id"]: image["Size"] for image in images}
        services, bound = {}, 0
        for name in ids:
            item = inspected[name]
            metadata = item["Config"]
            user = metadata.get("User") or None
            if user == "0":
                user = "root"
            if user and not re.fullmatch(r"[a-z_][a-z0-9_-]*[$]?", user):
                raise SandboxError(
                    f"Native Compose service {name!r} requires a named user without a group override"
                )
            argv = [*(metadata.get("Entrypoint") or []), *(metadata.get("Cmd") or [])]
            if not argv:
                raise SandboxError(f"Native Compose service {name!r} has no command")
            health = metadata.get("Healthcheck") or {}
            if health.get("Test", ["NONE"])[0] not in {"CMD", "CMD-SHELL", "NONE"}:
                raise SandboxError(
                    f"Native Compose service {name!r} has an unsupported health check"
                )
            definition = project.config["services"][name]
            # Bridge-mode services join no project network and keep their internet.
            definition.setdefault("networks", {})
            internal = bool(definition["networks"]) and all(
                project.config["networks"][network].get("internal", False)
                for network in definition["networks"]
            )
            if definition.get("cap_add") and user not in {None, "root"}:
                raise SandboxError(
                    f"Native Compose service {name!r} requires root for cap_add"
                )
            limits = definition.get("deploy", {}).get("resources", {}).get("limits", {})
            options = config.model_dump(
                exclude={"image", "workdir", "cpu", "memory", "disk"}
            )
            options.update(
                image=item["Image"], workdir=metadata.get("WorkingDir") or "/"
            )
            authored = {
                "cpu": definition.get("cpus", limits.get("cpus")),
                "memory": definition.get("mem_limit", limits.get("memory")),
            }
            for resource, scale in (("cpu", 1), ("memory", 1024**3)):
                default = PrimeConfig.model_fields[resource].default
                # The runtime config sizes main as it does task resources: an explicit
                # setting wins over Compose, which wins over the default.
                size = getattr(config, resource) if name == "main" else default
                if authored[resource] is not None and size == default:
                    # The VM also runs its OS: a container limit never shrinks it.
                    size = max(float(authored[resource]) / scale, default)
                options[resource] = size
            disk = (
                config.disk
                if name == "main"
                else PrimeConfig.model_fields["disk"].default
            )
            # The VM's disk holds its image, with room to write.
            options["disk"] = max(
                disk, math.ceil(image_sizes[item["Image"]] / 1024**3) + 1
            )
            if internal:
                options.update(allow=[], block=["*"])
            services[name] = NativeService(
                PrimeConfig(**options),
                argv,
                dict(value.split("=", 1) for value in metadata.get("Env") or []),
                health,
                metadata.get("Shell") or ["/bin/sh", "-c"],
                definition,
                user,
                metadata.get("StopSignal") or "SIGTERM",
                internal,
                await asyncio.to_thread(_mounts, definition, _BIND_MAX_BYTES - bound),
                image_env=image_envs[item["Image"]],
            )
            bound += sum(len(mount.archive or b"") for mount in services[name].mounts)
        for name, service in services.items():
            for dependency, options in service.definition.get("depends_on", {}).items():
                if options["condition"] == "service_healthy" and (
                    services[dependency].health.get("Test", ["NONE"])[0] == "NONE"
                ):
                    raise SandboxError(
                        f"Native Compose dependency {name!r} -> {dependency!r} requires a health check"
                    )
        # Validate every service before publishing any image.
        client = AsyncImageClient()
        try:
            images = {}
            for name, service in services.items():
                identity = f"{service.config.image}\n{service.config.workdir}"
                if identity not in images:
                    images[identity] = await _import_image(
                        client, project, ids[name], service.config
                    )
                service.config = service.config.model_copy(
                    update={"image": images[identity]}
                )
        finally:
            await run_shielded(client.client.aclose())
        return services


def _mounts(definition: dict, budget: int) -> list[NativeMount]:
    """A service's task binds and volumes, in Docker's mount order."""
    volumes = definition.get("volumes", [])
    targets = [PurePosixPath(volume["target"]) for volume in volumes]
    mounts = []
    for target, volume in zip(targets, volumes):
        if volume["type"] == "bind":
            source = Path(volume["source"]).resolve()
            # Nested mounts hide these files on Docker; they never reach the VM.
            hidden = [
                other.relative_to(target)
                for other in targets
                if other != target and other.is_relative_to(target)
            ]
            archive = _archive(source, hidden, budget)
            budget -= len(archive)
            mounts.append(
                NativeMount(
                    str(target),
                    bool(volume.get("read_only")),
                    archive,
                    source.is_file(),
                )
            )
        else:
            # Exporting a stopped container omits newly created volume directories.
            mounts.append(NativeMount(str(target)))
    return sorted(
        mounts, key=lambda mount: (len(PurePosixPath(mount.target).parts), mount.target)
    )


def _archive(source: Path, hidden: list[PurePosixPath], budget: int) -> bytes:
    """Pack a bind's source, including its own mode and owner as Docker shows them."""
    buffer = io.BytesIO()

    def include(info: tarfile.TarInfo) -> tarfile.TarInfo | None:
        if any(PurePosixPath(info.name).is_relative_to(path) for path in hidden):
            return None
        # Checked before each file is read, so oversized binds never fill memory.
        if buffer.tell() + info.size > budget:
            raise SandboxError(
                "Native Compose binds more than 256 MiB of task files; build them into an image"
            )
        return info

    with tarfile.open(fileobj=buffer, mode="w") as archive:
        if source.exists():
            archive.add(
                source, arcname="file" if source.is_file() else ".", filter=include
            )
        # Docker creates a missing source as an empty directory.
    return buffer.getvalue()


async def _mount(runtime, name: str, mounts: list[NativeMount]) -> None:
    """Mount task binds and initialized service volumes as Docker would."""
    unpack, attach, protect = [f"mkdir -p {_STAGING}"], [], []
    for index, mount in enumerate(mounts):
        source, target = f"{_STAGING}/{index}", shlex.quote(mount.target)
        if mount.archive is None:
            # Docker seeds a new volume from the image's directory, owner and mode.
            unpack.append(
                f"if [ -d {target} ]; then cp -a {target}/. {source}; else mkdir {source}; fi"
            )
        else:
            await runtime.write(f"{source}.tar", mount.archive)
            unpack.append(
                f"mkdir -p {source} && tar --numeric-owner -xf {source}.tar -C {source} && rm {source}.tar"
            )
        if mount.file:
            source += "/file"
            attach.append(f'mkdir -p "$(dirname {target})" && touch {target}')
        else:
            attach.append(f"mkdir -p {target}")
        attach.append(f"mount --bind {source} {target}")
        if mount.read_only:
            protect.append(f"mount -o remount,bind,ro {target}")
    # Read-only last, so nested mount points can still be made; then hide the staged
    # copies, which would otherwise bypass read-only mounts.
    script = " && ".join(
        [*unpack, *attach, *protect, f"mount -t tmpfs tmpfs {_STAGING}"]
    )
    result = await runtime.run(["sh", "-c", script], {})
    if result.exit_code:
        raise SandboxError(
            f"Native Compose could not mount task files on {name!r}: {result.stderr}"
        )


async def _import_image(client, project, container: str, config: PrimeConfig) -> str:
    """Cache a private VM image by the exact local image and container workdir."""
    identity = f"{config.image}\n{config.workdir}"
    name = "vf-compose-" + hashlib.sha256(identity.encode()).hexdigest()
    locks = CACHE_DIR / "compose-images"
    locks.mkdir(parents=True, exist_ok=True)
    # Env-server workers share the build cache; only one prepares a given image.
    with (locks / name).open("a") as lock:
        while True:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                await asyncio.sleep(1)
        async with asyncio.timeout(1800):
            while True:
                existing = [
                    image
                    for image in (await client.list(search=name)).data
                    if image.image_name == name
                    and image.image_tag == "v4"
                    and image.artifact_type == "VM_SANDBOX"
                ]
                for image in existing:
                    if image.status == "COMPLETED" and image.display_ref:
                        return image.display_ref
                # Another host or an earlier rollout is building it; a build ends on
                # its own. A PENDING upload may be abandoned, so it is not awaited.
                if not any(image.status == "BUILDING" for image in existing):
                    break
                await asyncio.sleep(5)
        rootfs = project.directory / "rootfs.tar"
        result = await cli(
            "docker", "export", "--output", str(rootfs), container, env=project.env
        )
        if result.exit_code:
            raise SandboxError(f"Compose filesystem export failed: {result.stderr}")
        dockerfile = project.directory / "Dockerfile"
        # Prime's command API launches /bin/bash, which minimal Alpine images omit.
        dockerfile.write_text(
            "FROM scratch\nADD rootfs.tar /\n"
            "RUN test -x /bin/bash || apk add --no-cache bash\n"
        )
        context = project.directory / "context.tar.gz"

        def package():
            with tarfile.open(context, "w:gz") as archive:
                archive.add(dockerfile, arcname="Dockerfile")
                archive.add(rootfs, arcname="rootfs.tar")

        await run_shielded(asyncio.to_thread(package))
        build = await client.initiate_build(
            BuildImageRequest(
                image_name=name,
                image_tag="v4",
                visibility="PRIVATE",
                team_id=client.client.config.team_id,
            )
        )

        def upload():
            try:
                with context.open("rb") as source:
                    response = httpx.put(
                        build.upload_url,
                        content=source,
                        headers={"Content-Type": "application/octet-stream"},
                        timeout=600,
                    )
            except httpx.HTTPError as error:
                # A presigned upload URL is a credential, not useful error context.
                raise SandboxError(
                    f"Compose image upload failed: {type(error).__name__}"
                ) from None
            if not response.is_success:
                raise SandboxError(
                    f"Compose image upload failed with HTTP {response.status_code}"
                )

        await run_shielded(asyncio.to_thread(upload))
        await client.start_build(build.build_id)
        async with asyncio.timeout(1800):
            while True:
                status = await client.get_build_status(build.build_id)
                if status["status"] == "COMPLETED":
                    return build.full_image_path
                if status["status"] in {"FAILED", "CANCELLED"}:
                    raise SandboxError(
                        f"Compose VM image {name} build {status['status']}: {status.get('errorMessage', '')}"
                    )
                await asyncio.sleep(5)


@dataclass
class NativeProcesses:
    services: dict[str, NativeService]
    processes: dict[str, RuntimeProcess] = field(default_factory=dict)

    async def start(self, deployment) -> None:
        for name, backend in deployment.target_backends.items():
            service = self.services[name]
            runtime = backend.bind().with_user(None).with_env({})
            peers = {
                peer: shared
                for peer, other in self.services.items()
                if (
                    shared := service.definition["networks"].keys()
                    & other.definition["networks"].keys()
                )
            }
            hosts = {
                lan_host(deployment.target_backends[peer]): peer
                for peer in peers
                if peer != name
            }
            # LAN DNS returns peer-specific proxy addresses, not local interfaces.
            # Prime forwards incoming TCP to loopback listeners, so self aliases
            # remain bindable and reachable by peers without resolving self in DNS.
            addresses = {name: "127.0.0.1"}
            if hosts:
                result = await runtime.run(["getent", "hosts", *hosts], {})
                if result.exit_code:
                    raise SandboxError(
                        f"Native Compose could not resolve peers on {name!r}: {result.stderr}"
                    )
                for line in result.stdout.splitlines():
                    address, host, *_ = line.split()
                    addresses[hosts[host]] = str(ipaddress.ip_address(address))
                if addresses.keys() != peers.keys():
                    raise SandboxError(
                        f"Native Compose could not resolve every peer on {name!r}"
                    )
            names = []
            for peer, shared in peers.items():
                definition = self.services[peer].definition
                aliases = [
                    *(
                        [definition["container_name"]]
                        if "container_name" in definition
                        else []
                    ),
                    *(
                        alias
                        for network in sorted(shared)
                        for alias in (definition["networks"][network] or {}).get(
                            "aliases", []
                        )
                    ),
                ]
                names.append(
                    f"{addresses[peer]} {' '.join(dict.fromkeys([peer, *aliases]))}"
                )
            hostname = service.definition.get("hostname")
            if hostname:
                names.append(f"127.0.0.1 {hostname}")
            host_entries = "\n".join(names) + "\n"
            backend.env = service.env
            backend.user = service.user
            if service.internal:
                # Internal-only services have no internet, including during setup.
                await backend.prepare_execution([])
            install = f"printf %s {shlex.quote(host_entries)} >> /etc/hosts"
            if hostname:
                install += (
                    f" && printf %s {shlex.quote(hostname)} > /proc/sys/kernel/hostname"
                )
            result = await runtime.run(["sh", "-c", install], {})
            if result.exit_code:
                raise SandboxError(
                    f"Native Compose could not install service names on {name!r}"
                )
            if service.mounts:
                await _mount(runtime, name, service.mounts)
            if size := service.definition.get("shm_size"):
                result = await runtime.run(
                    ["mount", "-o", f"remount,size={int(size)}", "/dev/shm"], {}
                )
                if result.exit_code:
                    raise SandboxError(
                        f"Native Compose could not resize /dev/shm on {name!r}: {result.stderr}"
                    )
            if requested := service.definition.get("cap_add"):
                result = await runtime.run(["cat", "/proc/self/status"], {})
                effective = int(
                    next(
                        line.split()[1]
                        for line in result.stdout.splitlines()
                        if line.startswith("CapEff:")
                    ),
                    16,
                )
                names = {cap.upper().removeprefix("CAP_") for cap in requested}
                # Docker already rejected names it does not know.
                missing = [
                    cap
                    for bit, cap in enumerate(_CAPABILITIES)
                    if ("ALL" in names or cap in names) and not effective >> bit & 1
                ]
                if missing:
                    raise SandboxError(
                        f"Native Compose VM {name!r} lacks {', '.join(missing)}"
                    )
        started = {name: asyncio.Event() for name in self.services}
        healthy = {}
        completed = {
            dependency
            for service in self.services.values()
            for dependency, options in service.definition.get("depends_on", {}).items()
            if options["condition"] == "service_completed_successfully"
        }
        completion_only = completed - {
            dependency
            for service in self.services.values()
            for dependency, options in service.definition.get("depends_on", {}).items()
            if options["condition"] == "service_healthy"
        }

        async def launch(name, service):
            for dependency, options in service.definition.get("depends_on", {}).items():
                await started[dependency].wait()
                if options["condition"] == "service_healthy":
                    await healthy[dependency]
                elif options["condition"] == "service_completed_successfully" and (
                    code := await self.processes[dependency].wait()
                ):
                    raise SandboxError(
                        f"Native Compose dependency {dependency!r} exited with {code}"
                    )
            runtime = deployment.target(name)
            # Output stays in the VM; streaming a daemon's logs would grow controller memory.
            self.processes[name] = await runtime.exec(
                [
                    "sh",
                    "-c",
                    'echo $$ > /tmp/vf-compose.pid; exec "$@" > /tmp/vf-compose.log 2>&1 < /dev/null',
                    "sh",
                    *service.argv,
                ],
                {},
            )
            started[name].set()

        async def ready(name, service):
            await started[name].wait()
            test = service.health.get("Test", ["NONE"])
            # Completion-only jobs are ready when they exit successfully, not healthy.
            if test[0] == "NONE" or name in completion_only:
                return
            argv = [*service.shell, test[1]] if test[0] == "CMD-SHELL" else test[1:]
            interval = (service.health.get("Interval") or 30_000_000_000) / 1e9
            start_interval = (
                service.health.get("StartInterval") or 5_000_000_000
            ) / 1e9
            timeout = (service.health.get("Timeout") or 30_000_000_000) / 1e9
            grace = service.health.get("StartPeriod", 0) / 1e9
            retries = service.health.get("Retries") or 3
            since = time.monotonic()
            failures = 0
            runtime = deployment.target(name)
            while failures < retries:
                await asyncio.sleep(
                    start_interval if time.monotonic() - since < grace else interval
                )
                if (code := await self.processes[name].poll()) is not None:
                    raise SandboxError(
                        f"Native Compose service {name!r} exited with {code} before becoming healthy"
                    )
                checked_at = time.monotonic()
                try:
                    async with asyncio.timeout(timeout):
                        # The base implementation kills a timed-out check; Prime's job API
                        # deliberately leaves ordinary background jobs running on cancellation.
                        result = await RuntimeBackend.run(
                            runtime.backend,
                            runtime,
                            ["sh", "-c", 'exec "$@" > /dev/null 2>&1', "sh", *argv],
                            {},
                        )
                    if result.exit_code == 0:
                        return
                except TimeoutError:
                    pass
                if checked_at - since >= grace:
                    failures += 1
            raise SandboxError(
                f"Native Compose service {name!r} failed its health check"
            )

        try:
            async with asyncio.TaskGroup() as group:
                for name, service in self.services.items():
                    group.create_task(launch(name, service))
                    healthy[name] = group.create_task(ready(name, service))
        except ExceptionGroup as failed:
            raise failed.exceptions[0]
        for name, process in self.processes.items():
            if (code := await process.poll()) is not None and (
                code != 0 or name not in completed
            ):
                raise SandboxError(
                    f"Native Compose service {name!r} exited with {code}"
                )

    async def stop(self, deployment, names) -> None:
        for name in names:
            process = self.processes.pop(name, None)
            if process is None or deployment.target_backends[name].stopped:
                continue
            try:
                async with asyncio.timeout(10):
                    if await process.poll() is None:
                        runtime = deployment.target(name).with_user(None)
                        result = await runtime.run(
                            [
                                "sh",
                                "-c",
                                'kill -s "$1" "$(cat /tmp/vf-compose.pid)"',
                                "sh",
                                self.services[name].stop_signal.removeprefix("SIG"),
                            ],
                            {},
                        )
                        if result.exit_code:
                            await process.kill()
                        await process.wait()
            except Exception:
                # The deployment still deletes every VM if graceful shutdown fails.
                logger.warning(
                    "Native Compose service %s did not stop cleanly",
                    name,
                    exc_info=True,
                )
