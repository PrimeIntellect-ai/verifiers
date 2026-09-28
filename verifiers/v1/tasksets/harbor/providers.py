"""Compatibility with the pinned upstream Harbor providers.

Harbor owns provisioning and cleanup. Private hooks needed for VF's service
discovery, interactive transport, and controller-env isolation stay here; they
are not an extension of Harbor's public API or of the core Runtime contract.
"""

import asyncio
import json
import os
import shlex
import uuid
from pathlib import Path
from typing import TypedDict

from verifiers.v1.errors import SandboxError
from verifiers.v1.runtimes.container import cli
from verifiers.v1.runtimes.prime import PrimeProcess
from verifiers.v1.utils.aio import run_shielded

DOCKER_ENV = (
    "PATH",
    "HOME",
    "DOCKER_HOST",
    "DOCKER_CONTEXT",
    "DOCKER_CONFIG",
    "DOCKER_TLS_VERIFY",
    "DOCKER_CERT_PATH",
)


class ServiceInfo(TypedDict):
    id: str
    image: str
    workdir: str


def _services(document: list[dict]) -> dict[str, ServiceInfo]:
    services = {}
    for container in document:
        config = container["Config"]
        name = config["Labels"]["com.docker.compose.service"]
        if name in services:
            raise SandboxError(f"Expected one container for Compose service {name!r}")
        services[name] = ServiceInfo(
            id=container["Id"],
            image=config["Image"],
            workdir=config["WorkingDir"] or "/",
        )
    if "main" not in services:
        raise SandboxError("Harbor Compose did not start a main service")
    return services


def network_service(services: dict, name: str) -> str:
    seen = set()
    while name not in seen:
        seen.add(name)
        if name not in services:
            raise SandboxError(f"Unknown Compose network service {name!r}")
        mode = services[name].get("network_mode", "")
        if not mode.startswith("service:"):
            return name
        name = mode.removeprefix("service:")
    raise SandboxError("Compose network namespaces contain a cycle")


def docker_provider():
    import yaml
    from harbor.environments.definition import should_use_prebuilt_docker_image
    from harbor.environments.docker.docker import DockerEnvironment

    class DockerServices(DockerEnvironment):
        def _compose_env_vars(self, include_os_env=True):
            controller = (
                {key: os.environ[key] for key in DOCKER_ENV if key in os.environ}
                if include_os_env
                else {}
            )
            return {**controller, **super()._compose_env_vars(include_os_env=False)}

        @property
        def _docker_compose_paths(self):
            paths = super()._docker_compose_paths
            # VF runtime defaults are fallbacks: an authored main image/build
            # owns image selection unless task/runtime data explicitly overrides it.
            authored = [
                self._environment_docker_compose_path,
                *self.extra_docker_compose_paths,
            ]
            if any(
                {"image", "build"}
                & (yaml.safe_load(path.read_text()) or {})
                .get("services", {})
                .get("main", {})
                .keys()
                for path in authored
                if path.is_file()
            ):
                base = (
                    self._DOCKER_COMPOSE_PREBUILT_PATH
                    if self._use_prebuilt
                    else self._DOCKER_COMPOSE_BUILD_PATH
                )
                document = yaml.safe_load(base.read_text())
                for key in ("image", "build", "pull_policy"):
                    document["services"]["main"].pop(key, None)
                replacement = self.trial_paths.trial_dir / "compose-base.json"
                replacement.write_text(json.dumps(document))
                paths = [replacement if path == base else path for path in paths]
            return paths

        async def compose_config(self):
            # Resolve the same generated overlays as start(), before creating
            # containers. Local policy validation must precede side effects.
            self._mounts_compose_path = self._write_mounts_compose_file()
            self._resources_compose_path = self._write_resources_compose_file()
            self._env_compose_path = self._write_env_compose_file()
            self._use_prebuilt = should_use_prebuilt_docker_image(
                self.environment_dir,
                docker_image=self.task_env_config.docker_image,
                force_build=False,
            )
            result = await self._run_docker_compose_command(
                ["config", "--format", "json"]
            )
            return json.loads(result.stdout)

        async def services(self):
            result = await self._run_docker_compose_command(["ps", "--all", "--quiet"])
            ids = (result.stdout or "").split()
            if not ids:
                raise SandboxError("Harbor Compose has no service containers")
            inspected = await cli(
                *self._engine_cmd("inspect", *ids), env=self._compose_env_vars()
            )
            if inspected.exit_code:
                raise SandboxError(
                    f"Cannot inspect Compose services: {inspected.stderr}"
                )
            return _services(json.loads(inspected.stdout))

        async def service_port(self, service, port):
            result = await self._run_docker_compose_command(
                ["port", service, str(port)]
            )
            return (result.stdout or "").strip()

    return DockerServices


def prime_provider():
    from harbor.environments.prime import PrimeEnvironment, _PrimeCompose

    from verifiers.v1.tasksets.harbor.process import open_service_process

    class DeclaredCompose(_PrimeCompose):
        def _referenced_host_env(self):
            # Harbor still resolves task-declared values and task-local .env.
            return {}

    class PrimeServices(PrimeEnvironment):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._compose = DeclaredCompose(self)

        async def services(self):
            result = await self._compose._compose_exec(["ps", "--all", "--quiet"])
            ids = (result.stdout or "").split()
            if result.return_code or not ids:
                raise SandboxError("Harbor Compose has no service containers")
            result = await self._host_exec(shlex.join(["docker", "inspect", *ids]))
            if result.return_code:
                raise SandboxError(f"Cannot inspect Compose services: {result.stderr}")
            return _services(json.loads(result.stdout))

        async def open_process(self, command, *, service, cwd, env):
            async def launch(wrapped):
                parts = ["exec", "-T"]
                if cwd:
                    parts.extend(["-w", cwd])
                # Startup values already live in main; do not resend credentials
                # in plain command strings. Sidecar env is explicitly scoped.
                startup = self._compose._startup_env if service == "main" else {}
                for key, value in env.items():
                    if startup.get(key) != value:
                        parts.extend(["-e", f"{key}={value}"])
                parts.extend([service, "sh", "-c", wrapped])
                client, sandbox_id = self._require_sandbox()
                return PrimeProcess(
                    await client.open_process(
                        sandbox_id, self._compose._compose_cmd(parts), working_dir="/"
                    )
                )

            async def execute(command):
                return await self.service_exec(command, service=service)

            return await open_service_process(command, launch=launch, execute=execute)

        async def service_upload_file(self, source: Path, target: str, *, service: str):
            if service == "main":
                return await self.upload_file(source, target)
            temporary = f"/tmp/vf-upload-{uuid.uuid4().hex}"
            try:
                await self._host_upload_file(source, temporary)
                result = await self._compose._compose_exec(
                    ["cp", temporary, f"{service}:{target}"]
                )
                if result.return_code:
                    raise SandboxError(f"Compose upload failed: {result.stderr}")
            finally:
                await run_shielded(
                    self._host_exec(f"rm -f {shlex.quote(temporary)}", timeout_sec=10)
                )

        async def set_network_rules(self, **rules):
            client, sandbox_id = self._require_sandbox()
            async with asyncio.timeout(30):
                status = await client.set_network(sandbox_id, **rules)
                while not status.applied:
                    await asyncio.sleep(1)
                    status = await client.get_network(sandbox_id)

    return PrimeServices
