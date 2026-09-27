"""Verifiers runtime views of services owned by a Harbor cloud environment."""

import asyncio
import json
import math
import os
import shlex
import tempfile
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, nullcontext
from functools import partial
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import urlsplit

from verifiers.v1.errors import SandboxError
from verifiers.v1.runtimes import ModalConfig, PrimeConfig, Runtime
from verifiers.v1.runtimes.base import SERVICE_PORT, ProgramResult
from verifiers.v1.runtimes.limiters import creation_limiter
from verifiers.v1.runtimes.modal import ModalRuntimeInfo, _egress_domain
from verifiers.v1.runtimes.prime import PrimeRuntimeInfo, validate_egress_lists
from verifiers.v1.tasksets.harbor.taskset import HarborTask
from verifiers.v1.utils.aio import run_shielded
from verifiers.v1.utils.prime import load_prime_config
from verifiers.v1.utils.scope import run_scope


class HarborRuntime(Runtime):
    is_local = False

    def __init__(self, environment, service, config, info):
        super().__init__()
        self.environment, self.service = environment, service
        self.config = config.model_copy(
            update={k: info[k] for k in ("image", "workdir")}
        )
        info_cls = (
            PrimeRuntimeInfo if isinstance(config, PrimeConfig) else ModalRuntimeInfo
        )
        self.info = info_cls(**self.config.model_dump(), id=info["id"], borrowed=True)

    @property
    def network_scope(self) -> object:
        return self.environment

    @property
    def published_port(self) -> int | None:
        return SERVICE_PORT if isinstance(self.config, ModalConfig) else None

    async def expose(self, port: int) -> str | None:
        if isinstance(self.config, ModalConfig):
            return await self.environment.expose(port)
        raise SandboxError("Prime VMs do not publish server ports; use a stdio server")

    async def start(self) -> None:
        raise RuntimeError("Harbor owns service creation")

    async def run(self, argv: list[str], env: dict[str, str]) -> ProgramResult:
        result = await self.environment.service_exec(
            shlex.join(argv),
            service=self.service,
            cwd=self.config.workdir,
            env=self.process_env(env),
        )
        return ProgramResult(
            result.return_code, result.stdout or "", result.stderr or ""
        )

    async def open_process(self, argv: list[str], env: dict[str, str]):
        return await self.environment.open_process(
            shlex.join(argv),
            service=self.service,
            cwd=self.config.workdir,
            env=self.process_env(env),
        )

    async def run_background(
        self, argv: list[str], env: dict[str, str], log: str
    ) -> None:
        result = await self.run(
            [
                "sh",
                "-c",
                f"nohup {shlex.join(argv)} > {shlex.quote(log)} 2>&1 </dev/null &",
            ],
            env,
        )
        if result.exit_code:
            raise SandboxError(f"Harbor background process failed: {result.stderr}")

    def _abs(self, path: str) -> str:
        return str(PurePosixPath(self.config.workdir) / path)

    async def write(self, path: str, data: bytes) -> None:
        target_path = self._abs(path)
        parent = await self.run(
            ["mkdir", "-p", str(PurePosixPath(target_path).parent)], {}
        )
        if parent.exit_code:
            raise SandboxError(f"Harbor write failed: {parent.stderr}")
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "data"
            source.write_bytes(data)
            await self.environment.service_upload_file(
                source, target_path, service=self.service
            )

    async def _read(self, path: str, max_bytes: int | None = None) -> bytes:
        if max_bytes is not None:
            return await super()._read(path, max_bytes)
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "data"
            await self.environment.service_download_file(
                self._abs(path), target, service=self.service
            )
            return target.read_bytes()

    async def prepare_execution(self, routes: list[str] | None) -> None:
        if not self.network_restricted:
            return
        from harbor.models.task.config import NetworkMode, NetworkPolicy

        if isinstance(self.config, PrimeConfig):
            if routes is None:
                rules = {"allow": ["*"]}
            elif self.config.allow == ["*"]:
                rules = {"deny": self.config.block}
            else:
                hosts = [urlsplit(route).hostname for route in routes]
                allow = list(
                    dict.fromkeys([*self.config.allow, *(h for h in hosts if h)])
                )
                validate_egress_lists(allow, None)
                rules = {"allow": allow} if allow else {"deny": ["*"]}
            await self.environment.set_network_rules(**rules)
        else:
            if routes is None:
                policy = NetworkPolicy(network_mode=NetworkMode.PUBLIC)
            else:
                domains = [_egress_domain(route, framework=True) for route in routes]
                domains.extend(_egress_domain(rule) for rule in self.config.allow)
                policy = NetworkPolicy(
                    network_mode=NetworkMode.ALLOWLIST,
                    allowed_hosts=list(dict.fromkeys(d for d in domains if d)),
                )
            await self.environment.set_network_policy(policy)


@asynccontextmanager
async def cloud_services(
    config: PrimeConfig | ModalConfig, task: HarborTask, *, setup_timeout=None
) -> AsyncIterator:
    import yaml
    from harbor.models.task.config import EnvironmentConfig, NetworkMode, NetworkPolicy
    from harbor.models.trial.paths import TrialPaths

    if config.gpu:
        raise ValueError("Harbor Compose currently supports CPU tasks")
    if isinstance(config, ModalConfig) and not config.network_access:
        raise ValueError("Harbor Compose on Modal requires network_access=True")
    authored = yaml.safe_load(
        (Path(task.data.task_dir) / "environment/docker-compose.yaml").read_text()
    )
    services = {"main": {}, **authored.get("services", {})}
    if any(
        service.get("scale", 1) != 1
        or service.get("deploy", {}).get("replicas", 1) != 1
        for service in services.values()
    ):
        raise ValueError("Harbor service runtimes require one container per service")
    options: dict[str, Any] = {"region": config.region}
    if isinstance(config, PrimeConfig):
        from harbor.environments.prime import PrimeEnvironment

        from verifiers.v1.runtimes import prime

        provider = PrimeEnvironment
        options.update(
            timeout_minutes=-1,
            idle_timeout_minutes=max(1, math.ceil(config.idle_timeout / 60))
            if config.idle_timeout is not None
            else None,
            labels=list(
                dict.fromkeys([*prime.BASE_LABELS, *config.labels, run_scope()])
            ),
        )
        options["team_id"] = os.environ.get("PRIME_TEAM_ID") or load_prime_config().get(
            "team_id"
        )
        if task.data.compose_host_image:
            options["compose_host_image"] = task.data.compose_host_image
    else:
        from harbor.environments.modal import ModalEnvironment

        provider = ModalEnvironment
        options.update(modal_vm_runtime=True, encrypted_ports=[SERVICE_PORT])
        if task.data.compose_host_image:
            options["dind_image"] = task.data.compose_host_image
    with tempfile.TemporaryDirectory(prefix="vf-harbor-cloud-") as directory:
        paths = TrialPaths(Path(directory) / "trial")
        paths.mkdir()
        main: dict[str, Any] = {}
        if task.data.image is not None:
            main["image"] = config.image
        if config.workdir is not None:
            main["working_dir"] = config.workdir
        overlay = {"services": {"main": main}}
        if isinstance(config, ModalConfig):
            owner = "main"
            seen = set()
            while (mode := services[owner].get("network_mode", "")).startswith(
                "service:"
            ):
                if owner in seen:
                    raise ValueError("Compose network cycle")
                seen.add(owner)
                owner = mode.removeprefix("service:")
            overlay["services"].setdefault(owner, {})["ports"] = [
                f"{SERVICE_PORT}:{SERVICE_PORT}"
            ]
        overlay_path = Path(directory) / "runtime.json"
        overlay_path.write_text(json.dumps(overlay))
        environment = provider(
            environment_dir=Path(task.data.task_dir) / "environment",
            environment_name=task.data.name or "verifiers",
            session_id=f"vf-{uuid.uuid4().hex}",
            trial_paths=paths,
            task_env_config=EnvironmentConfig(
                docker_image=task.data.image or config.image,
                memory_mb=int(config.memory * 1024),
                storage_mb=int(config.disk * 1024),
            ),
            override_cpus=config.cpu,
            persistent_env=task.runtime_env(),
            mounts=[],
            network_policy=NetworkPolicy(network_mode=NetworkMode.PUBLIC),
            phase_network_policies=[
                NetworkPolicy(network_mode=NetworkMode.ALLOWLIST, allowed_hosts=[])
            ]
            if config.network_restricted
            else [],
            extra_docker_compose=[overlay_path],
            **options,
        )
        runtimes = {}
        try:
            async with asyncio.timeout(setup_timeout):
                rate = (
                    (config.creates_per_min or 0) / 60
                    if isinstance(config, PrimeConfig)
                    else config.creates_per_sec
                )
                async with (
                    creation_limiter(rate, f"{config.type}-sandbox", run_scope())
                    or nullcontext()
                ):
                    await environment.start(force_build=False)
                for name in services:
                    runtimes[name] = HarborRuntime(
                        environment, name, config, await environment.service_info(name)
                    )
            yield runtimes, partial(environment.stop_service, "main")
        finally:
            for runtime in runtimes.values():
                runtime.stopped = True
            await run_shielded(environment.stop(delete=True))
