"""Adapt Harbor service execution and file APIs to a Verifiers runtime."""

import shlex
import tempfile
from pathlib import Path, PurePosixPath
from urllib.parse import urlsplit

from verifiers.v1.errors import SandboxError
from verifiers.v1.runtimes import ModalConfig, PrimeConfig, Runtime
from verifiers.v1.runtimes.base import SERVICE_PORT, ProgramResult
from verifiers.v1.runtimes.modal import ModalRuntimeInfo, _egress_domain
from verifiers.v1.runtimes.prime import PrimeRuntimeInfo, validate_egress_lists


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

    async def expose(self, port: int) -> str:
        if isinstance(self.config, ModalConfig):
            url = await self.environment.expose(port)
            if url is None:
                raise SandboxError(f"Harbor did not expose port {port}")
            return url
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
