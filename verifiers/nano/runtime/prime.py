"""Remote Prime sandbox runtime: run the program in a sandbox, server via tunnel."""

import shlex
from typing import Literal

from pydantic_config import BaseConfig

from verifiers.nano.runtime.base import ProgramResult, Runtime


class PrimeConfig(BaseConfig):
    kind: Literal["prime"] = "prime"
    image: str = "python:3.11-slim"
    workdir: str = "/app"
    cpu_cores: float = 1.0
    memory_gb: float = 2.0
    network_access: bool = True


class PrimeRuntime(Runtime):
    """Runs the program in a Prime sandbox; the server is reached via a tunnel."""

    def __init__(self, config: PrimeConfig) -> None:
        self.config = config
        self._client = None
        self._sandbox_id: str | None = None
        self._tunnel = None

    async def start(self, port: int) -> str:
        from prime_sandboxes import AsyncSandboxClient, CreateSandboxRequest
        from prime_tunnel import Tunnel

        self._client = AsyncSandboxClient()
        sandbox = await self._client.create(
            CreateSandboxRequest(
                name="vf-nano-program",
                docker_image=self.config.image,
                cpu_cores=self.config.cpu_cores,
                memory_gb=self.config.memory_gb,
                network_access=self.config.network_access,
            )
        )
        self._sandbox_id = sandbox.id
        await self._client.wait_for_creation(self._sandbox_id)
        await self._client.run_background_job(
            self._sandbox_id, f"mkdir -p {shlex.quote(self.config.workdir)}"
        )
        self._tunnel = Tunnel(local_port=port)
        url = str(await self._tunnel.start()).rstrip("/")
        return f"{url}/v1"

    async def run(self, argv: list[str], env: dict[str, str]) -> ProgramResult:
        result = await self._client.run_background_job(
            self._sandbox_id,
            shlex.join(argv),
            working_dir=self.config.workdir,
            env=env,
        )
        return ProgramResult(
            exit_code=result.exit_code or 0,
            stdout=result.stdout or "",
            stderr=result.stderr or "",
        )

    async def stop(self) -> None:
        if self._tunnel is not None:
            self._tunnel.sync_stop()
        if self._client is not None and self._sandbox_id is not None:
            await self._client.delete(self._sandbox_id)
        if self._client is not None:
            await self._client.aclose()
