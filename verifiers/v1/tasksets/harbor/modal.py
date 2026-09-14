"""Modal VM ownership for Harbor Compose, preserving Docker networking."""

import asyncio
import atexit

from verifiers.v1.runtimes import ModalRuntime
from verifiers.v1.runtimes.base import SERVICE_PORT


class ModalComposeVM(ModalRuntime):
    async def _create_sandbox(self, app) -> None:
        import modal

        # Keep Docker's control API on a private socket, unreachable over service networks.
        self._sandbox = await modal.Sandbox.create.aio(
            "dockerd",
            "--host=unix:///var/run/docker.sock",
            app=app,
            name=self.name,
            image=modal.Image.from_registry(self.config.image).entrypoint([]),
            workdir=self.config.workdir,
            cpu=self.config.cpu,
            memory=int(self.config.memory * 1024),
            region=self.config.region,
            # Seed both allowlist types so prepare_execution can update them.
            outbound_domain_allowlist=["*"] if self.network_restricted else None,
            outbound_cidr_allowlist=(
                ["0.0.0.0/0"] if self.network_restricted else None
            ),
            timeout=24 * 60 * 60,
            encrypted_ports=[SERVICE_PORT],
            experimental_options={"vm_runtime": True},
        )

    async def teardown(self) -> None:
        sandbox = self._sandbox
        if sandbox is None:
            return
        # Keep cleanup alive after a timeout even if the rollout releases this runtime.
        atexit.register(self.cleanup)
        async with asyncio.timeout(60):
            await sandbox.terminate.aio()
            await sandbox.wait.aio(raise_on_termination=False)
        self._sandbox = None
        atexit.unregister(self.cleanup)
