"""Modal VM ownership for Harbor Compose, preserving Docker networking."""

import asyncio

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
            timeout=24 * 60 * 60,
            encrypted_ports=[SERVICE_PORT],
            experimental_options={"vm_runtime": True},
        )

    async def teardown(self) -> None:
        sandbox = self._sandbox
        if sandbox is None:
            return
        # Preserve the handle for the cleanup backstop until termination is confirmed.
        async with asyncio.timeout(60):
            await sandbox.terminate.aio()
            await sandbox.wait.aio(raise_on_termination=False)
        self._sandbox = None
