"""Prime VM ownership and command transport for Harbor Compose."""

import asyncio
import shlex

from verifiers.v1.errors import SandboxError
from verifiers.v1.runtimes import PrimeRuntime
from verifiers.v1.runtimes.base import ProgramResult
from verifiers.v1.runtimes.prime import EFFECTIVELY_UNBOUNDED_SECONDS


class PrimeComposeVM(PrimeRuntime):
    async def run(self, argv: list[str], env: dict[str, str]) -> ProgramResult:
        # Compose already owns process lifetime. Stream its command to completion
        # without creating a second background job and polling its status.
        assert self._client is not None
        try:
            result = await self._client.execute_command(
                self.info.id,
                shlex.join(argv),
                working_dir=self.config.workdir,
                env=self.process_env(env),
                timeout=EFFECTIVELY_UNBOUNDED_SECONDS,
            )
        except Exception as error:
            raise SandboxError(f"Prime Compose command failed: {error}") from error
        return ProgramResult(
            result.exit_code or 0, result.stdout or "", result.stderr or ""
        )

    async def start(self) -> None:
        await super().start()
        install = await self.run(
            [
                "sh",
                "-c",
                (
                    "export DEBIAN_FRONTEND=noninteractive; apt-get update -qq && "
                    "apt-get install -y -qq --no-install-recommends docker.io docker-cli docker-compose iptables "
                    "> /tmp/docker-install.log 2>&1 || { tail -40 /tmp/docker-install.log; exit 1; }; "
                    "dockerd --host=unix:///var/run/docker.sock >/tmp/dockerd.log 2>&1 </dev/null &"
                ),
            ],
            {},
        )
        if install.exit_code:
            raise SandboxError(
                f"Docker bootstrap failed: {install.stderr} {install.stdout}"
            )

    async def expose(self, port: int) -> str | None:
        raise SandboxError(
            "The Prime SDK does not support port exposure from VM sandboxes"
        )

    async def teardown(self) -> None:
        from prime_sandboxes import APIError, AsyncSandboxClient

        await super().teardown()
        if self.info.id is None:
            return
        # PrimeRuntime logs deletion failures; separate grading must instead wait
        # for confirmed solver teardown before creating its fresh verifier.
        async with AsyncSandboxClient() as client, asyncio.timeout(60):
            while True:
                try:
                    sandbox = await client.get(self.info.id)
                except APIError as error:
                    if str(error).startswith("HTTP 404:"):
                        return
                    raise
                if str(sandbox.status) == "TERMINATED":
                    return
                await asyncio.sleep(1)
