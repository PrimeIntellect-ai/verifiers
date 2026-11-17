"""Runtime operations issued from A through the common workspace worker stream."""

import asyncio
import base64
import contextlib

from verifiers.v1.configs.runtime import CustomRuntimeConfig
from verifiers.v1.errors import SandboxError
from verifiers.v1.runtimes.backends.subprocess import SubprocessProcess
from verifiers.v1.runtimes.base import (
    BaseRuntimeInfo,
    RuntimeBackend,
    RuntimeProcess,
    collect_process,
)
from verifiers.v1.runtimes.runtime import Runtime
from verifiers.v1.runtimes.stdio_rpc import ProcessRPC


class RemoteProcess(RuntimeProcess):
    def __init__(self, rpc: ProcessRPC, identifier: str):
        self.rpc, self.identifier = rpc, identifier
        self.stdout, self.stderr = self._stream("stdout"), self._stream("stderr")
        self._stdin_lock = asyncio.Lock()

    async def _call(self, operation: str, **values):
        return await self.rpc.call(operation, process_id=self.identifier, **values)

    async def _stream(self, stream: str):
        while (data := await self._call("read", stream=stream)) is not None:
            yield base64.b64decode(data, validate=True)

    async def write(self, data: bytes) -> None:
        async with self._stdin_lock:
            for offset in range(0, len(data), 65536):
                await self._call(
                    "write",
                    data=base64.b64encode(data[offset : offset + 65536]).decode(),
                )

    async def wait(self) -> int:
        return await self._call("wait")

    async def poll(self) -> int | None:
        return await self._call("poll")

    async def terminate(self) -> None:
        await self._call("terminate")

    async def kill(self) -> None:
        await self._call("kill")


class RemoteBackend(RuntimeBackend):
    """Internal view of B's process worker, not a provisionable sandbox provider.

    B is bound to its task user. Resource ownership and port publication stay
    with the evaluator's deployment, outside this credential-free proxy.
    """

    def __init__(self, command: list[str]):
        super().__init__()
        self.command = command
        self.config = CustomRuntimeConfig(import_path=f"{__name__}.RemoteBackend")
        self.info = BaseRuntimeInfo()
        self.rpc: ProcessRPC | None = None

    async def start(self) -> None:
        process = await asyncio.create_subprocess_exec(
            *self.command,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=True,
        )
        self.rpc = ProcessRPC(SubprocessProcess(process))

    async def exec(
        self, runtime: Runtime, argv: list[str], env: dict[str, str]
    ) -> RuntimeProcess:
        assert self.rpc is not None
        identifier = await self.rpc.call(
            "exec", argv=argv, env=runtime.process_env(env)
        )
        return RemoteProcess(self.rpc, identifier)

    async def write(self, runtime: Runtime, path: str, data: bytes) -> None:
        process = await runtime.exec(
            [
                "/bin/sh",
                "-c",
                'mkdir -p -- "$(dirname -- "$1")" && head -c "$2" > "$1"',
                "workspace-write",
                path,
                str(len(data)),
            ],
            {},
        )
        output = asyncio.create_task(collect_process(process))
        try:
            await process.write(data)
            result = await output
        except BaseException:
            output.cancel()
            with contextlib.suppress(BaseException):
                await output
            raise
        if result.exit_code:
            raise SandboxError(f"write {path!r}: {result.stderr[-500:]}")

    async def teardown(self) -> None:
        if self.rpc is not None:
            try:
                async with asyncio.timeout(5):
                    await self.rpc.call("close")
            finally:
                await self.rpc.close()

    async def expose(self, port: int) -> str:
        raise SandboxError("workspace port publication is owned by the deployment")

    async def verify_stopped(self) -> None:
        raise SandboxError("workspace termination is verified by the deployment")
