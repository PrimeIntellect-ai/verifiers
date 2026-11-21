"""Run approved harness tools through the evaluator's bound workspace runtime."""

import asyncio
import base64
import contextlib
import uuid

import httpx

from verifiers.v1.configs.runtime import CustomRuntimeConfig
from verifiers.v1.errors import SandboxError
from verifiers.v1.runtimes.base import (
    BaseRuntimeInfo,
    RuntimeBackend,
    RuntimeProcess,
    collect_process,
)
from verifiers.v1.runtimes.runtime import Runtime


class RemoteProcess(RuntimeProcess):
    def __init__(self, backend: "RemoteBackend", identifier: str):
        self.backend, self.identifier = backend, identifier
        self.stdout, self.stderr = self._stream("stdout"), self._stream("stderr")
        self._stdin_lock = asyncio.Lock()

    async def _call(self, operation: str, **values):
        return await self.backend.call(operation, self.identifier, **values)

    async def _stream(self, stream: str):
        previous = None
        while True:
            request_id = uuid.uuid4().hex
            result = await self._call(
                "read", stream=stream, request_id=request_id, acknowledge_read=previous
            )
            if result["eof"]:
                return
            previous = request_id
            yield base64.b64decode(result["data"], validate=True)

    async def write(self, data: bytes) -> None:
        async with self._stdin_lock:
            for offset in range(0, len(data), 65536):
                await self._call(
                    "write",
                    data=base64.b64encode(data[offset : offset + 65536]).decode(),
                )

    async def wait(self) -> int:
        return (await self._call("wait"))["exit_code"]

    async def poll(self) -> int | None:
        return (await self._call("poll"))["exit_code"]

    async def terminate(self) -> None:
        await self._call("terminate")

    async def kill(self) -> None:
        await self._call("kill")


class RemoteBackend(RuntimeBackend):
    """Harness-side view of the task user and workspace bound by the evaluator."""

    def __init__(self, url: str, secret: str, adapter_key: str):
        super().__init__()
        self.url, self.adapter_key = url, adapter_key
        self.config = CustomRuntimeConfig(import_path=f"{__name__}.RemoteBackend")
        self.info = BaseRuntimeInfo()
        self.client = httpx.AsyncClient(
            headers={"Authorization": f"Bearer {secret}"},
            timeout=httpx.Timeout(None, connect=10),
            # Blocked stdout/stderr reads must leave a connection for stdin and signals.
            limits=httpx.Limits(max_connections=None, max_keepalive_connections=8),
        )

    async def start(self) -> None:
        """The evaluator already provisioned the workspace."""

    async def call(
        self,
        operation: str,
        process_id: str,
        *,
        request_id: str | None = None,
        **arguments,
    ) -> dict:
        body = dict(
            arguments,
            operation=operation,
            process_id=process_id,
            request_id=request_id or uuid.uuid4().hex,
        )
        for attempt in range(3):
            try:
                response = await self.client.post(self.url, json=body)
                response.raise_for_status()
                return response.json()
            except httpx.HTTPStatusError as error:
                raise SandboxError(
                    f"workspace {operation} failed: {error.response.text[:500]}"
                ) from error
            except httpx.TransportError as error:
                if attempt == 2:
                    raise SandboxError(
                        f"workspace {operation} failed: {error}"
                    ) from error
                await asyncio.sleep(0.2 * (attempt + 1))
        raise AssertionError("unreachable")

    async def exec(
        self, runtime: Runtime, argv: list[str], env: dict[str, str]
    ) -> RuntimeProcess:
        identifier = uuid.uuid4().hex
        await self.call(
            "exec",
            identifier,
            argv=argv,
            env=runtime.process_env(env),
            adapter_key=self.adapter_key,
        )
        return RemoteProcess(self, identifier)

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
        await self.client.aclose()

    async def expose(self, port: int) -> str:
        raise SandboxError("workspace port publication is owned by the deployment")

    async def verify_stopped(self) -> None:
        raise SandboxError("workspace termination is verified by the deployment")
