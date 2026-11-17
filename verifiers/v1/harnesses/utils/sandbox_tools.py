"""Run tool adapters in A, with workspace execution over the native worker bridge.

The evaluator sends only policy-approved calls and pins their results. Adapter
state and all Runtime operations live in A. Only explicit JSON configuration is
sent across the boundary; host Runtime objects and provider clients stay on host.
"""

import asyncio
import importlib
import inspect
from collections.abc import Callable
from contextlib import AsyncExitStack
from dataclasses import dataclass, field

from verifiers.v1.errors import HarnessError
from verifiers.v1.runtimes import Runtime
from verifiers.v1.runtimes.bridge import ExecutionBridge
from verifiers.v1.runtimes.stdio_rpc import ProcessRPC
from verifiers.v1.types import ToolCall, ToolMessage


@dataclass(frozen=True)
class RemoteToolsSpec:
    factory: Callable
    kwargs: dict = field(default_factory=dict)
    defer_until_gate: bool = False


class SandboxTools:
    def __init__(
        self, runtime: Runtime, connection: ExecutionBridge, spec: RemoteToolsSpec
    ):
        self.runtime, self.connection, self.spec = runtime, connection, spec
        self.defer_until_gate = spec.defer_until_gate
        self._stack = AsyncExitStack()
        self._python = ""
        self._rpc: ProcessRPC | None = None

    async def prepare(self) -> None:
        from verifiers.v1.harnesses.utils.launch import bundle_program
        from verifiers.v1.harnesses.utils.remote_runtime import REMOTE_RUNTIME_SOURCE
        from verifiers.v1.mcp.launch import install_in_runtime
        from verifiers.v1.runtimes import rpc_wire, workspace_worker

        self._python = await install_in_runtime(
            self.spec.factory, self.runtime, label="harness tool adapter"
        )
        source = bundle_program(inspect.getsource(workspace_worker), rpc_wire)
        command = await self.connection.runtime.prepare_uv_script(
            source, {}, activate=False
        )
        self.connection.register("tool-adapter", command, {})
        await self.runtime.prepare_uv_script(REMOTE_RUNTIME_SOURCE, {})

    async def connect(self, url: str, secret: str) -> None:
        from verifiers.v1.harnesses.utils.remote_runtime import prepare_remote_bridge

        command = await prepare_remote_bridge(self.runtime, url, secret, "tool-adapter")
        from verifiers.v1.runtimes.controller import current_job

        argv = [self._python, "-I", "-u", "-m", __name__]
        if current_job.get() and self.runtime.type == "prime":
            from verifiers.v1.runtimes import SubprocessConfig, provision_runtime
            from verifiers.v1.runtimes.lan_execution import LanExecution
            from verifiers.v1.runtimes.private_network import lan_host

            local = await self._stack.enter_async_context(
                provision_runtime(SubprocessConfig())
            )
            transport = LanExecution(local, lan_host(self.runtime.backend))
            self._stack.push_async_callback(transport.close)
            await transport.start(self.runtime)
            process = await transport.exec(self.runtime, argv, {})
        else:
            process = await self.runtime.exec(argv, {})
        self._rpc = ProcessRPC(process)
        await self._rpc.call(
            "configure",
            module=self.spec.factory.__module__,
            factory=self.spec.factory.__qualname__,
            kwargs=self.spec.kwargs,
            command=command,
        )

    async def __call__(self, call: ToolCall) -> ToolMessage:
        if self._rpc is None:
            raise HarnessError("sandbox tool adapter is not connected")
        result = await self._rpc.call("execute", call=call.model_dump())
        return ToolMessage.model_validate(result)

    async def aclose(self) -> None:
        try:
            if self._rpc is not None:
                try:
                    async with asyncio.timeout(10):
                        await self._rpc.call("close")
                finally:
                    await self._rpc.close()
                    self._rpc = None
        finally:
            await self._stack.aclose()


async def main() -> None:
    from verifiers.v1.runtimes.remote_backend import RemoteBackend
    from verifiers.v1.runtimes.rpc_wire import serve

    backend = None
    adapter = None

    async def close():
        nonlocal backend, adapter
        try:
            if close_adapter := getattr(adapter, "aclose", None):
                await close_adapter()
        finally:
            adapter = None
            if backend is not None:
                await backend.stop()
                backend = None

    async def handle(message):
        nonlocal backend, adapter
        operation = message["operation"]
        if operation == "configure":
            if backend is not None:
                raise HarnessError("sandbox tool adapter already configured")
            factory = importlib.import_module(message["module"])
            for part in message["factory"].split("."):
                factory = getattr(factory, part)
            backend = RemoteBackend(message["command"])
            await backend.start()
            adapter = factory(backend.bind(), **message["kwargs"])
            return None
        if operation == "execute":
            if adapter is None:
                raise HarnessError("sandbox tool adapter is not configured")
            return (
                await adapter(ToolCall.model_validate(message["call"]))
            ).model_dump()
        if operation == "close":
            await close()
            return None
        raise HarnessError("unknown sandbox tool adapter operation")

    try:
        await serve(handle)
    finally:
        await close()


if __name__ == "__main__":
    asyncio.run(main())
