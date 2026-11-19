"""Run tool adapters in A, with workspace execution over the native worker bridge.

The evaluator sends only policy-approved calls and pins their results. Adapter
state and all Runtime operations live in A. Only explicit JSON configuration is
sent across the boundary; host Runtime objects and provider clients stay on host.
"""

import asyncio
import importlib
import inspect
import sys
from collections.abc import Callable
from contextlib import AsyncExitStack
from dataclasses import dataclass, field

from verifiers.v1.errors import HarnessError
from verifiers.v1.runtimes import Runtime
from verifiers.v1.runtimes.base import RuntimeProcess
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
        self._endpoint: dict | None = None
        self._rpc: ProcessRPC | None = None
        self._configured = False
        self._abandon_on_close = False

    async def _start_adapter_process(self) -> RuntimeProcess:
        from verifiers.v1.runtimes.controller import current_job

        argv = [
            self._python,
            "-I",
            "-u",
            "-m",
            __name__,
            self.spec.factory.__module__,
            self.spec.factory.__qualname__,
        ]
        if current_job.get() and self.runtime.type == "prime":
            from verifiers.v1.runtimes.lan_execution import LanExecution
            from verifiers.v1.runtimes.private_network import lan_host

            transport = LanExecution(
                self.runtime, lan_host(self.runtime.backend), direct=True
            )
            self._stack.push_async_callback(
                lambda: (
                    transport.abandon() if self._abandon_on_close else transport.close()
                )
            )
            await transport.start(self.runtime)
            return await transport.exec(self.runtime, argv, {})
        return await self.runtime.exec(argv, {})

    async def prepare(self) -> None:
        from verifiers.v1.harnesses.utils.launch import bundle_program
        from verifiers.v1.mcp.launch import install_in_runtime
        from verifiers.v1.runtimes import rpc_wire, workspace_worker

        source = bundle_program(inspect.getsource(workspace_worker), rpc_wire)
        has_lan = getattr(self.connection, "transport", None) is not None

        async def install_and_spawn() -> ProcessRPC:
            self._python = await install_in_runtime(
                self.spec.factory, self.runtime, label="harness tool adapter"
            )
            process = await self._start_adapter_process()
            return ProcessRPC(process)

        async def register_and_prewarm() -> dict | None:
            worker_python = await self.connection.runtime.resolve_python()
            self.connection.register(
                "tool-adapter", [worker_python, "-I", "-u", "-c", source], {}
            )
            if has_lan:
                return await self.connection.prewarm("tool-adapter")
            return None

        try:
            async with asyncio.TaskGroup() as group:
                adapter_rpc = group.create_task(install_and_spawn())
                prewarmed = group.create_task(register_and_prewarm())
        except ExceptionGroup as failed:
            raise failed.exceptions[0]
        self._rpc = adapter_rpc.result()
        self._endpoint = prewarmed.result()

    async def connect(self, url: str, secret: str) -> None:
        if self._configured:
            return
        if self._rpc is None:
            process = await self._start_adapter_process()
            self._rpc = ProcessRPC(process)
        if self._endpoint is not None:
            await self._rpc.call(
                "configure",
                module=self.spec.factory.__module__,
                factory=self.spec.factory.__qualname__,
                kwargs=self.spec.kwargs,
                endpoint=self._endpoint,
            )
        else:
            from verifiers.v1.harnesses.utils.remote_runtime import (
                prepare_remote_bridge,
            )

            command = await prepare_remote_bridge(
                self.runtime, url, secret, "tool-adapter"
            )
            await self._rpc.call(
                "configure",
                module=self.spec.factory.__module__,
                factory=self.spec.factory.__qualname__,
                kwargs=self.spec.kwargs,
                command=command,
            )
        self._configured = True

    async def __call__(self, call: ToolCall) -> ToolMessage:
        if self._rpc is None or not self._configured:
            raise HarnessError("sandbox tool adapter is not connected")
        result = await self._rpc.call("execute", call=call.model_dump())
        return ToolMessage.model_validate(result)

    async def aclose(self) -> None:
        try:
            if self._rpc is not None:
                if self._abandon_on_close:
                    await self._rpc.process.abandon()
                    await super(ProcessRPC, self._rpc).close()
                    self._rpc = None
                else:
                    try:
                        async with asyncio.timeout(10):
                            await self._rpc.call("close")
                    finally:
                        await self._rpc.close()
                        self._rpc = None
        finally:
            await self._stack.aclose()


def _resolve_factory(module_name: str, qualname: str) -> Callable:
    target: object = importlib.import_module(module_name)
    for part in qualname.split("."):
        target = getattr(target, part)
    assert callable(target)
    return target


async def main() -> None:
    from verifiers.v1.runtimes.remote_backend import RemoteBackend
    from verifiers.v1.runtimes.rpc_wire import serve

    preloaded: Callable | None = None
    if len(sys.argv) >= 3:
        preloaded = _resolve_factory(sys.argv[1], sys.argv[2])

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
            factory = preloaded or _resolve_factory(
                message["module"], message["factory"]
            )
            if message.get("endpoint") is not None:
                backend = RemoteBackend(endpoint=message["endpoint"])
            else:
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
