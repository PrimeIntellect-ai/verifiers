"""Run tool adapters in `VM A`, with workspace execution over the native worker bridge.

The evaluator sends policy-approved calls to the adapter, which runs in the
harness and connects to the credential-free worker over the execution relay.
"""

import inspect
from collections.abc import Callable
from dataclasses import dataclass, field

from verifiers.v1.runtimes import Runtime
from verifiers.v1.runtimes.bridge import ExecutionBridge
from verifiers.v1.runtimes.harness_daemon import (
    HarnessDaemonClient,
    ensure_backend_daemon,
)
from verifiers.v1.types import ToolCall, ToolMessage


@dataclass(frozen=True)
class RemoteToolsSpec:
    factory: Callable
    kwargs: dict = field(default_factory=dict)
    defer_until_gate: bool = False

    native_tools: frozenset[str] | None = None


class SandboxTools:
    def __init__(
        self, runtime: Runtime, connection: ExecutionBridge, spec: RemoteToolsSpec
    ):
        self.runtime, self.connection, self.spec = runtime, connection, spec
        self.defer_until_gate = spec.defer_until_gate
        self.native_tools = spec.native_tools
        self._daemon: HarnessDaemonClient | None = None

    async def prepare(self) -> None:
        from verifiers.v1.harnesses.utils.launch import bundle_program
        from verifiers.v1.runtimes import rpc_wire, workspace_worker

        source = bundle_program(inspect.getsource(workspace_worker), rpc_wire)
        worker_python = await self.connection.runtime.resolve_python()
        self.connection.register(
            "tool-adapter", [worker_python, "-I", "-u", "-c", source], {}
        )
        self._daemon = await ensure_backend_daemon(
            self.runtime.backend, extra_cls=self.spec.factory
        )

    async def connect(self, url: str, secret: str) -> None:
        from verifiers.v1.harnesses.utils.remote_runtime import prepare_remote_bridge

        assert self._daemon is not None
        command = await prepare_remote_bridge(self.runtime, url, secret, "tool-adapter")
        await self._daemon.configure_tools(self.spec.factory, self.spec.kwargs, command)

    async def __call__(self, call: ToolCall) -> ToolMessage:
        assert self._daemon is not None
        return await self._daemon.execute_tool(call)

    async def aclose(self) -> None:
        if self._daemon is not None:
            daemon, self._daemon = self._daemon, None
            await daemon.close_tools()
