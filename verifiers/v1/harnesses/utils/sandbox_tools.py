"""Run approved tools in the harness and relay their workspace operations to the evaluator."""

from collections.abc import Awaitable, Callable
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
    prepare_tool: Callable[[ToolCall], Awaitable[None]] | None = None


class SandboxTools:
    def __init__(
        self, runtime: Runtime, connection: ExecutionBridge, spec: RemoteToolsSpec
    ):
        self.runtime, self.connection, self.spec = runtime, connection, spec
        self.defer_until_gate = spec.defer_until_gate
        self.native_tools = spec.native_tools
        self._daemon: HarnessDaemonClient | None = None

    async def prepare(self) -> None:
        self._daemon = await ensure_backend_daemon(
            self.runtime.backend, extra_cls=self.spec.factory
        )

    async def connect(self, url: str, secret: str) -> None:
        assert self._daemon is not None
        await self._daemon.configure_tools(
            self.spec.factory,
            self.spec.kwargs,
            url,
            secret,
            self.connection.adapter_key,
        )

    async def __call__(self, call: ToolCall) -> ToolMessage:
        assert self._daemon is not None
        if self.spec.prepare_tool is not None:
            try:
                await self.spec.prepare_tool(call)
            except Exception as error:  # noqa: BLE001 - preparation errors belong to this tool call
                return ToolMessage(
                    tool_call_id=call.id,
                    name=call.name,
                    content=f"error: could not prepare {call.name}: {error}",
                )
        return await self._daemon.execute_tool(call)

    async def aclose(self) -> None:
        if self._daemon is not None:
            daemon, self._daemon = self._daemon, None
            await daemon.close_tools()
