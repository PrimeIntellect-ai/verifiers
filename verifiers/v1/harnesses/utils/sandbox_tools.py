"""Run tool adapters in `VM A`, with workspace execution over the native worker bridge.

The evaluator sends only policy-approved calls and pins their results. Adapter
state and all Runtime operations live in `VM A` inside the shared `HarnessDaemon`.
Only explicit JSON configuration crosses the boundary; host Runtime objects and
provider clients stay on the host.
"""

import asyncio
import inspect
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field

from verifiers.v1.errors import HarnessError
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


class SandboxTools:
    def __init__(
        self, runtime: Runtime, connection: ExecutionBridge, spec: RemoteToolsSpec
    ):
        self.runtime, self.connection, self.spec = runtime, connection, spec
        self.defer_until_gate = spec.defer_until_gate
        self.uses_local_gate = not spec.defer_until_gate
        self.local_gate_url: str | None = None
        self._endpoint: dict | None = None
        self._daemon: HarnessDaemonClient | None = None
        self._slot_id = uuid.uuid4().hex
        self._configured = False

    async def prepare(self) -> None:
        from verifiers.v1.harnesses.utils.launch import bundle_program
        from verifiers.v1.runtimes import rpc_wire, workspace_worker

        source = bundle_program(inspect.getsource(workspace_worker), rpc_wire)
        has_lan = getattr(self.connection, "transport", None) is not None

        async def start_daemon() -> HarnessDaemonClient:
            daemon = await ensure_backend_daemon(
                self.runtime.backend, extra_cls=self.spec.factory
            )
            if daemon is None:
                raise HarnessError("split harness runtime does not support daemon RPC")
            return daemon

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
                daemon_task = group.create_task(start_daemon())
                prewarmed = group.create_task(register_and_prewarm())
        except ExceptionGroup as failed:
            raise failed.exceptions[0]
        self._daemon = daemon_task.result()
        self._endpoint = prewarmed.result()
        if self.uses_local_gate:
            self.local_gate_url = self._daemon.local_gate_url

    async def connect(
        self,
        url: str,
        secret: str,
        *,
        interception_url: str | None = None,
    ) -> None:
        if self._configured:
            return
        if self._daemon is None or not self._daemon.alive:
            self._daemon = await ensure_backend_daemon(
                self.runtime.backend, extra_cls=self.spec.factory
            )
            if self._daemon is None:
                raise HarnessError("split harness runtime does not support daemon RPC")
            if self.uses_local_gate:
                self.local_gate_url = self._daemon.local_gate_url
        for rt in (self.runtime, self.connection.runtime):
            await rt.wait_lan_ready()
        command: list[str] | None = None
        if self._endpoint is None:
            from verifiers.v1.harnesses.utils.remote_runtime import (
                prepare_remote_bridge,
            )

            command = await prepare_remote_bridge(
                self.runtime, url, secret, "tool-adapter"
            )
        await self._daemon.register_tool_slot(
            slot_id=self._slot_id,
            module=self.spec.factory.__module__,
            factory=self.spec.factory.__qualname__,
            kwargs=self.spec.kwargs,
            endpoint=self._endpoint,
            command=command,
            interception_url=interception_url if self.uses_local_gate else None,
            secret=secret,
        )
        self._configured = True

    async def __call__(self, call: ToolCall) -> ToolMessage:
        if not self._configured or self._daemon is None:
            raise HarnessError("sandbox tool adapter is not connected")
        result = await self._daemon.execute_tool_slot(self._slot_id, call.model_dump())
        return ToolMessage.model_validate(result)

    async def aclose(self) -> None:
        if self._daemon is not None:
            daemon, self._daemon = self._daemon, None
            await daemon.unregister_tool_slot(self._slot_id)
