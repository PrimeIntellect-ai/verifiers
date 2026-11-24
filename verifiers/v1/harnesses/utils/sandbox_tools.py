"""Run approved workspace tools on the evaluator against the bound workspace."""

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

from verifiers.v1.harnesses.utils.compaction import bound_tool_message
from verifiers.v1.runtimes import Runtime
from verifiers.v1.types import ToolCall, ToolMessage


@dataclass(frozen=True)
class RemoteToolsSpec:
    factory: Callable
    kwargs: dict = field(default_factory=dict)
    defer_until_gate: bool = False
    native_tools: frozenset[str] | None = None
    prepare_tool: Callable[[ToolCall], Awaitable[None]] | None = None


class SandboxTools:
    def __init__(self, workspace: Runtime, spec: RemoteToolsSpec):
        self.spec = spec
        self.defer_until_gate = spec.defer_until_gate
        self.native_tools = spec.native_tools
        self._adapter = spec.factory(workspace, **spec.kwargs)

    async def __call__(self, call: ToolCall) -> ToolMessage:
        if self.spec.prepare_tool is not None:
            try:
                await self.spec.prepare_tool(call)
            except Exception as error:  # noqa: BLE001 - preparation errors belong to this tool call
                return ToolMessage(
                    tool_call_id=call.id,
                    name=call.name,
                    content=f"error: could not prepare {call.name}: {error}",
                )
        result = await self._adapter(call)
        return ToolMessage.model_validate(
            bound_tool_message(result.model_dump(exclude_none=True))
        )

    async def aclose(self) -> None:
        if close := getattr(self._adapter, "aclose", None):
            await close()
