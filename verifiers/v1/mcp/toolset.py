"""`Toolset` + `ToolsetConfig`: a tool server authored as a vf-native class with `@vf.tool` methods.

A task gives the harness tools by declaring `Toolset`s from `Taskset.tools`. The config carries
placement (where the server runs); the class carries the `@vf.tool` methods the model calls.
"""

from __future__ import annotations

import asyncio
import inspect
from pathlib import Path
from typing import Annotated, Any, TYPE_CHECKING

import httpx
from pydantic import WithJsonSchema, model_validator
from pydantic_config import BaseConfig

from verifiers.v1.decorators import discover_decorated
from verifiers.v1.mcp.server import ConfigT, ServerBase
from verifiers.v1.runtimes import RuntimeConfig, SubprocessConfig
from verifiers.v1.state import StateT

if TYPE_CHECKING:
    from mcp.server.fastmcp import FastMCP


class ToolsetConfig(BaseConfig):
    """Where one tool server runs (placement). The default — its own host (`subprocess`)
    runtime — is the cheap one: the server runs in the eval process's environment, where
    `verifiers` and the env module are already installed (nothing to fetch), and the harness
    reaches it over the host network (docker `--network host`) or a tunnel (prime). `colocated`
    and `shared` trade that off:
      - neither (default): its own `runtime`, per rollout (host by default).
      - colocated: in the harness's OWN runtime, per rollout (no tunnel). In a sandbox this means
        the `verifiers` source + the env module are uploaded and `uv pip install`ed there, so it
        costs a per-rollout install.
      - shared: one instance for the whole eval, in its own `runtime` (pays `setup` once; per-rollout
        writable state stays isolated via `self.state`).
      - shared + fork: like shared, but a forked child per rollout (copy-on-write) also isolates
        state that can't live in `self.state`.
    See the placement/isolation section of `verifiers/v1/GUIDE.md` for the trade-offs of each.
    Subclass to add the server's own knobs (the data its `@tool` methods read). The server name is
    the class's `TOOL_PREFIX` ClassVar, not a field here — it's an identity (the model sees
    `<prefix>_<tool>`, baked into the taskset's prompt), not a tunable knob."""

    colocated: bool = False
    """Run the server inside the harness's runtime (reached in-sandbox, no tunnel). Off by
    default — on the host it's free, but in a sandbox the harness runtime must install the env
    package + `verifiers` (a per-rollout cost), so prefer the default own-runtime placement
    unless co-locating genuinely helps."""
    shared: bool = False
    """Run one server instance for the whole eval, shared across rollouts (in its own `runtime`) —
    pays an expensive `setup` (a corpus/index) once. It may still be writable: each rollout reads and
    writes its OWN `self.state` (the framework tags the per-rollout state channel onto the shared
    server's URL), so concurrent rollouts don't corrupt each other — provided the shared server runs
    on a local runtime (the default), which can reach the host's interception server. Mutually
    exclusive with `colocated`."""
    fork: bool = False
    """For a `shared` server: fork a child process per rollout (copy-on-write memory + a private
    working dir), so per-rollout state that can't live in `self.state` — module globals, a mutated
    in-memory object, relative-path on-disk writes — is isolated per rollout automatically. The
    expensive `setup` runs once in the parent; each child inherits it warm and runs `setup_task` for
    its rollout's task (see `verifiers.v1.mcp.multiplex`) — so a stateful per-rollout server pays its
    `setup` once yet stays isolated. Requires `shared`; works on any runtime (a remote one reaches the
    rollout's state/task channel over a host tunnel). Linux/fork only; not for CUDA/GPU state or
    background threads in the server."""
    runtime: RuntimeConfig = SubprocessConfig()
    """The server's own runtime, used unless `colocated` (host/subprocess by default — always
    reachable from any harness runtime; set docker/prime to isolate it in its own sandbox)."""
    url: str | None = None
    """An already-running streamable-HTTP MCP endpoint to connect to instead of launching a
    server (e.g. a public remote like DeepWiki). When set, placement is ignored — the toolset
    needs no `@tool` methods, the model just sees the remote's tools as `<name>_<tool>`."""

    @model_validator(mode="after")
    def _validate_placement(self) -> "ToolsetConfig":
        if self.colocated and self.shared:
            raise ValueError("colocated and shared are mutually exclusive")
        if self.fork and not self.shared:
            raise ValueError(
                "fork requires shared — a per-rollout server already runs in its own process"
            )
        return self


class JSONRPCToolsetConfig(ToolsetConfig):
    """A JSON-RPC endpoint exposing ``tools/list`` and ``tools/call``."""

    endpoint: str
    uds: str | None = None
    """Optional Unix socket used to reach the endpoint."""


class Toolset(ServerBase[ConfigT, StateT]):
    """A tool server authored as a class: write `@vf.tool` methods (the model calls them as
    `<prefix>_<method>`; the docstring is the tool description), reading config off `self.config` and
    optionally the rollout's shared `self.state`. Example:

        class GlossaryToolsetConfig(vf.ToolsetConfig):
            facts: dict[str, str] = {}

        class GlossaryToolset(vf.Toolset[GlossaryToolsetConfig]):
            @vf.tool
            def lookup(self, name: str) -> str:
                return self.config.facts.get(name.lower(), "unknown")

    Parameterize a stateful toolset with its `State` subclass too — `Toolset[Config, MyState]` — so
    `self.state` is typed; it defaults to the base `State`.
    """

    def _register(self, mcp: FastMCP) -> None:
        for fn in discover_decorated(self, "tool"):
            mcp.add_tool(
                self._with_state(fn),
                name=getattr(fn, "tool_name", None) or fn.__name__,
                description=(fn.__doc__ or "").strip() or None,
            )


class JSONRPCToolset(Toolset[JSONRPCToolsetConfig]):
    """Expose a simple JSON-RPC tool endpoint as a standard MCP server."""

    TOOL_PREFIX = "jsonrpc"

    async def setup(self) -> None:
        if self.config.uds:
            async with asyncio.timeout(60):
                while not Path(self.config.uds).exists():
                    await asyncio.sleep(0.1)
        transport = (
            httpx.AsyncHTTPTransport(uds=self.config.uds, retries=10)
            if self.config.uds
            else None
        )
        async with httpx.AsyncClient(transport=transport, timeout=30) as client:
            response = await client.post(
                self.config.endpoint,
                json={"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}},
            )
        response.raise_for_status()
        payload = response.json()
        if error := payload.get("error"):
            raise RuntimeError(error.get("message", str(error)))
        self.tools = payload["result"]["tools"]

    def _register(self, mcp: FastMCP) -> None:
        for tool in self.tools:

            async def call(_name=tool["name"], **arguments):
                transport = (
                    httpx.AsyncHTTPTransport(uds=self.config.uds)
                    if self.config.uds
                    else None
                )
                async with httpx.AsyncClient(transport=transport, timeout=30) as client:
                    response = await client.post(
                        self.config.endpoint,
                        json={
                            "jsonrpc": "2.0",
                            "id": 1,
                            "method": "tools/call",
                            "params": {
                                "name": _name,
                                "arguments": {
                                    key: value
                                    for key, value in arguments.items()
                                    if value is not None
                                },
                            },
                        },
                    )
                response.raise_for_status()
                payload = response.json()
                if error := payload.get("error"):
                    raise RuntimeError(error.get("message", str(error)))
                return payload.get("result")

            schema = (
                tool.get("inputSchema")
                or tool.get("input_schema")
                or {"type": "object", "properties": {}}
            )
            required = set(schema.get("required", []))
            call.__name__ = tool["name"]
            call.__doc__ = tool.get("description", "")
            call.__signature__ = inspect.Signature(  # type: ignore[attr-defined]
                [
                    inspect.Parameter(
                        name,
                        inspect.Parameter.KEYWORD_ONLY,
                        default=inspect.Parameter.empty
                        if name in required
                        else spec.get("default"),
                        annotation=Annotated[Any, WithJsonSchema(spec)],
                    )
                    for name, spec in schema.get("properties", {}).items()
                ],
                return_annotation=Any,
            )
            mcp.add_tool(
                call, name=tool["name"], description=tool.get("description") or None
            )


if __name__ == "__main__":
    JSONRPCToolset.run()
