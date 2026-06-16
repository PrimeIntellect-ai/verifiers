"""`Toolset` + `ToolsetConfig`: a tool server authored as a vf-native class with `@vf.tool` methods.

A task gives the harness tools by declaring `Toolset`s from `Taskset.tools`. The config carries
placement (where the server runs); the class carries the `@vf.tool` methods the model calls.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import model_validator
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
      - shared: one instance for the whole eval, in its own `runtime`.
    Subclass to add the server's own knobs (the data its `@tool` methods read). The server name is
    the class's `TOOL_PREFIX` ClassVar, not a field here — it's an identity (the model sees
    `<prefix>_<tool>`, baked into the taskset's instruction), not a tunable knob."""

    colocated: bool = False
    """Run the server inside the harness's runtime (reached in-sandbox, no tunnel). Off by
    default — on the host it's free, but in a sandbox the harness runtime must install the env
    package + `verifiers` (a per-rollout cost), so prefer the default own-runtime placement
    unless co-locating genuinely helps."""
    shared: bool = False
    """Run one server instance for the whole eval, shared across rollouts (in its own
    `runtime`). Mutually exclusive with `colocated`."""
    runtime: RuntimeConfig = SubprocessConfig()
    """The server's own runtime, used unless `colocated` (host/subprocess by default — always
    reachable from any harness runtime; set docker/prime to isolate it in its own sandbox)."""
    url: str | None = None
    """An already-running streamable-HTTP MCP endpoint to connect to instead of launching a
    server (e.g. a public remote like DeepWiki). When set, placement is ignored — the toolset
    needs no `@tool` methods, the model just sees the remote's tools as `<name>_<tool>`."""

    @model_validator(mode="after")
    def reject_colocated_and_shared(self) -> "ToolsetConfig":
        if self.colocated and self.shared:
            raise ValueError("colocated and shared are mutually exclusive")
        return self


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
