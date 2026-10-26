"""Nodes a rollout declares: where its parts run and who may reach them
(`rfcs/0001-nodes.md`). `verifiers.v1.runtimes.nodes` plans them."""

from __future__ import annotations

from typing import Any

from pydantic import Field, model_validator
from pydantic_config import BaseConfig

WORKSPACE = "main"
"""The agent's workspace. Its runtime is the agent's runtime, resolved for the task."""
HOST = "host"
"""The evaluator itself: no runtime is provisioned for it, and its egress is its own."""


class ProcessConfig(BaseConfig):
    """A long-running process a node runs besides the tool servers placed on it."""

    command: list[str] | None = None
    """None runs the image's own command."""
    environment: dict[str, str] = Field(default_factory=dict)
    """Reaches the node only, never the workspace."""
    passthrough: list[str] = Field(default_factory=list)
    """Variables whose values come from where the node runs, as a bare Compose `KEY`
    does: the rollout's environment for a node packed as a Compose service, the
    machine's own for a node on a machine of its own."""
    workdir: str | None = None
    readiness: list[str] | None = None
    """Exits 0 once the process serves. Runs as the process does: in its `workdir`,
    with its `environment`. On a packed node it replaces the image's healthcheck."""
    readiness_timeout: float = 600
    """Seconds the readiness check may take to pass before the rollout fails."""


class NodeConfig(BaseConfig):
    """A place where parts of a rollout run: a runtime with its own egress policy."""

    runtime: dict[str, Any] = Field(default_factory=dict)
    """Runtime fields that differ from the workspace's, such as `image`, `cpu` or
    `allow`. Setting `type` starts from that runtime's defaults instead."""
    process: ProcessConfig | None = None
    """A long-running process of the node's own. Without one, the node runs nothing of
    its own: a machine of its own, or a container that idles, for tool servers and
    hooks."""
    reachable_from: list[str] = Field(default_factory=lambda: [WORKSPACE])
    """Nodes that may open connections to this one. The framework adds the connections
    its own parts need: the harness to its tools and to where they execute."""
    isolated: bool = False
    """Give the node a machine of its own even when it could share one, e.g. to run
    agent-submitted code under a separate kernel."""


class TopologyConfig(BaseConfig):
    """Where a rollout's parts run. Declaring nothing is today's colocated rollout:
    one workspace, the harness inside it, tool servers on the evaluator."""

    nodes: dict[str, NodeConfig] = Field(default_factory=dict)
    """Nodes besides the evaluator. `main` may appear to set who reaches it; its
    runtime is the agent's."""
    harness_on: str = WORKSPACE
    """Where the harness's agent loop runs, holding the model credentials."""
    execution_on: str = WORKSPACE
    """Where the harness's native tools (shell, files, kernels) execute."""
    mcp_servers: dict[str, str] = Field(default_factory=dict)
    """The node each MCP server runs on."""
    compose: list[str] = Field(default_factory=list)
    """The nodes that are services of the task's Compose project, which runs them as
    authored."""

    @model_validator(mode="after")
    def _known_nodes(self):
        if HOST in self.nodes:
            raise ValueError(f"{HOST!r} is the evaluator, not a node to declare")
        if WORKSPACE in self.nodes and self.nodes[WORKSPACE].runtime:
            raise ValueError(
                f"{WORKSPACE!r} runs on the agent's runtime; set `agent.runtime` instead"
            )
        known = {WORKSPACE, HOST, *self.nodes}
        placed = {
            "harness_on": [self.harness_on],
            "execution_on": [self.execution_on],
            **{f"mcp_servers.{s}": [n] for s, n in self.mcp_servers.items()},
            **{
                f"nodes.{n}.reachable_from": c.reachable_from
                for n, c in self.nodes.items()
            },
        }
        for field, names in placed.items():
            if unknown := sorted(set(names) - known):
                raise ValueError(f"{field} names unknown nodes: {unknown}")
        if self.execution_on == HOST:
            raise ValueError(
                "native tools execute in a node, never on the evaluator: they run the "
                "agent's commands"
            )
        return self

    def node(self, name: str) -> NodeConfig:
        if name == WORKSPACE:
            # Nothing may open a connection to the workspace unless the task says so.
            return self.nodes.get(WORKSPACE, NodeConfig(reachable_from=[]))
        return self.nodes[name]
