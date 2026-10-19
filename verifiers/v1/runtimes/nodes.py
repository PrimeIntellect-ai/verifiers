"""Nodes: where each part of a rollout runs, and how the parts reach each other.

A design prototype for `rfcs/0001-nodes.md`. Code says what runs (the task, the
harness, tool servers); a topology says where. `plan` turns a topology into the
machines to provision and the connections between them, and refuses what the runtimes
cannot do before anything is provisioned. Nothing provisions from a plan yet:
`Deployment` still uses its strategies.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import Field, TypeAdapter, model_validator
from pydantic_config import BaseConfig

from verifiers.v1.configs.runtime import NetworkPolicyConfig
from verifiers.v1.runtimes import RuntimeConfig, SubprocessConfig, backend_cls

WORKSPACE = "main"
"""The agent's workspace. Its runtime is the agent's runtime, resolved for the task."""
HOST = "host"
"""The evaluator itself: no runtime is provisioned for it, and its egress is its own."""


class PlacementError(ValueError):
    """A topology the selected runtimes cannot realize."""


class ProcessConfig(BaseConfig):
    """A long-running process a node runs besides the tool servers placed on it."""

    command: list[str]
    environment: dict[str, str] = Field(default_factory=dict)
    """Reaches the node only, never the workspace."""
    workdir: str | None = None
    readiness: list[str] | None = None
    """Exits 0 once the process serves."""


class NodeConfig(BaseConfig):
    """A place where parts of a rollout run: a runtime with its own egress policy."""

    runtime: dict[str, Any] = Field(default_factory=dict)
    """Runtime fields that differ from the workspace's, such as `image`, `cpu` or
    `allow`. Setting `type` starts from that runtime's defaults instead."""
    process: ProcessConfig | None = None
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


@dataclass(frozen=True)
class Capabilities:
    """What a runtime can do for a topology, read from its backend's declarations."""

    network_policy: bool
    compose: bool
    private_network: bool
    port_exposure: bool


def capabilities(config: RuntimeConfig) -> Capabilities:
    cls = backend_cls(config)
    return Capabilities(
        network_policy=isinstance(config, NetworkPolicyConfig),
        compose=cls.SUPPORTS_COMPOSE,
        # Declared by a backend that can join machines on a private network, as
        # Prime's LAN does (verifiers#2876); no backend declares it yet.
        private_network=getattr(cls, "SUPPORTS_PRIVATE_NETWORK", False),
        port_exposure=cls.SUPPORTS_PORT_EXPOSURE,
    )


@dataclass(frozen=True)
class Machine:
    """One provisioned runtime. Several nodes on one machine run as its containers
    and share its egress policy; a single node boots its image as the machine."""

    nodes: tuple[str, ...]
    runtime: RuntimeConfig


Via = Literal["container network", "private network", "tunnel", "exposed port"]


@dataclass(frozen=True)
class Connection:
    source: str
    target: str
    purpose: str
    """`execution` (the agent loop drives its tools), `mcp:<server>`, or `service`."""
    via: Via


@dataclass(frozen=True)
class Plan:
    runtimes: Mapping[str, RuntimeConfig]
    """Each node's resolved runtime; the evaluator has none."""
    machines: tuple[Machine, ...]
    connections: tuple[Connection, ...]

    def machine(self, node: str) -> Machine | None:
        return next((m for m in self.machines if node in m.nodes), None)


def plan(
    topology: TopologyConfig,
    workspace: RuntimeConfig,
    capabilities: Callable[[RuntimeConfig], Capabilities] = capabilities,
) -> Plan:
    """The machines and connections that realize `topology`, with the workspace on
    `workspace` (the agent's runtime, resolved for the task). Raises `PlacementError`
    for anything the runtimes cannot do."""
    runtimes = {WORKSPACE: workspace} | {
        name: _runtime(name, node, workspace, capabilities)
        for name, node in topology.nodes.items()
        if name != WORKSPACE
    }
    machines = _machines(topology, runtimes, capabilities)
    machine_of = {node: m for m in machines for node in m.nodes}

    def via(source: str, target: str) -> Via:
        if target == HOST:
            return "tunnel"  # inward, as the model endpoint and host tool servers are
        if source == HOST:
            if not capabilities(runtimes[target]).port_exposure:
                raise PlacementError(
                    f"the evaluator cannot reach node {target!r}: the "
                    f"{runtimes[target].type} runtime cannot expose a port; place the "
                    "consumer on a node"
                )
            return "exposed port"
        if machine_of[source] is machine_of[target]:
            return "container network"
        for node in (source, target):
            if not capabilities(runtimes[node]).private_network:
                raise PlacementError(
                    f"nodes {source!r} and {target!r} run on separate machines, but "
                    f"the {runtimes[node].type} runtime cannot join machines on a "
                    "private network; use a supported runtime"
                )
        if runtimes[source].type != runtimes[target].type:
            raise PlacementError(
                f"nodes {source!r} and {target!r} need a private network across the "
                f"{runtimes[source].type} and {runtimes[target].type} runtimes"
            )
        return "private network"

    wanted: dict[tuple[str, str], str] = {}
    if topology.execution_on != topology.harness_on:
        wanted[(topology.harness_on, topology.execution_on)] = "execution"
    for server, node in topology.mcp_servers.items():
        if node != topology.harness_on:
            wanted.setdefault((topology.harness_on, node), f"mcp:{server}")
    for name in runtimes:
        for source in topology.node(name).reachable_from:
            wanted.setdefault((source, name), "service")
    connections = tuple(
        Connection(source, target, purpose, via(source, target))
        for (source, target), purpose in wanted.items()
        if source != target
    )
    return Plan(runtimes=runtimes, machines=machines, connections=connections)


def _runtime(
    name: str,
    node: NodeConfig,
    workspace: RuntimeConfig,
    capabilities: Callable[[RuntimeConfig], Capabilities],
) -> RuntimeConfig:
    overrides = dict(node.runtime)
    kind = overrides.pop("type", workspace.type)
    # A node's egress policy replaces the workspace's whole: an `allow` alone must not
    # keep the workspace's `block`.
    policy = (
        {"allow": overrides.pop("allow", ["*"]), "block": overrides.pop("block", [])}
        if {"allow", "block"} & overrides.keys()
        else {}
    )
    base = workspace.model_dump() if kind == workspace.type else {}
    adapter: TypeAdapter[Any] = TypeAdapter(RuntimeConfig)
    config = adapter.validate_python({**base, **overrides, "type": kind})
    if policy:
        if not capabilities(config).network_policy:
            raise PlacementError(
                f"node {name!r} sets an egress policy, but the {config.type} runtime "
                "does not enforce network policies; use a supported runtime"
            )
        config = adapter.validate_python({**config.model_dump(), **policy})
    return config


def _policy(config: RuntimeConfig) -> tuple[tuple[str, ...], tuple[str, ...]] | None:
    if isinstance(config, NetworkPolicyConfig):
        return tuple(config.allow), tuple(config.block)
    return None


def _machines(
    topology: TopologyConfig,
    runtimes: Mapping[str, RuntimeConfig],
    capabilities: Callable[[RuntimeConfig], Capabilities],
) -> tuple[Machine, ...]:
    # Egress belongs to a machine, so only nodes with one policy can share one.
    groups: dict[object, list[str]] = {}
    for name, config in runtimes.items():
        key = name if topology.node(name).isolated else (config.type, _policy(config))
        groups.setdefault(key, []).append(name)
    machines = []
    for names in groups.values():
        config = runtimes[names[0]]
        if len(names) == 1 or capabilities(config).compose:
            machines.append(Machine(tuple(names), config))
        elif capabilities(config).private_network:
            machines.extend(Machine((name,), runtimes[name]) for name in names)
        else:
            raise PlacementError(
                f"nodes {names} would share a machine, but the {config.type} runtime "
                "can neither run Compose services nor join machines on a private "
                "network; use a supported runtime"
            )
    return tuple(machines)


def topology_from(
    toolsets: Mapping[str, Any],
    compose: Mapping[str, Any] | None = None,
) -> TopologyConfig:
    """Today's placement knobs as a topology: each toolset's `colocated`, `runtime` or
    `service`, and a task's Compose services. It shows that the model can say
    everything they say; nothing calls it yet."""
    nodes = nodes_from_compose(compose) if compose is not None else {}
    servers = {}
    for name, config in toolsets.items():
        if config.service is not None:
            servers[name] = config.service
        elif config.colocated:
            servers[name] = WORKSPACE
        elif not isinstance(config.runtime, SubprocessConfig):
            # A runtime of its own is a machine of its own, as a tool VM is today.
            node = f"tools-{name}"
            nodes[node] = NodeConfig(runtime=config.runtime.model_dump(), isolated=True)
            servers[name] = node
        elif config.url is None:
            servers[name] = HOST
    return TopologyConfig(nodes=nodes, mcp_servers=servers)


def nodes_from_compose(
    project: Mapping[str, Any], workspace: str = WORKSPACE
) -> dict[str, NodeConfig]:
    """A Compose project's services as nodes: Compose is an input format. Services
    that share a network may reach each other, as Compose lets them; the workspace
    service becomes `main`."""
    services: Mapping[str, Mapping[str, Any]] = project.get("services", {})

    def name(service: str) -> str:
        return WORKSPACE if service == workspace else service

    def networks(service: Mapping[str, Any]) -> set[str]:
        declared = service.get("networks") or ["default"]
        return set(declared if isinstance(declared, list) else declared.keys())

    nodes = {}
    for service, spec in services.items():
        peers = [
            name(other)
            for other, other_spec in services.items()
            if other != service and networks(spec) & networks(other_spec)
        ]
        command = spec.get("command")
        nodes[name(service)] = NodeConfig(
            # The workspace runs on the agent's runtime, with the task's image.
            runtime=(
                {"image": spec["image"]}
                if "image" in spec and service != workspace
                else {}
            ),
            process=(
                ProcessConfig(
                    command=command
                    if isinstance(command, list)
                    else ["sh", "-c", command],
                    environment=dict(spec.get("environment") or {}),
                    workdir=spec.get("working_dir"),
                )
                if command is not None and service != workspace
                else None
            ),
            # Nothing reaches the workspace unless Compose let it.
            reachable_from=peers,
        )
    return nodes
