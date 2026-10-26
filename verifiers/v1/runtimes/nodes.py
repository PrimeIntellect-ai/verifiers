"""Nodes: where each part of a rollout runs, and how the parts reach each other
(`rfcs/0001-nodes.md`).

Code says what runs (the task, the harness, tool servers); a topology says where. A
rollout's topology comes from what it declares (`topology_from`): its Compose
services, the nodes its task declares (`TaskData.nodes`), and each tool server's
placement (`ToolsetConfig.on`). `plan` turns it into the machines to provision and
the connections between them, and refuses what the runtimes cannot do before anything
is provisioned; `Deployment` realizes the plan.
"""

from __future__ import annotations

import re
import shlex
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

from pydantic import TypeAdapter

from verifiers.v1.configs.nodes import (
    HOST,
    WORKSPACE,
    NodeConfig,
    ProcessConfig,
    TopologyConfig,
)
from verifiers.v1.configs.runtime import NetworkPolicyConfig

if TYPE_CHECKING:
    from verifiers.v1.runtimes.compose import ComposeSpec
from verifiers.v1.runtimes import (
    RuntimeBackend,
    RuntimeConfig,
    SubprocessConfig,
    backend_cls,
)

IDLE = ("sh", "-c", "trap 'exit 0' TERM; while :; do sleep 3600 & wait $!; done")
"""A packed node's command when it runs no process of its own."""


class PlacementError(ValueError):
    """A topology the selected runtimes cannot realize."""


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
        private_network=cls.SUPPORTS_PRIVATE_NETWORK,
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
        if _backend(runtimes[source]) is not _backend(runtimes[target]):
            raise PlacementError(
                f"nodes {source!r} and {target!r} need a private network across the "
                f"{_name(runtimes[source])} and {_name(runtimes[target])} runtimes"
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
    # Without `type`, a node starts from the workspace's runtime; with it, from that
    # runtime's defaults, even when it names the workspace's type.
    inherits = "type" not in overrides
    kind = overrides.pop("type", workspace.type)
    # A node's egress policy replaces the workspace's whole: an `allow` alone must not
    # keep the workspace's `block`.
    policy = (
        {"allow": overrides.pop("allow", ["*"]), "block": overrides.pop("block", [])}
        if {"allow", "block"} & overrides.keys()
        else {}
    )
    base = workspace.model_dump() if inherits else {}
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


def _backend(config: RuntimeConfig) -> type[RuntimeBackend]:
    # Runtime plugins share `type = "custom"`; their backends tell them apart.
    return backend_cls(config)


def _name(config: RuntimeConfig) -> str:
    return getattr(config, "import_path", config.type)


def _policy(config: RuntimeConfig) -> tuple[tuple[str, ...], tuple[str, ...]] | None:
    if isinstance(config, NetworkPolicyConfig):
        return tuple(config.allow), tuple(config.block)
    return None


def _machines(
    topology: TopologyConfig,
    runtimes: Mapping[str, RuntimeConfig],
    capabilities: Callable[[RuntimeConfig], Capabilities],
) -> tuple[Machine, ...]:
    # Egress belongs to a machine, so only nodes under the workspace's policy can
    # share its machine. Every other node gets a machine of its own.
    workspace = runtimes[WORKSPACE]

    def shares(name: str) -> bool:
        config = runtimes[name]
        return (
            not topology.node(name).isolated
            and not topology.node(WORKSPACE).isolated
            and _backend(config) is _backend(workspace)
            and _policy(config) == _policy(workspace)
        )

    packed = [WORKSPACE, *(n for n in runtimes if n != WORKSPACE and shares(n))]
    if len(packed) == 1 or capabilities(workspace).compose:
        machines = [Machine(tuple(packed), workspace)]
    elif capabilities(workspace).private_network:
        machines = [Machine((name,), runtimes[name]) for name in packed]
    else:
        raise PlacementError(
            f"nodes {packed} would share a machine, but the {workspace.type} runtime "
            "can neither run Compose services nor join machines on a private network; "
            "use a supported runtime"
        )
    machines += [Machine((n,), runtimes[n]) for n in runtimes if n not in packed]
    return tuple(machines)


def topology_from(
    toolsets: Mapping[str, Any],
    compose: Mapping[str, Any] | None = None,
    nodes: Mapping[str, NodeConfig] | None = None,
    *,
    workspace: RuntimeConfig | None = None,
    borrowed: bool = False,
    compose_workspace: str | None = None,
) -> TopologyConfig:
    """A rollout's topology: the services of its Compose project, the nodes its task
    declares, and where each tool server runs, by server name.

    A server runs where its `on` says (`colocated` and `service` are its older
    spellings). Otherwise, a server with a runtime of its own gets a machine of its own
    when the workspace (`workspace`, unborrowed and without Compose services) can join
    one on a private network; any other server runs on the evaluator, which provisions
    a runtime of its own at a published port. `compose_workspace` names the Compose
    service that is the workspace (`main` by default)."""
    authored = WORKSPACE if compose_workspace is None else compose_workspace
    services = nodes_from_compose(compose, authored) if compose is not None else {}
    if both := sorted(services.keys() & dict(nodes or {}).keys()):
        raise ValueError(f"nodes {both} are both declared and Compose services")
    declared = {**services, **(nodes or {})}
    own_machines = (
        workspace is not None
        and compose is None
        and not borrowed
        and backend_cls(workspace).SUPPORTS_PRIVATE_NETWORK
    )
    servers = {}
    for name, config in toolsets.items():
        if config.url is not None:
            continue  # a remote server: nothing to place
        if config.service is not None:
            if compose is None or config.service not in compose.get("services", {}):
                raise ValueError(
                    f"server {name!r} runs in service {config.service!r}, which needs "
                    "a task whose Compose services define it"
                )
            servers[name] = WORKSPACE if config.service == authored else config.service
        elif config.node is not None:
            servers[name] = WORKSPACE if config.node == authored else config.node
        elif isinstance(config.runtime, SubprocessConfig):
            servers[name] = HOST
        elif own_machines and _backend(config.runtime) is _backend(workspace):
            # A runtime of its own is a machine of its own, a node named after the
            # server, so task hooks reach it as `deployment.target(name)`.
            if name in (WORKSPACE, HOST) or name in declared:
                raise ValueError(
                    f"tool server {name!r} needs a machine of its own named after it, "
                    "but that name is the workspace's, the evaluator's, or a node the "
                    "task declares"
                )
            declared[name] = NodeConfig(
                runtime=config.runtime.model_dump(), isolated=True
            )
            servers[name] = name
        else:
            servers[name] = HOST
    return TopologyConfig(nodes=declared, mcp_servers=servers, compose=sorted(services))


def task_topology(
    toolsets: Mapping[str, Any],
    spec: ComposeSpec | None,
    nodes: Mapping[str, NodeConfig] | None,
    *,
    workspace: RuntimeConfig,
    borrowed: bool = False,
) -> TopologyConfig:
    """`topology_from` for a task: its Compose file as authored, if it has one, and the
    nodes its data declares."""
    from verifiers.v1.runtimes.compose import authored_project

    return topology_from(
        toolsets,
        authored_project(spec) if spec is not None else None,
        nodes,
        workspace=workspace,
        borrowed=borrowed,
        compose_workspace=spec.workspace if spec is not None else None,
    )


def compose_project(topology: TopologyConfig, result: Plan) -> dict[str, Any]:
    """A Compose project for the declared nodes that share the workspace's machine.

    Each service gets a network of its own for egress (`vf-egress-<node>`), and each
    connection between two of them a network of those two (`vf-link-<n>`). Container
    networks are symmetric: a target can open connections to its source too, as on any
    Compose network."""
    machine = result.machine(WORKSPACE)
    assert machine is not None
    names = [name for name in machine.nodes if name != WORKSPACE]
    if invalid := [n for n in names if not re.fullmatch(r"[a-z0-9][a-z0-9_.-]*", n)]:
        raise PlacementError(f"nodes {invalid} need names Compose accepts as services")
    # Only mounts a node sets: the workspace's own, which a node inherits, are refused
    # with the workspace's Compose services.
    if mounted := [n for n in names if topology.node(n).runtime.get("mounts")]:
        raise PlacementError(
            f"nodes {mounted} mount storage, which a Compose service on the "
            "workspace's machine does not take; `isolated = true` gives a node a "
            "machine of its own"
        )
    services: dict[str, dict[str, Any]] = {
        name: {"networks": [f"vf-egress-{name}"]} for name in [WORKSPACE, *names]
    }
    networks: dict[str, dict[str, Any]] = {f"vf-egress-{name}": {} for name in services}
    links = [c for c in result.connections if c.via == "container network"]
    for index, connection in enumerate(links):
        network = f"vf-link-{index}"
        networks[network] = {"internal": True}
        for name in (connection.source, connection.target):
            services[name]["networks"].append(network)
    for name in names:
        service = services[name]
        service["image"] = result.runtimes[name].image
        # Only a workdir the node sets: it runs its own image, whose own default the
        # workspace's workdir must not replace.
        if (workdir := topology.node(name).runtime.get("workdir")) is not None:
            service["working_dir"] = workdir
        process = topology.node(name).process
        if process is None or process.command is not None:
            # As on a machine of its own, the command runs as given, past any image
            # entrypoint; without a process, the container idles for tool servers and
            # hooks. The image's own healthcheck probes a command that no longer runs.
            command = list(IDLE) if process is None else process.command
            service["entrypoint"] = command
            service["healthcheck"] = {"disable": True}
        if process is None:
            continue
        if process.environment or process.passthrough:
            service["environment"] = [
                *(f"{key}={value}" for key, value in process.environment.items()),
                *process.passthrough,
            ]
        if process.workdir is not None:
            service["working_dir"] = process.workdir
        if process.readiness is not None:
            service["healthcheck"] = {
                "test": ["CMD", *process.readiness],
                "interval": "2s",
                "retries": max(1, int(process.readiness_timeout // 2)),
            }
    # Compose interpolates `$VAR` from the evaluator's environment when it reads the
    # file; a node's values are literal, as on a machine of its own.
    for name in names:
        services[name] = {
            key: value if key == "networks" else _literal(value)
            for key, value in services[name].items()
        }
    return {"services": services, "networks": networks}


def _literal(value: Any) -> Any:
    if isinstance(value, str):
        return value.replace("$", "$$")
    if isinstance(value, list):
        return [_literal(item) for item in value]
    if isinstance(value, dict):
        return {key: _literal(item) for key, item in value.items()}
    return value


def nodes_from_compose(
    project: Mapping[str, Any], workspace: str = WORKSPACE
) -> dict[str, NodeConfig]:
    """A Compose project's services as nodes: Compose is an input format. Services
    that share a network may reach each other, as Compose lets them; the workspace
    service becomes `main`."""
    services: Mapping[str, Mapping[str, Any]] = project.get("services", {})
    if workspace != WORKSPACE and WORKSPACE in services:
        raise ValueError(
            f"the Compose workspace service {workspace!r} is the {WORKSPACE!r} node, "
            f"so no other service can be named {WORKSPACE!r}"
        )

    def name(service: str) -> str:
        return WORKSPACE if service == workspace else service

    def networks(service: Mapping[str, Any]) -> set[str]:
        mode = service.get("network_mode")
        if mode == "none":
            return set()
        if isinstance(mode, str) and mode.startswith("service:"):
            # It shares that service's network namespace; Compose rejects cycles.
            return networks(services[mode.removeprefix("service:")])
        if mode == "bridge":
            # Docker's own default bridge, not a network of the project: it shares
            # one only with other services in bridge mode.
            return {"\0bridge"}
        if mode is not None:
            raise ValueError(f"network_mode {mode!r} has no node equivalent")
        declared = service.get("networks") or ["default"]
        return set(declared if isinstance(declared, list) else declared.keys())

    def environment(spec: Mapping[str, Any]) -> dict[str, str | None]:
        """`KEY: value` or `KEY=value`; None for a bare `KEY`."""
        declared = spec.get("environment") or {}
        if isinstance(declared, list):
            return {
                key: value if sep else None
                for key, sep, value in (item.partition("=") for item in declared)
            }
        return {
            key: None if value is None else str(value)
            for key, value in declared.items()
        }

    nodes = {}
    for service, spec in services.items():
        peers = [
            name(other)
            for other, other_spec in services.items()
            if other != service and networks(spec) & networks(other_spec)
        ]
        command = spec.get("command")
        if isinstance(command, str):
            command = shlex.split(command)  # as Compose splits it: no shell
        variables = environment(spec)
        nodes[name(service)] = NodeConfig(
            # The workspace runs on the agent's runtime, with the task's image.
            runtime=(
                {"image": spec["image"]}
                if "image" in spec and service != workspace
                else {}
            ),
            # Every other service runs a process: its command, or its image's own.
            process=(
                ProcessConfig(
                    command=command,
                    environment={k: v for k, v in variables.items() if v is not None},
                    passthrough=[k for k, v in variables.items() if v is None],
                    workdir=spec.get("working_dir"),
                )
                if service != workspace
                else None
            ),
            # Nothing reaches the workspace unless Compose let it.
            reachable_from=peers,
        )
    return nodes
