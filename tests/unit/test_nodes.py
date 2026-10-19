"""The node model (rfcs/0001-nodes.md): each case is a rollout shape we run today,
planned from config alone, plus the refusals a runtime's capabilities imply."""

from dataclasses import replace

import pytest

import verifiers.v1 as vf
from verifiers.v1.runtimes import DockerConfig, PrimeConfig, SubprocessConfig, nodes
from verifiers.v1.runtimes.nodes import (
    Connection,
    NodeConfig,
    PlacementError,
    ProcessConfig,
    TopologyConfig,
    nodes_from_compose,
    plan,
    topology_from,
)

ATLAS = "prime/prime/mcp-atlas:1.2.5"
FORTH = "prime/prime/forth:1"
OFFLINE = PrimeConfig(allow=[])  # a task with `network_allow = []`


def capabilities(*, private_network: bool = True):
    """What each backend declares, with Prime also declaring the private network its
    LAN already provides (the RFC's next step)."""

    def read(config) -> nodes.Capabilities:
        declared = nodes.capabilities(config)
        if config.type == "prime":
            return replace(declared, private_network=private_network)
        return declared

    return read


def test_capabilities_are_what_the_backends_declare():
    prime = nodes.capabilities(PrimeConfig())
    assert (prime.network_policy, prime.compose, prime.port_exposure) == (
        True,
        True,
        False,
    )
    assert not prime.private_network  # declared by no backend yet
    docker = nodes.capabilities(DockerConfig())
    assert (docker.network_policy, docker.compose, docker.port_exposure) == (
        True,
        True,
        True,
    )
    local = nodes.capabilities(SubprocessConfig())
    assert (local.network_policy, local.compose) == (False, False)


def shape(result):
    return [machine.nodes for machine in result.machines], set(result.connections)


def test_declaring_nothing_is_todays_colocated_rollout():
    result = plan(TopologyConfig(), OFFLINE, capabilities())
    assert shape(result) == ([("main",)], set())
    assert result.runtimes["main"] is OFFLINE


def test_a_tool_server_on_the_evaluator_is_reached_through_the_tunnel():
    topology = TopologyConfig(mcp_servers={"search": "host"})
    assert shape(plan(topology, OFFLINE, capabilities())) == (
        [("main",)],
        {Connection("main", "host", "mcp:search", "tunnel")},
    )


def test_atlas_gets_a_machine_of_its_own_because_its_egress_differs():
    topology = TopologyConfig(
        nodes={
            "atlas": NodeConfig(
                runtime={"image": ATLAS, "cpu": 4, "memory": 10, "allow": ["*"]}
            )
        },
        mcp_servers={"atlas": "atlas"},
    )
    result = plan(topology, OFFLINE, capabilities())
    assert shape(result) == (
        [("main",), ("atlas",)],
        {Connection("main", "atlas", "mcp:atlas", "private network")},
    )
    atlas = result.runtimes["atlas"]
    assert (atlas.image, atlas.cpu, atlas.allow) == (ATLAS, 4, ["*"])
    assert atlas.labels == OFFLINE.labels  # the rest follows the workspace

    with pytest.raises(PlacementError, match="cannot join machines on a private"):
        plan(topology, OFFLINE, capabilities(private_network=False))


EOG = {
    "services": {
        "main": {"image": "task", "networks": ["front"]},
        "tools": {"image": "tools", "networks": ["front", "back"]},
        "calendar": {"image": "gym-calendar", "command": "serve", "networks": ["back"]},
        "gmail": {"image": "gym-gmail", "command": ["serve"], "networks": ["back"]},
    }
}


def test_enterprise_ops_gym_packs_onto_one_machine_and_keeps_the_gyms_behind_tools():
    topology = TopologyConfig(
        nodes=nodes_from_compose(EOG), mcp_servers={"enterprise_ops": "tools"}
    )
    machines, connections = shape(plan(topology, OFFLINE, capabilities()))
    assert machines == [("main", "tools", "calendar", "gmail")]
    pairs = {(c.source, c.target) for c in connections}
    assert {("main", "tools"), ("tools", "calendar"), ("tools", "gmail")} <= pairs
    assert not {("main", "calendar"), ("main", "gmail")} & pairs
    assert {c.via for c in connections} == {"container network"}
    assert topology.nodes["calendar"].process.command == ["sh", "-c", "serve"]


def test_forth_runs_agent_code_on_a_machine_of_its_own_only_when_isolated():
    def topology(isolated):
        node = NodeConfig(runtime={"image": FORTH}, isolated=isolated)
        return TopologyConfig(nodes={"forth": node}, mcp_servers={"forth": "forth"})

    assert shape(plan(topology(True), OFFLINE, capabilities())) == (
        [("main",), ("forth",)],
        {Connection("main", "forth", "mcp:forth", "private network")},
    )
    # The same policy as the workspace: without isolation it shares the machine.
    packed = plan(topology(False), OFFLINE, capabilities())
    assert shape(packed) == (
        [("main", "forth")],
        {Connection("main", "forth", "mcp:forth", "container network")},
    )


def test_split_execution_is_the_agent_loop_on_another_node():
    harness = NodeConfig(runtime={"image": "harness"}, reachable_from=[])
    split = TopologyConfig(
        nodes={"harness": harness}, harness_on="harness", execution_on="main"
    )
    # Packed: the loop and its credentials in one container, the workspace in another.
    assert shape(plan(split, OFFLINE, capabilities())) == (
        [("main", "harness")],
        {Connection("harness", "main", "execution", "container network")},
    )
    # VM to VM, when the workspace needs a machine of its own.
    apart = split.model_copy(
        update={
            "nodes": {
                "harness": harness,
                "main": NodeConfig(isolated=True, reachable_from=[]),
            }
        }
    )
    assert shape(plan(apart, OFFLINE, capabilities())) == (
        [("main",), ("harness",)],
        {Connection("harness", "main", "execution", "private network")},
    )


def test_a_harness_on_the_evaluator_needs_a_workspace_that_exposes_a_port():
    hosted = TopologyConfig(harness_on="host")
    with pytest.raises(PlacementError, match="prime runtime cannot expose a port"):
        plan(hosted, OFFLINE, capabilities())
    assert shape(plan(hosted, DockerConfig(), capabilities())) == (
        [("main",)],
        {Connection("host", "main", "execution", "exposed port")},
    )


def test_refusals_name_the_runtime_and_the_gap():
    with pytest.raises(ValueError, match="never on the evaluator"):
        TopologyConfig(execution_on="host")
    with pytest.raises(ValueError, match="unknown nodes"):
        TopologyConfig(mcp_servers={"search": "nowhere"})
    with pytest.raises(ValueError, match="agent's runtime"):
        TopologyConfig(nodes={"main": NodeConfig(runtime={"cpu": 8})})

    restricted = TopologyConfig(nodes={"tools": NodeConfig(runtime={"allow": []})})
    with pytest.raises(PlacementError, match="subprocess runtime does not enforce"):
        plan(restricted, SubprocessConfig(), capabilities())

    sidecar = TopologyConfig(
        nodes={"db": NodeConfig(process=ProcessConfig(command=["postgres"]))}
    )
    with pytest.raises(PlacementError, match="neither run Compose services"):
        plan(sidecar, SubprocessConfig(), capabilities())


def test_todays_knobs_lower_to_a_topology():
    toolsets = {
        "search": vf.ToolsetConfig(),
        "files": vf.ToolsetConfig(colocated=True),
        "gym": vf.ToolsetConfig(service="tools"),
        "forth": vf.ToolsetConfig(runtime=PrimeConfig(image=FORTH, allow=[])),
    }
    topology = topology_from(toolsets, EOG)
    assert topology.mcp_servers == {
        "search": "host",
        "files": "main",
        "gym": "tools",
        "forth": "tools-forth",
    }
    # A server with a runtime of its own keeps a machine of its own, joined to the
    # workspace over the private network even beside Compose services.
    result = plan(topology, OFFLINE, capabilities())
    assert result.machine("tools-forth") is not result.machine("main")
    assert Connection("main", "tools-forth", "mcp:forth", "private network") in set(
        result.connections
    )
    assert result.machine("host") is None
