"""The node model (rfcs/0001-nodes.md): each case is a rollout shape we run today,
planned from config alone, plus the refusals a runtime's capabilities imply."""

from dataclasses import replace

import pytest
from pydantic import TypeAdapter

import verifiers.v1 as vf
from verifiers.v1.runtimes import (
    DockerConfig,
    PrimeConfig,
    RuntimeBackend,
    RuntimeConfig,
    SubprocessConfig,
    nodes,
)
from verifiers.v1.runtimes.nodes import (
    Connection,
    NodeConfig,
    PlacementError,
    ProcessConfig,
    TopologyConfig,
    compose_project,
    nodes_from_compose,
    plan,
    topology_from,
)

ATLAS = "prime/prime/mcp-atlas:1.2.5"
FORTH = "prime/prime/forth:1"
OFFLINE = PrimeConfig(allow=[])  # a task with `network_allow = []`


def without_private_network(config) -> nodes.Capabilities:
    return replace(nodes.capabilities(config), private_network=False)


def test_capabilities_are_what_the_backends_declare():
    prime = nodes.capabilities(PrimeConfig())
    assert (prime.network_policy, prime.compose, prime.private_network) == (
        True,
        True,
        True,
    )
    assert not prime.port_exposure
    docker = nodes.capabilities(DockerConfig())
    assert (docker.network_policy, docker.compose, docker.port_exposure) == (
        True,
        True,
        True,
    )
    assert not docker.private_network
    local = nodes.capabilities(SubprocessConfig())
    assert (local.network_policy, local.compose) == (False, False)


def shape(result):
    return [machine.nodes for machine in result.machines], set(result.connections)


def test_declaring_nothing_is_todays_colocated_rollout():
    result = plan(TopologyConfig(), OFFLINE)
    assert shape(result) == ([("main",)], set())
    assert result.runtimes["main"] is OFFLINE


def test_a_tool_server_on_the_evaluator_is_reached_through_the_tunnel():
    topology = TopologyConfig(mcp_servers={"search": "host"})
    assert shape(plan(topology, OFFLINE)) == (
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
    result = plan(topology, OFFLINE)
    assert shape(result) == (
        [("main",), ("atlas",)],
        {Connection("main", "atlas", "mcp:atlas", "private network")},
    )
    atlas = result.runtimes["atlas"]
    assert (atlas.image, atlas.cpu, atlas.allow) == (ATLAS, 4, ["*"])
    assert atlas.labels == OFFLINE.labels  # the rest follows the workspace

    with pytest.raises(PlacementError, match="cannot join machines on a private"):
        plan(topology, OFFLINE, without_private_network)


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
    machines, connections = shape(plan(topology, OFFLINE))
    assert machines == [("main", "tools", "calendar", "gmail")]
    pairs = {(c.source, c.target) for c in connections}
    assert {("main", "tools"), ("tools", "calendar"), ("tools", "gmail")} <= pairs
    assert not {("main", "calendar"), ("main", "gmail")} & pairs
    assert {c.via for c in connections} == {"container network"}
    assert topology.nodes["calendar"].process.command == ["serve"]


def test_forth_runs_agent_code_on_a_machine_of_its_own_only_when_isolated():
    def topology(isolated):
        node = NodeConfig(runtime={"image": FORTH}, isolated=isolated)
        return TopologyConfig(nodes={"forth": node}, mcp_servers={"forth": "forth"})

    assert shape(plan(topology(True), OFFLINE)) == (
        [("main",), ("forth",)],
        {Connection("main", "forth", "mcp:forth", "private network")},
    )
    # The same policy as the workspace: without isolation it shares the machine.
    packed = plan(topology(False), OFFLINE)
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
    assert shape(plan(split, OFFLINE)) == (
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
    assert shape(plan(apart, OFFLINE)) == (
        [("main",), ("harness",)],
        {Connection("harness", "main", "execution", "private network")},
    )


def test_a_harness_on_the_evaluator_needs_a_workspace_that_exposes_a_port():
    hosted = TopologyConfig(harness_on="host")
    with pytest.raises(PlacementError, match="prime runtime cannot expose a port"):
        plan(hosted, OFFLINE)
    assert shape(plan(hosted, DockerConfig())) == (
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
        plan(restricted, SubprocessConfig())

    sidecar = TopologyConfig(
        nodes={"db": NodeConfig(process=ProcessConfig(command=["postgres"]))}
    )
    with pytest.raises(PlacementError, match="neither run Compose services"):
        plan(sidecar, SubprocessConfig())


def test_todays_knobs_lower_to_a_topology():
    toolsets = {
        "search": vf.ToolsetConfig(),
        "files": vf.ToolsetConfig(colocated=True),
        "gym": vf.ToolsetConfig(service="tools"),
        "forth": vf.ToolsetConfig(runtime=PrimeConfig(image=FORTH, allow=[])),
    }
    topology = topology_from(toolsets, EOG, workspace=OFFLINE)
    # Beside Compose services, a server with a runtime of its own keeps its published
    # port, provisioned from the evaluator's side.
    assert topology.mcp_servers == {
        "search": "host",
        "files": "main",
        "gym": "tools",
        "forth": "host",
    }
    assert topology.compose == ["calendar", "gmail", "main", "tools"]
    # Without them, it gets a machine of its own on the private network.
    alone = topology_from({"forth": toolsets["forth"]}, workspace=OFFLINE)
    result = plan(alone, OFFLINE)
    assert result.machine("forth") is not result.machine("main")
    assert set(result.connections) == {
        Connection("main", "forth", "mcp:forth", "private network")
    }
    assert result.machine("host") is None


def test_compose_services_keep_their_compose_meaning():
    project = {
        "services": {
            "main": {"image": "task"},
            "db": {
                "image": "postgres",
                "environment": ["POSTGRES_PASSWORD=pw", "PGDATA"],
                "working_dir": "/data",
            },
            "web": {
                "image": "web",
                "command": "serve --root '$HOME'",
                "environment": {"PORT": 8080},
            },
            "batch": {"image": "batch", "network_mode": "none"},
        }
    }
    services = nodes_from_compose(project)
    # The image's own command, with the service's environment and working directory;
    # a bare `KEY` takes its value from the node's environment.
    assert services["db"].process == ProcessConfig(
        environment={"POSTGRES_PASSWORD": "pw"}, passthrough=["PGDATA"], workdir="/data"
    )
    # Split as Compose splits it, with no shell to expand `$HOME`.
    assert services["web"].process.command == ["serve", "--root", "$HOME"]
    assert services["web"].process.environment == {"PORT": "8080"}
    # No network: nothing reaches it, and it reaches nothing.
    assert services["batch"].reachable_from == []
    assert "batch" not in services["db"].reachable_from
    bridged = {"services": {"main": {}, "x": {"network_mode": "bridge"}}}
    assert nodes_from_compose(bridged)["x"].reachable_from == ["main"]
    host = {"services": {"main": {}, "x": {"image": "x", "network_mode": "host"}}}
    with pytest.raises(ValueError, match="network_mode 'host' has no node"):
        nodes_from_compose(host)


def test_a_node_that_names_its_type_starts_from_that_runtimes_defaults():
    workspace = PrimeConfig(allow=[], labels=["run"])
    topology = TopologyConfig(
        nodes={
            "inherits": NodeConfig(runtime={"image": ATLAS}),
            "fresh": NodeConfig(runtime={"type": "prime", "image": ATLAS}),
        }
    )
    runtimes = plan(topology, workspace).runtimes
    assert (runtimes["inherits"].labels, runtimes["inherits"].allow) == (["run"], [])
    assert (runtimes["fresh"].labels, runtimes["fresh"].allow) == (
        PrimeConfig().labels,
        PrimeConfig().allow,
    )


class Plugin(RuntimeBackend):
    SUPPORTS_PRIVATE_NETWORK = True


class OtherPlugin(Plugin):
    pass


def test_runtime_plugins_are_told_apart_by_their_backends():
    def plugin(name):
        path = f"{__name__}.{name}"
        return TypeAdapter(RuntimeConfig).validate_python(
            {"type": "custom", "import_path": path}
        )

    other = NodeConfig(runtime={"import_path": f"{__name__}.OtherPlugin"})
    topology = TopologyConfig(nodes={"tools": other}, mcp_servers={"tools": "tools"})
    # Both are `type = "custom"`, but they cannot share a machine or a network.
    with pytest.raises(PlacementError, match="across the .*Plugin and .*OtherPlugin"):
        plan(topology, plugin("Plugin"))
    same = TopologyConfig(
        nodes={"tools": NodeConfig(isolated=True)}, mcp_servers={"tools": "tools"}
    )
    assert shape(plan(same, plugin("Plugin"))) == (
        [("main",), ("tools",)],
        {Connection("main", "tools", "mcp:tools", "private network")},
    )


def test_a_remote_toolset_is_not_placed_and_tool_machines_take_free_names():
    remote = vf.ToolsetConfig(url="https://tools.example/mcp", runtime=PrimeConfig())
    assert topology_from({"remote": remote}, workspace=OFFLINE).mcp_servers == {}
    declared = {"search": NodeConfig(runtime={"image": "search"})}
    own = vf.ToolsetConfig(runtime=PrimeConfig())
    with pytest.raises(ValueError, match="'search' needs a machine of its own named"):
        topology_from({"search": own}, nodes=declared, workspace=OFFLINE)
    with pytest.raises(ValueError, match="'main' needs a machine of its own named"):
        topology_from({"main": own}, workspace=OFFLINE)
    authored = {"services": {"main": {}, "db": {}}}
    with pytest.raises(ValueError, match=r"nodes \['db'\] are both declared"):
        topology_from({}, authored, {"db": NodeConfig()})


def test_only_the_workspace_s_machine_is_shared():
    online = NodeConfig(runtime={"allow": ["*"]})
    topology = TopologyConfig(
        nodes={"atlas": online, "search": online},
        mcp_servers={"atlas": "atlas", "search": "search"},
    )
    # The same policy as each other, but not the workspace's: a machine each.
    assert shape(plan(topology, OFFLINE))[0] == [("main",), ("atlas",), ("search",)]


def test_declared_nodes_on_the_workspace_s_machine_become_a_compose_project():
    db = NodeConfig(
        runtime={"image": "postgres"},
        process=ProcessConfig(
            environment={"POSTGRES_PASSWORD": "pw"},
            passthrough=["PGDATA"],
            readiness=["pg_isready"],
        ),
    )
    cache = NodeConfig(
        runtime={"image": "redis"},
        process=ProcessConfig(command=["redis-server"], workdir="/data"),
        reachable_from=["db"],
    )
    topology = TopologyConfig(nodes={"db": db, "cache": cache})
    project = compose_project(topology, plan(topology, OFFLINE))
    services, networks = project["services"], project["networks"]
    # A network of its own for egress, and one per connection, shared by its two ends.
    links = {
        frozenset(n for n, s in services.items() if link in s["networks"])
        for link in networks
        if link.startswith("vf-link-")
    }
    assert links == {frozenset({"main", "db"}), frozenset({"db", "cache"})}
    for name, service in services.items():
        assert service["networks"][0] == f"vf-egress-{name}"
        assert networks[f"vf-egress-{name}"] == {}
    assert all(
        networks[link] == {"internal": True} for link in networks if "link" in link
    )
    assert services["db"]["image"] == "postgres" and "command" not in services["db"]
    assert services["db"]["environment"] == ["POSTGRES_PASSWORD=pw", "PGDATA"]
    assert services["db"]["healthcheck"]["test"] == ["CMD", "pg_isready"]
    assert services["cache"]["command"] == ["redis-server"]
    assert services["cache"]["working_dir"] == "/data"
    with pytest.raises(PlacementError, match="names Compose accepts"):
        named = TopologyConfig(nodes={"DB!": NodeConfig()})
        compose_project(named, plan(named, OFFLINE))


def test_egress_networks_and_connection_networks_never_share_a_name():
    # `main-db`'s egress network must not be the main -> db connection's network.
    nodes = {"db": NodeConfig(), "main-db": NodeConfig()}
    topology = TopologyConfig(nodes=nodes)
    networks = compose_project(topology, plan(topology, OFFLINE))["networks"]
    assert len(networks) == 5  # three egress networks, two connections


def test_a_renamed_workspace_service_cannot_leave_another_named_main():
    project = {"services": {"app": {}, "main": {"image": "other"}}}
    with pytest.raises(ValueError, match="no other service can be named 'main'"):
        nodes_from_compose(project, workspace="app")


def test_a_packed_node_keeps_its_own_workdir_and_on_follows_the_workspace_rename():
    nodes = {
        "db": NodeConfig(runtime={"image": "pg", "workdir": "/data"}),
        "cache": NodeConfig(runtime={"image": "redis"}),
        "web": NodeConfig(
            runtime={"image": "web", "workdir": "/srv"},
            process=ProcessConfig(workdir="/srv/app"),
        ),
    }
    workspace = PrimeConfig(allow=[], workdir="/workspace")
    topology = TopologyConfig(nodes=nodes)
    services = compose_project(topology, plan(topology, workspace))["services"]
    assert services["db"]["working_dir"] == "/data"
    assert "working_dir" not in services["cache"]  # its image's own, not /workspace
    assert services["web"]["working_dir"] == "/srv/app"  # the process's wins
    project = {"services": {"app": {}, "tools": {"image": "tools"}}}
    servers = topology_from(
        {"app": vf.ToolsetConfig(on="app")}, project, compose_workspace="app"
    ).mcp_servers
    assert servers == {"app": "main"}
