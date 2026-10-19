import asyncio
from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest

import verifiers.v1 as vf
from verifiers.v1.errors import ToolsetError
from verifiers.v1.mcp import launch
from verifiers.v1.runtimes import (
    DockerBackend,
    DockerConfig,
    PrimeBackend,
    PrimeConfig,
    private_network,
)
from verifiers.v1.runtimes.backends import prime
from verifiers.v1.runtimes.nodes import (
    NodeConfig,
    PlacementError,
    ProcessConfig,
    TopologyConfig,
    plan,
    topology_from,
)
from verifiers.v1.runtimes.strategies import (
    _ColocatedDeployment,
    _ContainerDeployment,
    _deployment_strategy,
    _PrivateNetworkDeployment,
    start_process,
)

ATLAS = PrimeConfig(image="atlas:1", cpu=4, memory=10)  # open egress by default
FORTH = PrimeConfig(image="forth:1", allow=[])


def toolset(name, **config):
    return SimpleNamespace(server_name=name, config=vf.ToolsetConfig(**config))


def test_tool_servers_with_runtimes_of_their_own_get_machines_beside_prime():
    toolsets = {
        "atlas": vf.ToolsetConfig(runtime=ATLAS),
        "search": vf.ToolsetConfig(),  # the evaluator
        "files": vf.ToolsetConfig(colocated=True, runtime=ATLAS),  # the workspace
        "remote": vf.ToolsetConfig(url="https://tools.example/mcp"),
        "local": vf.ToolsetConfig(runtime=DockerConfig()),  # a published port
    }
    topology = topology_from(toolsets, workspace=PrimeConfig())
    assert topology.mcp_servers == {
        "atlas": "atlas",
        "search": "host",
        "files": "main",
        "local": "host",
    }
    atlas = topology.nodes["atlas"]
    assert atlas.isolated and atlas.runtime == ATLAS.model_dump()
    # Without a Prime workspace of the rollout's own and without Compose services,
    # a server keeps its runtime's published port, on the evaluator's side.
    compose = {"services": {"main": {}, "tools": {"image": "tools"}}}
    for args, kwargs in [
        ((), {"workspace": DockerConfig()}),
        ((), {"workspace": PrimeConfig(), "borrowed": True}),
        ((compose,), {"workspace": PrimeConfig()}),
    ]:
        assert topology_from(toolsets, *args, **kwargs).mcp_servers["atlas"] == "host"


def test_a_server_runs_on_the_node_its_placement_names():
    nodes = {"web": NodeConfig(runtime={"allow": ["*"]})}
    compose = {"services": {"app": {}, "tools": {"image": "tools"}}}
    servers = topology_from(
        {
            "on": vf.ToolsetConfig(on="tools"),
            "gym": vf.ToolsetConfig(service="tools"),
            "app": vf.ToolsetConfig(service="app"),  # the workspace's own service
        },
        compose,
        compose_workspace="app",
    ).mcp_servers
    assert servers == {"on": "tools", "gym": "tools", "app": "main"}
    assert topology_from(
        {"web": vf.ToolsetConfig(on="web")}, nodes=nodes
    ).mcp_servers == {"web": "web"}
    with pytest.raises(ValueError, match="which needs a task whose Compose services"):
        topology_from({"gym": vf.ToolsetConfig(service="tools")}, nodes=nodes)
    with pytest.raises(ValueError, match="unknown nodes: \\['db'\\]"):
        topology_from({"db": vf.ToolsetConfig(on="db")}, nodes=nodes)


def test_a_placement_takes_one_spelling_and_a_node_s_runtime():
    with pytest.raises(ValueError, match="takes no `colocated`, `service` or `url`"):
        vf.ToolsetConfig(on="web", colocated=True)
    with pytest.raises(ValueError, match="on node 'web' runs on that node"):
        vf.ToolsetConfig(on="web", runtime=ATLAS)
    # The evaluator provisions a runtime of the server's own, as without `on`.
    assert vf.ToolsetConfig(on="host", runtime=ATLAS).node == "host"


OFFLINE = PrimeConfig(allow=[])


def realize(topology, spec=None, workspace=OFFLINE):
    return _deployment_strategy(
        None, PrimeBackend, spec, plan(topology, workspace), topology
    )


def test_the_plan_picks_how_the_deployment_realizes_it():
    assert isinstance(realize(TopologyConfig()), _ColocatedDeployment)
    tools = topology_from({"atlas": vf.ToolsetConfig(runtime=ATLAS)}, workspace=OFFLINE)
    own = realize(tools)
    assert isinstance(own, _PrivateNetworkDeployment)
    assert own._machines["atlas"].image == "atlas:1"
    assert own._rules == [("main", "atlas")]
    assert "atlas" in own._served
    # The task's Compose services run as authored.
    authored = topology_from({}, {"services": {"main": {}, "db": {"image": "pg"}}})
    composed = realize(authored, vf.ComposeSpec("task"))
    assert isinstance(composed, _ContainerDeployment) and composed._project is None
    # Declared nodes under the workspace's policy join a Compose project of their own.
    sidecar = TopologyConfig(nodes={"db": NodeConfig(runtime={"image": "pg"})})
    generated = realize(sidecar)
    assert isinstance(generated, _ContainerDeployment)
    assert generated._project["services"]["db"]["image"] == "pg"


def test_a_plan_the_deployment_cannot_realize_is_refused_before_provisioning():
    sidecar = {"db": NodeConfig(runtime={"image": "pg"})}
    online = {"web": NodeConfig(runtime={"allow": ["*"]})}
    authored = {"services": {"main": {}, "tools": {"image": "tools"}}}
    with pytest.raises(PlacementError, match="declare them as services of that"):
        realize(topology_from({}, authored, sidecar), vf.ComposeSpec("task"))
    with pytest.raises(PlacementError, match="cannot join it on a private network"):
        realize(TopologyConfig(nodes={**sidecar, **online}))
    imaged = {"web": NodeConfig(runtime={"allow": ["*"]}, process=ProcessConfig())}
    with pytest.raises(PlacementError, match="set `process.command`"):
        realize(TopologyConfig(nodes=imaged))


def test_the_private_network_needs_a_runtime_that_declares_one():
    workspace = DockerBackend(DockerConfig(), name="ws")

    async def enter():
        members = {"tools": DockerConfig()}
        async with private_network.private_network(workspace, members, None):
            pass

    with pytest.raises(ValueError, match="the docker runtime cannot join machines"):
        asyncio.run(enter())


def test_a_prime_network_joins_only_prime_machines_by_known_names(monkeypatch):
    monkeypatch.setattr(prime, "ensure_prime_auth", lambda: None)
    vm = PrimeBackend(ATLAS)

    async def enter(members, rules):
        async with PrimeBackend.private_network(members, rules):
            pass

    with pytest.raises(ValueError, match=r"only Prime machines, not \['ws'\]"):
        asyncio.run(enter({"ws": DockerBackend(DockerConfig()), "vm": vm}, []))
    with pytest.raises(ValueError, match=r"unknown members: \['db'\]"):
        asyncio.run(enter({"vm": vm}, [("vm", "db")]))


class FakeLanClient:
    def __init__(self, events):
        self.events = events

    async def create_lan_network(self, name, rules):
        self.events.append(("network", [(rule.from_, rule.to) for rule in rules]))
        return SimpleNamespace(id="net1")

    async def delete_lan_network(self, network_id):
        self.events.append("network deleted")


def boot(monkeypatch, members):
    """Enter `private_network` with a fake LAN client and Prime VMs that record their
    boot and stop there."""
    events = []

    @asynccontextmanager
    async def leased_client():
        yield FakeLanClient(events)

    async def start(self):
        events.append(("boot", self.name, self.config, self.lan))
        raise RuntimeError("booted")

    monkeypatch.setattr(prime, "leased_client", leased_client)
    monkeypatch.setattr(prime, "ensure_prime_auth", lambda: None)
    monkeypatch.setattr(PrimeBackend, "start", start)
    workspace = PrimeBackend(PrimeConfig(allow=[]), name="ws")

    async def enter():
        async with private_network.private_network(workspace, members, workspace.start):
            pass

    with pytest.raises(RuntimeError, match="booted"):
        asyncio.run(enter())
    return events


def test_each_vm_boots_its_server_runtime_as_given(monkeypatch):
    events = boot(monkeypatch, {"atlas": ATLAS, "forth": FORTH})
    vms = {e[1]: e[2] for e in events if isinstance(e, tuple) and e[0] == "boot"}

    def shape(config):
        return config.image, config.cpu, config.memory, config.allow

    assert shape(vms["ws-node-0"]) == ("atlas:1", 4, 10, ["*"])
    assert shape(vms["ws-node-1"]) == ("forth:1", FORTH.cpu, FORTH.memory, [])
    assert vms["ws"].allow == []  # the workspace keeps its own policy


def test_the_network_starts_with_its_rules_and_every_vm_joins_as_it_boots(monkeypatch):
    events = boot(monkeypatch, {"atlas": ATLAS, "forth": FORTH})
    # Only the workspace may open connections, to each server's VM.
    assert events[0] == (
        "network",
        [("tag:vf-0", "tag:vf-1"), ("tag:vf-0", "tag:vf-2")],
    )
    boots = {name: lan for kind, name, _, lan in events[1:-1] if kind == "boot"}
    assert boots == {
        "ws": ("net1", ["vf-0"]),
        "ws-node-0": ("net1", ["vf-1"]),
        "ws-node-1": ("net1", ["vf-2"]),
    }
    assert events[-1] == "network deleted"


def test_a_tool_machine_keeps_only_its_state_channel_and_the_harness_its_routes():
    applied = []

    def backend(name):
        async def prepare_execution(routes):
            applied.append((name, routes))

        def host_url(url):
            return url.replace("http://127.0.0.1:9000", "https://tunnel.example")

        return SimpleNamespace(prepare_execution=prepare_execution, host_url=host_url)

    strategy = _PrivateNetworkDeployment(
        {"forth": FORTH, "db": FORTH}, [], served=["forth"]
    )
    strategy._service_backends = {"forth": backend("forth"), "db": backend("db")}
    deployment = SimpleNamespace(harness_backend=backend("main"))
    # The harness reaches the model and a remote tool server; the tool machine needs
    # neither, and a machine without a tool server needs no state channel either.
    routes = ["https://tunnel.example/v1", "https://tools.example/mcp"]
    asyncio.run(
        strategy.prepare_execution(
            deployment, routes, state_route="http://127.0.0.1:9000/state"
        )
    )
    assert sorted(applied) == [
        ("db", []),
        ("forth", ["https://tunnel.example/state"]),
        ("main", routes),
    ]


def test_the_workspace_stops_before_tool_evidence_and_is_released_regardless():
    events = []

    async def stop():
        events.append("workspace stopped")

    async def aclose():
        events.append("tool VMs released")
        raise RuntimeError("network deletion failed")

    strategy = _PrivateNetworkDeployment({}, [])
    strategy._services = SimpleNamespace(aclose=aclose)
    deployment = SimpleNamespace(execution_backend=SimpleNamespace(stop=stop))

    # As Compose stops its workspace, before finalize_services reads the targets.
    asyncio.run(strategy.stop_execution(deployment))
    assert events == ["workspace stopped"]

    # A failed tool VM or network release neither skips the workspace's nor raises.
    events.clear()
    asyncio.run(strategy.prepare_verifier(deployment))
    assert events == ["workspace stopped", "tool VMs released"]


def served_url(monkeypatch, server, **kwargs):
    async def serve_in_runtime(server, runtime, **options):
        assert options["listen_all"]
        return 1234

    monkeypatch.setattr(launch, "serve_in_runtime", serve_in_runtime)

    async def url():
        async with launch._serve(server, **kwargs) as served:
            return served.url

    return asyncio.run(url())


def test_a_tool_server_on_its_own_vm_is_reached_by_its_lan_name(monkeypatch):
    runtime = SimpleNamespace(is_local=False, host_url=lambda url: url)
    # Its restricted runtime is enforced on its own VM, so it is not refused.
    server = toolset("forth", runtime=FORTH)
    assert (
        served_url(
            monkeypatch,
            server,
            hosts={"forth": "sb1.sandbox.internal"},
            placements={"forth": runtime},
        )
        == "http://sb1.sandbox.internal:1234/mcp"
    )
    # A Compose service keeps its name.
    compose = toolset("echo", service="tools")
    assert served_url(monkeypatch, compose, services={"tools": runtime}) == (
        "http://tools:1234/mcp"
    )


def test_an_explicit_service_names_only_a_compose_service(monkeypatch):
    # Another server's machine is a deployment target, not a service.
    runtime = SimpleNamespace(is_local=False, host_url=lambda url: url)
    with pytest.raises(ToolsetError, match="Compose services define it"):
        served_url(
            monkeypatch,
            toolset("echo", service="forth"),
            services={},
            hosts={"forth": "sb1.sandbox.internal"},
            placements={"forth": runtime},
        )
    with pytest.raises(ToolsetError, match="needs a task that declares that node"):
        served_url(monkeypatch, toolset("echo", on="db"), services={})


def test_a_restricted_runtime_without_a_private_network_is_refused(monkeypatch):
    with pytest.raises(ToolsetError, match="joins the rollout's private network"):
        served_url(monkeypatch, toolset("forth", runtime=FORTH))


def test_a_node_s_process_starts_in_its_workdir_and_waits_until_ready():
    calls = []

    class Machine:
        async def run_background(self, argv, env, log):
            calls.append(("start", argv, env))

        async def run(self, argv, env):
            calls.append(("check", argv))
            return SimpleNamespace(exit_code=0 if len(calls) == 3 else 1)

    process = ProcessConfig(
        command=["serve"], environment={"A": "1"}, workdir="/srv", readiness=["ready"]
    )
    asyncio.run(start_process(Machine(), process))
    assert calls == [
        (
            "start",
            ["sh", "-c", 'cd "$1" && shift && exec "$@"', "sh", "/srv", "serve"],
            {"A": "1"},
        ),
        ("check", ["ready"]),
        ("check", ["ready"]),
    ]
