import asyncio
from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest

import verifiers.v1 as vf
from verifiers.v1.errors import ToolsetError
from verifiers.v1.mcp import launch
from verifiers.v1.runtimes import (
    DockerConfig,
    PrimeBackend,
    PrimeConfig,
    prime,
    private_network,
)
from verifiers.v1.runtimes.strategies import (
    _ColocatedDeployment,
    _ContainerDeployment,
    _deployment_strategy,
    _PrivateNetworkDeployment,
)

ATLAS = PrimeConfig(image="atlas:1", cpu=4, memory=10)  # open egress by default
FORTH = PrimeConfig(image="forth:1", allow=[])


def toolset(name, **config):
    return SimpleNamespace(server_name=name, config=vf.ToolsetConfig(**config))


def test_tool_servers_on_prime_runtimes_beside_a_prime_workspace_get_vms():
    toolsets = [
        toolset("atlas", runtime=ATLAS),
        toolset("search"),  # the evaluator
        toolset("files", colocated=True, runtime=ATLAS),  # the workspace
        toolset("remote", url="https://tools.example/mcp"),
        toolset("gym", service="tools"),  # a Compose service
        toolset("local", runtime=DockerConfig()),  # a published port
    ]
    vms = private_network.tool_vms
    assert vms(toolsets, PrimeConfig(), composed=False, borrowed=False) == {
        "atlas": ATLAS
    }
    # Without a Prime workspace of the rollout's own, servers keep their published port.
    for workspace, composed, borrowed in [
        (DockerConfig(), False, False),
        (PrimeConfig(), True, False),
        (PrimeConfig(), False, True),
    ]:
        assert vms(toolsets, workspace, composed=composed, borrowed=borrowed) == {}
    with pytest.raises(ValueError, match="cannot be named 'main'"):
        vms(
            [toolset("main", runtime=ATLAS)],
            PrimeConfig(),
            composed=False,
            borrowed=False,
        )


def test_the_deployment_joins_tool_vms_over_a_private_network():
    assert isinstance(
        _deployment_strategy(None, PrimeConfig(), None, {"atlas": ATLAS}),
        _PrivateNetworkDeployment,
    )
    assert isinstance(
        _deployment_strategy(None, PrimeConfig(), None, {}), _ColocatedDeployment
    )
    spec = vf.ComposeSpec("task")
    assert isinstance(
        _deployment_strategy(None, PrimeConfig(), spec), _ContainerDeployment
    )


def test_the_private_network_needs_a_prime_workspace():
    workspace = SimpleNamespace(config=DockerConfig())

    async def enter():
        async with private_network.private_network(workspace, {"atlas": ATLAS}, None):
            pass

    with pytest.raises(TypeError, match="needs the workspace on Prime, not docker"):
        asyncio.run(enter())


class FakeLanClient:
    def __init__(self, events):
        self.events = events

    async def create_lan_network(self, name):
        self.events.append("network")
        return SimpleNamespace(id="net1")

    async def set_lan_rules(self, network_id, rules):
        self.events.append(("rules", [(rule.from_, rule.to) for rule in rules]))
        return SimpleNamespace(synced=False)  # not yet enforced

    async def get_lan_network(self, network_id):
        self.events.append("rules applied")
        return SimpleNamespace(synced=True)

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

    monkeypatch.setattr(private_network, "leased_client", leased_client)
    monkeypatch.setattr(private_network, "SYNC_POLL_SECONDS", 0)
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

    assert shape(vms["ws-vf-tool-0"]) == ("atlas:1", 4, 10, ["*"])
    assert shape(vms["ws-vf-tool-1"]) == ("forth:1", FORTH.cpu, FORTH.memory, [])
    assert vms["ws"].allow == []  # the workspace keeps its own policy


def test_every_vm_joins_the_lan_as_it_boots_once_its_rules_are_enforced(monkeypatch):
    events = boot(monkeypatch, {"atlas": ATLAS, "forth": FORTH})
    assert events[:3] == [
        "network",
        (
            "rules",
            [
                ("tag:vf-workspace", "tag:vf-tool-0"),
                ("tag:vf-workspace", "tag:vf-tool-1"),
            ],
        ),
        "rules applied",
    ]
    boots = {name: lan for kind, name, _, lan in events[3:-1] if kind == "boot"}
    assert boots == {
        "ws": ("net1", ["vf-workspace"]),
        "ws-vf-tool-0": ("net1", ["vf-tool-0"]),
        "ws-vf-tool-1": ("net1", ["vf-tool-1"]),
    }
    assert events[-1] == "network deleted"


def test_each_vm_enforces_its_own_policy_with_the_framework_routes():
    applied = []

    def backend(name):
        async def prepare_execution(routes):
            applied.append((name, routes))

        return SimpleNamespace(prepare_execution=prepare_execution)

    strategy = _PrivateNetworkDeployment()
    strategy._service_backends = [backend("forth")]
    deployment = SimpleNamespace(harness_backend=backend("main"))
    asyncio.run(strategy.prepare_execution(deployment, ["https://tunnel.example"]))
    assert sorted(applied) == [
        ("forth", ["https://tunnel.example"]),
        ("main", ["https://tunnel.example"]),
    ]


def test_the_workspace_stops_before_tool_evidence_and_is_released_regardless():
    events = []

    async def stop():
        events.append("workspace stopped")

    async def aclose():
        events.append("tool VMs released")
        raise RuntimeError("network deletion failed")

    strategy = _PrivateNetworkDeployment()
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
            services={"forth": runtime},
            hosts={"forth": "sb1.sandbox.internal"},
            placements={"forth": "forth"},
        )
        == "http://sb1.sandbox.internal:1234/mcp"
    )
    # A Compose service keeps its name.
    compose = toolset("echo", service="tools")
    assert served_url(monkeypatch, compose, services={"tools": runtime}) == (
        "http://tools:1234/mcp"
    )


def test_a_restricted_runtime_without_a_private_network_is_refused(monkeypatch):
    with pytest.raises(ToolsetError, match="joins the rollout's private network"):
        served_url(monkeypatch, toolset("forth", runtime=FORTH))
