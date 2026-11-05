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
from verifiers.v1.runtimes.deployment import Deployment
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
        toolset("remote", url="https://tools.example/mcp"),
        toolset("gym", service="tools"),  # a named task service
        toolset("local", runtime=DockerConfig()),  # a published port
        toolset("shared", runtime=ATLAS, shared=True),
    ]
    vms = private_network.tool_vms
    assert vms(toolsets, PrimeConfig()) == {"atlas": ATLAS}
    assert vms(toolsets, DockerConfig()) == {}
    with pytest.raises(ValueError, match="cannot be named 'main'"):
        vms(
            [toolset("main", runtime=ATLAS)],
            PrimeConfig(),
        )


def test_tool_vms_do_not_choose_the_workspace_deployment_strategy():
    assert isinstance(
        _deployment_strategy(None, PrimeBackend, None), _ColocatedDeployment
    )
    spec = vf.ComposeSpec("task")
    assert isinstance(
        _deployment_strategy(None, DockerBackend, spec), _ContainerDeployment
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

    monkeypatch.setattr(private_network, "leased_client", leased_client)
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


def test_the_network_starts_with_its_rules_and_every_vm_joins_as_it_boots(monkeypatch):
    events = boot(monkeypatch, {"atlas": ATLAS, "forth": FORTH})
    assert events[0] == (
        "network",
        [("tag:vf-workspace", "tag:vf-tool-0"), ("tag:vf-workspace", "tag:vf-tool-1")],
    )
    boots = {name: lan for kind, name, _, lan in events[1:-1] if kind == "boot"}
    assert boots == {
        "ws": ("net1", ["vf-workspace"]),
        "ws-vf-tool-0": ("net1", ["vf-tool-0"]),
        "ws-vf-tool-1": ("net1", ["vf-tool-1"]),
    }
    assert events[-1] == "network deleted"


def test_a_tool_vm_keeps_only_its_state_channel_and_the_harness_its_routes():
    applied = []

    def backend(name):
        async def prepare_execution(routes):
            applied.append((name, routes))

        def host_url(url):
            return url.replace("http://127.0.0.1:9000", "https://tunnel.example")

        return SimpleNamespace(prepare_execution=prepare_execution, host_url=host_url)

    strategy = _PrivateNetworkDeployment()
    deployment = SimpleNamespace(
        target_backends={"main": backend("main"), "forth": backend("forth")},
        tool_vms={},
    )
    # The harness reaches the model and a remote tool server; the tool VM needs neither.
    routes = ["https://tunnel.example/v1", "https://tools.example/mcp"]
    asyncio.run(
        strategy.prepare_execution(
            deployment, routes, state_route="http://127.0.0.1:9000/state"
        )
    )
    assert sorted(applied) == [
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
            runtime=runtime,
            host="sb1.sandbox.internal",
        )
        == "http://sb1.sandbox.internal:1234/mcp"
    )
    # A Compose service keeps its name.
    compose = toolset("echo", service="tools")
    assert served_url(monkeypatch, compose, runtime=runtime, host="tools") == (
        "http://tools:1234/mcp"
    )


@pytest.mark.parametrize("service", ["main", "forth"])
def test_an_explicit_service_needs_a_task_service_spec(monkeypatch, service):
    # Without a service spec, these are targets for independent tool placement.
    runtime = SimpleNamespace(is_local=False, host_url=lambda url: url)
    with pytest.raises(ToolsetError, match="services define"):
        Deployment.tool_target(
            SimpleNamespace(
                spec=None, target_backends={"forth": runtime}, tool_vms={"forth": FORTH}
            ),
            toolset("echo", service=service),
        )


def test_a_restricted_runtime_without_a_private_network_is_refused(monkeypatch):
    with pytest.raises(ToolsetError, match="private network"):
        served_url(monkeypatch, toolset("forth", runtime=FORTH))
