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


def test_a_rollouts_own_prime_tool_servers_get_vms_on_its_lan():
    toolsets = [
        toolset("atlas", runtime=ATLAS),
        toolset("search"),  # the evaluator
        toolset("index", shared=True, runtime=ATLAS),  # the eval LAN
        toolset("remote", url="https://tools.example/mcp"),
        toolset("gym", service="tools"),  # a named task service
        toolset("local", runtime=DockerConfig()),  # a published port
    ]
    vms = private_network.tool_vms
    assert vms(toolsets, PrimeConfig(), borrowed=False) == {"atlas": ATLAS}
    # Only a Prime workspace the rollout provisions itself has a LAN of its own.
    for workspace, borrowed in [(DockerConfig(), False), (PrimeConfig(), True)]:
        assert vms(toolsets, workspace, borrowed=borrowed) == {}
    with pytest.raises(ValueError, match="cannot be named 'main'"):
        vms([toolset("main", runtime=ATLAS)], PrimeConfig(), borrowed=False)


@pytest.mark.parametrize(
    "config",
    [
        {"shared": True, "service": "tools"},  # a service belongs to one rollout
        {"url": "https://tools.example/mcp", "service": "tools"},
        {"runtime": {"type": "prime"}, "service": "tools"},
        {"runtime": {"type": "docker", "allow": []}},  # no policy off a Prime VM
        {"shared": True, "runtime": {"type": "prime", "allow": []}},
    ],
)
def test_a_tool_server_has_one_placement(config):
    with pytest.raises(ValueError):
        vf.ToolsetConfig(**config)


def test_a_rollouts_own_prime_vm_keeps_its_own_network_policy():
    assert vf.ToolsetConfig(runtime=FORTH).runtime.allow == []


def test_the_deployment_joins_tool_vms_over_a_private_network():
    assert isinstance(
        _deployment_strategy(None, PrimeBackend, None, {"atlas": ATLAS}),
        _PrivateNetworkDeployment,
    )
    assert isinstance(
        _deployment_strategy(None, PrimeBackend, None, {}), _ColocatedDeployment
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
        async with private_network.private_network(
            workspace, members, workspace.start, tools=frozenset(members)
        ):
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
        target_backends={"main": backend("main"), "forth": backend("forth")}
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


def served_url(
    monkeypatch, server, *, targets=(), hosts=None, workspace=None, shared=None
):
    """The URL `serve_tools` hands the harness for one `server`, in a deployment whose
    `targets` run every server as on port 1234."""
    runtime = SimpleNamespace(is_local=False, host_url=lambda url: url)

    async def serve_in_runtime(server, runtime, **options):
        assert options["listen_all"]
        return 1234

    monkeypatch.setattr(launch, "serve_in_runtime", serve_in_runtime)
    deployment = SimpleNamespace(
        target_backends=dict.fromkeys(targets),
        target=lambda name: runtime,
        hosts=hosts or {},
    )

    async def url():
        async with launch.serve_tools(
            [server],
            workspace or SimpleNamespace(is_local=False, backend=None, type="prime"),
            deployment,
            shared or {},
            {},
            state_base="https://tunnel.example",
            state_route="trace1",
            proxy_mcp=proxy,
        ) as urls:
            return urls[server.server_name]

    return asyncio.run(url())


@asynccontextmanager
async def proxy(url):
    yield f"https://tunnel.example/mcp/cap?backend={url}"


def test_a_tool_server_on_its_own_vm_is_reached_by_its_lan_name(monkeypatch):
    # Its restricted runtime is enforced on its own VM, so it is not refused.
    server = toolset("forth", runtime=FORTH)
    hosts = {"forth": "sb1.sandbox.internal"}
    assert served_url(monkeypatch, server, targets=["forth"], hosts=hosts) == (
        "http://sb1.sandbox.internal:1234/mcp"
    )
    # A Compose service keeps its name.
    compose = toolset("echo", service="tools")
    assert served_url(monkeypatch, compose, targets=["main", "tools"]) == (
        "http://tools:1234/mcp"
    )


def test_a_service_placement_needs_the_task_to_define_it(monkeypatch):
    with pytest.raises(ToolsetError, match="services do not define"):
        served_url(monkeypatch, toolset("echo", service="tools"), targets=["main"])


def test_a_prime_tool_vm_needs_a_prime_workspace_of_the_rollouts_own(monkeypatch):
    with pytest.raises(ToolsetError, match="needs a Prime workspace"):
        served_url(monkeypatch, toolset("forth", runtime=FORTH), targets=["main"])


def test_a_shared_server_must_be_running(monkeypatch):
    with pytest.raises(ToolsetError, match="is not running"):
        served_url(monkeypatch, toolset("index", shared=True))


def test_a_remote_workspace_reaches_a_shared_host_server_through_its_interception(
    monkeypatch,
):
    runtime = SimpleNamespace(host_url=lambda url: url)
    shared = {
        "index": launch.SharedToolServer(
            "http://127.0.0.1:7000/mcp", runtime, state_secret="s"
        )
    }
    url = served_url(monkeypatch, toolset("index", shared=True), shared=shared)
    assert url.startswith("https://tunnel.example/mcp/cap?")
    # The rollout's state coordinates ride the proxied URL, signed for the server.
    assert "vf_state_route=trace1" in url
    assert launch.state_signature("s", "https://tunnel.example/state", "trace1") in url


def test_a_prime_workspace_joins_the_eval_lan_of_a_shared_vm(monkeypatch):
    joined = []

    async def join_lan(workspace, network, hosts):
        joined.append((workspace.backend, network, hosts))

    monkeypatch.setattr(launch, "join_lan", join_lan)
    monkeypatch.setattr(prime, "ensure_prime_auth", lambda: None)
    backend = PrimeBackend(PrimeConfig())
    workspace = SimpleNamespace(is_local=False, backend=backend, type="prime")
    vm = launch.SharedToolServer(
        "http://sb9.sandbox.internal:1234/mcp", SimpleNamespace(), lan="eval1"
    )
    toolsets = [toolset("index", shared=True, runtime=ATLAS), toolset("search")]

    async def join():
        await launch.join_shared_lans(toolsets, {"index": vm}, workspace)

    asyncio.run(join())
    assert joined == [(backend, "eval1", {"sb9.sandbox.internal"})]
    # The joined workspace reaches the server by its LAN name.
    url = served_url(
        monkeypatch, toolsets[0], workspace=workspace, shared={"index": vm}
    )
    assert url == vm.url
    # Any other workspace cannot reach the eval LAN, which shows before any setup.
    docker = SimpleNamespace(is_local=True, backend=None, type="docker")
    with pytest.raises(ToolsetError, match="only from a Prime workspace"):
        launch.join_shared_lans(toolsets, {"index": vm}, docker)


class FakeEvalLanClient(FakeLanClient):
    def __init__(self, events, syncs):
        super().__init__(events)
        self.syncs = syncs

    async def add_lan_member(self, network, sandbox, tags):
        self.events.append(("join", network, sandbox, tags))
        return SimpleNamespace(synced=self.syncs.pop(0))

    async def get_lan_network(self, network):
        raise AssertionError("a read lists every member of the eval LAN")


def test_shared_vms_boot_onto_an_eval_lan_that_workspaces_join(monkeypatch):
    events = []
    syncs = [False, True]

    @asynccontextmanager
    async def leased_client():
        yield FakeEvalLanClient(events, syncs)

    async def start(self):
        events.append(("boot", self.lan))

    async def stop(self):
        events.append("stop")

    monkeypatch.setattr(private_network, "leased_client", leased_client)
    monkeypatch.setattr(private_network, "_JOIN_POLL_SECONDS", 0)
    monkeypatch.setattr(prime, "ensure_prime_auth", lambda: None)
    monkeypatch.setattr(PrimeBackend, "start", start)
    monkeypatch.setattr(PrimeBackend, "stop", stop)
    backend = PrimeBackend(PrimeConfig())
    backend.info.id = "ws1"

    async def resolve(argv, env):
        events.append(("resolve", argv[4:]))
        return SimpleNamespace(exit_code=0)

    workspace = SimpleNamespace(backend=backend, run=resolve)

    async def run():
        async with private_network.eval_lan({"atlas": ATLAS}) as (network, vms):
            assert set(vms) == {"atlas"}
            await private_network.join_lan(workspace, network, ["sb9.sandbox.internal"])

    asyncio.run(run())
    assert events == [
        ("network", [("tag:vf-workspace", "tag:vf-tool-0")]),
        ("boot", ("net1", ["vf-tool-0"])),
        # The join's own response says when it applies; it is retried, never read.
        ("join", "net1", "ws1", ["vf-workspace"]),
        ("join", "net1", "ws1", ["vf-workspace"]),
        # The workspace's DNS can lag the join, so it waits to resolve the server.
        ("resolve", ["sb9.sandbox.internal"]),
        "stop",
        "network deleted",
    ]
