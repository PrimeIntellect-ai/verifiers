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


def test_tool_servers_on_prime_runtimes_beside_a_prime_workspace_get_vms():
    toolsets = [
        toolset("atlas", runtime=ATLAS),
        toolset("search"),  # the evaluator
        toolset("remote", url="https://tools.example/mcp"),
        toolset("gym", service="tools"),  # a named task service
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


@pytest.mark.parametrize("service", ["main", "forth"])
def test_an_explicit_service_needs_a_task_service_spec(monkeypatch, service):
    # Without a service spec, these are targets for independent tool placement.
    runtime = SimpleNamespace(is_local=False, host_url=lambda url: url)
    with pytest.raises(ToolsetError, match="services define it"):
        served_url(
            monkeypatch,
            toolset("echo", service=service),
            services={},
            hosts={"forth": "sb1.sandbox.internal"},
            placements={"forth": runtime},
        )


def test_a_restricted_runtime_without_a_private_network_is_refused(monkeypatch):
    with pytest.raises(ToolsetError, match="joins the rollout's private network"):
        served_url(monkeypatch, toolset("forth", runtime=FORTH))


def test_reuse_harness_pools_across_rollouts_and_detaches_each_lan(monkeypatch):
    from verifiers.v1.runtimes.harness_pool import harness_pool_scope

    events = []
    net_seq = 0
    sb_seq = 0

    class PoolLanClient:
        async def create_lan_network(self, name, rules):
            nonlocal net_seq
            net_seq += 1
            return SimpleNamespace(id=f"net{net_seq}")

        async def remove_lan_member(self, network_id, sandbox_id):
            events.append(("detach", network_id, sandbox_id))

        async def delete_lan_network(self, network_id):
            events.append(("delete_net", network_id))

    @asynccontextmanager
    async def leased_client():
        yield PoolLanClient()

    async def start(self):
        nonlocal sb_seq
        sb_seq += 1
        self.info.id = f"sb-{self.name}-{sb_seq}"
        self._client = PoolLanClient()
        self._created_workdirs.add(self.config.workdir)
        events.append(("start", self.name, self.info.id))

    async def teardown(self):
        if self._client is not None:
            self._client = None
            events.append(("stop", self.name, self.info.id))

    monkeypatch.setattr(private_network, "leased_client", leased_client)
    monkeypatch.setattr(prime, "ensure_prime_auth", lambda: None)
    monkeypatch.setattr(PrimeBackend, "start", start)
    monkeypatch.setattr(PrimeBackend, "teardown", teardown)
    monkeypatch.setattr(
        prime,
        "join_lan",
        lambda backend, network_id, tags: asyncio.sleep(0),
    )

    cfg = vf.DeploymentConfig()

    async def run_two():
        async with harness_pool_scope():
            for idx in (1, 2):
                ws = PrimeBackend(PrimeConfig(), name=f"ws{idx}")
                dep = ws.deploy(config=cfg, split=True)
                await dep.start()
                h_id = dep.harness_backend.info.id
                await dep.close()
                events.append(("rollout_done", idx, h_id))

    asyncio.run(run_two())
    harness_starts = [e for e in events if e[0] == "start" and "harness" in e[1]]
    assert len(harness_starts) == 1
    harness_id = harness_starts[0][2]
    assert ("rollout_done", 1, harness_id) in events
    assert ("rollout_done", 2, harness_id) in events
    assert ("detach", "net1", harness_id) in events
    assert ("detach", "net2", harness_id) in events
    assert events[-1] == ("stop", "ws1-harness", harness_id)


def test_set_network_skips_open_reset_when_only_control_hosts_are_blocked():
    calls = []

    class FakeBackend:
        name = "ws"
        network_restricted = True

        def __init__(self, block, control_hosts):
            self.config = PrimeConfig(allow=["*"], block=block)
            self._control_blocked_hosts = frozenset(control_hosts)

        async def prepare_execution(self, routes, network=None):
            calls.append((routes, network))

    open_ws = vf.Runtime(
        FakeBackend(["t-0.tunnel.pinfra.io"], ["t-0.tunnel.pinfra.io"])
    )
    asyncio.run(open_ws.set_network(vf.NetworkPolicyConfig()))
    assert calls == []

    restricted_ws = vf.Runtime(
        FakeBackend(
            ["t-0.tunnel.pinfra.io", "pypi.org"],
            ["t-0.tunnel.pinfra.io"],
        )
    )
    asyncio.run(restricted_ws.set_network(vf.NetworkPolicyConfig()))
    assert calls == [(None, None)]


def test_multiplex_shares_harness_vm_across_concurrent_rollouts_with_isolated_workdirs(
    monkeypatch,
):
    from verifiers.v1.runtimes.harness_pool import harness_pool_scope

    events = []
    net_seq = 0
    sb_seq = 0

    class MuxLanClient:
        async def create_lan_network(self, name, rules):
            nonlocal net_seq
            net_seq += 1
            return SimpleNamespace(id=f"net{net_seq}")

        async def remove_lan_member(self, network_id, sandbox_id):
            events.append(("detach", network_id, sandbox_id))

        async def delete_lan_network(self, network_id):
            events.append(("delete_net", network_id))

    @asynccontextmanager
    async def leased_client():
        yield MuxLanClient()

    async def start(self):
        nonlocal sb_seq
        sb_seq += 1
        await asyncio.sleep(0.01)
        self.info.id = f"sb-{self.name}-{sb_seq}"
        self._client = MuxLanClient()
        self._created_workdirs.add(self.config.workdir)
        events.append(("start", self.name, self.info.id))

    async def run_cmd(self, runtime, argv, env):
        events.append(("mkdir", self.info.id, runtime.workdir or self.config.workdir))
        return vf.ProgramResult(exit_code=0, stdout="", stderr="")

    async def teardown(self):
        if self._client is not None:
            self._client = None
            events.append(("stop", self.name, self.info.id))

    monkeypatch.setattr(private_network, "leased_client", leased_client)
    monkeypatch.setattr(prime, "ensure_prime_auth", lambda: None)
    monkeypatch.setattr(PrimeBackend, "start", start)
    monkeypatch.setattr(PrimeBackend, "run", run_cmd)
    monkeypatch.setattr(PrimeBackend, "teardown", teardown)
    monkeypatch.setattr(
        prime,
        "join_lan",
        lambda backend, network_id, tags: asyncio.sleep(0),
    )

    cfg = vf.DeploymentConfig(multiplex=2)
    barrier = None

    async def one_rollout(idx: int, workdir: str):
        ws = PrimeBackend(PrimeConfig(workdir=workdir), name=f"ws{idx}")
        dep = ws.deploy(config=cfg, split=True)
        await dep.start()
        assert dep._harness_runtime.info.workdir == workdir
        assert dep.harness_backend.config.workdir == "/app"
        await barrier.wait()
        h_id = dep.harness_backend.info.id
        await dep.close()
        return h_id

    async def run_four_concurrent():
        nonlocal barrier
        barrier = asyncio.Barrier(4)
        async with harness_pool_scope():
            ids = await asyncio.gather(
                one_rollout(1, "/app"),
                one_rollout(2, "/workspace"),
                one_rollout(3, "/srv/task"),
                one_rollout(4, "/app"),
            )
            # While pool scope is still open, neither shared harness VM was stopped yet.
            harness_stops_mid = [
                e for e in events if e[0] == "stop" and "harness" in e[1]
            ]
            assert harness_stops_mid == []
            return ids

    assigned = asyncio.run(run_four_concurrent())
    # 4 concurrent rollouts with multiplex=2 pack onto exactly 2 harness VMs (2 slots each).
    harness_starts = [e for e in events if e[0] == "start" and "harness" in e[1]]
    assert len(harness_starts) == 2
    assert len(set(assigned)) == 2
    for hid in set(assigned):
        assert assigned.count(hid) == 2
    # Both pooled harness VMs stop cleanly on scope exit.
    harness_stops = [e for e in events if e[0] == "stop" and "harness" in e[1]]
    assert len(harness_stops) == 2


def test_harness_pool_auto_checkpoints_first_vm_for_scale_out_and_cleans_up(
    monkeypatch,
):
    from verifiers.v1.runtimes.harness_pool import harness_pool_scope

    events = []
    sb_seq = 0

    class AutoCpClient:
        async def create_lan_network(self, name, rules):
            return SimpleNamespace(id="net1")

        async def remove_lan_member(self, network_id, sandbox_id):
            pass

        async def delete_lan_network(self, network_id):
            pass

        async def checkpoint(self, sandbox_id, ttl_seconds=None):
            events.append(("checkpoint", sandbox_id))
            return SimpleNamespace(id=f"auto-cp-{sandbox_id}")

        async def wait_for_checkpoint(self, checkpoint_id, timeout_seconds=300):
            events.append(("wait_cp", checkpoint_id))

        async def delete_checkpoint(self, checkpoint_id):
            events.append(("delete_cp", checkpoint_id))

    @asynccontextmanager
    async def leased_client():
        yield AutoCpClient()

    async def start(self):
        nonlocal sb_seq
        sb_seq += 1
        self.info.id = f"sb-{sb_seq}"
        self._client = AutoCpClient()
        self._created_workdirs.add(self.config.workdir)
        events.append(("start", self.name, self.info.id, self.config.checkpoint))

    async def teardown(self):
        if self._client is not None:
            self._client = None
            events.append(("stop", self.info.id))

    monkeypatch.setattr(private_network, "leased_client", leased_client)
    monkeypatch.setattr(prime, "leased_client", leased_client)
    monkeypatch.setattr(prime, "ensure_prime_auth", lambda: None)
    monkeypatch.setattr(PrimeBackend, "start", start)
    monkeypatch.setattr(PrimeBackend, "teardown", teardown)
    monkeypatch.setattr(
        prime,
        "join_lan",
        lambda backend, network_id, tags: asyncio.sleep(0),
    )

    cfg = vf.DeploymentConfig(multiplex=1)

    async def run_scale_out():
        async with harness_pool_scope() as pool:
            ws1 = PrimeBackend(PrimeConfig(), name="ws1")
            dep1 = ws1.deploy(config=cfg, split=True)
            await dep1.start()
            dep1._harness_lease.notify_setup_complete()
            while pool._auto_checkpoint_tasks:
                await asyncio.sleep(0.01)

            # While dep1 still holds VM A #1 (multiplex=1), dep2 scales out to VM A #2
            # using the auto-minted checkpoint from VM A #1!
            ws2 = PrimeBackend(PrimeConfig(), name="ws2")
            dep2 = ws2.deploy(config=cfg, split=True)
            await dep2.start()
            await dep1.close()
            await dep2.close()

    asyncio.run(run_scale_out())
    harness_starts = [
        (sid, cp)
        for kind, name, sid, cp in (e for e in events if e[0] == "start")
        if "harness" in name
    ]
    assert harness_starts == [
        ("sb-1", None),
        ("sb-3", "auto-cp-sb-1"),
    ]
    assert ("delete_cp", "auto-cp-sb-1") in events


def test_harness_pool_evicts_expired_auto_checkpoint_and_cleans_cancelled_mint(
    monkeypatch,
):
    from verifiers.v1.runtimes.harness_pool import HarnessPool, harness_pool_scope

    events = []
    sb_seq = 0
    now = 1000.0
    wait_started = asyncio.Event()

    class ClockCpClient:
        async def create_lan_network(self, name, rules):
            return SimpleNamespace(id="net1")

        async def remove_lan_member(self, network_id, sandbox_id):
            pass

        async def delete_lan_network(self, network_id):
            pass

        async def checkpoint(self, sandbox_id, ttl_seconds=None):
            events.append(("checkpoint", sandbox_id))
            return SimpleNamespace(id=f"cp-{sandbox_id}")

        async def wait_for_checkpoint(self, checkpoint_id, timeout_seconds=300):
            if checkpoint_id == "cp-sb-5":
                wait_started.set()
                await asyncio.sleep(60)
            events.append(("wait_cp", checkpoint_id))

        async def delete_checkpoint(self, checkpoint_id):
            events.append(("delete_cp", checkpoint_id))

    @asynccontextmanager
    async def leased_client():
        yield ClockCpClient()

    async def start(self):
        nonlocal sb_seq
        sb_seq += 1
        self.info.id = f"sb-{sb_seq}"
        self._client = ClockCpClient()
        self._created_workdirs.add(self.config.workdir)
        events.append(("start", self.name, self.info.id, self.config.checkpoint))

    async def teardown(self):
        if self._client is not None:
            self._client = None

    monkeypatch.setattr(private_network, "leased_client", leased_client)
    monkeypatch.setattr(prime, "leased_client", leased_client)
    monkeypatch.setattr(prime, "ensure_prime_auth", lambda: None)
    monkeypatch.setattr(PrimeBackend, "start", start)
    monkeypatch.setattr(PrimeBackend, "teardown", teardown)
    monkeypatch.setattr(prime, "join_lan", lambda *_: asyncio.sleep(0))

    cfg = vf.DeploymentConfig(multiplex=1)

    async def run_expiry_and_cancel():
        nonlocal now
        pool1 = HarnessPool()
        pool1._clock = lambda: now
        async with harness_pool_scope(pool1):
            ws1 = PrimeBackend(PrimeConfig(), name="ws1")
            dep1 = ws1.deploy(config=cfg, split=True)
            await dep1.start()
            dep1._harness_lease.notify_setup_complete()
            while pool1._auto_checkpoint_tasks:
                await asyncio.sleep(0.01)

            # Advance past usable TTL: scale-out falls back to cold boot (`checkpoint=None`).
            now += 3601.0
            ws2 = PrimeBackend(PrimeConfig(), name="ws2")
            dep2 = ws2.deploy(config=cfg, split=True)
            await dep2.start()
            await dep1.close()
            await dep2.close()
        await pool1.aclose()

        # Start a new pool whose mint is cancelled mid-wait on scope exit; cp is still deleted.
        async with harness_pool_scope():
            ws3 = PrimeBackend(PrimeConfig(image="other:1"), name="ws3")
            dep3 = ws3.deploy(config=cfg, split=True)
            await dep3.start()
            dep3._harness_lease.notify_setup_complete()
            await asyncio.wait_for(wait_started.wait(), timeout=2.0)
            await dep3.close()

    asyncio.run(run_expiry_and_cancel())
    harness_starts = [
        (sid, cp)
        for kind, name, sid, cp in (e for e in events if e[0] == "start")
        if "harness" in name
    ]
    assert harness_starts == [
        ("sb-1", None),
        ("sb-3", None),
        ("sb-5", None),
    ]
    assert ("delete_cp", "cp-sb-1") in events
    assert ("delete_cp", "cp-sb-5") in events
