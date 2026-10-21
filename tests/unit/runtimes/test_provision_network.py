"""A restricted runtime provisioned on its own is under its network policy once it is
up: a caller that never reaches `prepare_execution` must not get open egress."""

from types import SimpleNamespace

import pytest

import verifiers.v1 as vf
from verifiers.v1.runtimes import PrimeConfig, provision_runtime
from verifiers.v1.runtimes.backends import prime
from verifiers.v1.runtimes.base import _LIVE, ProgramResult
from verifiers.v1.runtimes.compose import INSTALL_DOCKER, ComposeSpec, provision_vm_host
from verifiers.v1.runtimes.deployment import _DeploymentStrategy


class FakePrimeClient:
    """Records the policies a sandbox is put under; every policy applies at once."""

    def __init__(self):
        self.policies: list[dict] = []

    async def create(self, request):
        return SimpleNamespace(id="sb1", pending_image_build_id=None)

    async def wait_for_creation(self, sandbox_id, max_attempts):
        pass

    async def execute_command(self, sandbox_id, command):
        pass

    async def set_network(self, sandbox_id, **policy):
        self.policies.append(policy)
        return SimpleNamespace(applied=True)

    async def delete(self, sandbox_id):
        pass


@pytest.fixture
def client(monkeypatch):
    client = FakePrimeClient()

    async def release(_):
        pass

    monkeypatch.setattr(prime, "ensure_prime_auth", lambda: None)
    monkeypatch.setattr(prime, "_lease_client", lambda: client)
    monkeypatch.setattr(prime, "_release_client", release)
    yield client
    # The fake boxes never existed: keep the atexit backstop from deleting them.
    for backend in list(_LIVE):
        if isinstance(backend, prime.PrimeBackend):
            _LIVE.discard(backend)


async def test_a_restricted_prime_box_is_closed_once_provisioned(client):
    async with provision_runtime(PrimeConfig(allow=[])):
        assert client.policies == [{"deny": ["*"]}]
    client.policies.clear()
    async with provision_runtime(PrimeConfig(allow=["pypi.org"])):
        assert client.policies == [{"allow": ["pypi.org"]}]


async def test_trusted_setup_reopens_a_provisioned_box_until_execution(client):
    async with provision_runtime(PrimeConfig(allow=[])) as runtime:
        await runtime.backend.prepare_setup()
        await runtime.backend.prepare_execution(["https://tunnel.example/v1"])
    assert client.policies == [
        {"deny": ["*"]},
        {"allow": ["*"]},
        {"allow": ["tunnel.example"]},
    ]


async def test_an_agent_provisioned_box_is_closed_until_a_run_sets_it_up(client):
    agent = vf.Agent(vf.AgentConfig(model="m", runtime=PrimeConfig(allow=[])))
    async with agent.provision() as runtime:
        assert client.policies == [{"deny": ["*"]}]
        await runtime.backend.prepare_setup()  # what a run placed into the box does
    assert client.policies == [{"deny": ["*"]}, {"allow": ["*"]}]


async def test_open_boxes_stay_open(client):
    async with provision_runtime(PrimeConfig()):
        pass
    assert client.policies == []


async def test_the_compose_vm_host_installs_docker_before_any_policy(
    client, monkeypatch
):
    """The VM host installs Docker over the network; its services apply the policy
    later. Closing it at provision would fail every restricted Compose task."""
    events = []

    async def run(self, runtime, argv, env):
        events.append(argv[-1][:24])
        return ProgramResult(0, "", "")

    async def run_background(self, runtime, argv, env, log):
        events.append(argv[0])

    monkeypatch.setattr(prime.PrimeBackend, "run", run)
    monkeypatch.setattr(prime.PrimeBackend, "run_background", run_background)
    async with provision_vm_host(PrimeConfig(allow=[]), ComposeSpec(directory=".")):
        pass
    assert events[0] == INSTALL_DOCKER[:24]
    assert client.policies == []


async def test_the_isolated_verifier_is_staged_open_then_closed_once(client):
    """Grading stages the verifier, then applies its policy itself: one policy call."""
    strategy = _DeploymentStrategy()
    async with strategy.verifier(None, PrimeConfig(allow=[]), {}) as backend:
        assert client.policies == []
        await backend.prepare_setup()  # a fresh box: already open
        await backend.prepare_execution([])
    assert client.policies == [{"deny": ["*"]}]
