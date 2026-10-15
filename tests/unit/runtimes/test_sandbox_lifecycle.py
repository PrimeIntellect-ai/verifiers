"""Run labels on Prime sandboxes and tunnels, label sweeps, and image deletes, against
fakes of the Prime SDK."""

import contextlib
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import ClassVar

import prime_sandboxes
import prime_tunnel
import pytest
from prime_sandboxes import APIError
from prime_sandboxes.models import (
    BulkDeleteSandboxResponse,
)
from prime_tunnel.core import client as tunnel_client_module
from prime_tunnel.models import TunnelInfo, TunnelListPage

import verifiers.v1.interception.tunnel.prime as tunnel_prime
from verifiers.v1.runtimes import prime, sweep
from verifiers.v1.runtimes.prime_images import delete_image

NOW = datetime(2026, 10, 13, tzinfo=UTC)


@pytest.fixture(autouse=True)
def run_id(monkeypatch):
    monkeypatch.setenv("VF_RUN_ID", "run-1")
    monkeypatch.setattr(prime, "BASE_LABELS", [])


def test_run_labels_are_base_labels_then_the_run_scope():
    prime.set_base_sandbox_labels(["campaign", "run-1"])
    assert prime.run_labels() == ["campaign", "run-1"]
    prime.set_base_sandbox_labels(["campaign"])
    assert prime.run_labels() == ["campaign", "run-1"]


async def test_tunnels_carry_the_run_labels(monkeypatch):
    created: list[dict] = []

    class FakeTunnel:
        def __init__(self, **kwargs) -> None:
            created.append(kwargs)

        async def start(self) -> str:
            return "https://t.example/"

        def sync_stop(self) -> None:
            pass

    @contextlib.asynccontextmanager
    async def unlimited():
        yield

    monkeypatch.setattr(prime_tunnel, "Tunnel", FakeTunnel)
    monkeypatch.setattr(tunnel_prime, "ensure_prime_auth", lambda: None)
    monkeypatch.setattr(tunnel_prime, "tunnel_limiter", unlimited)
    prime.set_base_sandbox_labels(["campaign"])
    async with tunnel_prime.PrimeTunnel().expose(8000) as url:
        assert url == "https://t.example"
    assert created == [{"local_port": 8000, "labels": ["campaign", "run-1"]}]


# -- sweeps ------------------------------------------------------------------------------


class FakeSandboxClient:
    def __init__(self, outcomes: dict[str, object]) -> None:
        self.outcomes = outcomes
        self.calls: list[list[str]] = []
        self.closed = False

    async def bulk_delete(self, labels):
        self.calls.append(labels)
        outcome = self.outcomes[labels[0]]
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    async def aclose(self):
        self.closed = True


def deleted(ids, failed=()):
    return BulkDeleteSandboxResponse(
        succeeded=list(ids), failed=[{"id": i} for i in failed], message=""
    )


async def test_sandbox_sweep_deletes_one_label_per_call_and_survives_a_failure(
    monkeypatch,
):
    client = FakeSandboxClient(
        {
            "a": deleted(["s1", "s2"]),
            "b": RuntimeError("gateway"),
            "c": deleted([], ["s3"]),
        }
    )
    monkeypatch.setattr(prime_sandboxes, "AsyncSandboxClient", lambda: client)
    assert await sweep.sweep_sandboxes(["a", "b", "a", "c"]) == {
        "a": 2,
        "b": -1,
        "c": 0,
    }
    assert client.calls == [["a"], ["b"], ["c"]]
    assert client.closed


def tunnel(tunnel_id, labels, user="me", terminated=False):
    return TunnelInfo(
        tunnel_id=tunnel_id,
        hostname="h",
        url="https://h",
        frp_token="",
        server_host="",
        labels=labels,
        expires_at=NOW,
        user_id=user,
        terminated_at=NOW if terminated else None,
    )


class FakeTunnelClient:
    pages: ClassVar[dict[str, object]] = {}

    def __init__(self) -> None:
        self.config = SimpleNamespace(user_id="me")
        self.deletes: list[str] = []
        self.closed = False
        FakeTunnelClient.last = self

    async def list_tunnels_page(self, labels, page, per_page):
        pages = self.pages[labels[0]]
        if isinstance(pages, Exception):
            raise pages
        return TunnelListPage(tunnels=pages[page - 1], has_next=page < len(pages))

    async def delete_tunnel(self, tunnel_id):
        self.deletes.append(tunnel_id)
        if tunnel_id == "broken":
            raise RuntimeError("timeout")
        return tunnel_id != "gone"

    async def close(self):
        self.closed = True


async def test_tunnel_sweep_deletes_only_this_users_live_tunnels_with_the_label(
    monkeypatch,
):
    FakeTunnelClient.pages = {
        "a": [
            [
                tunnel("t1", ["a"]),
                tunnel("t2", ["other"]),
                tunnel("t3", ["a"], user="them"),
            ],
            [
                tunnel("t4", ["a", "x"]),
                tunnel("t5", ["a"], terminated=True),
                tunnel("gone", ["a"]),
            ],
        ],
        "b": RuntimeError("list failed"),
        "c": [[tunnel("broken", ["c"])]],
    }
    monkeypatch.setattr(tunnel_client_module, "TunnelClient", FakeTunnelClient)
    assert await sweep.sweep_tunnels(["a", "b", "c"]) == {"a": 3, "b": -1, "c": 0}
    assert sorted(FakeTunnelClient.last.deletes) == ["broken", "gone", "t1", "t4"]
    assert FakeTunnelClient.last.closed


async def test_sweep_reports_both_kinds_and_skips_no_labels(monkeypatch):
    monkeypatch.setattr(
        prime_sandboxes,
        "AsyncSandboxClient",
        lambda: FakeSandboxClient({"a": deleted(["s"])}),
    )
    FakeTunnelClient.pages = {"a": [[tunnel("t", ["a"])]]}
    monkeypatch.setattr(tunnel_client_module, "TunnelClient", FakeTunnelClient)
    report = await sweep.sweep(["a"])
    assert report.sandboxes == {"a": 1} and report.tunnels == {"a": 1}
    assert await sweep.sweep([]) == sweep.SweepReport()


def test_prime_authenticated_reads_the_key(monkeypatch):
    from verifiers.v1.utils import prime

    monkeypatch.setattr(prime, "load_prime_config", dict)
    monkeypatch.delenv("PRIME_API_KEY", raising=False)
    assert not prime.prime_authenticated()
    monkeypatch.setenv("PRIME_API_KEY", "k")
    assert prime.prime_authenticated()


# -- images ------------------------------------------------------------------------------


async def test_delete_image(monkeypatch):
    calls: list[tuple] = []

    class FakeApi:
        async def request(self, method, endpoint, params=None):
            calls.append((method, endpoint, params))
            if endpoint.endswith("missing"):
                raise APIError("HTTP 404: not found")
            if endpoint.endswith("broken"):
                raise APIError("HTTP 500: boom")
            return {}

        async def aclose(self):
            pass

    monkeypatch.setattr(prime_sandboxes, "AsyncAPIClient", FakeApi)
    monkeypatch.setenv("PRIME_TEAM_ID", "t1")
    assert await delete_image("task", "v1")
    assert not await delete_image("task", "missing")
    with pytest.raises(APIError):
        await delete_image("task", "broken")
    assert calls[0] == ("DELETE", "/images/task/v1", {"teamId": "t1"})


@pytest.mark.asyncio
async def test_prime_reports_the_stored_bytes_of_the_checkpoints_it_still_holds(
    monkeypatch,
):
    from types import SimpleNamespace

    import prime_sandboxes

    from verifiers.v1.runtimes import PrimeConfig
    from verifiers.v1.runtimes.prime import PrimeBackend

    asked: list[list[str]] = []

    class Client:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return None

        async def get_checkpoints(self, ids):
            asked.append(list(ids))
            held = [
                SimpleNamespace(id=i, stored_bytes=len(i)) for i in ids if i != "gone"
            ]
            return SimpleNamespace(checkpoints=held)

    monkeypatch.setattr(prime_sandboxes, "AsyncSandboxClient", Client)
    ids = [f"ck{n}" for n in range(150)] + ["gone"]
    sizes = await PrimeBackend(PrimeConfig()).checkpoint_sizes(ids)
    assert [len(batch) for batch in asked] == [100, 51]
    assert len(sizes) == 150 and sizes["ck7"] == 3 and "gone" not in sizes
