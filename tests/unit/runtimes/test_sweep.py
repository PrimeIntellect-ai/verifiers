"""Sweeping a run's leftover Prime sandboxes and tunnels, against fake Prime clients:
only what carries the label, belongs to this user, was created before the cutoff and is
not held by this process is deleted."""

import contextlib
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import prime_tunnel
import prime_tunnel.core.client
import pytest

from verifiers.v1.interception.tunnel import prime as tunnel_prime
from verifiers.v1.runtimes import PrimeBackend, PrimeConfig, run_labels
from verifiers.v1.runtimes.backends import prime
from verifiers.v1.runtimes.base import _LIVE, register
from verifiers.v1.runtimes.sweep import sweep, sweep_sandboxes

LAUNCH = datetime(2026, 10, 20, 12, 0, tzinfo=UTC)
EARLIER = LAUNCH - timedelta(hours=1)
LATER = LAUNCH + timedelta(seconds=5)


def box(id, created=EARLIER, labels=("run-1",), user="me"):
    return SimpleNamespace(id=id, created_at=created, labels=list(labels), user_id=user)


def tunnel(id, created=EARLIER, labels=("run-1",), user="me", terminated=None):
    return SimpleNamespace(
        tunnel_id=id,
        created_at=created,
        labels=list(labels),
        user_id=user,
        terminated_at=terminated,
    )


class FakeSandboxes:
    """A team-wide listing at its worst: every team box, whatever the label or user
    asked for, in pages of two. Records the ids each bulk delete names."""

    def __init__(self, boxes, fail=False):
        self.boxes = boxes
        self.fail = fail
        self.deleted: list[str] = []
        self.client = SimpleNamespace(config=SimpleNamespace(user_id="me"))

    async def list(self, labels, page, per_page, exclude_terminated, user_id):
        if self.fail:
            raise RuntimeError("HTTP 503: unavailable")
        return SimpleNamespace(
            sandboxes=self.boxes[(page - 1) * 2 : page * 2],
            has_next=page * 2 < len(self.boxes),
        )

    async def bulk_delete(self, sandbox_ids):
        self.deleted += sandbox_ids
        return SimpleNamespace(succeeded=sandbox_ids, failed=[])


class FakeTunnels:
    def __init__(self, tunnels):
        self.tunnels = tunnels
        self.deleted: list[str] = []
        self.config = SimpleNamespace(user_id="me")

    async def list_tunnels_page(self, labels, page, per_page):
        return SimpleNamespace(tunnels=self.tunnels, has_next=False)

    async def delete_tunnel(self, tunnel_id):
        self.deleted.append(tunnel_id)
        return tunnel_id != "t-gone"  # the SDK's False: already gone

    async def close(self):
        pass


@pytest.fixture
def provider(monkeypatch):
    sandboxes, tunnels = FakeSandboxes([]), FakeTunnels([])

    @contextlib.asynccontextmanager
    async def leased_client():
        yield sandboxes

    monkeypatch.setattr(prime, "leased_client", leased_client)
    monkeypatch.setattr(prime_tunnel.core.client, "TunnelClient", lambda: tunnels)
    monkeypatch.setenv("VF_RUN_ID", "run-1")
    return sandboxes, tunnels


async def test_a_sweep_deletes_only_the_runs_leftovers(provider):
    sandboxes, tunnels = provider
    sandboxes.boxes = [
        box("old-1"),
        box("old-2"),
        box("old-3"),  # on the second page
        box("other-run", labels=["run-2"]),
        box("teammate", user="them"),
        box("unowned", user=None),
        box("at-the-cutoff", created=LAUNCH),
    ]
    tunnels.tunnels = [
        tunnel("t-old"),
        tunnel("t-other", labels=["run-2"]),
        tunnel("t-stopped", terminated=EARLIER),
        tunnel("t-unknown-age", created=None),
        tunnel("t-teammate", user="them"),
        tunnel("t-gone"),
    ]
    report = await sweep(before=LAUNCH)
    assert sorted(sandboxes.deleted) == ["old-1", "old-2", "old-3"]
    assert tunnels.deleted == ["t-old", "t-gone"]
    # A tunnel that was already gone is not counted as swept.
    assert (report.sandboxes, report.tunnels) == ({"run-1": 3}, {"run-1": 1})


async def test_without_a_user_id_a_team_listing_is_never_swept(provider):
    """CI and launchers often configure only a team key: the listing then covers the
    whole team, and a teammate's run can carry the same label (a fixed run id)."""
    sandboxes, tunnels = provider
    sandboxes.client.config.user_id = tunnels.config.user_id = None
    sandboxes.boxes = [box("mine", user=None), box("teammates", user="them")]
    tunnels.tunnels = [tunnel("t-mine", user=None), tunnel("t-teammates", user="them")]
    report = await sweep(before=LAUNCH)
    assert sandboxes.deleted == tunnels.deleted == []
    assert (report.sandboxes, report.tunnels) == ({"run-1": -1}, {"run-1": -1})
    # A caller whose label is minted for this launch vouches for it.
    report = await sweep(before=LAUNCH, unique_labels=True)
    assert sandboxes.deleted == ["mine", "teammates"]
    assert tunnels.deleted == ["t-mine", "t-teammates"]


async def test_nothing_the_launch_created_is_swept_whenever_it_sweeps(
    provider, monkeypatch
):
    """A launcher takes `before` as it starts, then its setup creates boxes before the
    sweep runs (flow #77's order): those are after the cutoff, and a box this process
    holds is spared even when the provider's clock puts it before the cutoff."""
    sandboxes, tunnels = provider
    monkeypatch.setattr(prime, "ensure_prime_auth", lambda: None)
    held = PrimeBackend(PrimeConfig())
    held.info.id = "held"
    register(held)

    class HeldTunnel:
        tunnel_id = "t-held"

    monkeypatch.setattr(tunnel_prime, "LIVE_TUNNELS", {HeldTunnel()})
    sandboxes.boxes = [box("leftover"), box("from-setup", created=LATER), box("held")]
    tunnels.tunnels = [tunnel("t-leftover"), tunnel("t-held")]
    try:
        await sweep(before=LAUNCH)
    finally:
        _LIVE.discard(held)  # no atexit delete of a box that never existed
    assert sandboxes.deleted == ["leftover"]
    assert tunnels.deleted == ["t-leftover"]


async def test_a_failed_sweep_reports_minus_one_and_never_raises(provider):
    sandboxes, _ = provider
    sandboxes.fail = True
    report = await sweep(["run-1", "run-2"], before=LAUNCH)
    assert report.sandboxes == {"run-1": -1, "run-2": -1}
    assert report.tunnels == {"run-1": 0, "run-2": 0}


async def test_a_sweep_refuses_a_naive_cutoff_and_shared_base_labels(
    provider, monkeypatch
):
    with pytest.raises(ValueError, match="timezone-aware"):
        await sweep(before=LAUNCH.replace(tzinfo=None))
    monkeypatch.setattr(prime, "BASE_LABELS", ["team-evals", "run-1"])
    with pytest.raises(ValueError, match="team-evals"):
        await sweep_sandboxes(["team-evals"], before=LAUNCH)
    # The run's own scope stays sweepable even when it is also a base label.
    assert await sweep_sandboxes(["run-1"], before=LAUNCH) == {"run-1": 0}


async def test_tunnels_carry_the_run_labels(monkeypatch):
    opened = {}

    class FakeTunnel:
        def __init__(self, local_port, labels):
            opened["labels"] = labels
            self.tunnel_id = "t1"

        async def start(self):
            return "https://t1.tunnel.example"

        def sync_stop(self):
            pass

    monkeypatch.setenv("VF_RUN_ID", "run-1")
    monkeypatch.setattr(prime, "BASE_LABELS", ["trainer"])
    monkeypatch.setattr(prime_tunnel, "Tunnel", FakeTunnel)
    monkeypatch.setattr(tunnel_prime, "ensure_prime_auth", lambda: None)
    monkeypatch.setattr(tunnel_prime, "tunnel_limiter", contextlib.nullcontext)
    async with tunnel_prime.PrimeTunnel().expose(8000):
        assert [t.tunnel_id for t in tunnel_prime.LIVE_TUNNELS] == ["t1"]
    assert opened["labels"] == run_labels() == ["trainer", "run-1"]
    assert not tunnel_prime.LIVE_TUNNELS


async def test_a_known_user_id_decides_even_for_unique_labels(provider):
    """A launch label can still collide (a fixed hostname and path in two pods): when
    the owner is known, a teammate's box is never deleted."""
    sandboxes, tunnels = provider
    sandboxes.boxes = [box("mine"), box("teammates", user="them")]
    tunnels.tunnels = [tunnel("t-mine"), tunnel("t-teammates", user="them")]
    await sweep(before=LAUNCH, unique_labels=True)
    assert sandboxes.deleted == ["mine"]
    assert tunnels.deleted == ["t-mine"]


async def test_an_empty_user_id_is_no_user_id(provider, caplog):
    """`PRIME_USER_ID=""` makes the SDK report "" as the user id: it names nobody, so
    nothing is swept, the label reports -1 and the warning says why (not a silent 0)."""
    sandboxes, tunnels = provider
    sandboxes.client.config.user_id = tunnels.config.user_id = ""
    sandboxes.boxes = [box("mine", user="me"), box("blank", user="")]
    tunnels.tunnels = [tunnel("t-mine", user="me"), tunnel("t-blank", user="")]
    with caplog.at_level("WARNING", logger="verifiers.v1.runtimes.sweep"):
        report = await sweep(before=LAUNCH)
    assert sandboxes.deleted == tunnels.deleted == []
    assert (report.sandboxes, report.tunnels) == ({"run-1": -1}, {"run-1": -1})
    assert "no Prime user id is configured" in caplog.text
