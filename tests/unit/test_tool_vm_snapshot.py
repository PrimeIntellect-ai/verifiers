"""A rollout snapshots its tool server VMs with its box, and its teardown frees every
snapshot nothing restores."""

import contextlib
from types import SimpleNamespace

import pytest

import verifiers.v1.rollout as rollout_module
from verifiers.v1.errors import SandboxError
from verifiers.v1.rollout import Rollout
from verifiers.v1.runtimes import private_network
from verifiers.v1.runtimes.backends.prime import PrimeBackend
from verifiers.v1.runtimes.base import Snapshot


class VM:
    """A VM whose snapshot is named, then fails while the VM resumes when `fails`."""

    def __init__(self, snapshot_id: str, *, fails: bool = False):
        self.snapshot_id = snapshot_id
        self.fails = fails
        self.stray_snapshots: list[Snapshot] = []

    async def snapshot(self, taken=None) -> Snapshot:
        snapshot = Snapshot(id=self.snapshot_id)
        self.stray_snapshots.append(snapshot)
        if self.fails:
            raise SandboxError("resume failed")
        return snapshot


def make_rollout(box: VM, vms: dict[str, VM]) -> Rollout:
    run = Rollout.__new__(Rollout)
    run.runtime = SimpleNamespace(backend=box)  # type: ignore[assignment]
    run.deployment = SimpleNamespace(  # type: ignore[assignment]
        tool_vms=dict.fromkeys(vms), target_backends=vms
    )
    return run


async def test_snapshot_carries_the_tool_vms_and_leaves_only_it_to_free(monkeypatch):
    rejoined: list = []

    async def rejoin(backends):
        rejoined.append(backends)

    monkeypatch.setattr(rollout_module, "rejoin", rejoin)
    box, tool = VM("s-box"), VM("s-tool")
    taken: list[Snapshot] = []
    snapshot = await make_rollout(box, {"t": tool})._snapshot(taken.append)
    assert snapshot.peers == {"t": Snapshot(id="s-tool")}
    assert taken == [snapshot] and box.stray_snapshots == [snapshot]
    assert tool.stray_snapshots == [] and rejoined == [[box, tool]]


async def test_a_failed_tool_vm_snapshot_is_freed_with_the_box(monkeypatch):
    async def rejoin(backends):
        return None

    monkeypatch.setattr(rollout_module, "rejoin", rejoin)
    box, good, bad = VM("s-box"), VM("s-good"), VM("s-bad", fails=True)
    run = make_rollout(box, {"good": good, "bad": bad})
    with pytest.raises(SandboxError):
        await run._snapshot(lambda snapshot: None)
    assert {s.id for s in box.stray_snapshots} == {"s-box", "s-good", "s-bad"}
    assert good.stray_snapshots == [] and bad.stray_snapshots == []


async def test_rejoin_brings_back_every_vm_that_left_though_one_fails(monkeypatch):
    calls: list[str] = []

    class Client:
        async def remove_lan_member(self, network, sandbox):
            calls.append(f"leave:{sandbox}")
            if sandbox == "b":
                raise RuntimeError("platform error")

        async def _join_lan(self, network, sandbox, tags):
            calls.append(f"join:{sandbox}")

    @contextlib.asynccontextmanager
    async def leased_client():
        yield Client()

    monkeypatch.setattr(private_network, "leased_client", leased_client)
    backends = []
    for sandbox in "abc":
        backend = PrimeBackend.__new__(PrimeBackend)
        backend.lan = ("net", ["tag"])
        backend.info = SimpleNamespace(id=sandbox)
        backends.append(backend)
    with pytest.raises(SandboxError):
        await private_network.rejoin(backends)
    assert calls == ["leave:a", "leave:b", "leave:c", "join:a", "join:c"]
