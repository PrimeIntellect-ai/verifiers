import asyncio
from types import SimpleNamespace

import pytest
from prime_sandboxes import SandboxNotRunningError

from verifiers.v1.errors import SandboxError
from verifiers.v1.runtimes import PrimeBackend, PrimeConfig
from verifiers.v1.runtimes.backends import prime


class FakeClient:
    """Creates boxes from a memory snapshot; the first one ends before it runs."""

    def __init__(self, durable: bool = True):
        self.created: list[str] = []
        self.deleted: list[str] = []
        self.durable = durable

    async def wait_for_checkpoint(self, checkpoint_id, timeout_seconds):
        if not self.durable:
            raise RuntimeError(
                f"Checkpoint {checkpoint_id} cannot become durable (state=FAILED)"
            )
        return SimpleNamespace(id=checkpoint_id, state="DURABLE")

    async def create(self, request):
        self.created.append(f"sb{len(self.created)}")
        return SimpleNamespace(id=self.created[-1], pending_image_build_id=None)

    async def wait_for_creation(self, sandbox_id, max_attempts):
        if sandbox_id == "sb0":
            raise SandboxNotRunningError(sandbox_id, "TERMINATED")

    async def delete(self, sandbox_id):
        self.deleted.append(sandbox_id)


def test_a_memory_restore_retry_deletes_the_box_that_died(monkeypatch):
    client = FakeClient()
    monkeypatch.setattr(prime, "ensure_prime_auth", lambda: None)
    monkeypatch.setattr(prime, "_lease_client", lambda: client)
    backend = PrimeBackend(PrimeConfig(checkpoint="ckpt", restore_memory=True))
    asyncio.run(backend.start())
    assert client.created == ["sb0", "sb1"]
    assert client.deleted == ["sb0"]
    assert backend.info.id == "sb1"


def test_a_memory_restore_waits_for_a_durable_snapshot(monkeypatch):
    """A save returns before its snapshot is uploaded: a restore from one that never
    becomes durable fails before it creates a box, and the rollout starts over."""
    client = FakeClient(durable=False)
    monkeypatch.setattr(prime, "ensure_prime_auth", lambda: None)
    monkeypatch.setattr(prime, "_lease_client", lambda: client)
    backend = PrimeBackend(PrimeConfig(checkpoint="ckpt", restore_memory=True))
    with pytest.raises(SandboxError, match="cannot become durable"):
        asyncio.run(backend.start())
    assert client.created == []


def test_attach_waits_until_sandbox_is_awake(monkeypatch):
    class FakeProcess:
        session_uuid = "sess-1"

        async def _empty(self):
            if False:
                yield b""

        @property
        def stdout(self):
            return self._empty()

        @property
        def stderr(self):
            return self._empty()

        async def wait(self):
            return 0

        async def aclose(self):
            return None

    events: list[str] = []

    class AttachClient:
        async def attach_process(self, sandbox_id, session_uuid):
            events.append(f"attach:{sandbox_id}:{session_uuid}")
            return FakeProcess()

    async def scenario():
        monkeypatch.setattr(prime, "ensure_prime_auth", lambda: None)
        backend = PrimeBackend(PrimeConfig())
        backend._client = AttachClient()
        backend.info.id = "sb1"
        backend._awake.clear()

        task = asyncio.create_task(
            backend.attach(backend.bind(), {"session": "sess-1"})
        )
        await asyncio.sleep(0)
        assert events == []
        events.append("awake")
        backend._awake.set()
        result = await task
        assert result.exit_code == 0
        assert events == ["awake", "attach:sb1:sess-1"]

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "lookup,resumed",
    [
        ("PAUSED", True),
        (RuntimeError("HTTP 429: Too Many Requests"), True),
        ("RUNNING", False),
        ("TERMINATED", False),
        (RuntimeError("HTTP 404: sandbox not found"), False),
    ],
)
def test_a_lost_pause_resumes_only_a_box_that_may_be_paused(
    monkeypatch, lookup, resumed
):
    """A pause that raised may still have stopped the box: a box that may be paused
    runs again, a running, dead or missing one is left as it is."""
    calls: list[str] = []

    class Client:
        async def pause(self, sandbox_id):
            raise RuntimeError("connection lost")

        async def get(self, sandbox_id):
            if isinstance(lookup, Exception):
                raise lookup
            return SimpleNamespace(status=lookup)

    async def resume(self):
        calls.append("resume")

    async def scenario():
        monkeypatch.setattr(prime, "ensure_prime_auth", lambda: None)
        monkeypatch.setattr(PrimeBackend, "_resume", resume)
        backend = PrimeBackend(PrimeConfig())
        backend._client = Client()
        backend.info.id = "sb1"
        with pytest.raises(SandboxError, match="connection lost"):
            await backend.snapshot()
        assert calls == (["resume"] if resumed else [])
        assert backend._awake.is_set()

    asyncio.run(scenario())


@pytest.mark.parametrize("taken_fails", [False, True])
def test_a_snapshot_is_named_before_the_box_stops(monkeypatch, taken_fails):
    """`taken` gets the snapshot once the pause is accepted, before the box has
    stopped. The box runs again whatever `taken` does."""
    calls: list[str] = []

    class Client:
        async def pause(self, sandbox_id):
            calls.append("pause")
            return SimpleNamespace(checkpoint_id="ck1")

        async def wait_for_pause(self, sandbox_id, timeout_seconds):
            calls.append("paused")

    async def resume(self):
        calls.append("resume")

    def taken(snapshot):
        calls.append(f"taken:{snapshot.id}")
        if taken_fails:
            raise RuntimeError("taken failed")

    async def scenario():
        monkeypatch.setattr(prime, "ensure_prime_auth", lambda: None)
        monkeypatch.setattr(PrimeBackend, "_resume", resume)
        backend = PrimeBackend(PrimeConfig())
        backend._client = Client()
        backend.info.id = "sb1"
        if taken_fails:
            with pytest.raises(SandboxError, match="taken failed"):
                await backend.snapshot(taken)
            assert calls == ["pause", "taken:ck1", "resume"]
        else:
            assert (await backend.snapshot(taken)).id == "ck1"
            assert calls == ["pause", "taken:ck1", "paused", "resume"]
        assert backend._awake.is_set()

    asyncio.run(scenario())


@pytest.mark.parametrize("settles", [True, False])
def test_a_box_kept_stopped_runs_again_only_on_wake(monkeypatch, settles):
    """A snapshot that keeps the box stopped leaves calls into it waiting until
    `wake`. A box whose stop was not seen to finish runs again at once."""
    calls: list[str] = []

    class Client:
        async def pause(self, sandbox_id):
            calls.append("pause")
            return SimpleNamespace(checkpoint_id="ck1")

        async def wait_for_pause(self, sandbox_id, timeout_seconds):
            if not settles:
                raise TimeoutError("still pausing")
            calls.append("paused")

    async def resume(self):
        calls.append("resume")

    async def scenario():
        monkeypatch.setattr(prime, "ensure_prime_auth", lambda: None)
        monkeypatch.setattr(PrimeBackend, "_resume", resume)
        backend = PrimeBackend(PrimeConfig())
        backend._client = Client()
        backend.info.id = "sb1"
        if not settles:
            with pytest.raises(SandboxError, match="still pausing"):
                await backend.snapshot(stay_stopped=True)
            assert calls == ["pause", "resume"] and backend._awake.is_set()
            return
        assert (await backend.snapshot(stay_stopped=True)).id == "ck1"
        assert calls == ["pause", "paused"] and not backend._awake.is_set()
        await backend.wake()
        assert calls == ["pause", "paused", "resume"] and backend._awake.is_set()

    asyncio.run(scenario())
