import asyncio
from types import SimpleNamespace

from prime_sandboxes import SandboxNotRunningError

from verifiers.v1.runtimes import PrimeBackend, PrimeConfig
from verifiers.v1.runtimes.backends import prime


class FakeClient:
    """Creates boxes from a memory snapshot; the first one ends before it runs."""

    def __init__(self):
        self.created: list[str] = []
        self.deleted: list[str] = []

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
