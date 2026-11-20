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
