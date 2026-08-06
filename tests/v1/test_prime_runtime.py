from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from verifiers.v1.runtimes.prime import PrimeConfig, PrimeRuntime


def test_prime_lifetime_timeout_is_configurable() -> None:
    config = PrimeConfig(vm=True, lifetime_timeout=600)

    assert config.lifetime_timeout == 600
    assert config.idle_timeout == 3_600


def test_prime_idle_timeout_cannot_exceed_lifetime() -> None:
    with pytest.raises(ValidationError, match="must not exceed"):
        PrimeConfig(lifetime_timeout=3_600, idle_timeout=3_601)


@pytest.mark.asyncio
async def test_prime_runtime_uses_configured_lifetime(monkeypatch) -> None:
    import prime_sandboxes

    class Client:
        def __init__(self) -> None:
            self.request = None
            self.run_timeout = None

        async def create(self, request):
            self.request = request
            return SimpleNamespace(id="sandbox", pending_image_build_id=None)

        async def wait_for_creation(self, sandbox_id: str) -> None:
            assert sandbox_id == "sandbox"

        async def execute_command(self, sandbox_id: str, command: str):
            del sandbox_id, command
            return SimpleNamespace(exit_code=0, stdout="", stderr="")

        async def run_background_job(self, sandbox_id: str, command: str, **kwargs):
            del sandbox_id, command
            self.run_timeout = kwargs["timeout"]
            return SimpleNamespace(exit_code=0, stdout="", stderr="")

    client = Client()
    monkeypatch.setattr(prime_sandboxes, "AsyncSandboxClient", lambda: client)
    runtime = PrimeRuntime(PrimeConfig(vm=True, lifetime_timeout=43_200), name="test")

    await runtime.start()
    await runtime.run(["true"], {})

    assert client.request.timeout_minutes == 720
    assert client.request.idle_timeout_minutes is None
    assert client.run_timeout == 43_200
