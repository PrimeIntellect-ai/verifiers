from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from verifiers.v1.errors import HarnessError, TunnelError
from verifiers.v1.harnesses.bash import BashHarness, BashHarnessConfig
from verifiers.v1.interception.pool import ElasticInterceptionPool
from verifiers.v1.runtimes import ProgramResult


TUNNEL_GONE_PAGE = """<!doctype html>
<html>
  <body>
    <h1>404</h1>
    <p>Tunnel not found or no longer active.</p>
  </body>
</html>"""


@pytest.mark.asyncio
async def test_harness_classifies_prime_tunnel_gone_page() -> None:
    harness = BashHarness(BashHarnessConfig(id="bash"))
    runtime = SimpleNamespace(alive=AsyncMock(return_value=True))
    trace = SimpleNamespace(stop_condition=None)

    with pytest.raises(TunnelError, match="prime tunnel disappeared"):
        await harness._check_result(
            trace,
            runtime,
            ProgramResult(exit_code=1, stdout="", stderr=TUNNEL_GONE_PAGE),
        )

    runtime.alive.assert_not_awaited()


@pytest.mark.asyncio
async def test_harness_keeps_unrelated_404_as_harness_error() -> None:
    harness = BashHarness(BashHarnessConfig(id="bash"))
    runtime = SimpleNamespace(alive=AsyncMock(return_value=True))
    trace = SimpleNamespace(stop_condition=None)

    with pytest.raises(HarnessError):
        await harness._check_result(
            trace,
            runtime,
            ProgramResult(exit_code=1, stdout="", stderr="404: task file missing"),
        )


class FakeServer:
    def __init__(self) -> None:
        self.base_url = "https://dead.tunnel.example"
        self._load = 0
        self.stopped = False

    @property
    def load(self) -> int:
        return self._load

    def register(self, session) -> tuple[str, str]:
        self._load += 1
        return f"model-{self._load}", f"state-{self._load}"

    def unregister(self, model_secret: str, state_secret: str) -> None:
        self._load -= 1

    async def stop(self) -> None:
        self.stopped = True


def session() -> SimpleNamespace:
    return SimpleNamespace(trace=SimpleNamespace(errors=[]))


@pytest.mark.asyncio
async def test_elastic_pool_retires_failed_tunnel_after_active_rollouts_drain() -> None:
    pool = ElasticInterceptionPool(requires_tunnel=True)
    server = FakeServer()
    pool.servers.append(server)
    first = session()
    second = session()

    async with pool.acquire(first):
        async with pool.acquire(second):
            second.trace.errors.append(SimpleNamespace(type="TunnelError"))
        assert server not in pool.servers
        assert not server.stopped

    assert server.stopped
