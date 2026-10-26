"""Interception pool selection must not reuse a slot whose tunnel died."""

import pytest

from verifiers.v1.interception.pool import (
    ElasticInterceptionPool,
    ElasticInterceptionPoolConfig,
)


class Slot:
    def __init__(self, healthy, load=0):
        self.healthy = healthy
        self.load = load


@pytest.mark.asyncio
async def test_elastic_server_skips_unhealthy_slots():
    pool = ElasticInterceptionPool(ElasticInterceptionPoolConfig(multiplex=1))
    healthy = Slot(healthy=True)
    pool.servers = [Slot(healthy=False), healthy]
    assert await pool._server() is healthy
