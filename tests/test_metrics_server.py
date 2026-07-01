import asyncio
import http.client

import pytest

from verifiers.serve.server.env_router import EnvRouterStats
from verifiers.serve.server.env_worker import EnvWorkerStats
from verifiers.serve.server.metrics import MetricsServer, render_prometheus_text


def test_render_prometheus_text_smoke():
    stats = EnvRouterStats(
        workers={
            0: EnvWorkerStats(worker_id=0, timestamp=0.0, active_tasks=3),
            1: EnvWorkerStats(worker_id=1, timestamp=0.0, active_tasks=5),
        }
    )

    body = render_prometheus_text(stats, env_id="math-python", version="0.1.15")

    assert "verifiers_env_active_tasks 8" in body
    assert "verifiers_env_workers_total 2" in body
    assert 'verifiers_env_worker_active_tasks{worker_id="0"} 3' in body
    assert 'verifiers_env_worker_active_tasks{worker_id="1"} 5' in body
    assert 'verifiers_env_server_info{env_id="math-python",version="0.1.15"} 1' in body
    assert 'verifiers_env_loop_lag_seconds{worker_id="0",quantile="p95"} 0.0' in body
    assert body.endswith("\n")


@pytest.mark.asyncio
async def test_metrics_server_serves_metrics(unused_tcp_port):
    class StubRouter:
        stats = EnvRouterStats(
            workers={0: EnvWorkerStats(worker_id=0, timestamp=0.0, active_tasks=2)}
        )

    server = MetricsServer(
        StubRouter(), env_id="test", version="dev", port=unused_tcp_port
    )
    await server.start()
    try:

        def fetch():
            conn = http.client.HTTPConnection("127.0.0.1", unused_tcp_port, timeout=2)
            conn.request("GET", "/metrics")
            response = conn.getresponse()
            return response.status, response.read().decode()

        status, body = await asyncio.get_running_loop().run_in_executor(None, fetch)

        assert status == 200
        assert "verifiers_env_active_tasks 2" in body
        assert "verifiers_env_workers_total 1" in body
    finally:
        await server.close()


@pytest.mark.asyncio
async def test_metrics_server_404_on_other_path(unused_tcp_port):
    class StubRouter:
        stats = EnvRouterStats(workers={})

    server = MetricsServer(StubRouter(), env_id="t", version="x", port=unused_tcp_port)
    await server.start()
    try:

        def fetch():
            conn = http.client.HTTPConnection("127.0.0.1", unused_tcp_port, timeout=2)
            conn.request("GET", "/")
            return conn.getresponse().status

        status = await asyncio.get_running_loop().run_in_executor(None, fetch)

        assert status == 404
    finally:
        await server.close()
