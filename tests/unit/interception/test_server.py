import logging
from contextlib import asynccontextmanager

import httpx

from verifiers.v1.interception.server import InterceptionServer


class FakeTunnel:
    bind_host = "127.0.0.1"
    bind_port = 0
    tunnel_id = "tunnel-test"

    def __init__(self):
        self.exposures = 0

    @asynccontextmanager
    async def expose(self, port):
        self.exposures += 1
        yield f"http://localhost:{port}"


async def test_local_and_public_health_probes_report_server_and_tunnel(caplog):
    server = InterceptionServer()
    tunnel = FakeTunnel()
    server.tunnel = tunnel
    with caplog.at_level(logging.INFO, logger="verifiers.v1.interception.server"):
        async with server:
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{server.base_url}/healthz")
            assert response.status_code == 200
            assert response.json() == {
                "status": "ok",
                "server_id": server.server_id,
                "tunnel_id": "tunnel-test",
            }
    assert "scope=local" in caplog.text
    assert "scope=public" in caplog.text
    assert "probe=local" in caplog.text
    assert "probe=public" in caplog.text
    assert "matched=True" in caplog.text
    assert f"server_id={server.server_id}" in caplog.text
    assert "tunnel_id=tunnel-test" in caplog.text


async def test_unmatched_not_found_is_logged_without_payload_or_recreation(caplog):
    server = InterceptionServer()
    tunnel = FakeTunnel()
    server.tunnel = tunnel
    async with server:
        caplog.clear()
        with caplog.at_level(logging.INFO, logger="verifiers.v1.interception.server"):
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{server.base_url}/v1/wrong-path?key=url-secret",
                    content=b"prompt-secret",
                    headers={
                        "Authorization": "Bearer auth-secret",
                        "X-ACP-Model-Request-ID": "request-1",
                    },
                )
    assert response.status_code == 404
    assert "status=404 route=unmatched" in caplog.text
    assert "request_id='request-1'" in caplog.text
    assert tunnel.exposures == 1
    for secret in ("prompt-secret", "auth-secret", "url-secret"):
        assert secret not in caplog.text
