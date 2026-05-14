from __future__ import annotations

import asyncio

import pytest
from aiohttp import ClientSession, web

from verifiers.utils.serve_utils import get_free_port
from verifiers.v1.packages.harnesses.nemo_gym import (
    NeMoGymModelProxy,
    ROLLOUT_ID_HEADER,
    build_nemo_gym_global_config,
)


@pytest.mark.asyncio
async def test_nemo_gym_proxy_routes_concurrent_rollouts_by_header():
    upstream_a = await _start_upstream("a")
    upstream_b = await _start_upstream("b")
    proxy = NeMoGymModelProxy()
    await proxy.start()

    try:
        async with (
            upstream_a,
            upstream_b,
            proxy.activate(
                "rollout-a",
                {
                    "base_url": upstream_a.base_url,
                    "api_key": "key-a",
                    "model": "model-a",
                },
            ),
            proxy.activate(
                "rollout-b",
                {
                    "base_url": upstream_b.base_url,
                    "api_key": "key-b",
                    "model": "model-b",
                },
            ),
        ):
            async with ClientSession() as session:
                response_a, response_b = await asyncio.gather(
                    _proxy_response(session, proxy, "rollout-a"),
                    _proxy_response(session, proxy, "rollout-b"),
                )

        assert response_a == {"label": "a", "model": "model-a"}
        assert response_b == {"label": "b", "model": "model-b"}
        assert upstream_a.authorizations == ["Bearer key-a"]
        assert upstream_b.authorizations == ["Bearer key-b"]
        assert upstream_a.rollout_headers == [None]
        assert upstream_b.rollout_headers == [None]
    finally:
        await proxy.stop()


@pytest.mark.asyncio
async def test_nemo_gym_proxy_rejects_unrouted_request_with_multiple_rollouts():
    upstream_a = await _start_upstream("a")
    upstream_b = await _start_upstream("b")
    proxy = NeMoGymModelProxy()
    await proxy.start()

    try:
        async with (
            upstream_a,
            upstream_b,
            proxy.activate(
                "rollout-a",
                {
                    "base_url": upstream_a.base_url,
                    "api_key": "key-a",
                    "model": "model-a",
                },
            ),
            proxy.activate(
                "rollout-b",
                {
                    "base_url": upstream_b.base_url,
                    "api_key": "key-b",
                    "model": "model-b",
                },
            ),
        ):
            async with ClientSession() as session:
                response = await session.post(
                    f"http://{proxy.host}:{proxy.port}/v1/responses",
                    headers={"Authorization": f"Bearer {proxy.secret}"},
                    json={"model": "ignored"},
                )
                body = await response.json()

        assert response.status == 409
        assert ROLLOUT_ID_HEADER in body["error"]
    finally:
        await proxy.stop()


def test_nemo_gym_global_config_forwards_rollout_header():
    config = build_nemo_gym_global_config(
        config_paths=["agent.yaml"],
        endpoint_config={
            "base_url": "http://127.0.0.1:1/v1",
            "api_key": "secret",
            "model": "proxy-model",
        },
        global_config={"forward_request_headers": ["x-team-id"]},
    )

    assert config["forward_request_headers"] == ["x-team-id", ROLLOUT_ID_HEADER]


async def _proxy_response(
    session: ClientSession, proxy: NeMoGymModelProxy, rollout_id: str
) -> dict[str, str]:
    response = await session.post(
        f"http://{proxy.host}:{proxy.port}/v1/responses",
        headers={
            "Authorization": f"Bearer {proxy.secret}",
            ROLLOUT_ID_HEADER: rollout_id,
        },
        json={"model": "ignored"},
    )
    assert response.status == 200
    return await response.json()


class UpstreamServer:
    def __init__(
        self,
        *,
        label: str,
        runner: web.AppRunner,
        site: web.TCPSite,
        base_url: str,
        authorizations: list[str],
        rollout_headers: list[str | None],
    ) -> None:
        self.label = label
        self.runner = runner
        self.site = site
        self.base_url = base_url
        self.authorizations = authorizations
        self.rollout_headers = rollout_headers

    async def __aenter__(self) -> "UpstreamServer":
        return self

    async def __aexit__(self, *args: object) -> None:
        await self.runner.cleanup()


async def _start_upstream(label: str) -> UpstreamServer:
    authorizations: list[str] = []
    rollout_headers: list[str | None] = []

    async def handle_response(request: web.Request) -> web.Response:
        authorizations.append(request.headers["Authorization"])
        rollout_headers.append(request.headers.get(ROLLOUT_ID_HEADER))
        body = await request.json()
        return web.json_response({"label": label, "model": body["model"]})

    app = web.Application()
    app.router.add_post("/v1/responses", handle_response)
    runner = web.AppRunner(app)
    await runner.setup()
    port = get_free_port()
    site = web.TCPSite(runner, "127.0.0.1", port)
    await site.start()
    return UpstreamServer(
        label=label,
        runner=runner,
        site=site,
        base_url=f"http://127.0.0.1:{port}/v1",
        authorizations=authorizations,
        rollout_headers=rollout_headers,
    )
