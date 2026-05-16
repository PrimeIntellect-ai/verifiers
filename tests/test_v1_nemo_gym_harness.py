from __future__ import annotations

import asyncio
import gzip
import os

import pytest
from aiohttp import ClientSession, web

from verifiers.utils.serve_utils import get_free_port
from verifiers.v1.packages.harnesses.nemo_gym import (
    NEMO_GYM_EXTERNAL_POLICY_MODEL_ENTRYPOINT,
    NEMO_GYM_POLICY_MODEL_SERVER_NAME,
    NEMO_GYM_POLICY_MODEL_TYPE_NAME,
    NeMoGymHarness,
    NeMoGymModelProxy,
    PersistentNeMoGymRunner,
    build_nemo_gym_global_config,
    build_nemo_gym_policy_model_config,
    disable_ray_uv_run_runtime_env,
    nemo_gym_proxy_model_name,
    set_nemo_gym_proxy_model,
    skip_nemo_gym_policy_model_process,
)


@pytest.mark.asyncio
async def test_nemo_gym_proxy_routes_concurrent_rollouts_by_model():
    upstream_a = await _start_upstream("a")
    upstream_b = await _start_upstream("b")
    proxy = NeMoGymModelProxy()
    await proxy.start()
    routing_model_a = nemo_gym_proxy_model_name("rollout-a")
    routing_model_b = nemo_gym_proxy_model_name("rollout-b")

    try:
        async with (
            upstream_a,
            upstream_b,
            proxy.activate(
                routing_model_a,
                {
                    "base_url": upstream_a.base_url,
                    "api_key": "key-a",
                    "model": "model-a",
                },
            ),
            proxy.activate(
                routing_model_b,
                {
                    "base_url": upstream_b.base_url,
                    "api_key": "key-b",
                    "model": "model-b",
                },
            ),
        ):
            async with ClientSession() as session:
                response_a, response_b = await asyncio.gather(
                    _proxy_response(session, proxy, routing_model_a),
                    _proxy_response(session, proxy, routing_model_b),
                )

        assert response_a == {"label": "a", "model": "model-a"}
        assert response_b == {"label": "b", "model": "model-b"}
        assert upstream_a.authorizations == ["Bearer key-a"]
        assert upstream_b.authorizations == ["Bearer key-b"]
    finally:
        await proxy.stop()


@pytest.mark.asyncio
async def test_nemo_gym_proxy_rejects_unrouted_request_with_multiple_rollouts():
    upstream_a = await _start_upstream("a")
    upstream_b = await _start_upstream("b")
    proxy = NeMoGymModelProxy()
    await proxy.start()
    routing_model_a = nemo_gym_proxy_model_name("rollout-a")
    routing_model_b = nemo_gym_proxy_model_name("rollout-b")

    try:
        async with (
            upstream_a,
            upstream_b,
            proxy.activate(
                routing_model_a,
                {
                    "base_url": upstream_a.base_url,
                    "api_key": "key-a",
                    "model": "model-a",
                },
            ),
            proxy.activate(
                routing_model_b,
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
        assert "model" in body["error"]
    finally:
        await proxy.stop()


@pytest.mark.asyncio
async def test_nemo_gym_proxy_falls_back_when_only_one_rollout_is_active():
    upstream = await _start_upstream("single")
    proxy = NeMoGymModelProxy()
    await proxy.start()

    try:
        async with (
            upstream,
            proxy.activate(
                nemo_gym_proxy_model_name("rollout"),
                {
                    "base_url": upstream.base_url,
                    "api_key": "key",
                    "model": "real-model",
                },
            ),
        ):
            async with ClientSession() as session:
                response = await session.post(
                    f"http://{proxy.host}:{proxy.port}/v1/responses",
                    headers={"Authorization": f"Bearer {proxy.secret}"},
                    json={"model": "verifiers-nemo-gym-proxy"},
                )
                body = await response.json()

        assert response.status == 200
        assert body == {"label": "single", "model": "real-model"}
    finally:
        await proxy.stop()


@pytest.mark.asyncio
async def test_nemo_gym_proxy_strips_content_encoding_after_decompression():
    async def handle_response(request: web.Request) -> web.Response:
        body = await request.json()
        return web.Response(
            body=gzip.compress(
                f'{{"model":"{body["model"]}","ok":true}}'.encode()
            ),
            headers={
                "Content-Encoding": "gzip",
                "Content-Type": "application/json",
            },
        )

    app = web.Application()
    app.router.add_post("/v1/responses", handle_response)
    runner = web.AppRunner(app)
    await runner.setup()
    port = get_free_port()
    site = web.TCPSite(runner, "127.0.0.1", port)
    await site.start()

    proxy = NeMoGymModelProxy()
    await proxy.start()
    routing_model = nemo_gym_proxy_model_name("rollout")
    try:
        async with proxy.activate(
            routing_model,
            {
                "base_url": f"http://127.0.0.1:{port}/v1",
                "api_key": "key",
                "model": "real-model",
            },
        ):
            async with ClientSession(auto_decompress=False) as session:
                response = await session.post(
                    f"http://{proxy.host}:{proxy.port}/v1/responses",
                    headers={"Authorization": f"Bearer {proxy.secret}"},
                    json={"model": routing_model},
                )
                body = await response.read()

        assert response.status == 200
        assert "Content-Encoding" not in response.headers
        assert body == b'{"model":"real-model","ok":true}'
    finally:
        await proxy.stop()
        await runner.cleanup()


def test_nemo_gym_global_config_uses_proxy_endpoint_without_header_forwarding():
    config = build_nemo_gym_global_config(
        config_paths=["agent.yaml"],
        endpoint_config={
            "base_url": "http://127.0.0.1:12345/v1",
            "api_key": "secret",
            "model": "proxy-model",
        },
        global_config={"custom": "value"},
    )

    assert config["policy_base_url"] == "http://127.0.0.1:12345/v1"
    assert config["policy_api_key"] == "secret"
    assert config["policy_model_name"] == "proxy-model"
    assert config["custom"] == "value"
    assert "forward_request_headers" not in config
    assert config[NEMO_GYM_POLICY_MODEL_SERVER_NAME] == {
        "responses_api_models": {
            NEMO_GYM_POLICY_MODEL_TYPE_NAME: {
                "entrypoint": NEMO_GYM_EXTERNAL_POLICY_MODEL_ENTRYPOINT,
                "host": "127.0.0.1",
                "port": 12345,
            }
        }
    }


def test_nemo_gym_harness_does_not_add_a_model_server_config_path():
    harness = NeMoGymHarness(
        config_paths=["agent.yaml"],
        server_name="agent_server",
        agent_name="agent",
    )

    assert harness._config_paths() == ["agent.yaml"]


def test_build_nemo_gym_policy_model_config_requires_explicit_port():
    with pytest.raises(ValueError, match="host and port"):
        build_nemo_gym_policy_model_config(
            {
                "base_url": "https://api.openai.com/v1",
                "api_key": "secret",
                "model": "model",
            }
        )


def test_set_nemo_gym_proxy_model_preserves_row_without_mutating_create_params():
    create_params = {"input": [{"role": "user", "content": "hi"}], "temperature": 0.2}
    row = {"responses_create_params": create_params}

    set_nemo_gym_proxy_model(row, "proxy-rollout")

    assert row["responses_create_params"] == {
        "input": [{"role": "user", "content": "hi"}],
        "temperature": 0.2,
        "model": "proxy-rollout",
    }
    assert create_params == {"input": [{"role": "user", "content": "hi"}], "temperature": 0.2}


def test_disable_ray_uv_run_runtime_env_sets_and_restores_env(monkeypatch):
    monkeypatch.delenv("RAY_ENABLE_UV_RUN_RUNTIME_ENV", raising=False)

    with disable_ray_uv_run_runtime_env():
        assert os.environ["RAY_ENABLE_UV_RUN_RUNTIME_ENV"] == "0"

    assert "RAY_ENABLE_UV_RUN_RUNTIME_ENV" not in os.environ


def test_skip_nemo_gym_policy_model_process_leaves_other_processes_untouched():
    class FakeNemoCliModule:
        def __init__(self) -> None:
            self.calls = []
            self.setup_calls = []

        def setup_env_command(
            self, dir_path: object, global_config_dict: object, prefix: str
        ) -> str:
            self.setup_calls.append((dir_path, global_config_dict, prefix))
            return "real-setup"

        def run_command(self, command: str, working_dir_path: object):
            self.calls.append((command, working_dir_path))
            return "real-process"

    module = FakeNemoCliModule()
    with skip_nemo_gym_policy_model_process(module):
        assert module.setup_env_command(".", {}, "policy_model") == "true"
        assert (
            module.setup_env_command(".", {}, "example_single_tool_call")
            == "real-setup"
        )
        proxy_process = module.run_command(
            "NEMO_GYM_CONFIG_PATH=policy_model python app.py",
            ".",
        )
        real_process = module.run_command(
            "NEMO_GYM_CONFIG_PATH=example_single_tool_call python app.py",
            ".",
        )

    assert proxy_process.poll() is None
    proxy_process.send_signal(2)
    assert proxy_process.poll() == 0
    assert real_process == "real-process"
    assert module.setup_calls == [(".", {}, "example_single_tool_call")]
    assert module.calls == [("NEMO_GYM_CONFIG_PATH=example_single_tool_call python app.py", ".")]


@pytest.mark.asyncio
async def test_nemo_gym_runner_uses_own_head_server_config_for_rollouts():
    head_server_config = object()
    collector = FakeRolloutCollector({"reward": 1.0})
    runner = PersistentNeMoGymRunner()
    runner._helper = FakeRunHelper()
    runner._rollout_collector = collector
    runner._proxy = NeMoGymModelProxy()
    runner._head_server_config = head_server_config

    result = await runner._run_once(
        {
            "agent_ref": {"type": "responses_api_agents", "name": "agent"},
            "responses_create_params": {"input": "hi"},
        },
        server_name=None,
        agent_name=None,
        endpoint_config={
            "base_url": "http://127.0.0.1:1/v1",
            "api_key": "key",
            "model": "real-model",
        },
    )

    assert result == {"reward": 1.0}
    assert collector.head_server_config is head_server_config
    assert collector.row["responses_create_params"]["model"].startswith(
        "verifiers-nemo-gym-proxy-"
    )


async def _proxy_response(
    session: ClientSession, proxy: NeMoGymModelProxy, routing_model: str
) -> dict[str, str]:
    response = await session.post(
        f"http://{proxy.host}:{proxy.port}/v1/responses",
        headers={"Authorization": f"Bearer {proxy.secret}"},
        json={"model": routing_model},
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
    ) -> None:
        self.label = label
        self.runner = runner
        self.site = site
        self.base_url = base_url
        self.authorizations = authorizations

    async def __aenter__(self) -> "UpstreamServer":
        return self

    async def __aexit__(self, *args: object) -> None:
        await self.runner.cleanup()


async def _start_upstream(label: str) -> UpstreamServer:
    authorizations: list[str] = []

    async def handle_response(request: web.Request) -> web.Response:
        authorizations.append(request.headers["Authorization"])
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
    )


class FakeRunHelper:
    def poll(self) -> None:
        return None


class FakeRolloutCollector:
    def __init__(self, result: dict[str, float]) -> None:
        self.result = result
        self.row: dict | None = None
        self.head_server_config: object | None = None

    def run_examples(
        self, rows: list[dict], *, head_server_config: object | None = None
    ):
        self.row = rows[0]
        self.head_server_config = head_server_config
        future = asyncio.get_running_loop().create_future()
        future.set_result((self.row, self.result))
        return iter([future])
