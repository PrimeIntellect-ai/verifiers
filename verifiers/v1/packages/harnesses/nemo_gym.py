from __future__ import annotations

import asyncio
import contextlib
import inspect
import json
import logging
import os
import secrets
from collections.abc import Mapping, Sequence
from copy import deepcopy
from typing import Any, Protocol, cast
from urllib.parse import urlparse

from aiohttp import ClientSession, web
from pydantic import Field

from verifiers.types import AssistantMessage, ToolCall, ToolMessage
from verifiers.utils.serve_utils import get_free_port

from ...config import HarnessConfig
from ...harness import Harness
from ...state import State
from ...task import Task
from ..nemo_gym import (
    agent_ref_name,
    first_nemo_gym_agent,
    resolve_nemo_gym_config_path,
)

logger = logging.getLogger(__name__)

NEMO_GYM_POLICY_MODEL_SERVER_NAME = "policy_model"
NEMO_GYM_POLICY_MODEL_TYPE_NAME = "verifiers_proxy"
NEMO_GYM_EXTERNAL_POLICY_MODEL_ENTRYPOINT = "__verifiers_external_policy_model__.py"
PROXY_MODEL_NAME = "verifiers-nemo-gym-proxy"
_NEMO_GYM_GLOBALS_LOCK = asyncio.Lock()
_NEMO_GYM_ACTIVE_RUNNERS = 0
_NEMO_GYM_OWNS_AIOHTTP_CLIENT = False
_RAY_ENABLE_UV_RUN_RUNTIME_ENV = "RAY_ENABLE_UV_RUN_RUNTIME_ENV"
_NEMO_GYM_CONFIG_PATH_ENV_VAR_NAME = "NEMO_GYM_CONFIG_PATH"


class NeMoGymHarnessConfig(HarnessConfig):
    nemo_env: str | None = None
    config_name: str | None = None
    config_paths: list[str] = Field(default_factory=list)
    server_name: str | None = None
    agent_name: str | None = None
    timeout_seconds: float | None = None
    global_config: dict[str, object] = Field(default_factory=dict)


class NeMoGymRunner(Protocol):
    async def run(
        self,
        row: Mapping[str, Any],
        *,
        config_paths: Sequence[str],
        server_name: str | None,
        agent_name: str | None,
        endpoint_config: Mapping[str, str],
        timeout_seconds: float | None,
        global_config: Mapping[str, object],
    ) -> Mapping[str, Any]: ...


class NeMoGymHarness(Harness):
    """Run a NeMo Gym row from a Verifiers rollout.

    The default runner keeps one NeMo Gym server stack alive per harness instance.
    NeMo Gym agents call a stable local Verifiers proxy registered as their
    policy model server, and that proxy routes each model request into the
    matching Verifiers rollout endpoint.
    """

    config_type = NeMoGymHarnessConfig

    def __init__(
        self,
        *,
        nemo_env: str | None = None,
        config_name: str | None = None,
        config_paths: Sequence[str] | None = None,
        server_name: str | None = None,
        agent_name: str | None = None,
        timeout_seconds: float | None = None,
        global_config: Mapping[str, object] | None = None,
        runner: NeMoGymRunner | None = None,
        config: NeMoGymHarnessConfig | Mapping[str, object] | None = None,
        **kwargs: Any,
    ):
        harness_config = type(self).config_type.from_config(
            config,
            nemo_env=nemo_env,
            config_name=config_name,
            config_paths=list(config_paths) if config_paths is not None else None,
            server_name=server_name,
            agent_name=agent_name,
            timeout_seconds=timeout_seconds,
            global_config=dict(global_config or {})
            if global_config is not None
            else None,
        )
        configure_nemo_gym_harness_config(harness_config)
        self.runner = runner or PersistentNeMoGymRunner()
        super().__init__(
            program=self._run_nemo_gym,
            config=harness_config,
            **kwargs,
        )

    async def teardown(self) -> None:
        teardown = getattr(self.runner, "teardown", None)
        if callable(teardown):
            result = teardown()
            if inspect.isawaitable(result):
                await result
        await super().teardown()

    async def _run_nemo_gym(self, task: Task, state: State) -> State:
        endpoint_config = state.get_endpoint_config(api="responses")
        row = nemo_gym_row_from_task(task, self.config.agent_name)
        result = await self.runner.run(
            row,
            config_paths=self._config_paths(),
            server_name=self.config.server_name,
            agent_name=self.config.agent_name,
            endpoint_config=endpoint_config,
            timeout_seconds=self.config.timeout_seconds,
            global_config=self.config.global_config,
        )
        apply_nemo_gym_result(state, result)
        return state

    def _config_paths(self) -> list[str]:
        paths = list(self.config.config_paths)
        if not paths:
            raise ValueError("NeMoGymHarness requires at least one config path.")
        return paths


def configure_nemo_gym_harness_config(config: NeMoGymHarnessConfig) -> None:
    if config.nemo_env and not config.config_paths:
        config.config_paths = [
            str(resolve_nemo_gym_config_path(config.nemo_env, config.config_name))
        ]

    if config.server_name is not None and config.agent_name is not None:
        return
    inferred = first_nemo_gym_agent(tuple(config.config_paths))
    if inferred is None:
        return
    inferred_server_name, inferred_agent_name = inferred
    if config.server_name is None:
        config.server_name = inferred_server_name
    if config.agent_name is None:
        config.agent_name = inferred_agent_name


class PersistentNeMoGymRunner:
    """Run one NeMo Gym server stack and route model calls per rollout."""

    def __init__(self) -> None:
        self._lifecycle_lock = asyncio.Lock()
        self._helper: Any | None = None
        self._rollout_collector: Any | None = None
        self._proxy: NeMoGymModelProxy | None = None
        self._config_key: str | None = None
        self._head_server_config: Any | None = None

    async def run(
        self,
        row: Mapping[str, Any],
        *,
        config_paths: Sequence[str],
        server_name: str | None,
        agent_name: str | None,
        endpoint_config: Mapping[str, str],
        timeout_seconds: float | None,
        global_config: Mapping[str, object],
    ) -> Mapping[str, Any]:
        async with self._lifecycle_lock:
            await self._ensure_started(
                config_paths=config_paths,
                global_config=global_config,
            )

        run_once = self._run_once(
            row,
            server_name=server_name,
            agent_name=agent_name,
            endpoint_config=endpoint_config,
        )
        if timeout_seconds is None:
            return await run_once
        return await asyncio.wait_for(run_once, timeout=timeout_seconds)

    async def _ensure_started(
        self, *, config_paths: Sequence[str], global_config: Mapping[str, object]
    ) -> None:
        global _NEMO_GYM_ACTIVE_RUNNERS, _NEMO_GYM_OWNS_AIOHTTP_CLIENT

        key = json.dumps(
            {
                "config_paths": list(config_paths),
                "global_config": jsonable(dict(global_config)),
            },
            sort_keys=True,
        )
        if self._helper is not None and self._config_key == key:
            return
        if self._helper is not None:
            await self.teardown()

        try:
            from omegaconf import OmegaConf

            from nemo_gym import cli as nemo_cli
            from nemo_gym.global_config import GlobalConfigDictParserConfig
            from nemo_gym import global_config as nemo_global_config
            from nemo_gym import server_utils as nemo_server_utils
            from nemo_gym.rollout_collection import RolloutCollectionHelper
            from nemo_gym.server_utils import (
                GlobalAIOHTTPAsyncClientConfig,
                is_global_aiohttp_client_setup,
                set_global_aiohttp_client,
            )
        except ImportError as exc:
            raise ImportError(
                "NeMoGymHarness requires nemo-gym. Install with: uv add nemo-gym"
            ) from exc

        proxy = await self._ensure_proxy()
        config = build_nemo_gym_global_config(
            config_paths=config_paths,
            endpoint_config=proxy.endpoint_config(),
            global_config=global_config,
        )
        parser_config = GlobalConfigDictParserConfig(
            initial_global_config_dict=OmegaConf.create(config),
            skip_load_from_cli=True,
            skip_load_from_dotenv=True,
        )

        helper = nemo_cli.RunHelper()
        async with _NEMO_GYM_GLOBALS_LOCK:
            reset_nemo_gym_global_config(nemo_global_config)
            try:
                with disable_ray_uv_run_runtime_env():
                    with skip_nemo_gym_policy_model_process(nemo_cli):
                        await asyncio.to_thread(helper.start, parser_config)
                if not is_global_aiohttp_client_setup():
                    set_global_aiohttp_client(
                        GlobalAIOHTTPAsyncClientConfig.model_validate(
                            helper._server_client.global_config_dict
                        )
                    )
                    _NEMO_GYM_OWNS_AIOHTTP_CLIENT = True
                _NEMO_GYM_ACTIVE_RUNNERS += 1
            except Exception:
                with contextlib.suppress(Exception):
                    await asyncio.to_thread(helper.shutdown)
                if _NEMO_GYM_ACTIVE_RUNNERS == 0 and _NEMO_GYM_OWNS_AIOHTTP_CLIENT:
                    await close_nemo_gym_aiohttp_client(nemo_server_utils)
                    _NEMO_GYM_OWNS_AIOHTTP_CLIENT = False
                reset_nemo_gym_global_config(nemo_global_config)
                raise

        self._helper = helper
        self._rollout_collector = RolloutCollectionHelper()
        self._config_key = key
        self._head_server_config = helper._server_client.head_server_config

    async def _ensure_proxy(self) -> "NeMoGymModelProxy":
        if self._proxy is None:
            self._proxy = NeMoGymModelProxy(require_auth=False)
        await self._proxy.start()
        return self._proxy

    async def _run_once(
        self,
        row: Mapping[str, Any],
        *,
        server_name: str | None,
        agent_name: str | None,
        endpoint_config: Mapping[str, str],
    ) -> Mapping[str, Any]:
        if (
            self._helper is None
            or self._rollout_collector is None
            or self._proxy is None
            or self._head_server_config is None
        ):
            raise RuntimeError("NeMo Gym runner has not been started.")

        self._helper.poll()
        rollout_id = secrets.token_urlsafe(16)
        routing_model = nemo_gym_proxy_model_name(rollout_id)
        async with self._proxy.activate(routing_model, endpoint_config):
            request_row = prepare_nemo_gym_rollout_collection_row(
                row,
                server_name=server_name,
                agent_name=agent_name,
            )
            set_nemo_gym_proxy_model(request_row, routing_model)
            futures = self._rollout_collector.run_examples(
                [request_row],
                head_server_config=self._head_server_config,
            )
            try:
                future = next(futures)
            except StopIteration as exc:
                raise RuntimeError(
                    "NeMo Gym rollout collector returned no tasks."
                ) from exc
            _, result = await future
            return cast(Mapping[str, Any], result)

    async def teardown(self) -> None:
        global _NEMO_GYM_ACTIVE_RUNNERS, _NEMO_GYM_OWNS_AIOHTTP_CLIENT

        helper = self._helper
        proxy = self._proxy
        self._helper = None
        self._rollout_collector = None
        self._proxy = None
        self._config_key = None
        self._head_server_config = None

        if helper is None:
            if proxy is not None:
                await proxy.stop()
            return

        try:
            from nemo_gym import global_config as nemo_global_config
            from nemo_gym import server_utils as nemo_server_utils
        except ImportError:
            nemo_global_config = None
            nemo_server_utils = None

        try:
            await asyncio.to_thread(helper.shutdown)
        except Exception:
            logger.exception("Failed to shut down NeMo Gym runner")
        if nemo_global_config is not None:
            async with _NEMO_GYM_GLOBALS_LOCK:
                _NEMO_GYM_ACTIVE_RUNNERS = max(0, _NEMO_GYM_ACTIVE_RUNNERS - 1)
                if _NEMO_GYM_ACTIVE_RUNNERS == 0:
                    if nemo_server_utils is not None and _NEMO_GYM_OWNS_AIOHTTP_CLIENT:
                        await close_nemo_gym_aiohttp_client(nemo_server_utils)
                        _NEMO_GYM_OWNS_AIOHTTP_CLIENT = False
                    reset_nemo_gym_global_config(nemo_global_config)
        if proxy is not None:
            await proxy.stop()


class NeMoGymModelProxy:
    """Stable OpenAI-compatible proxy used by a persistent NeMo Gym stack."""

    def __init__(
        self,
        *,
        host: str = "127.0.0.1",
        port: int | None = None,
        require_auth: bool = True,
    ) -> None:
        self.host = host
        self.port = get_free_port() if port is None else port
        self.require_auth = require_auth
        self.secret = secrets.token_urlsafe(32)
        self._app: web.Application | None = None
        self._runner: web.AppRunner | None = None
        self._site: web.TCPSite | None = None
        self._session: ClientSession | None = None
        self._lock = asyncio.Lock()
        self._endpoints_by_model: dict[str, dict[str, str]] = {}

    async def start(self) -> None:
        async with self._lock:
            if self._runner is not None:
                return
            app = web.Application()
            app.router.add_get("/health", lambda _: web.json_response({"status": "ok"}))
            app.router.add_get("/v1/models", self._handle_models)
            app.router.add_route("*", "/v1/{tail:.*}", self._handle_openai_request)
            runner = web.AppRunner(app)
            await runner.setup()
            site = web.TCPSite(runner, self.host, self.port)
            await site.start()
            self._app = app
            self._runner = runner
            self._site = site
            self._session = ClientSession()

    async def stop(self) -> None:
        async with self._lock:
            session = self._session
            runner = self._runner
            self._session = None
            self._runner = None
            self._site = None
            self._app = None
            self._endpoints_by_model.clear()
        if session is not None:
            await session.close()
        if runner is not None:
            await runner.cleanup()

    def endpoint_config(self) -> dict[str, str]:
        return {
            "base_url": f"http://{self.host}:{self.port}/v1",
            "api_base": f"http://{self.host}:{self.port}/v1",
            "api_key": self.secret,
            "model": PROXY_MODEL_NAME,
            "api_client_type": "openai_responses",
        }

    @contextlib.asynccontextmanager
    async def activate(self, routing_model: str, endpoint_config: Mapping[str, str]):
        async with self._lock:
            if routing_model in self._endpoints_by_model:
                raise RuntimeError(
                    f"NeMo Gym model proxy already has model route {routing_model!r}."
                )
            self._endpoints_by_model[routing_model] = {
                "base_url": str(endpoint_config["base_url"]),
                "api_key": str(endpoint_config["api_key"]),
                "model": str(endpoint_config["model"]),
            }
        try:
            yield
        finally:
            async with self._lock:
                self._endpoints_by_model.pop(routing_model, None)

    async def _handle_models(self, request: web.Request) -> web.Response:
        if not self._authorized(request):
            return web.json_response({"error": "Unauthorized"}, status=401)
        async with self._lock:
            models = list(self._endpoints_by_model) or [PROXY_MODEL_NAME]
        return web.json_response(
            {
                "object": "list",
                "data": [
                    {
                        "id": model,
                        "object": "model",
                        "created": 0,
                        "owned_by": "verifiers",
                    }
                    for model in models
                ],
            }
        )

    async def _handle_openai_request(self, request: web.Request) -> web.Response:
        if not self._authorized(request):
            return web.json_response({"error": "Unauthorized"}, status=401)
        json_payload = await self._request_json(request)
        endpoint = await self._endpoint_for_request(json_payload)
        if endpoint is None:
            return web.json_response(
                {
                    "error": (
                        "Missing or unknown model for NeMo Gym model proxy request."
                    )
                },
                status=409,
            )
        session = self._session
        if session is None:
            return web.json_response({"error": "Proxy not started"}, status=503)

        suffix = request.path.removeprefix("/v1")
        upstream_url = f"{endpoint['base_url'].rstrip('/')}{suffix}"
        if request.query_string:
            upstream_url = f"{upstream_url}?{request.query_string}"

        headers = self._upstream_headers(request, endpoint["api_key"])
        if json_payload is not None:
            json_payload["model"] = endpoint["model"]
            request_kwargs: dict[str, Any] = {"json": json_payload}
        else:
            request_kwargs = {"data": await request.read()}

        async with session.request(
            request.method,
            upstream_url,
            headers=headers,
            **request_kwargs,
        ) as response:
            body = await response.read()
            return web.Response(
                status=response.status,
                body=body,
                headers=self._response_headers(response.headers),
            )

    async def _endpoint_for_request(
        self, json_payload: Mapping[str, Any] | None
    ) -> dict[str, str] | None:
        routing_model = (
            json_payload.get("model") if isinstance(json_payload, Mapping) else None
        )
        async with self._lock:
            if isinstance(routing_model, str):
                endpoint = self._endpoints_by_model.get(routing_model)
                if endpoint:
                    return dict(endpoint)
            if len(self._endpoints_by_model) == 1:
                endpoint = next(iter(self._endpoints_by_model.values()))
                return dict(endpoint)
            return None

    def _authorized(self, request: web.Request) -> bool:
        if not self.require_auth:
            return True
        auth = request.headers.get("Authorization", "")
        api_key = request.headers.get("x-api-key", "")
        return auth == f"Bearer {self.secret}" or api_key == self.secret

    def _upstream_headers(self, request: web.Request, api_key: str) -> dict[str, str]:
        headers = {
            key: value
            for key, value in request.headers.items()
            if key.lower()
            not in {
                "authorization",
                "content-length",
                "host",
                "x-api-key",
            }
        }
        headers["Authorization"] = f"Bearer {api_key}"
        return headers

    async def _request_json(self, request: web.Request) -> dict[str, Any] | None:
        content_type = request.headers.get("content-type", "")
        if "json" not in content_type:
            return None
        payload = await request.json()
        if not isinstance(payload, dict):
            raise TypeError("OpenAI-compatible proxy requests must be JSON objects.")
        return cast(dict[str, Any], payload)

    def _response_headers(self, headers: Mapping[str, str]) -> dict[str, str]:
        return {
            key: value
            for key, value in headers.items()
            if key.lower()
            not in {
                "content-length",
                "content-encoding",
                "connection",
                "keep-alive",
                "transfer-encoding",
            }
        }


def build_nemo_gym_global_config(
    *,
    config_paths: Sequence[str],
    endpoint_config: Mapping[str, str],
    global_config: Mapping[str, object],
) -> dict[str, object]:
    config = dict(global_config)
    config["config_paths"] = list(config_paths)
    config["policy_base_url"] = str(endpoint_config["base_url"])
    config["policy_api_key"] = str(endpoint_config["api_key"])
    config["policy_model_name"] = str(endpoint_config["model"])
    config[NEMO_GYM_POLICY_MODEL_SERVER_NAME] = build_nemo_gym_policy_model_config(
        endpoint_config
    )
    config.setdefault("head_server", {"host": "127.0.0.1", "port": get_free_port()})
    return config


def build_nemo_gym_policy_model_config(
    endpoint_config: Mapping[str, str],
) -> dict[str, object]:
    parsed_url = urlparse(str(endpoint_config["base_url"]))
    if not parsed_url.hostname or parsed_url.port is None:
        raise ValueError(
            "NeMo Gym Verifiers proxy base_url must include an explicit host and port."
        )
    return {
        "responses_api_models": {
            NEMO_GYM_POLICY_MODEL_TYPE_NAME: {
                "entrypoint": NEMO_GYM_EXTERNAL_POLICY_MODEL_ENTRYPOINT,
                "host": parsed_url.hostname,
                "port": parsed_url.port,
            }
        }
    }


def nemo_gym_proxy_model_name(rollout_id: str) -> str:
    return f"{PROXY_MODEL_NAME}-{rollout_id}"


class _NoopPolicyModelProcess:
    pid = 0

    def __init__(self) -> None:
        self._running = True

    def poll(self) -> int | None:
        return None if self._running else 0

    def communicate(self):
        self._running = False
        return b"", b""

    def send_signal(self, signal: int) -> None:
        self._running = False

    def wait(self, timeout: float | None = None) -> int:
        self._running = False
        return 0

    def kill(self) -> None:
        self._running = False


@contextlib.contextmanager
def skip_nemo_gym_policy_model_process(nemo_cli_module: object):
    original_run_command = getattr(nemo_cli_module, "run_command")
    original_setup_env_command = getattr(nemo_cli_module, "setup_env_command")

    def setup_env_command(
        dir_path: object, global_config_dict: object, prefix: str
    ) -> str:
        if prefix == NEMO_GYM_POLICY_MODEL_SERVER_NAME:
            return "true"
        return original_setup_env_command(dir_path, global_config_dict, prefix)

    def run_command(command: str, working_dir_path: object):
        if (
            f"{_NEMO_GYM_CONFIG_PATH_ENV_VAR_NAME}="
            f"{NEMO_GYM_POLICY_MODEL_SERVER_NAME}"
        ) in command:
            return _NoopPolicyModelProcess()
        return original_run_command(command, working_dir_path)

    setattr(nemo_cli_module, "setup_env_command", setup_env_command)
    setattr(nemo_cli_module, "run_command", run_command)
    try:
        yield
    finally:
        setattr(nemo_cli_module, "run_command", original_run_command)
        setattr(nemo_cli_module, "setup_env_command", original_setup_env_command)


def reset_nemo_gym_global_config(module: object) -> None:
    if hasattr(module, "_GLOBAL_CONFIG_DICT"):
        setattr(module, "_GLOBAL_CONFIG_DICT", None)


async def close_nemo_gym_aiohttp_client(module: object) -> None:
    client = getattr(module, "_GLOBAL_AIOHTTP_CLIENT", None)
    if client is None:
        return
    close = getattr(client, "close", None)
    if callable(close):
        result = close()
        if inspect.isawaitable(result):
            await result
    setattr(module, "_GLOBAL_AIOHTTP_CLIENT", None)


@contextlib.contextmanager
def disable_ray_uv_run_runtime_env():
    original_env = os.environ.get(_RAY_ENABLE_UV_RUN_RUNTIME_ENV)
    os.environ[_RAY_ENABLE_UV_RUN_RUNTIME_ENV] = "0"
    ray_constants: object | None = None
    original_constant: object | None = None
    try:
        import ray._private.ray_constants as ray_constants

        original_constant = getattr(ray_constants, _RAY_ENABLE_UV_RUN_RUNTIME_ENV)
        setattr(ray_constants, _RAY_ENABLE_UV_RUN_RUNTIME_ENV, False)
    except (AttributeError, ImportError):
        pass
    try:
        yield
    finally:
        if original_env is None:
            os.environ.pop(_RAY_ENABLE_UV_RUN_RUNTIME_ENV, None)
        else:
            os.environ[_RAY_ENABLE_UV_RUN_RUNTIME_ENV] = original_env
        if ray_constants is not None and original_constant is not None:
            setattr(ray_constants, _RAY_ENABLE_UV_RUN_RUNTIME_ENV, original_constant)


def prepare_nemo_gym_request_row(
    row: Mapping[str, Any],
    agent_name: str | None,
) -> dict[str, Any]:
    prepared = deepcopy(dict(row))
    if agent_name and not agent_ref_name(prepared.get("agent_ref")):
        prepared["agent_ref"] = {
            "type": "responses_api_agents",
            "name": agent_name,
        }
    if not agent_ref_name(prepared.get("agent_ref")):
        raise ValueError(
            "NeMo Gym row has no agent_ref.name; pass NeMoGymHarness(agent_name=...) "
            "or include agent_ref in each row."
        )
    return prepared


def prepare_nemo_gym_rollout_collection_row(
    row: Mapping[str, Any],
    *,
    server_name: str | None,
    agent_name: str | None,
) -> dict[str, Any]:
    prepared = prepare_nemo_gym_request_row(row, agent_name)
    target_server_name = nemo_gym_server_name(prepared, server_name)
    agent_ref = dict(cast(Mapping[str, Any], prepared["agent_ref"]))
    agent_ref["name"] = target_server_name
    prepared["agent_ref"] = agent_ref
    return prepared


def set_nemo_gym_proxy_model(row: dict[str, Any], routing_model: str) -> None:
    create_params = row.get("responses_create_params")
    if not isinstance(create_params, Mapping):
        raise ValueError("NeMo Gym row requires responses_create_params.")
    row["responses_create_params"] = dict(create_params)
    row["responses_create_params"]["model"] = routing_model


def nemo_gym_server_name(row: Mapping[str, Any], server_name: str | None) -> str:
    if server_name:
        return server_name
    name = agent_ref_name(row.get("agent_ref"))
    if name is None:
        raise ValueError(
            "NeMo Gym row has no agent_ref.name; pass "
            "NeMoGymHarness(server_name=...) or include agent_ref in each row."
        )
    return name


def nemo_gym_row_from_task(
    task: Mapping[str, Any], agent_name: str | None
) -> dict[str, Any]:
    raw_row = task.get("nemo_gym_row")
    if isinstance(raw_row, Mapping):
        return prepare_nemo_gym_request_row(
            cast(Mapping[str, Any], raw_row), agent_name
        )
    if "responses_create_params" not in task:
        raise ValueError(
            "NeMoGymHarness tasks must contain nemo_gym_row or responses_create_params."
        )
    ignored = {
        "prompt",
        "info",
        "example_id",
        "task_id",
        "taskset_id",
        "runtime",
    }
    row = {key: deepcopy(value) for key, value in task.items() if key not in ignored}
    return prepare_nemo_gym_request_row(row, agent_name)


def apply_nemo_gym_result(state: State, result: Mapping[str, Any]) -> None:
    result_dict = jsonable_mapping(result)
    state["nemo_gym_result"] = result_dict
    response = result_dict.get("response")
    if isinstance(response, Mapping):
        completion = messages_from_nemo_gym_response(response)
        if completion:
            state["completion"] = completion
    reward = result_dict.get("reward")
    if reward is not None:
        state["reward"] = float(reward)
    metrics = state.setdefault("metrics", {})
    if isinstance(metrics, dict):
        for key, value in result_dict.items():
            if key in {"responses_create_params", "response", "reward"}:
                continue
            if isinstance(value, bool):
                metrics[key] = float(value)
            elif isinstance(value, int | float):
                metrics[key] = float(value)
    state.stop("nemo_gym_completed")


def messages_from_nemo_gym_response(
    response: Mapping[str, Any],
) -> list[dict[str, Any]]:
    output = response.get("output")
    if not isinstance(output, list):
        return []
    messages: list[dict[str, Any]] = []
    for item in output:
        if not isinstance(item, Mapping):
            continue
        item_type = item.get("type")
        if item_type == "function_call":
            call_id = item.get("call_id") or item.get("id")
            name = item.get("name")
            arguments = item.get("arguments")
            if isinstance(call_id, str) and isinstance(name, str):
                messages.append(
                    AssistantMessage(
                        content=None,
                        tool_calls=[
                            ToolCall(
                                id=call_id,
                                name=name,
                                arguments=arguments
                                if isinstance(arguments, str)
                                else "{}",
                            )
                        ],
                    ).model_dump(exclude_none=True)
                )
            continue
        if item_type == "function_call_output":
            call_id = item.get("call_id")
            if isinstance(call_id, str):
                messages.append(
                    ToolMessage(
                        tool_call_id=call_id,
                        content=nemo_output_text(item.get("output")),
                    ).model_dump(exclude_none=True)
                )
            continue
        if item_type == "message":
            messages.append(
                AssistantMessage(
                    content=nemo_content_text(item.get("content")),
                ).model_dump(exclude_none=True)
            )
    return messages


def nemo_content_text(content: object) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, Mapping) and isinstance(part.get("text"), str):
                parts.append(str(part["text"]))
        return "\n".join(parts)
    return "" if content is None else str(content)


def nemo_output_text(output: object) -> str:
    if isinstance(output, str):
        return output
    return json.dumps(jsonable(output), sort_keys=True)


def jsonable_mapping(value: Mapping[str, Any]) -> dict[str, Any]:
    return cast(dict[str, Any], jsonable(dict(value)))


def jsonable(value: Any) -> Any:
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        return jsonable(model_dump(exclude_none=True))
    if isinstance(value, Mapping):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [jsonable(item) for item in value]
    return json.loads(json.dumps(value))
