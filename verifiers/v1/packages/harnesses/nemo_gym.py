from __future__ import annotations

import asyncio
import contextlib
import inspect
import json
import logging
import secrets
from collections.abc import Mapping, Sequence
from copy import deepcopy
from typing import Any, Protocol, cast

from aiohttp import ClientSession, web
from pydantic import Field

from verifiers.types import AssistantMessage, ToolCall, ToolMessage
from verifiers.utils.serve_utils import get_free_port

from ...config import HarnessConfig
from ...harness import Harness
from ...state import State
from ...task import Task
from ..nemo_gym import first_nemo_gym_agent, resolve_nemo_gym_config_path

logger = logging.getLogger(__name__)

DEFAULT_NEMO_GYM_MODEL_CONFIG = (
    "responses_api_models/openai_model/configs/openai_model.yaml"
)
PROXY_MODEL_NAME = "verifiers-nemo-gym-proxy"
ROLLOUT_ID_HEADER = "x-verifiers-rollout-id"
FORWARD_REQUEST_HEADERS_CONFIG_KEY = "forward_request_headers"


class NeMoGymHarnessConfig(HarnessConfig):
    nemo_env: str | None = None
    config_name: str | None = None
    config_paths: list[str] = Field(default_factory=list)
    server_name: str | None = None
    agent_name: str | None = None
    include_openai_model_config: bool = True
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
    NeMo's OpenAI-compatible model server calls a stable local proxy, and that
    proxy routes each model request into the matching Verifiers rollout endpoint.
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
        include_openai_model_config: bool | None = None,
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
            include_openai_model_config=include_openai_model_config,
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
        if (
            self.config.include_openai_model_config
            and DEFAULT_NEMO_GYM_MODEL_CONFIG not in paths
        ):
            paths.append(DEFAULT_NEMO_GYM_MODEL_CONFIG)
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

            from nemo_gym.cli import RunHelper
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

        _require_nemo_gym_header_support(RolloutCollectionHelper, nemo_server_utils)
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

        helper = RunHelper()
        reset_nemo_gym_global_config(nemo_global_config)
        try:
            await asyncio.to_thread(helper.start, parser_config)
            if not is_global_aiohttp_client_setup():
                set_global_aiohttp_client(
                    GlobalAIOHTTPAsyncClientConfig.model_validate(
                        helper._server_client.global_config_dict
                    )
                )
        except Exception:
            with contextlib.suppress(Exception):
                await asyncio.to_thread(helper.shutdown)
            await close_nemo_gym_aiohttp_client(nemo_server_utils)
            reset_nemo_gym_global_config(nemo_global_config)
            raise

        self._helper = helper
        self._rollout_collector = RolloutCollectionHelper()
        self._config_key = key

    async def _ensure_proxy(self) -> "NeMoGymModelProxy":
        if self._proxy is None:
            self._proxy = NeMoGymModelProxy()
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
        ):
            raise RuntimeError("NeMo Gym runner has not been started.")

        self._helper.poll()
        rollout_id = secrets.token_urlsafe(16)
        async with self._proxy.activate(rollout_id, endpoint_config):
            request_row = prepare_nemo_gym_rollout_collection_row(
                row,
                server_name=server_name,
                agent_name=agent_name,
            )
            futures = self._rollout_collector.run_examples(
                [request_row],
                headers={ROLLOUT_ID_HEADER: rollout_id},
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
        try:
            from nemo_gym import global_config as nemo_global_config
            from nemo_gym import server_utils as nemo_server_utils
        except ImportError:
            nemo_global_config = None
            nemo_server_utils = None

        helper = self._helper
        self._helper = None
        self._rollout_collector = None
        self._config_key = None

        if helper is not None:
            try:
                await asyncio.to_thread(helper.shutdown)
            except Exception:
                logger.exception("Failed to shut down NeMo Gym runner")
        if nemo_server_utils is not None:
            await close_nemo_gym_aiohttp_client(nemo_server_utils)
        if nemo_global_config is not None:
            reset_nemo_gym_global_config(nemo_global_config)
        if self._proxy is not None:
            await self._proxy.stop()
            self._proxy = None


class NeMoGymModelProxy:
    """Stable OpenAI-compatible proxy used by a persistent NeMo Gym stack."""

    def __init__(self, *, host: str = "127.0.0.1", port: int | None = None) -> None:
        self.host = host
        self.port = get_free_port() if port is None else port
        self.secret = secrets.token_urlsafe(32)
        self._app: web.Application | None = None
        self._runner: web.AppRunner | None = None
        self._site: web.TCPSite | None = None
        self._session: ClientSession | None = None
        self._lock = asyncio.Lock()
        self._endpoints_by_rollout_id: dict[str, dict[str, str]] = {}

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
            self._endpoints_by_rollout_id.clear()
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
    async def activate(self, rollout_id: str, endpoint_config: Mapping[str, str]):
        async with self._lock:
            if rollout_id in self._endpoints_by_rollout_id:
                raise RuntimeError(
                    f"NeMo Gym model proxy already has rollout {rollout_id!r}."
                )
            self._endpoints_by_rollout_id[rollout_id] = {
                "base_url": str(endpoint_config["base_url"]),
                "api_key": str(endpoint_config["api_key"]),
                "model": str(endpoint_config["model"]),
            }
        try:
            yield
        finally:
            async with self._lock:
                self._endpoints_by_rollout_id.pop(rollout_id, None)

    async def _handle_models(self, request: web.Request) -> web.Response:
        if not self._authorized(request):
            return web.json_response({"error": "Unauthorized"}, status=401)
        endpoint = await self._endpoint_for_request(request)
        model = endpoint["model"] if endpoint is not None else PROXY_MODEL_NAME
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
                ],
            }
        )

    async def _handle_openai_request(self, request: web.Request) -> web.Response:
        if not self._authorized(request):
            return web.json_response({"error": "Unauthorized"}, status=401)
        endpoint = await self._endpoint_for_request(request)
        if endpoint is None:
            return web.json_response(
                {
                    "error": (
                        f"Missing or unknown {ROLLOUT_ID_HEADER} for NeMo Gym "
                        "model proxy request."
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
        json_payload = await self._request_json(request)
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
        self, request: web.Request
    ) -> dict[str, str] | None:
        rollout_id = request.headers.get(ROLLOUT_ID_HEADER)
        async with self._lock:
            if rollout_id:
                endpoint = self._endpoints_by_rollout_id.get(rollout_id)
                return dict(endpoint) if endpoint else None
            if len(self._endpoints_by_rollout_id) == 1:
                endpoint = next(iter(self._endpoints_by_rollout_id.values()))
                return dict(endpoint)
            return None

    def _authorized(self, request: web.Request) -> bool:
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
                ROLLOUT_ID_HEADER,
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
    _ensure_forward_request_header(config)
    config.setdefault("head_server", {"host": "127.0.0.1", "port": get_free_port()})
    return config


def _ensure_forward_request_header(
    config: dict[str, object],
    header: str = ROLLOUT_ID_HEADER,
) -> None:
    raw_headers = config.get(FORWARD_REQUEST_HEADERS_CONFIG_KEY)
    if raw_headers is None:
        config[FORWARD_REQUEST_HEADERS_CONFIG_KEY] = [header]
        return
    if isinstance(raw_headers, str):
        headers = [raw_headers]
    elif isinstance(raw_headers, Sequence):
        headers = list(raw_headers)
    else:
        raise TypeError(
            f"{FORWARD_REQUEST_HEADERS_CONFIG_KEY} must be a string or sequence."
        )

    normalized_headers = {str(value).strip().lower() for value in headers}
    if header.lower() not in normalized_headers:
        headers.append(header)
    config[FORWARD_REQUEST_HEADERS_CONFIG_KEY] = headers


def _require_nemo_gym_header_support(
    rollout_collection_helper_cls: object,
    server_utils_module: object,
) -> None:
    run_examples = getattr(rollout_collection_helper_cls, "run_examples", None)
    parameters = (
        inspect.signature(run_examples).parameters if callable(run_examples) else {}
    )
    supports_rollout_headers = "headers" in parameters
    supports_forwarding = hasattr(
        server_utils_module, "FORWARD_REQUEST_HEADERS_KEY_NAME"
    )
    if supports_rollout_headers and supports_forwarding:
        return
    raise RuntimeError(
        "NeMoGymHarness requires a nemo-gym build with request header forwarding. "
        "Install nemo-gym from a branch that supports "
        "RolloutCollectionHelper.run_examples(headers=...) and "
        "server_utils.forward_request_headers."
    )


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


def prepare_nemo_gym_request_row(
    row: Mapping[str, Any],
    agent_name: str | None,
) -> dict[str, Any]:
    prepared = deepcopy(dict(row))
    if agent_name and not _agent_ref_name(prepared.get("agent_ref")):
        prepared["agent_ref"] = {
            "type": "responses_api_agents",
            "name": agent_name,
        }
    if not _agent_ref_name(prepared.get("agent_ref")):
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


def nemo_gym_server_name(row: Mapping[str, Any], server_name: str | None) -> str:
    if server_name:
        return server_name
    name = _agent_ref_name(row.get("agent_ref"))
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


def _agent_ref_name(value: object) -> str | None:
    if not isinstance(value, Mapping):
        return None
    name = value.get("name")
    return name if isinstance(name, str) and name else None


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
