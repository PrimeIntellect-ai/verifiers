from __future__ import annotations

import asyncio
import logging
import os
import time
import uuid
from collections.abc import Mapping
from typing import cast

from prime_tunnel import Tunnel

from verifiers.errors import TunnelError
from verifiers.types import ClientType
from verifiers.types import State
from verifiers.utils.serve_utils import get_free_port

from verifiers.envs.experimental.channels.channel import (
    Channel,
    ChannelConfig,
    ChannelContext,
    single_config,
)


class Endpoint:
    """Environment-scoped LLM endpoint server and optional tunnel."""

    TUNNEL_CHECK_INTERVAL = 60.0

    def __init__(
        self,
        *,
        port: int | None = None,
        url: str | None = None,
        secret: str | None = None,
        api_client_type: ClientType = "openai_chat_completions",
        use_tunnel: bool = False,
        logger: logging.Logger | None = None,
    ):
        from verifiers.utils.interception_utils import InterceptionServer

        self.port = get_free_port() if port is None else port
        self.url = url
        self.secret = secret or os.environ.get("ENDPOINT_SECRET")
        self.api_client_type = api_client_type
        self.use_tunnel = use_tunnel
        self.logger = logger or logging.getLogger(__name__)
        self.server = InterceptionServer(self.port, secret=self.secret)
        self._tunnel: Tunnel | None = None
        self._tunnel_lock = asyncio.Lock()
        self._tunnel_last_checked = 0.0
        self._request_queues: dict[str, asyncio.Queue] = {}

    async def start(self) -> None:
        if self.api_client_type != "openai_chat_completions":
            raise NotImplementedError(
                "Endpoint currently exposes an OpenAI chat completions endpoint."
            )
        await self.server.start()

    async def register_rollout(self, state: State) -> str:
        await self.start()
        request_key = f"rollout_{uuid.uuid4().hex[:8]}"
        request_queue = self.server.register_rollout(request_key, state=state)
        self._request_queues[request_key] = request_queue
        state["endpoint_request_key"] = request_key
        state["endpoint_base_url"] = f"{await self.url_base()}/rollout/{request_key}/v1"
        return state["endpoint_base_url"]

    def unregister_rollout(self, request_key: str) -> None:
        self._request_queues.pop(request_key, None)
        self.server.unregister_rollout(request_key)

    def request_queue(self, request_key: str) -> asyncio.Queue:
        return self._request_queues[request_key]

    def get_request(self, request_id: str) -> dict[str, object]:
        return self.server.intercepts[request_id]

    async def url_base(self) -> str:
        if self.url is not None:
            return self.url.rstrip("/")
        if self.use_tunnel:
            return await self.get_tunnel_url()
        return f"http://127.0.0.1:{self.port}"

    async def get_tunnel_url(self) -> str:
        async with self._tunnel_lock:
            if self._tunnel is not None and not self._tunnel.is_running:
                frpc_output = "\n".join(self._tunnel.recent_output)
                self.logger.warning(
                    f"Tunnel process died, recreating. frpc output:\n{frpc_output}"
                )
                self._tunnel.sync_stop()
                self._tunnel = None

            if self._tunnel is not None:
                now = time.time()
                if now - self._tunnel_last_checked > self.TUNNEL_CHECK_INTERVAL:
                    self._tunnel_last_checked = now
                    try:
                        registered = await self._tunnel.check_registered()
                        if not registered:
                            self.logger.warning(
                                "Tunnel registration expired server-side, recreating."
                            )
                            self._tunnel.sync_stop()
                            self._tunnel = None
                    except Exception as e:
                        self.logger.warning(f"Tunnel health check failed: {e}")

            if self._tunnel is None:
                self._tunnel = Tunnel(local_port=self.port)
                url = await self._tunnel.start()
                self._tunnel_last_checked = time.time()
                return url

            if self._tunnel.url is None:
                raise TunnelError("Tunnel started but URL is unavailable.")
            return self._tunnel.url

    async def check_tunnel(self) -> None:
        if self._tunnel is not None and not self._tunnel.is_running:
            frpc_output = "\n".join(self._tunnel.recent_output)
            raise TunnelError(
                f"Tunnel process died during rollout. frpc output:\n{frpc_output}"
            )

    async def teardown(self) -> None:
        async with self._tunnel_lock:
            if self._tunnel is not None:
                self._tunnel.sync_stop()
                self._tunnel = None
        await self.server.stop()


def resolve_endpoint(
    configs: list[ChannelConfig], context: ChannelContext
) -> dict[str, object]:
    config = single_config("endpoint", configs)
    if config is None:
        return {}
    if config is True:
        config = {}
    if not isinstance(config, Mapping):
        raise TypeError("The endpoint channel expects a mapping config.")
    if context.phase != "env":
        return {}
    endpoint = Endpoint(
        port=optional_int(config.get("port")),
        url=optional_str(config.get("url")),
        secret=optional_str(config.get("secret")),
        api_client_type=endpoint_api_client_type(config.get("api_client_type")),
        use_tunnel=bool(config.get("use_tunnel", False)),
        logger=channel_logger(context),
    )
    return {
        "endpoint": endpoint,
        "teardown_handlers": [endpoint.teardown],
    }


def optional_int(value: object) -> int | None:
    if value is None:
        return None
    return int(value)


def optional_str(value: object) -> str | None:
    if value is None:
        return None
    return str(value)


def endpoint_api_client_type(value: object) -> ClientType:
    if value is None:
        return "openai_chat_completions"
    if value not in {
        "openai_chat_completions",
        "anthropic_messages",
        "openai_completions",
    }:
        raise ValueError(f"Unsupported endpoint API client type: {value!r}")
    return cast(ClientType, value)


def channel_logger(context: ChannelContext) -> logging.Logger:
    for owner in context.owners:
        logger = getattr(owner, "logger", None)
        if isinstance(logger, logging.Logger):
            return logger
    return logging.getLogger(__name__)


endpoint_channel = Channel(
    name="endpoint",
    outputs=("endpoint",),
    resolve_fn=resolve_endpoint,
)
