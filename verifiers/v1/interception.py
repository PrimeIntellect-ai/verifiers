from __future__ import annotations

import secrets
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import cast

from aiohttp import web
from pydantic import BaseModel, Field

from verifiers.types import Messages, Response, Tool

from .state import State
from .task import Task
from .types import JsonData, JsonValue, Context

StopCheck = Callable[[], Awaitable[str | None]]


@dataclass(frozen=True)
class ProtocolRoute:
    method: str
    path: str


class InterceptedRequest(BaseModel, extra="forbid"):
    protocol: str
    prompt: Messages
    model: str | None = None
    sampling_args: dict[str, JsonValue] = Field(default_factory=dict)
    tools: list[Tool] | None = None
    body: JsonData = Field(default_factory=dict)


class EndpointProtocol(ABC):
    name: str
    routes: Sequence[ProtocolRoute]

    def env(self, *, base_url: str, api_key: str, model: str) -> dict[str, str]:
        _ = base_url, api_key, model
        return {}

    @abstractmethod
    async def parse(
        self, request: web.Request, body: JsonData
    ) -> InterceptedRequest: ...

    @abstractmethod
    def serialize(
        self, response: Response, request: InterceptedRequest
    ) -> JsonData: ...

    def serialize_error(self, error: BaseException) -> tuple[int, JsonData]:
        return 502, {"error": str(error)}


class InterceptionServer:
    def __init__(
        self,
        context: Context,
        task: Task,
        state: State,
        *,
        protocols: Sequence[EndpointProtocol] | None = None,
        stop_check: StopCheck | None = None,
    ):
        if protocols is None:
            from .protocols import default_protocols

            protocols = default_protocols()
        self.context = context
        self.task = task
        self.state = state
        self.protocols = list(protocols)
        self.stop_check = stop_check
        self.secret = secrets.token_urlsafe(16)
        self.port = 0
        self.runner: web.AppRunner | None = None

    async def __aenter__(self) -> "InterceptionServer":
        app = web.Application()
        for protocol in self.protocols:
            for route in protocol.routes:
                app.router.add_route(route.method, route.path, self.handler(protocol))
        self.runner = web.AppRunner(app)
        await self.runner.setup()
        site = web.TCPSite(self.runner, "127.0.0.1", 0)
        await site.start()
        sockets = getattr(site, "_server").sockets
        self.port = int(sockets[0].getsockname()[1])
        return self

    async def __aexit__(self, *exc: object) -> None:
        if self.runner is not None:
            await self.runner.cleanup()

    def env(self, *, base_url: str, model: str) -> dict[str, str]:
        env: dict[str, str] = {}
        for protocol in self.protocols:
            env.update(
                protocol.env(base_url=base_url, api_key=self.secret, model=model)
            )
        return env

    def handler(self, protocol: EndpointProtocol):
        async def handle(request: web.Request) -> web.Response:
            if request.headers.get("Authorization") != f"Bearer {self.secret}":
                return web.json_response({"error": "unauthorized"}, status=401)
            try:
                body = await json_body(request)
                intercepted = await protocol.parse(request, body)
                stop_condition = await self.check_stop()
                if stop_condition is not None:
                    self.state.stop(stop_condition)
                    return web.json_response(
                        {"error": f"rollout stopped: {stop_condition}"},
                        status=400,
                    )
                response = await self.context.model_client.get_response(
                    prompt=intercepted.prompt,
                    model=intercepted.model or self.context.model,
                    sampling_args={
                        **self.context.sampling_args,
                        **intercepted.sampling_args,
                    },
                    tools=intercepted.tools,
                    state=self.state,
                )
                await self.state.add_response_turn(intercepted.prompt, response)
                return web.json_response(protocol.serialize(response, intercepted))
            except BaseException as exc:
                status, payload = protocol.serialize_error(exc)
                return web.json_response(payload, status=status)

        return handle

    async def check_stop(self) -> str | None:
        if self.stop_check is None:
            return None
        return await self.stop_check()


async def json_body(request: web.Request) -> JsonData:
    body = await request.json()
    if not isinstance(body, dict):
        raise TypeError("Protocol request body must be a JSON object.")
    return cast(JsonData, body)
