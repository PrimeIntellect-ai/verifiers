from __future__ import annotations

import contextlib
import contextvars
import functools
import inspect
import logging
import os
from typing import TYPE_CHECKING, Callable, ClassVar, Generic, TypeVar

from pydantic import TypeAdapter, ValidationError
from pydantic_config import BaseConfig

from verifiers.v1.state import State, StateT, state_cls
from verifiers.v1.utils.generic import generic_type

if TYPE_CHECKING:
    from httpx import AsyncClient, Response
    from mcp.server.fastmcp import FastMCP

ConfigT = TypeVar("ConfigT", bound=BaseConfig)
logger = logging.getLogger(__name__)

# State calls may cross a tunnel, so allow transient startup failures without hanging forever.
STATE_TIMEOUT = 30.0  # seconds per request
STATE_RETRIES = 4


async def _channel_request(
    method: str,
    url: str,
    secret: str,
    *,
    content: bytes | None = None,
    client: AsyncClient | None = None,
) -> Response:
    """Retry tunnel transport errors and 5xx responses, but not invalid 4xx requests."""
    import httpx
    from tenacity import (
        AsyncRetrying,
        retry_if_exception,
        stop_after_attempt,
        wait_exponential_jitter,
    )

    def transient(e: BaseException) -> bool:
        return isinstance(e, httpx.TransportError) or (
            isinstance(e, httpx.HTTPStatusError) and e.response.status_code >= 500
        )

    async def request() -> Response:
        manager = (
            contextlib.nullcontext(client)
            if client is not None
            else httpx.AsyncClient(timeout=STATE_TIMEOUT)
        )
        async with manager as request_client:
            headers = {"Authorization": f"Bearer {secret}"}
            if content is not None:
                headers["Content-Type"] = "application/json"
            resp = await request_client.request(
                method, url, content=content, headers=headers
            )
            resp.raise_for_status()
            return resp

    return await AsyncRetrying(
        stop=stop_after_attempt(STATE_RETRIES + 1),
        wait=wait_exponential_jitter(initial=0.5, max=30),
        retry=retry_if_exception(transient),
        reraise=True,
    )(request)


# Shared servers receive the calling rollout's state channel in URL parameters because one
# process cannot carry a single rollout's channel in its environment.
STATE_URL_PARAM = "vf_state_url"
STATE_SECRET_PARAM = "vf_state_secret"

# A context variable isolates the state seen by concurrent calls on one shared server.
_call_state: contextvars.ContextVar[State | None] = contextvars.ContextVar(
    "vf_call_state", default=None
)


def _request_query(name: str) -> str | None:
    from mcp.server.lowlevel.server import request_ctx

    try:
        request = request_ctx.get().request
    except LookupError:
        return None
    return request.query_params.get(name) if request is not None else None


def _die_with_parent() -> None:
    """On Linux, prevent a server from outliving its launcher."""
    import ctypes
    import signal

    with contextlib.suppress(Exception):
        ctypes.CDLL(None).prctl(1, signal.SIGKILL)  # PR_SET_PDEATHSIG


def _import_ref(ref: str) -> object:
    import importlib

    module_name, _, qualname = ref.partition(":")
    obj: object = importlib.import_module(module_name)
    for attr in qualname.split("."):
        obj = getattr(obj, attr)
    return obj


class ServerBase(Generic[ConfigT, StateT]):
    TOOL_PREFIX: ClassVar[str | None] = ""
    """The empty value falls back to the snake-cased class name. None advertises the server's
    tools bare (no `<server>_` prefix); name collisions across servers are then the taskset
    author's concern."""

    EXTRAS: ClassVar[tuple[str, ...]] = ()
    """Package extras the server's module needs, applied at sandbox install."""

    def __init__(self, config: ConfigT) -> None:
        self.config = config
        self._state_cls = state_cls(type(self))
        self._state_adapter = TypeAdapter(self._state_cls)
        self._inert_state: StateT = self._state_cls()  # type: ignore[assignment]
        self._state_client: AsyncClient | None = None
        self._exit_stack = contextlib.AsyncExitStack()

    @property
    def state(self) -> StateT:
        current = _call_state.get()
        return current if current is not None else self._inert_state  # type: ignore[return-value]

    def _state_channel(self) -> tuple[str | None, str]:
        """Prefer per-call coordinates for shared servers, then the process environment."""
        url = _request_query(STATE_URL_PARAM) or os.environ.get("VF_STATE_URL")
        secret = _request_query(STATE_SECRET_PARAM) or os.environ.get(
            "VF_STATE_SECRET", ""
        )
        return url, secret

    async def _pull_state(self) -> State:
        url, secret = self._state_channel()
        if not url:
            return self._state_cls()
        response = await _channel_request("GET", url, secret, client=self._state_client)
        try:
            return self._state_adapter.validate_json(response.content)
        except ValidationError as e:
            logger.warning(
                "state pull rejected for %s: %s", self._state_cls.__name__, e
            )
            raise

    async def _push_state(self, before: bytes) -> None:
        url, secret = self._state_channel()
        if not url:
            return
        state = _call_state.get()
        current = state if state is not None else self._inert_state
        after = self._state_adapter.dump_json(current)
        if after == before:
            return
        await _channel_request(
            "PUT", url, secret, content=after, client=self._state_client
        )

    async def _fetch_task(self, state_url: str | None, secret: str):
        """Fetch the rollout task; shared task-agnostic servers have no task channel."""
        if not state_url:
            return None
        task_url = (
            state_url[: -len("/state")] + "/task"
            if state_url.endswith("/state")
            else state_url
        )
        data = (await _channel_request("GET", task_url, secret)).json()
        return _import_ref(data["cls"]).model_validate_json(data["task"])

    async def _setup_task_from_channel(
        self, state_url: str | None, secret: str
    ) -> None:
        task = await self._fetch_task(state_url, secret)
        if task is not None:
            await self.setup_task(task)

    def _with_state(self, fn: Callable) -> Callable:
        """Sync state around one call, isolated by a context variable.

        Updates replace the whole state, so concurrent writes are last-write-wins. Mutations that
        must compose need to run sequentially.
        """

        @functools.wraps(fn)
        async def wrapper(*args, **kwargs):
            # Base State has no fields and rejects extras, so only subclasses need channel sync.
            sync_state = self._state_cls is not State
            state = await self._pull_state() if sync_state else State()
            token = _call_state.set(state)
            try:
                before = self._state_adapter.dump_json(state) if sync_state else None
                result = fn(*args, **kwargs)
                if inspect.isawaitable(result):
                    result = await result
                if before is not None:
                    await self._push_state(before)
                return result
            finally:
                _call_state.reset(token)

        # FastMCP must advertise the wrapped tool's parameters, not `*args, **kwargs`.
        wrapper.__signature__ = inspect.signature(fn)  # type: ignore[attr-defined]
        return wrapper

    @property
    def server_name(self) -> str:
        if self.TOOL_PREFIX is None:
            return ""
        return self.TOOL_PREFIX or "".join(
            ("_" + c.lower() if c.isupper() else c) for c in type(self).__name__
        ).lstrip("_")

    async def setup(self) -> None:
        """Initialize task-agnostic server state."""

    async def setup_task(self, task) -> None:
        """Initialize per-task state; taskset-scoped servers skip this hook."""

    def _register(self, mcp: FastMCP) -> None:
        raise NotImplementedError

    def _serve(self) -> None:
        import asyncio
        import socket
        from pathlib import Path

        import uvicorn
        from mcp.server.fastmcp import FastMCP

        _die_with_parent()
        host = os.environ.get("MCP_HOST", "127.0.0.1")
        # Remote runtimes require their forwarded port; local servers let the OS choose. Report it
        # before setup so an expensive setup does not block port discovery.
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((host, int(os.environ.get("MCP_PORT", 0))))
        port_file = os.environ.get("MCP_PORT_FILE")
        if port_file:
            Path(port_file).write_text(str(sock.getsockname()[1]))

        async def serve() -> None:
            import httpx
            from mcp.server.transport_security import TransportSecuritySettings

            try:
                async with (
                    self._exit_stack,
                    httpx.AsyncClient(timeout=STATE_TIMEOUT) as client,
                ):
                    self._state_client = client
                    await self.setup()
                    await self._setup_task_from_channel(*self._state_channel())
                    # These servers are reached through localhost or a tunnel, never a browser.
                    security = TransportSecuritySettings(
                        enable_dns_rebinding_protection=False
                    )
                    mcp = FastMCP(
                        self.server_name,
                        json_response=True,
                        stateless_http=True,
                        transport_security=security,
                    )
                    self._register(mcp)
                    app = mcp.streamable_http_app()
                    server = uvicorn.Server(uvicorn.Config(app, log_level="critical"))
                    await server.serve(sockets=[sock])
            finally:
                self._state_client = None

        asyncio.run(serve())

    @classmethod
    def _config_cls(cls) -> type[BaseConfig]:
        """Resolve the server's config specialization through its MRO."""
        if config_cls := generic_type(cls, BaseConfig):
            return config_cls
        raise TypeError(
            f"{cls.__name__} must parameterize its config, e.g. Toolset[MyConfig]"
        )

    @classmethod
    def run(cls) -> None:
        config_cls = cls._config_cls()
        if "VF_CONFIG" in os.environ:
            config = config_cls.model_validate_json(os.environ["VF_CONFIG"])
        else:
            from pydantic_config import cli

            config = cli(config_cls)
        cls(config)._serve()
