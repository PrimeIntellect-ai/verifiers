"""`ServerBase`: the vf-native server base — a class authored from a config, shared by `Toolset`
and `User`.

A server's env module is self-runnable: its `__main__` calls `ServerBase.run()`, which rebuilds the
server from the environment the framework set and serves it over MCP (`_serve`). The host side that
starts these in a runtime lives in `launch`.
"""

from __future__ import annotations

import contextlib
import contextvars
import functools
import inspect
import logging
import os
from typing import TYPE_CHECKING, Callable, ClassVar, Generic, TypeVar, get_args

from pydantic import TypeAdapter, ValidationError
from pydantic_config import BaseConfig

from verifiers.v1.state import State, StateT, state_cls

if TYPE_CHECKING:
    from httpx import AsyncClient, Response
    from mcp.server.fastmcp import FastMCP

ConfigT = TypeVar("ConfigT", bound=BaseConfig)
logger = logging.getLogger(__name__)

STATE_TIMEOUT = 30.0
"""Seconds for a state-channel GET/PUT (localhost, or a tunnel to the host)."""
STATE_RETRIES = 4
"""Retries for a state/task channel request. Behind a remote runtime the channel is reached over a
host tunnel that can return a transient 5xx or reset the connection while it settles (e.g. just
after the sandbox/tunnel comes up), so a single blip must not fail the tool call."""


async def _channel_request(
    method: str,
    url: str,
    secret: str,
    *,
    content: bytes | None = None,
    client: AsyncClient | None = None,
) -> Response:
    """One bearer-authed call to the interception `/state` or `/task` channel, retried with
    exponential backoff + jitter on transient failures (connection resets / 5xx from the host
    tunnel). A 4xx is a real error (e.g. an invalid state PUT) and is not retried. Returns the
    response so typed callers can validate its JSON bytes directly."""
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


# A `shared` server is one process for the whole eval, so it can't take a per-rollout state channel
# from its environment like a per-rollout server does. Instead the framework tags each rollout's URL
# to a shared server with that rollout's channel coordinates as these query params (`serve_tools`);
# `_state_channel` reads them back per call from the MCP request, so `self.state` is the CALLING
# rollout's state. The secret is the bearer token the harness already holds — no new exposure.
STATE_URL_PARAM = "vf_state_url"
STATE_SECRET_PARAM = "vf_state_secret"

# The in-flight call's pulled `self.state`, set per call by `_with_state`. Per-call (not an instance
# attribute) so concurrent calls — including different rollouts on one `shared` server — each see
# their own state and never clobber each other.
_call_state: contextvars.ContextVar[State | None] = contextvars.ContextVar(
    "vf_call_state", default=None
)


def _request_query(name: str) -> str | None:
    """A query param of the in-flight tool/respond call's HTTP request, read from MCP's per-call
    request context (streamable HTTP threads the originating request through to the handler) — or
    None outside an HTTP call. How a `shared` server reads the per-rollout state coordinates the
    framework tagged on its URL."""
    from mcp.server.lowlevel.server import request_ctx

    try:
        request = request_ctx.get().request
    except LookupError:
        return None
    return request.query_params.get(name) if request is not None else None


def _die_with_parent() -> None:
    """Ask the kernel to SIGKILL this process when its parent (the launcher) dies, so a server is
    never orphaned if its launcher is torn down without stopping it (a backstop to the runtime's
    own cleanup). Linux-only; a no-op elsewhere. Cleared across `fork`, so a forked child must call
    it again."""
    import ctypes
    import signal

    with contextlib.suppress(Exception):
        ctypes.CDLL(None).prctl(1, signal.SIGKILL)  # PR_SET_PDEATHSIG


def _import_ref(ref: str) -> object:
    """Resolve a `module:qualname` reference (e.g. `glossary_v1:GlossaryTask`) to the object."""
    import importlib

    module_name, _, qualname = ref.partition(":")
    obj: object = importlib.import_module(module_name)
    for attr in qualname.split("."):
        obj = getattr(obj, attr)
    return obj


class ServerBase(Generic[ConfigT, StateT]):
    """A vf-native server authored as a class, initialized from its config — the same shape as
    `Taskset`/`TasksetConfig`: the config (a `ToolsetConfig`/`UserConfig` subclass) is the
    serializable data (placement + the server's own knobs); the class is the behaviour. The
    framework launches it by running its env module (`python -m <module>`), whose `__main__` calls
    `cls.run()` to rebuild `cls(config)` from the environment and serve over MCP — no FastMCP
    boilerplate. Subclassed by `Toolset` (`@tool` methods) and `User` (a `respond` hook). Build
    expensive/non-serializable state in `setup` — set it as plain instance attributes (it runs in
    the server process). The server's deps come from its env package's `pyproject` (the install in
    a sandbox), so the class may freely `import verifiers`, import siblings, and use module
    globals.

    `self.state` is the rollout's shared `State` (see `verifiers.v1.state`): a `@vf.tool` / `respond`
    reads+writes it, and each call is bracketed (`_with_state`) to pull the latest from the
    interception server and push back any change — so tools and the user sim share state, and a
    taskset can end the trajectory from it via a `@vf.stop` over a flag a server sets. Parameterize a
    stateful server with its `State` subclass (`Toolset[Config, MyState]`); defaults to base `State`."""

    TOOL_PREFIX: ClassVar[str] = ""
    """Prefix the model sees on this server's tools (`<TOOL_PREFIX>_<tool>`), set on the class,
    not the config. Empty falls back to the class name snake-cased — set it explicitly for a
    toolset the model calls (e.g. `wiki` -> `wiki_search`)."""

    def __init__(self, config: ConfigT) -> None:
        self.config = config
        self._state_cls = state_cls(type(self))
        # Reuse Pydantic's public byte serializer/validator across state-channel calls.
        self._state_adapter = TypeAdapter(self._state_cls)
        self._inert_state: StateT = self._state_cls()  # type: ignore[assignment]
        """A fresh state used outside a tool/respond call (setup, or a manual debug run) — there's
        no channel to sync, so writes to it don't escape the process (matches the no-channel case)."""
        self._state_client: AsyncClient | None = None

    @property
    def state(self) -> StateT:
        """The rollout's shared runtime state for the in-flight tool/respond call, refreshed from the
        interception server's channel before the call and pushed back after (see `_with_state`).
        Backed by a per-call contextvar, so concurrent calls — including different rollouts on one
        `shared` server — each see their own state and never clobber each other. Outside a call it's
        a fresh, inert `State`."""
        current = _call_state.get()
        return current if current is not None else self._inert_state  # type: ignore[return-value]

    def _state_channel(self) -> tuple[str | None, str]:
        """The interception server's state channel `(url, secret)` for the in-flight call: the
        per-rollout coordinates the framework tagged on a `shared` server's URL (read per call from
        the MCP request), else the per-process channel it set in the environment (a per-rollout
        server), else `(None, "")` (no channel — a manual debug run)."""
        url = _request_query(STATE_URL_PARAM) or os.environ.get("VF_STATE_URL")
        secret = _request_query(STATE_SECRET_PARAM) or os.environ.get(
            "VF_STATE_SECRET", ""
        )
        return url, secret

    async def _pull_state(self) -> State:
        """Fetch the in-flight call's shared state from the channel (so it reflects other servers'
        writes); a fresh inert state when there's no channel."""
        url, secret = self._state_channel()
        if not url:
            return self._state_cls()
        response = await _channel_request("GET", url, secret, client=self._state_client)
        try:
            return self._state_adapter.validate_json(response.content)
        except ValidationError as e:
            # Excessively nested or otherwise invalid state is a broken channel contract.
            logger.warning(
                "state pull rejected for %s: %s", self._state_cls.__name__, e
            )
            raise

    async def _push_state(self, before: bytes) -> None:
        """Push the in-flight call's `self.state` back to the shared channel if it changed. No-op
        without a channel or when nothing changed."""
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
        """Fetch this rollout's task from the interception server's `/task` channel (the sibling of
        `/state`, keyed by the same bearer secret) and rebuild it — how EVERY launched server gets its
        task to run `setup_task`. `None` when there's no channel (a shared, task-agnostic server)."""
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
        """Fetch this rollout's task over the state channel and run `setup_task` for it — a no-op
        without a channel or task (a shared, task-agnostic server). The single place task setup
        happens: `_serve` calls it with the env channel; a forked child calls it with its per-request
        channel (see `verifiers.v1.mcp.multiplex`)."""
        task = await self._fetch_task(state_url, secret)
        if task is not None:
            await self.setup_task(task)

    def _with_state(self, fn: Callable) -> Callable:
        """Wrap a tool/respond callable so each invocation pulls the latest shared state into
        `self.state` before running and pushes back any change after — the read/write channel a
        `@vf.tool` and `respond` use via `self.state`. The pulled state lives in a per-call
        contextvar, so concurrent calls (and concurrent rollouts on a `shared` server) don't share
        it. Preserves `fn`'s signature so FastMCP advertises the tool unchanged. A no-op (fresh inert
        state) when the server runs outside a rollout (no channel).

        This is a whole-object read-modify-write, so it is **last-write-wins**: calls a harness runs
        concurrently (several `tool_calls` in one turn) race and can lose each other's writes; calls
        run sequentially compose correctly. Keep shared-state mutations on the sequential path."""

        @functools.wraps(fn)
        async def wrapper(*args, **kwargs):
            state = await self._pull_state()
            token = _call_state.set(state)
            try:
                # Reuse the second serialization as the PUT body when the callback mutates state.
                before = self._state_adapter.dump_json(state)
                result = fn(*args, **kwargs)
                if inspect.isawaitable(result):
                    result = await result
                await self._push_state(before)
                return result
            finally:
                _call_state.reset(token)

        wrapper.__signature__ = inspect.signature(fn)  # type: ignore[attr-defined]
        return wrapper

    @property
    def server_name(self) -> str:
        """The server's identity (MCP name, log + namespace key): `TOOL_PREFIX`, else the class
        name snake-cased."""
        return self.TOOL_PREFIX or "".join(
            ("_" + c.lower() if c.isupper() else c) for c in type(self).__name__
        ).lstrip("_")

    async def setup(self) -> None:
        """Task-agnostic setup, in the server process: global state (a corpus / index / graph loaded
        from disk or a dataset) as plain instance attributes (`self.x = ...`). Runs for every server
        — shared or per-rollout. Config knobs stay on `self.config`."""

    async def setup_task(self, task) -> None:
        """Per-rollout setup, in the server process: per-task input read off `task` (this rollout's
        task) and initial per-rollout mutable state (counters, paths). Runs only when the server has
        a task — SKIPPED for a `shared` server (one instance for the whole eval), so don't override
        it on a shared server (the framework warns loudly if you do)."""

    def _register(self, mcp: FastMCP) -> None:
        raise NotImplementedError

    def _serve(self) -> None:
        """Run this server's MCP server: bind its port, `setup` (always) + `setup_task` for the
        rollout's task (fetched from the interception `/task` channel — skipped for a shared server,
        which has no channel), build a FastMCP from the registered tools, and serve it over streamable
        HTTP on `MCP_HOST`. Called by `run()`."""
        import asyncio
        import socket
        from pathlib import Path

        import uvicorn
        from mcp.server.fastmcp import FastMCP

        _die_with_parent()  # never outlive the launcher that started us, even if it skips cleanup
        host = os.environ.get("MCP_HOST", "127.0.0.1")
        # Bind our own socket up front: `MCP_PORT` when the framework fixed one (a self-publishing
        # runtime's forwarded port), else 0 = an OS-assigned free port — guaranteed free in whatever
        # environment we run in (host or sandbox), so the launcher never probes for a free port.
        # Report the bound port back via `MCP_PORT_FILE` before setup, so the launcher learns it
        # without waiting on a slow `setup` (its readiness probe absorbs that).
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((host, int(os.environ.get("MCP_PORT", 0))))
        port_file = os.environ.get("MCP_PORT_FILE")
        if port_file:
            Path(port_file).write_text(str(sock.getsockname()[1]))

        async def _setup() -> None:
            await self.setup()
            # Per-rollout servers carry the rollout's state channel in their env; fetch this rollout's
            # task over it and run `setup_task`. A shared server has no env channel (its forked
            # children fetch /task per-request instead — see multiplex), so this is a no-op for it.
            await self._setup_task_from_channel(*self._state_channel())

        asyncio.run(_setup())
        # Relax FastMCP's DNS-rebinding guard: it 421s a non-localhost Host, but our servers are
        # reached only by our own harness over localhost or a tunnel (never a browser).
        from mcp.server.transport_security import TransportSecuritySettings

        security = TransportSecuritySettings(enable_dns_rebinding_protection=False)
        mcp = FastMCP(self.server_name, transport_security=security)
        self._register(mcp)
        app = mcp.streamable_http_app()
        mcp_lifespan = app.router.lifespan_context

        @contextlib.asynccontextmanager
        async def serving_lifespan(starlette):
            import httpx

            try:
                async with (
                    httpx.AsyncClient(timeout=STATE_TIMEOUT) as client,
                    mcp_lifespan(starlette),
                ):
                    self._state_client = client
                    yield
            finally:
                self._state_client = None

        app.router.lifespan_context = serving_lifespan
        if getattr(self.config, "fork", False):
            # `setup` ran once above (warm); fork a child per rollout that inherits it and runs
            # `setup_task` for the rollout's task (see multiplex).
            from verifiers.v1.mcp.multiplex import serve_forked

            serve_forked(app, sock, self)
            return
        server = uvicorn.Server(uvicorn.Config(app, log_level="critical"))
        asyncio.run(server.serve(sockets=[sock]))

    @classmethod
    def _config_cls(cls) -> type[BaseConfig]:
        """The config type from the `Toolset[Config]` / `User[Config]` generic parameter. Walks the
        MRO, so a further subclass that doesn't re-parameterize (`class B(MyToolset)`) inherits the
        config from where it was set."""
        for klass in cls.__mro__:
            for base in getattr(klass, "__orig_bases__", ()):
                for arg in get_args(base):
                    if isinstance(arg, type) and issubclass(arg, BaseConfig):
                        return arg
        raise TypeError(
            f"{cls.__name__} must parameterize its config, e.g. Toolset[MyConfig]"
        )

    @classmethod
    def run(cls) -> None:
        """Entry point a server module calls from `if __name__ == "__main__"`: rebuild this server
        from the config the framework set (`VF_CONFIG` JSON) and serve it over MCP. The rollout's task
        is NOT passed in the environment — the server fetches it from the interception `/task` channel
        at startup (see `_serve` / `_fetch_task`). With no `VF_CONFIG` the config is parsed from the CLI
        instead (`cli(config)`), so the module is runnable by hand for debugging (no channel, no task)."""
        config_cls = cls._config_cls()
        if "VF_CONFIG" in os.environ:
            config = config_cls.model_validate_json(os.environ["VF_CONFIG"])
        else:
            from pydantic_config import cli

            config = cli(config_cls)
        cls(config)._serve()
