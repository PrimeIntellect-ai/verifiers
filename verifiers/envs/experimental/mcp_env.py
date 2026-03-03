from __future__ import annotations

import asyncio
import threading
from typing import Any, Callable, Literal, cast

from mcp.types import Tool as MCPTool

import verifiers as vf
from verifiers.decorators import cleanup as cleanup_handler
from verifiers.decorators import teardown as teardown_handler
from verifiers.types import Messages, State, ToolMessage
from verifiers.utils.async_utils import maybe_await
from verifiers.utils.mcp_utils.mcp_env_utils import create_transport, validate_config
from verifiers.utils.mcp_utils.mcp_tool_wrapper import (
    MCPToolProxy,
    RegisteredMCPTool,
    build_registered_tool,
)
from verifiers.utils.mcp_utils.models import MCPServerConfig, MCPTransportConfig
from verifiers.utils.mcp_utils.transports.base import MCPTransport

MCPTransportType = Literal["stdio", "http", "sandbox"]
MCPConnectionScope = Literal["shared", "rollout"]
MCP_TEARDOWN_TIMEOUT_SECONDS = 5.0


class MCPEnv(vf.StatefulToolEnv):
    """Transport-backed MCP tool environment.

    The default path shares stateless stdio/http transports across rollouts.
    For stateful MCP servers, use ``connection_scope="rollout"`` or the
    sandbox transport, which defaults to isolated per-rollout state.
    """

    def __init__(
        self,
        mcp_servers: list[MCPServerConfig | dict] | None = None,
        tools: list[Callable] | None = None,
        transport_type: MCPTransportType = "stdio",
        connection_scope: MCPConnectionScope | None = None,
        http_urls: dict[str, str] | None = None,
        http_timeout: float = 30.0,
        http_max_retries: int = 3,
        sandbox_image: str = "python:3.11-slim",
        sandbox_start_command: str = "tail -f /dev/null",
        sandbox_cpu_cores: int = 1,
        sandbox_memory_gb: int = 2,
        sandbox_disk_size_gb: int = 5,
        sandbox_timeout_minutes: int = 60,
        sandbox_environment_vars: dict[str, str] | None = None,
        sandbox_port_to_expose: int = 8000,
        transport_factory: Callable[[MCPTransportConfig], Any] | None = None,
        max_turns: int = 10,
        error_formatter: Callable[[Exception], str] = lambda e: f"Error: {str(e)}",
        stop_errors: list[type[Exception]] | None = None,
        **kwargs,
    ):
        self.mcp_servers = [
            MCPServerConfig(**server) if isinstance(server, dict) else server
            for server in (mcp_servers or [])
        ]
        self.transport_type = transport_type
        self.connection_scope = (
            connection_scope
            if connection_scope is not None
            else ("rollout" if transport_type == "sandbox" else "shared")
        )
        self.http_urls = http_urls or {}
        self.http_timeout = http_timeout
        self.http_max_retries = http_max_retries
        self.sandbox_image = sandbox_image
        self.sandbox_start_command = sandbox_start_command
        self.sandbox_cpu_cores = sandbox_cpu_cores
        self.sandbox_memory_gb = sandbox_memory_gb
        self.sandbox_disk_size_gb = sandbox_disk_size_gb
        self.sandbox_timeout_minutes = sandbox_timeout_minutes
        self.sandbox_environment_vars = sandbox_environment_vars or {}
        self.sandbox_port_to_expose = sandbox_port_to_expose
        self.transport_factory = transport_factory or create_transport

        validate_config(
            self.transport_type,
            self.mcp_servers,
            http_urls=self.http_urls,
        )

        self._shared_transports: dict[str, MCPTransport] = {}
        self._registered_mcp_tools: dict[str, RegisteredMCPTool] = {}
        # Keep MCP tool aliases stable so reconnects do not rename tools mid-run.
        self._mcp_public_names: dict[tuple[str, str], str] = {}
        self._shared_setup_lock: asyncio.Lock | None = None
        self._shared_bg_loop: asyncio.AbstractEventLoop | None = None
        self._shared_bg_thread: threading.Thread | None = None

        super().__init__(
            tools=tools or [],
            max_turns=max_turns,
            error_formatter=error_formatter,
            stop_errors=stop_errors,
            **kwargs,
        )

        if self.connection_scope == "shared" and self.mcp_servers:
            self._bootstrap_shared_transports()

    def _run_shared_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        asyncio.set_event_loop(loop)
        loop.run_forever()

    def _run_on_shared_loop_sync(self, coro: Any) -> Any:
        if self._shared_bg_loop is None:
            raise RuntimeError("Shared MCP loop is not initialized")
        future = asyncio.run_coroutine_threadsafe(coro, self._shared_bg_loop)
        return future.result()

    def _bootstrap_shared_transports(self) -> None:
        if self._shared_bg_loop is not None:
            return

        # Shared MCP sessions live on their own loop so they can stay connected
        # across rollout event loops while still supporting eager tool discovery.
        self._shared_bg_loop = asyncio.new_event_loop()
        self._shared_bg_thread = threading.Thread(
            target=self._run_shared_loop,
            args=(self._shared_bg_loop,),
            daemon=True,
        )
        self._shared_bg_thread.start()

        try:
            transports = self._run_on_shared_loop_sync(
                self._create_connected_transports()
            )
        except Exception:
            self._shutdown_shared_loop()
            raise

        self._shared_transports = transports
        self._register_discovered_tools(transports)

    async def _make_transport(
        self, transport_config: MCPTransportConfig
    ) -> MCPTransport:
        transport = await maybe_await(self.transport_factory, transport_config)
        if not isinstance(transport, MCPTransport):
            raise TypeError(
                "transport_factory must return an MCPTransport instance, "
                f"got {type(transport).__name__}"
            )
        return transport

    def _build_transport_config(
        self, server_config: MCPServerConfig
    ) -> MCPTransportConfig:
        return MCPTransportConfig(
            server_config=server_config,
            transport_type=self.transport_type,
            http_urls=self.http_urls,
            http_timeout=self.http_timeout,
            http_max_retries=self.http_max_retries,
            sandbox_image=self.sandbox_image,
            sandbox_start_command=self.sandbox_start_command,
            sandbox_environment_vars=self.sandbox_environment_vars,
            sandbox_cpu_cores=self.sandbox_cpu_cores,
            sandbox_memory_gb=self.sandbox_memory_gb,
            sandbox_disk_size_gb=self.sandbox_disk_size_gb,
            sandbox_timeout_minutes=self.sandbox_timeout_minutes,
            sandbox_port_to_expose=self.sandbox_port_to_expose,
        )

    async def _create_connected_transports(self) -> dict[str, MCPTransport]:
        transports: dict[str, MCPTransport] = {}
        try:
            for server_config in self.mcp_servers:
                transport = await self._make_transport(
                    self._build_transport_config(server_config)
                )
                await transport.connect()
                transports[server_config.name] = transport
            return transports
        except Exception:
            try:
                await self._disconnect_transports(transports)
            except Exception as cleanup_exc:
                self.logger.warning(
                    f"Failed to clean up partially initialized MCP transports: {cleanup_exc}"
                )
            raise

    async def _disconnect_transports(self, transports: dict[str, MCPTransport]) -> None:
        errors: list[Exception] = []
        for transport in transports.values():
            try:
                await transport.disconnect()
            except Exception as exc:
                errors.append(exc)
        if errors:
            raise errors[0]

    async def _disconnect_transports_with_timeout(
        self,
        transports: dict[str, MCPTransport],
        *,
        context: str,
    ) -> None:
        try:
            await asyncio.wait_for(
                self._disconnect_transports(transports),
                timeout=MCP_TEARDOWN_TIMEOUT_SECONDS,
            )
        except asyncio.TimeoutError:
            self.logger.warning(f"Timed out while {context}")
        except Exception as exc:
            self.logger.warning(f"Failed while {context}: {exc}")

    async def _all_transports_connected(
        self, transports: dict[str, MCPTransport]
    ) -> bool:
        for transport in transports.values():
            try:
                if not await transport.is_connected():
                    return False
            except Exception:
                return False
        return True

    async def _shared_transports_connected(self) -> bool:
        if not self._shared_transports:
            return False
        if self._shared_bg_loop is not None:
            future = asyncio.run_coroutine_threadsafe(
                self._all_transports_connected(self._shared_transports),
                self._shared_bg_loop,
            )
            try:
                return await asyncio.wait_for(
                    asyncio.wrap_future(future),
                    timeout=MCP_TEARDOWN_TIMEOUT_SECONDS,
                )
            except Exception:
                future.cancel()
                return False
        return await self._all_transports_connected(self._shared_transports)

    async def _reset_shared_transports(self) -> None:
        if not self._shared_transports:
            return
        transports = self._shared_transports
        self._shared_transports = {}
        if self._shared_bg_loop is not None:
            future = asyncio.run_coroutine_threadsafe(
                self._disconnect_transports(transports),
                self._shared_bg_loop,
            )
            try:
                await asyncio.wait_for(
                    asyncio.wrap_future(future),
                    timeout=MCP_TEARDOWN_TIMEOUT_SECONDS,
                )
            except Exception as exc:
                future.cancel()
                self.logger.warning(
                    f"Failed to reset stale shared MCP transports: {exc}"
                )
            return
        await self._disconnect_transports_with_timeout(
            transports,
            context="resetting stale shared MCP transports",
        )

    def _compute_public_tool_names(
        self, discovered_tools: list[tuple[str, MCPTool]]
    ) -> dict[tuple[str, str], str]:
        discovered_names: dict[str, int] = {}
        for _, tool in discovered_tools:
            discovered_names[tool.name] = discovered_names.get(tool.name, 0) + 1
        public_names: dict[tuple[str, str], str] = {}
        assigned_names = set(self._mcp_public_names.values())
        reserved_names = {
            tool.name
            for tool in (self.tool_defs or [])
            if tool.name not in assigned_names
        }
        reserved_names.update(assigned_names)

        for server_name, tool in discovered_tools:
            key = (server_name, tool.name)
            if key in self._mcp_public_names:
                public_name = self._mcp_public_names[key]
            else:
                candidate = tool.name
                if candidate in reserved_names or discovered_names[tool.name] > 1:
                    candidate = f"{server_name}__{tool.name}"
                public_name = self._allocate_tool_name(candidate, reserved_names)
                self._mcp_public_names[key] = public_name

            public_names[key] = public_name
            reserved_names.add(public_name)

        return public_names

    def _allocate_tool_name(self, candidate: str, reserved_names: set[str]) -> str:
        if candidate not in reserved_names:
            return candidate

        suffix = 2
        while f"{candidate}__{suffix}" in reserved_names:
            suffix += 1
        return f"{candidate}__{suffix}"

    def _get_shared_setup_lock(self) -> asyncio.Lock:
        if self._shared_setup_lock is None:
            self._shared_setup_lock = asyncio.Lock()
        return self._shared_setup_lock

    def _register_discovered_tools(self, transports: dict[str, MCPTransport]) -> None:
        discovered_tools: list[tuple[str, MCPTool]] = []
        for server_name, transport in transports.items():
            for tool in transport.tools.values():
                discovered_tools.append((server_name, tool))

        public_names = self._compute_public_tool_names(discovered_tools)

        for server_name, tool in discovered_tools:
            public_name = public_names[(server_name, tool.name)]
            if public_name in self._registered_mcp_tools:
                continue

            registration = build_registered_tool(
                public_name=public_name,
                server_name=server_name,
                tool=tool,
            )
            proxy = MCPToolProxy(registration, self)
            self.tools.append(proxy)
            if self.tool_defs is None:
                self.tool_defs = []
            self.tool_defs.append(registration.to_tool_def())
            self.tool_map[public_name] = proxy
            self.skipped_args[public_name] = ["state"]
            self.tool_monitor_rubric.add_tool_metric(proxy)
            self._registered_mcp_tools[public_name] = registration

    async def _ensure_shared_transports(self) -> None:
        if await self._shared_transports_connected():
            return
        async with self._get_shared_setup_lock():
            if await self._shared_transports_connected():
                return
            await self._reset_shared_transports()
            if self._shared_bg_loop is not None:
                transports = await asyncio.wrap_future(
                    asyncio.run_coroutine_threadsafe(
                        self._create_connected_transports(),
                        self._shared_bg_loop,
                    )
                )
            else:
                transports = await self._create_connected_transports()
            self._register_discovered_tools(transports)
            self._shared_transports = transports

    async def setup_state(self, state: State, **kwargs) -> State:
        state = await super().setup_state(state, **kwargs)

        if self.connection_scope == "shared":
            await self._ensure_shared_transports()
        else:
            rollout_transports = await self._create_connected_transports()
            if not self._registered_mcp_tools:
                self._register_discovered_tools(rollout_transports)
            state["_mcp_rollout_transports"] = rollout_transports

        state["tool_defs"] = list(self.tool_defs or [])
        return state

    def update_tool_args(
        self,
        tool_name: str,
        tool_args: dict,
        messages: Messages,
        state: State,
        **kwargs,
    ) -> dict:
        if tool_name in self._registered_mcp_tools:
            return {**tool_args, "state": state}
        return tool_args

    def _resolve_transport(self, public_tool_name: str, state: State) -> MCPTransport:
        registration = self._registered_mcp_tools[public_tool_name]
        if self.connection_scope == "shared":
            transport = self._shared_transports.get(registration.server_name)
        else:
            rollout_transports = state.get("_mcp_rollout_transports", {})
            transport = rollout_transports.get(registration.server_name)
        if transport is None:
            raise RuntimeError(
                f"MCP transport for server '{registration.server_name}' is not available"
            )
        return transport

    async def call_mcp_tool(
        self,
        *,
        public_tool_name: str,
        tool_args: dict,
        state: State,
    ) -> Any:
        registration = self._registered_mcp_tools[public_tool_name]
        transport = self._resolve_transport(public_tool_name, state)
        if self.connection_scope == "shared" and self._shared_bg_loop is not None:
            return await asyncio.wrap_future(
                asyncio.run_coroutine_threadsafe(
                    transport.call_tool(registration.remote_name, tool_args),
                    self._shared_bg_loop,
                )
            )
        return await transport.call_tool(registration.remote_name, tool_args)

    async def call_tool(
        self,
        tool_name: str,
        tool_args: dict,
        tool_call_id: str,
        state: State | None = None,
        **kwargs,
    ) -> ToolMessage:
        if tool_name in self._registered_mcp_tools and "state" not in tool_args:
            resolved_state = state or cast(State | None, kwargs.get("state"))
            if resolved_state is None:
                if self.connection_scope == "shared":
                    resolved_state = cast(State, {})
                else:
                    raise ValueError(
                        f"MCP tool '{tool_name}' requires rollout state when "
                        "connection_scope='rollout'"
                    )
            tool_args = {**tool_args, "state": resolved_state}

        try:
            return await super().call_tool(tool_name, tool_args, tool_call_id, **kwargs)
        except Exception as exc:
            if self._should_stop_for_error(exc):
                raise vf.ToolCallError from exc
            return ToolMessage(
                role="tool",
                content=self.error_formatter(exc),
                tool_call_id=tool_call_id,
            )

    async def cleanup(self) -> None:
        """Backward-compatible shared cleanup hook."""
        if self.connection_scope == "shared":
            await self.teardown_shared_transports()

    @cleanup_handler
    async def cleanup_rollout_transports(self, state: State) -> None:
        rollout_transports = cast(
            dict[str, MCPTransport] | None, state.pop("_mcp_rollout_transports", None)
        )
        if rollout_transports:
            try:
                await self._disconnect_transports(rollout_transports)
            except Exception as exc:
                self.logger.warning(f"Failed to clean up rollout MCP transports: {exc}")

    @teardown_handler
    async def teardown_shared_transports(self) -> None:
        if not self._shared_transports:
            self._shutdown_shared_loop()
            return
        transports = self._shared_transports
        self._shared_transports = {}
        if self._shared_bg_loop is not None:
            disconnect_future = asyncio.run_coroutine_threadsafe(
                self._disconnect_transports(transports),
                self._shared_bg_loop,
            )
            try:
                await asyncio.wait_for(
                    asyncio.wrap_future(disconnect_future),
                    timeout=MCP_TEARDOWN_TIMEOUT_SECONDS,
                )
            except asyncio.TimeoutError:
                disconnect_future.cancel()
                self.logger.warning(
                    "Timed out while disconnecting shared MCP transports during teardown"
                )
            except Exception as exc:
                self.logger.warning(
                    f"Failed to disconnect shared MCP transports during teardown: {exc}"
                )
            self._shutdown_shared_loop()
            return
        await self._disconnect_transports_with_timeout(
            transports,
            context="disconnecting shared MCP transports during teardown",
        )

    def _shutdown_shared_loop(self) -> None:
        if self._shared_bg_loop is None:
            return
        loop = self._shared_bg_loop
        thread = self._shared_bg_thread
        loop.call_soon_threadsafe(loop.stop)
        if thread is not None:
            thread.join(timeout=MCP_TEARDOWN_TIMEOUT_SECONDS)
            if thread.is_alive():
                self.logger.warning(
                    "Shared MCP event loop did not stop within the teardown timeout"
                )
                self._shared_bg_loop = None
                self._shared_bg_thread = None
                return
        loop.close()
        self._shared_bg_loop = None
        self._shared_bg_thread = None


__all__ = ["MCPEnv", "MCPServerConfig", "MCPConnectionScope", "MCPTransportType"]
