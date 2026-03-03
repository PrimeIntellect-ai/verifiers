"""Tests for the experimental MCPEnv foundation."""

import asyncio
import json
from contextlib import asynccontextmanager

import pytest

pytest.importorskip("mcp")

import verifiers as vf
from verifiers.parsers.parser import Parser
from verifiers.rubrics.rubric import Rubric
from verifiers.types import ToolCall
from verifiers.utils.mcp_utils.models import MCPTransportConfig
from verifiers.utils.mcp_utils.transports.streaming_http import StreamingHTTPTransport
from verifiers.utils.mcp_utils.transports.synthetic_transport import (
    SyntheticTransport,
    create_tool,
)


def add_one(x: int) -> int:
    return x + 1


def local_lookup(query: str) -> str:
    return query


def prefixed_lookup(query: str) -> str:
    return query


local_lookup.__name__ = "lookup"
prefixed_lookup.__name__ = "search__lookup"


class CountingSyntheticTransport(SyntheticTransport):
    def __init__(self, *, label: str, tools, handlers):
        super().__init__(tools=tools, handlers=handlers, name=label)
        self.label = label
        self.connect_count = 0
        self.disconnect_count = 0

    async def connect(self):
        self.connect_count += 1
        return await super().connect()

    async def disconnect(self):
        self.disconnect_count += 1
        await super().disconnect()


class MCPStopError(RuntimeError):
    pass


def _build_env(mock_client, sample_chat_dataset, **kwargs):
    return vf.MCPEnv(
        client=mock_client,
        model="test-model",
        dataset=sample_chat_dataset,
        parser=Parser(),
        rubric=Rubric(),
        **kwargs,
    )


async def _setup_state(env, make_input, mock_client):
    state = await env.init_state(make_input(), mock_client, "test-model", {})
    return await env.setup_state(state)


class TestMCPEnv:
    def test_duplicate_server_names_are_rejected(
        self, mock_client, sample_chat_dataset
    ):
        with pytest.raises(ValueError, match="unique 'name'"):
            _build_env(
                mock_client,
                sample_chat_dataset,
                mcp_servers=[
                    {"name": "dup", "command": "dummy"},
                    {"name": "dup", "command": "dummy"},
                ],
            )

    @pytest.mark.asyncio
    async def test_shared_scope_reuses_transports_and_routes_calls(
        self, mock_client, sample_chat_dataset, make_input
    ):
        created: list[CountingSyntheticTransport] = []

        def transport_factory(config: MCPTransportConfig):
            tool = create_tool(
                name="lookup",
                description="Lookup a value",
                parameters={"query": {"type": "string"}},
                required=["query"],
            )
            transport = CountingSyntheticTransport(
                label=config.server_config.name,
                tools={"lookup": tool},
                handlers={
                    "lookup": lambda data, args, name=config.server_config.name: (
                        f"{name}:{args['query']}"
                    )
                },
            )
            created.append(transport)
            return transport

        env = _build_env(
            mock_client,
            sample_chat_dataset,
            mcp_servers=[{"name": "search", "command": "dummy"}],
            tools=[add_one],
            transport_factory=transport_factory,
        )

        assert [tool.name for tool in env.tool_defs] == ["add_one", "lookup"]

        state_one = await _setup_state(env, make_input, mock_client)
        state_two = await _setup_state(env, make_input, mock_client)

        assert len(created) == 1
        assert created[0].label == "search"
        assert created[0].connect_count == 1
        assert [tool.name for tool in state_one["tool_defs"]] == ["add_one", "lookup"]
        assert [tool.name for tool in state_two["tool_defs"]] == ["add_one", "lookup"]

        tool_message = await env.call_tool("lookup", {"query": "ping"}, "call_0")
        assert tool_message.content == "search:ping"

        await env.teardown_shared_transports()
        assert env._shared_transports == {}
        assert created[0].disconnect_count == 1

    @pytest.mark.asyncio
    async def test_duplicate_tool_names_are_prefixed(
        self, mock_client, sample_chat_dataset, make_input
    ):
        def transport_factory(config: MCPTransportConfig):
            tool = create_tool(
                name="search",
                description="Search data",
                parameters={"query": {"type": "string"}},
                required=["query"],
            )
            return CountingSyntheticTransport(
                label=config.server_config.name,
                tools={"search": tool},
                handlers={"search": lambda data, args: args["query"]},
            )

        env = _build_env(
            mock_client,
            sample_chat_dataset,
            mcp_servers=[
                {"name": "alpha", "command": "dummy"},
                {"name": "beta", "command": "dummy"},
            ],
            transport_factory=transport_factory,
        )

        state = await _setup_state(env, make_input, mock_client)
        assert [tool.name for tool in state["tool_defs"]] == [
            "alpha__search",
            "beta__search",
        ]

    @pytest.mark.asyncio
    async def test_prefixed_aliases_still_avoid_existing_local_tool_names(
        self, mock_client, sample_chat_dataset, make_input
    ):
        def transport_factory(config: MCPTransportConfig):
            tool = create_tool(
                name="lookup",
                description="Lookup data",
                parameters={"query": {"type": "string"}},
                required=["query"],
            )
            return CountingSyntheticTransport(
                label=config.server_config.name,
                tools={"lookup": tool},
                handlers={"lookup": lambda data, args: args["query"]},
            )

        env = _build_env(
            mock_client,
            sample_chat_dataset,
            mcp_servers=[{"name": "search", "command": "dummy"}],
            tools=[local_lookup, prefixed_lookup],
            transport_factory=transport_factory,
        )

        state = await _setup_state(env, make_input, mock_client)
        assert [tool.name for tool in state["tool_defs"]] == [
            "lookup",
            "search__lookup",
            "search__lookup__2",
        ]

    @pytest.mark.asyncio
    async def test_mcp_tool_named_state_preserves_model_argument(
        self, mock_client, sample_chat_dataset, make_input
    ):
        def transport_factory(config: MCPTransportConfig):
            tool = create_tool(
                name="inspect",
                description="Return the model-provided state argument",
                parameters={"state": {"type": "string"}},
                required=["state"],
            )
            return CountingSyntheticTransport(
                label=config.server_config.name,
                tools={"inspect": tool},
                handlers={"inspect": lambda data, args: args["state"]},
            )

        env = _build_env(
            mock_client,
            sample_chat_dataset,
            mcp_servers=[{"name": "search", "command": "dummy"}],
            transport_factory=transport_factory,
        )

        state = await _setup_state(env, make_input, mock_client)
        schema = next(tool for tool in state["tool_defs"] if tool.name == "inspect")
        assert schema.parameters["properties"]["state"] == {"type": "string"}
        assert schema.parameters["required"] == ["state"]

        tool_message = await env.call_tool(
            "inspect",
            {"state": "model-value"},
            "call_state",
            state=state,
        )
        assert tool_message.content == "model-value"

    @pytest.mark.asyncio
    async def test_rollout_scope_creates_and_cleans_isolated_transports(
        self, mock_client, sample_chat_dataset, make_input
    ):
        created: list[CountingSyntheticTransport] = []

        def transport_factory(config: MCPTransportConfig):
            tool = create_tool(
                name="lookup",
                description="Lookup a value",
                parameters={"query": {"type": "string"}},
                required=["query"],
            )
            transport = CountingSyntheticTransport(
                label=f"{config.server_config.name}-{len(created)}",
                tools={"lookup": tool},
                handlers={
                    "lookup": lambda data,
                    args,
                    label=f"{config.server_config.name}-{len(created)}": (
                        f"{label}:{args['query']}"
                    )
                },
            )
            created.append(transport)
            return transport

        env = _build_env(
            mock_client,
            sample_chat_dataset,
            mcp_servers=[{"name": "db", "command": "dummy"}],
            connection_scope="rollout",
            transport_factory=transport_factory,
        )

        state_one = await _setup_state(env, make_input, mock_client)
        state_two = await _setup_state(env, make_input, mock_client)

        assert len(created) == 2
        assert created[0] is not created[1]

        msg_one = await env.call_tool(
            "lookup", {"query": "one"}, "call_1", state=state_one
        )
        msg_two = await env.call_tool(
            "lookup", {"query": "two"}, "call_2", state=state_two
        )

        assert msg_one.content == "db-0:one"
        assert msg_two.content == "db-1:two"

        await env.cleanup_rollout_transports(state_one)
        assert created[0].disconnect_count == 1
        assert created[1].disconnect_count == 0

        await env.cleanup_rollout_transports(state_two)
        assert created[1].disconnect_count == 1

    @pytest.mark.asyncio
    async def test_rollout_call_tool_keeps_explicit_falsey_state(
        self, mock_client, sample_chat_dataset, make_input
    ):
        created: list[CountingSyntheticTransport] = []

        def transport_factory(config: MCPTransportConfig):
            tool = create_tool(
                name="lookup",
                description="Lookup a value",
                parameters={"query": {"type": "string"}},
                required=["query"],
            )
            transport = CountingSyntheticTransport(
                label=f"{config.server_config.name}-{len(created)}",
                tools={"lookup": tool},
                handlers={
                    "lookup": lambda data,
                    args,
                    label=f"{config.server_config.name}-{len(created)}": (
                        f"{label}:{args['query']}"
                    )
                },
            )
            created.append(transport)
            return transport

        class FalseyState(vf.State):
            def __bool__(self) -> bool:
                return False

        env = _build_env(
            mock_client,
            sample_chat_dataset,
            mcp_servers=[{"name": "db", "command": "dummy"}],
            connection_scope="rollout",
            transport_factory=transport_factory,
        )

        state = await _setup_state(env, make_input, mock_client)
        falsey_state = FalseyState(state)

        tool_message = await env.call_tool(
            "lookup",
            {"query": "ping"},
            "call_falsey",
            state=falsey_state,
        )
        assert tool_message.content == "db-0:ping"

        await env.cleanup_rollout_transports(state)

    @pytest.mark.asyncio
    async def test_cleanup_disconnects_shared_transports(
        self, mock_client, sample_chat_dataset, make_input
    ):
        created: list[CountingSyntheticTransport] = []

        def transport_factory(config: MCPTransportConfig):
            tool = create_tool(
                name="lookup",
                description="Lookup a value",
                parameters={"query": {"type": "string"}},
                required=["query"],
            )
            transport = CountingSyntheticTransport(
                label=config.server_config.name,
                tools={"lookup": tool},
                handlers={"lookup": lambda data, args: args["query"]},
            )
            created.append(transport)
            return transport

        env = _build_env(
            mock_client,
            sample_chat_dataset,
            mcp_servers=[{"name": "search", "command": "dummy"}],
            transport_factory=transport_factory,
        )

        await _setup_state(env, make_input, mock_client)
        assert len(created) == 1
        assert created[0].disconnect_count == 0

        await env.cleanup()
        assert created[0].disconnect_count == 1

    @pytest.mark.asyncio
    async def test_shared_reconnect_keeps_stable_tool_names(
        self, mock_client, sample_chat_dataset, make_input
    ):
        def transport_factory(config: MCPTransportConfig):
            tool = create_tool(
                name="lookup",
                description="Lookup a value",
                parameters={"query": {"type": "string"}},
                required=["query"],
            )
            return CountingSyntheticTransport(
                label=config.server_config.name,
                tools={"lookup": tool},
                handlers={"lookup": lambda data, args: args["query"]},
            )

        env = _build_env(
            mock_client,
            sample_chat_dataset,
            mcp_servers=[{"name": "search", "command": "dummy"}],
            transport_factory=transport_factory,
        )

        state_one = await _setup_state(env, make_input, mock_client)
        first_names = [tool.name for tool in state_one["tool_defs"]]
        assert first_names == ["lookup"]

        await env.cleanup()

        state_two = await _setup_state(env, make_input, mock_client)
        second_names = [tool.name for tool in state_two["tool_defs"]]
        assert second_names == ["lookup"]
        assert [tool.name for tool in env.tool_defs] == ["lookup"]

    @pytest.mark.asyncio
    async def test_shared_scope_recreates_background_loop_after_cleanup(
        self, mock_client, sample_chat_dataset, make_input
    ):
        class LoopTrackingSyntheticTransport(CountingSyntheticTransport):
            def __init__(self, *, label: str, tools, handlers):
                super().__init__(label=label, tools=tools, handlers=handlers)
                self.connect_loop: asyncio.AbstractEventLoop | None = None
                self.call_loop: asyncio.AbstractEventLoop | None = None

            async def connect(self):
                self.connect_loop = asyncio.get_running_loop()
                return await super().connect()

            async def call_tool(self, tool_name: str, arguments: dict):
                self.call_loop = asyncio.get_running_loop()
                return await super().call_tool(tool_name, arguments)

        created: list[LoopTrackingSyntheticTransport] = []

        def transport_factory(config: MCPTransportConfig):
            tool = create_tool(
                name="lookup",
                description="Lookup a value",
                parameters={"query": {"type": "string"}},
                required=["query"],
            )
            transport = LoopTrackingSyntheticTransport(
                label=f"{config.server_config.name}-{len(created)}",
                tools={"lookup": tool},
                handlers={
                    "lookup": lambda data,
                    args,
                    label=f"{config.server_config.name}-{len(created)}": (
                        f"{label}:{args['query']}"
                    )
                },
            )
            created.append(transport)
            return transport

        env = _build_env(
            mock_client,
            sample_chat_dataset,
            mcp_servers=[{"name": "search", "command": "dummy"}],
            transport_factory=transport_factory,
        )

        initial_loop = env._shared_bg_loop
        assert initial_loop is not None

        await _setup_state(env, make_input, mock_client)
        assert len(created) == 1
        assert created[0].connect_loop is initial_loop

        await env.cleanup()
        assert env._shared_bg_loop is None

        rollout_loop = asyncio.get_running_loop()
        await _setup_state(env, make_input, mock_client)
        assert len(created) == 2

        recreated_loop = env._shared_bg_loop
        assert recreated_loop is not None
        assert recreated_loop is not rollout_loop
        assert created[1].connect_loop is recreated_loop

        tool_message = await env.call_tool("lookup", {"query": "ping"}, "call_3")
        assert tool_message.content == "search-1:ping"
        assert created[1].call_loop is recreated_loop

        await env.cleanup()

    @pytest.mark.asyncio
    async def test_shared_scope_reconnects_after_unexpected_disconnect(
        self, mock_client, sample_chat_dataset, make_input
    ):
        created: list[CountingSyntheticTransport] = []

        def transport_factory(config: MCPTransportConfig):
            tool = create_tool(
                name="lookup",
                description="Lookup a value",
                parameters={"query": {"type": "string"}},
                required=["query"],
            )
            transport = CountingSyntheticTransport(
                label=f"{config.server_config.name}-{len(created)}",
                tools={"lookup": tool},
                handlers={
                    "lookup": lambda data,
                    args,
                    label=f"{config.server_config.name}-{len(created)}": (
                        f"{label}:{args['query']}"
                    )
                },
            )
            created.append(transport)
            return transport

        env = _build_env(
            mock_client,
            sample_chat_dataset,
            mcp_servers=[{"name": "search", "command": "dummy"}],
            transport_factory=transport_factory,
        )

        await _setup_state(env, make_input, mock_client)
        assert len(created) == 1

        await created[0].disconnect()
        assert not await created[0].is_connected()

        state = await _setup_state(env, make_input, mock_client)
        assert len(created) == 2
        assert [tool.name for tool in state["tool_defs"]] == ["lookup"]

        tool_message = await env.call_tool("lookup", {"query": "ping"}, "call_3")
        assert tool_message.content == "search-1:ping"

    @pytest.mark.asyncio
    async def test_streaming_http_retry_clears_stale_session(self, monkeypatch):
        transport_module = pytest.importorskip(
            "verifiers.utils.mcp_utils.transports.streaming_http"
        )
        models_module = pytest.importorskip("verifiers.utils.mcp_utils.models")

        backoff_checked = asyncio.Event()
        hold_backoff = asyncio.Event()
        sleep_calls = 0
        real_sleep = asyncio.sleep

        tool = create_tool(
            name="lookup",
            description="Lookup a value",
            parameters={"query": {"type": "string"}},
            required=["query"],
        )

        @asynccontextmanager
        async def fake_streamablehttp_client(url):
            yield object(), object(), None

        class FakeClientSession:
            def __init__(self, read, write):
                self.read = read
                self.write = write

            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb):
                return False

            async def initialize(self):
                return None

            async def list_tools(self):
                return type("ToolsResponse", (), {"tools": [tool]})()

            async def call_tool(self, tool_name, arguments):
                return f"{tool_name}:{arguments['query']}"

        async def fake_sleep(delay):
            nonlocal sleep_calls
            sleep_calls += 1
            if sleep_calls == 1:
                raise RuntimeError("connection dropped")
            if sleep_calls == 2:
                assert transport.session is None
                assert transport.tools == {}
                backoff_checked.set()
                await hold_backoff.wait()
                return
            await real_sleep(0)

        monkeypatch.setattr(
            transport_module,
            "streamablehttp_client",
            fake_streamablehttp_client,
        )
        monkeypatch.setattr(
            transport_module,
            "ClientSession",
            FakeClientSession,
        )
        monkeypatch.setattr(transport_module.asyncio, "sleep", fake_sleep)

        transport = StreamingHTTPTransport(
            models_module.MCPServerConfig(name="remote", url="http://example/mcp"),
            url="http://example/mcp",
            max_retries=2,
        )

        await transport.connect()
        await asyncio.wait_for(backoff_checked.wait(), timeout=1)
        hold_backoff.set()
        await transport.disconnect()

    @pytest.mark.asyncio
    async def test_streaming_http_max_retries_allows_one_retry(self, monkeypatch):
        transport_module = pytest.importorskip(
            "verifiers.utils.mcp_utils.transports.streaming_http"
        )
        models_module = pytest.importorskip("verifiers.utils.mcp_utils.models")

        attempts = 0
        tool = create_tool(
            name="lookup",
            description="Lookup a value",
            parameters={"query": {"type": "string"}},
            required=["query"],
        )

        @asynccontextmanager
        async def fake_streamablehttp_client(url):
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                raise ConnectionError("not ready yet")
            yield object(), object(), None

        class FakeClientSession:
            def __init__(self, read, write):
                self.read = read
                self.write = write

            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb):
                return False

            async def initialize(self):
                return None

            async def list_tools(self):
                return type("ToolsResponse", (), {"tools": [tool]})()

            async def call_tool(self, tool_name, arguments):
                return f"{tool_name}:{arguments['query']}"

        real_sleep = asyncio.sleep

        async def fake_sleep(delay):
            await real_sleep(0)

        monkeypatch.setattr(
            transport_module,
            "streamablehttp_client",
            fake_streamablehttp_client,
        )
        monkeypatch.setattr(
            transport_module,
            "ClientSession",
            FakeClientSession,
        )
        monkeypatch.setattr(transport_module.asyncio, "sleep", fake_sleep)

        transport = StreamingHTTPTransport(
            models_module.MCPServerConfig(name="remote", url="http://example/mcp"),
            url="http://example/mcp",
            max_retries=1,
        )

        assert await transport.connect() == {"lookup": tool}
        assert attempts == 2
        await transport.disconnect()

    @pytest.mark.asyncio
    async def test_streaming_http_reconnect_exhaustion_raises_disconnect_error(
        self, monkeypatch
    ):
        transport_module = pytest.importorskip(
            "verifiers.utils.mcp_utils.transports.streaming_http"
        )
        models_module = pytest.importorskip("verifiers.utils.mcp_utils.models")

        attempts = 0
        sleep_calls = 0
        allow_drop = asyncio.Event()
        real_sleep = asyncio.sleep
        tool = create_tool(
            name="lookup",
            description="Lookup a value",
            parameters={"query": {"type": "string"}},
            required=["query"],
        )

        @asynccontextmanager
        async def fake_streamablehttp_client(url):
            nonlocal attempts
            attempts += 1
            if attempts > 1:
                raise ConnectionError("reconnect failed")
            yield object(), object(), None

        class FakeClientSession:
            def __init__(self, read, write):
                self.read = read
                self.write = write

            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb):
                return False

            async def initialize(self):
                return None

            async def list_tools(self):
                return type("ToolsResponse", (), {"tools": [tool]})()

            async def call_tool(self, tool_name, arguments):
                return f"{tool_name}:{arguments['query']}"

        async def fake_sleep(delay):
            nonlocal sleep_calls
            sleep_calls += 1
            if sleep_calls == 1:
                await allow_drop.wait()
                raise RuntimeError("connection dropped")
            await real_sleep(0)

        monkeypatch.setattr(
            transport_module,
            "streamablehttp_client",
            fake_streamablehttp_client,
        )
        monkeypatch.setattr(
            transport_module,
            "ClientSession",
            FakeClientSession,
        )
        monkeypatch.setattr(transport_module.asyncio, "sleep", fake_sleep)

        transport = StreamingHTTPTransport(
            models_module.MCPServerConfig(name="remote", url="http://example/mcp"),
            url="http://example/mcp",
            max_retries=1,
        )

        assert await transport.connect() == {"lookup": tool}
        allow_drop.set()
        await asyncio.wait_for(transport._connection_task, timeout=1)
        assert attempts == 2
        with pytest.raises(ConnectionError, match="Lost connection to MCP server"):
            await transport.call_tool("lookup", {"query": "ping"})
        await transport.disconnect()

    @pytest.mark.asyncio
    async def test_stop_errors_propagate_through_env_response(
        self, mock_client, sample_chat_dataset, make_input
    ):
        def transport_factory(config: MCPTransportConfig):
            tool = create_tool(
                name="explode",
                description="Raise an MCP stop error",
                parameters={},
                required=[],
            )
            return CountingSyntheticTransport(
                label=config.server_config.name,
                tools={"explode": tool},
                handlers={
                    "explode": lambda data, args: (_ for _ in ()).throw(
                        MCPStopError("boom")
                    )
                },
            )

        env = _build_env(
            mock_client,
            sample_chat_dataset,
            mcp_servers=[{"name": "search", "command": "dummy"}],
            transport_factory=transport_factory,
            stop_errors=[MCPStopError],
        )

        tool_call = ToolCall(id="call_0", name="explode", arguments=json.dumps({}))
        mock_client.add_response(
            messages=[{"role": "user", "content": "Invoke"}],
            response="Using tool",
            tool_calls=[
                {
                    "id": tool_call.id,
                    "function": {
                        "name": tool_call.name,
                        "arguments": tool_call.arguments,
                    },
                }
            ],
        )

        state = await env.rollout(
            input=make_input(
                prompt=[{"role": "user", "content": "Invoke"}],
                answer="",
                task="",
            ),
            client=mock_client,
            model="test-model",
        )

        assert isinstance(state["error"], vf.ToolCallError)
        assert isinstance(state["error"].__cause__, MCPStopError)
        assert state["stop_condition"] == "has_error"
        await env.cleanup()

    @pytest.mark.asyncio
    async def test_sandbox_connect_waits_for_exposed_service(self, monkeypatch):
        sandbox_module = pytest.importorskip(
            "verifiers.utils.mcp_utils.transports.sandbox"
        )
        models_module = pytest.importorskip("verifiers.utils.mcp_utils.models")

        events: list[str] = []

        class RecordingSandboxTransport(sandbox_module.SandboxTransport):
            def __init__(self):
                super().__init__(
                    models_module.MCPServerConfig(name="sandbox", command="dummy"),
                    sandbox_image="python:3.11-slim",
                    sandbox_start_command="tail -f /dev/null",
                    sandbox_environment_vars={},
                    sandbox_cpu_cores=1,
                    sandbox_memory_gb=1,
                    sandbox_disk_size_gb=1,
                    sandbox_timeout_minutes=1,
                    port_to_expose=8000,
                )

            async def create_sandbox(self) -> str:
                events.append("create")
                self.sandbox_id = "sandbox-1"
                return self.sandbox_id

            async def run_setup_commands(self) -> None:
                events.append("setup")

            async def start_mcp_server(self) -> None:
                events.append("start")

            async def expose_port(self) -> str:
                events.append("expose")
                self.url = "https://sandbox.example/mcp"
                return self.url

            async def wait_for_service_ready(self) -> None:
                events.append("ready")

        async def fake_http_connect(self):
            events.append("connect")
            return {}

        monkeypatch.setattr(
            sandbox_module.StreamingHTTPTransport,
            "connect",
            fake_http_connect,
        )

        transport = RecordingSandboxTransport()
        assert await transport.connect() == {}
        assert events == ["create", "setup", "start", "expose", "ready", "connect"]

    @pytest.mark.asyncio
    async def test_sandbox_wait_for_service_ready_polls_exposed_url(self, monkeypatch):
        sandbox_module = pytest.importorskip(
            "verifiers.utils.mcp_utils.transports.sandbox"
        )
        models_module = pytest.importorskip("verifiers.utils.mcp_utils.models")

        class DummyWriter:
            def __init__(self):
                self.closed = False

            def close(self):
                self.closed = True

            def is_closing(self) -> bool:
                return self.closed

            async def wait_closed(self):
                return None

        attempts: list[tuple[str, int, bool]] = []

        async def fake_open_connection(host: str, port: int, ssl: bool):
            attempts.append((host, port, ssl))
            if len(attempts) == 1:
                raise ConnectionError("not ready yet")
            return object(), DummyWriter()

        monkeypatch.setattr(
            sandbox_module.asyncio,
            "open_connection",
            fake_open_connection,
        )
        transport = sandbox_module.SandboxTransport(
            models_module.MCPServerConfig(name="sandbox", command="dummy"),
            sandbox_image="python:3.11-slim",
            sandbox_start_command="tail -f /dev/null",
            sandbox_environment_vars={},
            sandbox_cpu_cores=1,
            sandbox_memory_gb=1,
            sandbox_disk_size_gb=1,
            sandbox_timeout_minutes=1,
            port_to_expose=8000,
        )
        transport.url = "https://sandbox.example/mcp"

        await transport.wait_for_service_ready()

        assert attempts == [
            ("sandbox.example", 443, True),
            ("sandbox.example", 443, True),
        ]

    @pytest.mark.asyncio
    async def test_sandbox_expose_port_uses_tcp_endpoint(self, monkeypatch):
        sandbox_module = pytest.importorskip(
            "verifiers.utils.mcp_utils.transports.sandbox"
        )
        models_module = pytest.importorskip("verifiers.utils.mcp_utils.models")

        class FakeExposure:
            external_endpoint = "sandbox-host.example:12345"
            url = "tcp://sandbox-host.example:12345"

        class FakeClient:
            def __init__(self):
                self.calls = []

            async def expose(self, sandbox_id, port, name=None, protocol="HTTP"):
                self.calls.append(
                    {
                        "sandbox_id": sandbox_id,
                        "port": port,
                        "name": name,
                        "protocol": protocol,
                    }
                )
                return FakeExposure()

        fake_client = FakeClient()
        monkeypatch.setattr(
            sandbox_module.SandboxTransport,
            "get_client",
            classmethod(lambda cls: fake_client),
        )

        transport = sandbox_module.SandboxTransport(
            models_module.MCPServerConfig(name="sandbox", command="dummy"),
            sandbox_image="python:3.11-slim",
            sandbox_start_command="tail -f /dev/null",
            sandbox_environment_vars={},
            sandbox_cpu_cores=1,
            sandbox_memory_gb=1,
            sandbox_disk_size_gb=1,
            sandbox_timeout_minutes=1,
            port_to_expose=8000,
        )
        transport.sandbox_id = "sandbox-1"

        assert await transport.expose_port() == "http://sandbox-host.example:12345/mcp"
        assert transport.url == "http://sandbox-host.example:12345/mcp"
        assert fake_client.calls == [
            {
                "sandbox_id": "sandbox-1",
                "port": 8000,
                "name": None,
                "protocol": "TCP",
            }
        ]

    @pytest.mark.asyncio
    async def test_sandbox_startup_error_reports_cleanup_failure(self):
        sandbox_module = pytest.importorskip(
            "verifiers.utils.mcp_utils.transports.sandbox"
        )
        models_module = pytest.importorskip("verifiers.utils.mcp_utils.models")

        class FailingSandboxTransport(sandbox_module.SandboxTransport):
            def __init__(self):
                super().__init__(
                    models_module.MCPServerConfig(name="sandbox", command="dummy"),
                    sandbox_image="python:3.11-slim",
                    sandbox_start_command="tail -f /dev/null",
                    sandbox_environment_vars={},
                    sandbox_cpu_cores=1,
                    sandbox_memory_gb=1,
                    sandbox_disk_size_gb=1,
                    sandbox_timeout_minutes=1,
                    port_to_expose=8000,
                )

            async def create_sandbox(self) -> str:
                self.sandbox_id = "sandbox-1"
                return self.sandbox_id

            async def run_setup_commands(self) -> None:
                return None

            async def start_mcp_server(self) -> None:
                raise RuntimeError("server failed to start")

            async def read_log_tail(self) -> str | None:
                return "sandbox log tail"

            async def disconnect(self) -> None:
                raise RuntimeError("sandbox cleanup failed")

        transport = FailingSandboxTransport()
        with pytest.raises(RuntimeError) as exc_info:
            await transport.connect()

        message = str(exc_info.value)
        assert "server failed to start" in message
        assert "sandbox log tail" in message
        assert "sandbox cleanup failed" in message
