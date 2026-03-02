"""Tests for the experimental MCPEnv foundation."""

import pytest

pytest.importorskip("mcp")

import verifiers as vf
from verifiers.parsers.parser import Parser
from verifiers.rubrics.rubric import Rubric
from verifiers.utils.mcp_utils.models import MCPTransportConfig
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
                    "lookup": lambda data, args, label=f"{config.server_config.name}-{len(created)}": (
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
