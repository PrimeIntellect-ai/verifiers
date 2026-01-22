"""
Scaffold abstraction for managing how an LLM interacts with tools.

A scaffold wraps an LLM client and defines:
- What tools are available (via MCP or native functions)
- How tool calls are executed
- The tool loop (call LLM, execute tools, repeat until done)

Environments define tasks and rewards; scaffolds define agent capabilities.
"""

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Callable

from openai import AsyncOpenAI, NOT_GIVEN
from openai.types.chat import ChatCompletion

from verifiers.types import Messages, SamplingArgs, State

logger = logging.getLogger(__name__)


@dataclass
class ToolCall:
    """Represents a tool call from the model."""
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class ScaffoldResult:
    """Result from scaffold.generate(), includes full message history."""
    response: ChatCompletion
    messages: Messages  # full conversation including tool calls
    tool_calls_made: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


class Scaffold:
    """
    Base scaffold - wraps an LLM client with no tools.

    This is the "null" scaffold that just passes through to the LLM.
    """

    def __init__(
        self,
        client: AsyncOpenAI,
        model: str,
        sampling_args: SamplingArgs | None = None,
    ):
        self.client = client
        self.model = model
        self.sampling_args = sampling_args or {}
        self.logger = logging.getLogger(self.__class__.__name__)

    async def setup(self):
        """Initialize scaffold resources. Override in subclasses."""
        pass

    async def teardown(self):
        """Clean up scaffold resources. Override in subclasses."""
        pass

    async def generate(
        self,
        messages: Messages,
        state: State | None = None,
    ) -> ScaffoldResult:
        """
        Generate a response from the LLM.

        Args:
            messages: The conversation history to send to the model
            state: Optional rollout state for tracking metadata

        Returns:
            ScaffoldResult with response and full message history
        """
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            **self.sampling_args,
        )

        # Build final messages (original + assistant response)
        final_messages = list(messages)
        if response.choices and response.choices[0].message:
            final_messages.append(response.choices[0].message.model_dump())

        return ScaffoldResult(
            response=response,
            messages=final_messages,
            tool_calls_made=0,
        )


class MCPScaffold(Scaffold):
    """
    Scaffold that connects to MCP servers for tool execution.

    Handles the full tool loop:
    1. Call LLM with available tools
    2. If tool calls in response, execute them via MCP
    3. Append results and call LLM again
    4. Repeat until no tool calls or max_tool_turns reached
    """

    def __init__(
        self,
        client: AsyncOpenAI,
        model: str,
        mcp_servers: list[dict] | None = None,
        max_tool_turns: int = 10,
        sampling_args: SamplingArgs | None = None,
        error_formatter: Callable[[Exception], str] = lambda e: f"Error: {e}",
    ):
        super().__init__(client, model, sampling_args)
        self.mcp_server_configs = mcp_servers or []
        self.max_tool_turns = max_tool_turns
        self.error_formatter = error_formatter

        # Will be populated on setup()
        self.oai_tools: list[dict] = []
        self.tool_map: dict[str, Any] = {}  # tool_name -> MCPToolWrapper
        self._connections: dict[str, Any] = {}  # server_name -> connection
        self._setup_complete = False

    async def setup(self):
        """Connect to MCP servers and register tools."""
        if self._setup_complete:
            return

        # Import MCP components here to avoid hard dependency
        try:
            from verifiers.envs.experimental.mcp_env import (
                MCPServerConfig,
                MCPServerConnection,
                MCPToolWrapper,
            )
        except ImportError as e:
            raise ImportError(
                "MCP dependencies not installed. Install with: pip install mcp"
            ) from e

        for server_config in self.mcp_server_configs:
            if isinstance(server_config, dict):
                config = MCPServerConfig(
                    name=server_config["name"],
                    command=server_config["command"],
                    args=server_config.get("args"),
                    env=server_config.get("env"),
                    description=server_config.get("description", ""),
                )
            else:
                config = server_config

            connection = MCPServerConnection(config, self.logger)
            tools = await connection.connect()

            self._connections[config.name] = connection

            for tool in tools.values():
                wrapper = MCPToolWrapper(config.name, tool, connection)
                self.tool_map[wrapper.__name__] = wrapper
                self.oai_tools.append(wrapper.to_oai_tool())
                self.logger.info(f"Registered tool: {wrapper.__name__} from {config.name}")

        self._setup_complete = True

    async def teardown(self):
        """Disconnect from MCP servers."""
        for connection in self._connections.values():
            await connection.disconnect()
        self._connections.clear()
        self.tool_map.clear()
        self.oai_tools.clear()
        self._setup_complete = False

    def _has_tool_calls(self, response: ChatCompletion) -> bool:
        """Check if response contains tool calls."""
        if not response.choices:
            return False
        message = response.choices[0].message
        return bool(message.tool_calls)

    def _parse_tool_calls(self, response: ChatCompletion) -> list[ToolCall]:
        """Extract tool calls from response."""
        if not response.choices:
            return []
        message = response.choices[0].message
        if not message.tool_calls:
            return []

        tool_calls = []
        for tc in message.tool_calls:
            try:
                args = json.loads(tc.function.arguments)
            except json.JSONDecodeError:
                args = {}
            tool_calls.append(ToolCall(
                id=tc.id,
                name=tc.function.name,
                arguments=args,
            ))
        return tool_calls

    async def call_tool(self, tool_call: ToolCall) -> str:
        """Execute a single tool call via MCP."""
        tool_name = tool_call.name

        if tool_name not in self.tool_map:
            return f"Error: Tool '{tool_name}' not found"

        try:
            wrapper = self.tool_map[tool_name]
            result = await wrapper(**tool_call.arguments)
            return str(result)
        except Exception as e:
            return self.error_formatter(e)

    async def generate(
        self,
        messages: Messages,
        state: State | None = None,
    ) -> ScaffoldResult:
        """
        Generate with tool loop.

        Calls LLM, executes any tool calls, and repeats until:
        - Model responds without tool calls, or
        - max_tool_turns is reached
        """
        if not self._setup_complete:
            await self.setup()

        current_messages = list(messages)
        total_tool_calls = 0

        tools_arg = self.oai_tools if self.oai_tools else NOT_GIVEN

        for turn in range(self.max_tool_turns):
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=current_messages,
                tools=tools_arg,
                **self.sampling_args,
            )

            # Append assistant message
            if response.choices and response.choices[0].message:
                assistant_msg = response.choices[0].message.model_dump()
                current_messages.append(assistant_msg)

            # Check for tool calls
            if not self._has_tool_calls(response):
                # Done - no more tool calls
                return ScaffoldResult(
                    response=response,
                    messages=current_messages,
                    tool_calls_made=total_tool_calls,
                    metadata={"tool_turns": turn + 1},
                )

            # Execute tool calls
            tool_calls = self._parse_tool_calls(response)
            for tc in tool_calls:
                result = await self.call_tool(tc)
                tool_message = {
                    "role": "tool",
                    "content": result,
                    "tool_call_id": tc.id,
                }
                current_messages.append(tool_message)
                total_tool_calls += 1
                self.logger.debug(f"Tool {tc.name}: {result[:100]}...")

        # Hit max turns
        self.logger.warning(f"Hit max_tool_turns ({self.max_tool_turns})")
        return ScaffoldResult(
            response=response,
            messages=current_messages,
            tool_calls_made=total_tool_calls,
            metadata={"tool_turns": self.max_tool_turns, "max_turns_hit": True},
        )


class ToolScaffold(Scaffold):
    """
    Scaffold with native Python tools (no MCP).

    Useful for simple tools that don't need external infrastructure.
    """

    def __init__(
        self,
        client: AsyncOpenAI,
        model: str,
        tools: list[Callable] | None = None,
        max_tool_turns: int = 10,
        sampling_args: SamplingArgs | None = None,
        error_formatter: Callable[[Exception], str] = lambda e: f"Error: {e}",
    ):
        super().__init__(client, model, sampling_args)
        self.tools = tools or []
        self.max_tool_turns = max_tool_turns
        self.error_formatter = error_formatter

        # Build tool map and schemas
        from verifiers.utils.tool_utils import convert_func_to_oai_tool

        self.tool_map = {
            getattr(t, "__name__", t.__class__.__name__): t
            for t in self.tools
        }
        self.oai_tools = [convert_func_to_oai_tool(t) for t in self.tools]

    def _has_tool_calls(self, response: ChatCompletion) -> bool:
        if not response.choices:
            return False
        message = response.choices[0].message
        return bool(message.tool_calls)

    def _parse_tool_calls(self, response: ChatCompletion) -> list[ToolCall]:
        if not response.choices:
            return []
        message = response.choices[0].message
        if not message.tool_calls:
            return []

        tool_calls = []
        for tc in message.tool_calls:
            try:
                args = json.loads(tc.function.arguments)
            except json.JSONDecodeError:
                args = {}
            tool_calls.append(ToolCall(
                id=tc.id,
                name=tc.function.name,
                arguments=args,
            ))
        return tool_calls

    async def call_tool(self, tool_call: ToolCall) -> str:
        """Execute a native Python tool."""
        from verifiers.utils.async_utils import maybe_await

        tool_name = tool_call.name
        if tool_name not in self.tool_map:
            return f"Error: Tool '{tool_name}' not found"

        try:
            tool_func = self.tool_map[tool_name]
            result = await maybe_await(tool_func, **tool_call.arguments)
            return str(result)
        except Exception as e:
            return self.error_formatter(e)

    async def generate(
        self,
        messages: Messages,
        state: State | None = None,
    ) -> ScaffoldResult:
        """Generate with tool loop using native Python tools."""
        current_messages = list(messages)
        total_tool_calls = 0

        tools_arg = self.oai_tools if self.oai_tools else NOT_GIVEN

        for turn in range(self.max_tool_turns):
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=current_messages,
                tools=tools_arg,
                **self.sampling_args,
            )

            if response.choices and response.choices[0].message:
                assistant_msg = response.choices[0].message.model_dump()
                current_messages.append(assistant_msg)

            if not self._has_tool_calls(response):
                return ScaffoldResult(
                    response=response,
                    messages=current_messages,
                    tool_calls_made=total_tool_calls,
                    metadata={"tool_turns": turn + 1},
                )

            tool_calls = self._parse_tool_calls(response)
            for tc in tool_calls:
                result = await self.call_tool(tc)
                tool_message = {
                    "role": "tool",
                    "content": result,
                    "tool_call_id": tc.id,
                }
                current_messages.append(tool_message)
                total_tool_calls += 1

        return ScaffoldResult(
            response=response,
            messages=current_messages,
            tool_calls_made=total_tool_calls,
            metadata={"tool_turns": self.max_tool_turns, "max_turns_hit": True},
        )
