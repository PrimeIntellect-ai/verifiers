"""
Scaffold abstraction for managing how an LLM interacts with tools.

A scaffold wraps an LLM client and defines:
- What tools are available (via MCP or native functions)
- How tool calls are executed
- The tool loop (call LLM, execute tools, repeat until done)

Environments define tasks and rewards; scaffolds define agent capabilities.
"""

import functools
import json
import logging
from dataclasses import dataclass, field
from typing import Any, Callable

from openai import AsyncOpenAI, BadRequestError, NOT_GIVEN
from openai.types.chat import ChatCompletion

import verifiers as vf
from verifiers.types import ChatCompletionToolParam, Messages, SamplingArgs, State

logger = logging.getLogger(__name__)


@dataclass
class ToolCall:
    """Represents a tool call from the model."""
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class ScaffoldResult:
    """Result from scaffold.generate(), includes full message history.

    When pending_tool_calls is set, the scaffold is yielding control back
    to the environment to handle env-owned tools (the action space).
    """
    response: ChatCompletion
    messages: Messages  # full conversation including tool calls
    tool_calls_made: int = 0
    pending_tool_calls: list[ToolCall] | None = None  # env tools needing execution
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def has_pending_tool_calls(self) -> bool:
        """Check if there are env tool calls waiting to be handled."""
        return self.pending_tool_calls is not None and len(self.pending_tool_calls) > 0


class Scaffold:
    """
    Base scaffold - wraps an LLM client with no tools.

    This is the "null" scaffold that just passes through to the LLM.
    Handles all the complexity of making API calls including:
    - Error handling for overlong prompts
    - Interleaved rollouts (pre-tokenized prompts for PRIME-RL)
    - Sampling args normalization
    """

    def __init__(
        self,
        client: AsyncOpenAI,
        model: str,
        sampling_args: SamplingArgs | None = None,
        interleaved_rollouts: bool = False,
        max_seq_len: int | None = None,
        oai_tools: list[ChatCompletionToolParam] | None = None,
    ):
        self.client = client
        self.model = model
        self.sampling_args = sampling_args or {}
        self.interleaved_rollouts = interleaved_rollouts
        self.max_seq_len = max_seq_len
        self.oai_tools = oai_tools
        self.logger = logging.getLogger(self.__class__.__name__)
        self._setup_complete = False

    async def setup(self):
        """Initialize scaffold resources. Override in subclasses. Idempotent."""
        self._setup_complete = True

    async def teardown(self):
        """Clean up scaffold resources. Override in subclasses. Idempotent."""
        self._setup_complete = False

    def _normalize_sampling_args(self, sampling_args: SamplingArgs) -> SamplingArgs:
        """Normalize sampling arguments for the API call."""
        sampling_args = dict(sampling_args)  # copy
        # Rename max_tokens to max_completion_tokens for chat API
        if "max_tokens" in sampling_args:
            if sampling_args["max_tokens"] is None:
                sampling_args.pop("max_tokens")
            else:
                sampling_args["max_completion_tokens"] = sampling_args.pop("max_tokens")
        if (
            "max_completion_tokens" in sampling_args
            and sampling_args["max_completion_tokens"] is None
        ):
            sampling_args.pop("max_completion_tokens")
        # Drop None values
        return {k: v for k, v in sampling_args.items() if v is not None}

    async def _call_api(
        self,
        messages: Messages,
        state: State | None,
        sampling_args: SamplingArgs,
        tools: list[ChatCompletionToolParam] | None = None,
    ) -> ChatCompletion:
        """
        Make the actual API call, handling interleaved rollouts if needed.

        This is the core method that subclasses can rely on for API calls.
        """
        from verifiers.utils.message_utils import strip_nones_from_content

        # Normalize sampling args
        sampling_args = self._normalize_sampling_args(sampling_args)

        # Prepare for interleaved rollouts if enabled
        if self.interleaved_rollouts:
            from verifiers.utils.token_utils import prepare_sampling_args_for_token_prompts
            sampling_args = prepare_sampling_args_for_token_prompts(sampling_args)

        # Check if we should use token-based prompts (interleaved rollouts, not first turn)
        use_tokens = (
            self.interleaved_rollouts
            and state is not None
            and len(state.get("trajectory", [])) > 0
        )

        # Strip None content values
        if isinstance(messages, list):
            messages = strip_nones_from_content(messages)

        if use_tokens:
            # Token-based prompt for interleaved rollouts
            from verifiers.utils.token_utils import get_prompt_ids

            prompt_ids = await get_prompt_ids(state, messages, self.client)
            response = await self._call_api_with_tokens(
                messages, prompt_ids, sampling_args, tools
            )
        else:
            # Standard message-based prompt
            response = await self._call_api_with_messages(
                messages, sampling_args, tools
            )

        return response

    async def _call_api_with_messages(
        self,
        messages: Messages,
        sampling_args: SamplingArgs,
        tools: list[ChatCompletionToolParam] | None = None,
    ) -> ChatCompletion:
        """Make a standard API call with messages (chat) or string prompt (completion)."""
        try:
            # Handle completion mode (string prompt)
            if isinstance(messages, str):
                if tools:
                    raise ValueError("Tools are not supported for completion mode")
                # For completion mode, convert max_completion_tokens back to max_tokens
                completion_args = dict(sampling_args)
                if "max_completion_tokens" in completion_args:
                    completion_args["max_tokens"] = completion_args.pop("max_completion_tokens")
                response = await self.client.completions.create(
                    model=self.model,
                    prompt=messages,
                    **completion_args,
                )
                self._validate_response(response)
                return response

            # Chat mode (list of messages)
            # Detect audio parts and force text-only modality if caller didn't set one
            if "modalities" not in sampling_args:
                has_audio = False
                try:
                    for m in messages:
                        c = m.get("content")  # type: ignore[union-attr]
                        if isinstance(c, list):
                            for p in c:
                                if isinstance(p, dict) and str(p.get("type", "")).startswith("input_audio"):
                                    has_audio = True
                                    break
                        if has_audio:
                            break
                except Exception:
                    has_audio = False
                if has_audio:
                    sampling_args = {**sampling_args, "modalities": ["text"]}

            if tools:
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    tools=tools,
                    **sampling_args,
                )
            else:
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    **sampling_args,
                )
            self._validate_response(response)
            return response
        except BadRequestError as e:
            self._handle_bad_request(e)
            raise  # re-raise if not handled
        except (vf.EmptyModelResponseError, vf.OverlongPromptError, vf.ModelError):
            raise  # Let verifiers errors bubble up unwrapped
        except Exception as e:
            raise vf.ModelError from e

    async def _call_api_with_tokens(
        self,
        messages: Messages,
        prompt_ids: list[int],
        sampling_args: SamplingArgs,
        tools: list[ChatCompletionToolParam] | None = None,
    ) -> ChatCompletion:
        """Make an API call with pre-tokenized prompt (PRIME-RL extension)."""
        try:
            extra_body = sampling_args.pop("extra_body", {})
            body = dict(
                model=self.model,
                messages=messages,
                tools=tools,
                tokens=prompt_ids,
                **sampling_args,
                **extra_body,
            )
            response = await self.client.post(
                "/chat/completions/tokens",
                body=body,
                cast_to=ChatCompletion,
            )
            self._validate_response(response)
            return response
        except BadRequestError as e:
            self._handle_bad_request(e)
            raise
        except (vf.EmptyModelResponseError, vf.OverlongPromptError, vf.ModelError):
            raise  # Let verifiers errors bubble up unwrapped
        except Exception as e:
            raise vf.ModelError from e

    def _handle_bad_request(self, e: BadRequestError):
        """Handle BadRequestError, raising appropriate custom errors."""
        error_text = e.response.text.lower()
        context_length_phrases = [
            "this model's maximum context length is",
            "is longer than the model's context length",
            "exceeds the model's context length",
            "exceed the configured limit",
            "exceeds the configured limit",
            "exceeded model",
        ]
        if any(phrase in error_text for phrase in context_length_phrases):
            self.logger.debug("Caught overlong prompt.")
            raise vf.OverlongPromptError from e
        raise vf.ModelError from e

    def _validate_response(self, response: ChatCompletion):
        """Validate the API response."""
        if response is None:
            raise vf.EmptyModelResponseError from ValueError(
                "Model returned no response"
            )
        if response.choices is None:
            raise vf.EmptyModelResponseError from ValueError(
                "Model returned no response choices"
            )

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
        # Get tools from state if not set on scaffold
        tools = self.oai_tools
        if tools is None and state is not None:
            tools = state.get("oai_tools")

        response = await self._call_api(messages, state, self.sampling_args, tools)

        # Build final messages (original + assistant response)
        # For completion mode (string prompt), messages stays as-is
        if isinstance(messages, str):
            final_messages = messages
        else:
            final_messages = list(messages)
            # For chat mode, append assistant message if present
            if response.choices:
                choice = response.choices[0]
                if hasattr(choice, 'message') and choice.message:
                    final_messages.append(choice.message.model_dump())

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
        """Disconnect from MCP servers. Idempotent."""
        if not self._setup_complete:
            return
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
        - Model responds without tool calls
        - Model calls an env tool (yields back with pending_tool_calls)
        - max_tool_turns is reached
        """
        if not self._setup_complete:
            await self.setup()

        current_messages = list(messages)
        total_tool_calls = 0

        tools = self.oai_tools if self.oai_tools else None

        for turn in range(self.max_tool_turns):
            response = await self._call_api(current_messages, state, self.sampling_args, tools)

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

            tool_calls = self._parse_tool_calls(response)

            # Split into internal (scaffold) vs external (env) tools
            internal_tools = [tc for tc in tool_calls if tc.name in self.tool_map]
            external_tools = [tc for tc in tool_calls if tc.name not in self.tool_map]

            # If there are env tools, yield control back to environment
            if external_tools:
                # First execute any internal tools in this response
                for tc in internal_tools:
                    result = await self.call_tool(tc)
                    tool_message = {
                        "role": "tool",
                        "content": result,
                        "tool_call_id": tc.id,
                    }
                    current_messages.append(tool_message)
                    total_tool_calls += 1
                    self.logger.debug(f"Tool {tc.name}: {result[:100]}...")

                # Yield back with pending env tool calls
                return ScaffoldResult(
                    response=response,
                    messages=current_messages,
                    tool_calls_made=total_tool_calls,
                    pending_tool_calls=external_tools,
                    metadata={"tool_turns": turn + 1, "yielded_for_env_tools": True},
                )

            # All internal tools - execute and continue
            for tc in internal_tools:
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
        """Generate with tool loop using native Python tools.

        Internal (scaffold) tools are executed within this loop.
        External (env) tools cause the scaffold to yield back with
        pending_tool_calls, letting the environment handle them.
        """
        current_messages = list(messages)
        total_tool_calls = 0

        tools = self.oai_tools if self.oai_tools else None

        for turn in range(self.max_tool_turns):
            response = await self._call_api(current_messages, state, self.sampling_args, tools)

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

            # Split into internal (scaffold) vs external (env) tools
            internal_tools = [tc for tc in tool_calls if tc.name in self.tool_map]
            external_tools = [tc for tc in tool_calls if tc.name not in self.tool_map]

            # If there are env tools, yield control back to environment
            if external_tools:
                # First execute any internal tools in this response
                for tc in internal_tools:
                    result = await self.call_tool(tc)
                    tool_message = {
                        "role": "tool",
                        "content": result,
                        "tool_call_id": tc.id,
                    }
                    current_messages.append(tool_message)
                    total_tool_calls += 1

                # Yield back with pending env tool calls
                return ScaffoldResult(
                    response=response,
                    messages=current_messages,
                    tool_calls_made=total_tool_calls,
                    pending_tool_calls=external_tools,
                    metadata={"tool_turns": turn + 1, "yielded_for_env_tools": True},
                )

            # All internal tools - execute and continue
            for tc in internal_tools:
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
