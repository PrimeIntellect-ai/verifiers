"""
MultiAgentEnv + tool registration with hidden-arg injection.

Adds tool registration and hidden arg injection on top of MultiAgentEnv.
The tool execution loop lives in MultiAgentEnv.rollout() — this class
just provides the env_response() that executes registered tools.

Usage:
    class MyEnv(MultiAgentStatefulToolEnv):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.add_tool(self.my_tool, args_to_skip=["state"])

        def update_tool_args(self, tool_name, tool_args, messages, state):
            return {**tool_args, "state": state}

    task = MyTask(...)  # TaskSet with roles, build_prompt, etc.
    agents = {"solver": Agent(id="solver")}
    env = MyEnv(task=task, agents=agents)
"""

from __future__ import annotations

import inspect
import json
from abc import abstractmethod
from typing import Callable

from verifiers.envs.multiagent_env import MultiAgentEnv
from verifiers.types import Messages, State, ToolMessage
from verifiers.utils.async_utils import maybe_await
from verifiers.utils.tool_utils import convert_func_to_tool_def


# =============================================================================
# Helper Functions
# =============================================================================


def _filter_signature(func: Callable, args_to_skip: list[str]) -> Callable:
    """
    Create wrapper with filtered signature for schema generation.

    Hidden args are removed from the schema shown to the model,
    but the original function still receives them via update_tool_args().
    """
    if not args_to_skip:
        return func

    sig = inspect.signature(func)
    filtered_params = [
        p for name, p in sig.parameters.items()
        if name not in args_to_skip and name != "self"
    ]

    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    wrapper.__name__ = getattr(func, "__name__", "unknown")
    wrapper.__doc__ = getattr(func, "__doc__", None)
    wrapper.__signature__ = sig.replace(parameters=filtered_params)
    wrapper.__annotations__ = {
        k: v for k, v in getattr(func, "__annotations__", {}).items()
        if k not in args_to_skip
    }
    return wrapper


# =============================================================================
# MultiAgentStatefulToolEnv
# =============================================================================


class MultiAgentStatefulToolEnv(MultiAgentEnv):
    """
    MultiAgentEnv with tool registration and hidden-arg injection.

    The tool execution loop lives in MultiAgentEnv.rollout().
    This class adds:
    - Tool registration via add_tool() with optional hidden args
    - Tool execution via env_response() (called by parent's tool loop)
    - Hidden arg injection via update_tool_args()

    Subclasses must implement:
    - update_tool_args(): inject hidden state into tool calls
    """

    def __init__(
        self,
        tools: list[Callable] | None = None,
        max_turns: int = 50,
        **kwargs,
    ):
        # Tool storage — initialize before super().__init__
        self.tools: list[Callable] = []
        self.tool_map: dict[str, Callable] = {}
        self.skipped_args: dict[str, list[str]] = {}

        # Register initial tools
        if tools:
            for tool in tools:
                self._register_tool(tool, [])

        super().__init__(
            tool_defs=self._rebuild_tool_defs() or None,
            max_turns=max_turns,
            **kwargs,
        )

    # =========================================================================
    # Tool Registration
    # =========================================================================

    def _register_tool(self, tool: Callable, args_to_skip: list[str]) -> None:
        """Internal tool registration."""
        self.tools.append(tool)
        tool_name = getattr(tool, "__name__", tool.__class__.__name__)
        self.tool_map[tool_name] = tool
        self.skipped_args[tool_name] = args_to_skip

    def _rebuild_tool_defs(self) -> list:
        """Rebuild tool_defs list from registered tools."""
        defs = []
        for t in self.tools:
            t_name = getattr(t, "__name__", t.__class__.__name__)
            skip = self.skipped_args.get(t_name, [])
            filtered = _filter_signature(t, skip)
            defs.append(convert_func_to_tool_def(filtered))
        return defs

    def add_tool(self, tool: Callable, args_to_skip: list[str] | None = None) -> None:
        """
        Add tool with optional hidden args.

        Hidden args are removed from the schema shown to the model but can be
        injected at call time via update_tool_args().

        Args:
            tool: Callable to register as a tool
            args_to_skip: Parameter names to hide from model (injected via update_tool_args)

        Example:
            self.add_tool(self.python_repl, args_to_skip=["state", "session_id"])
        """
        self._register_tool(tool, args_to_skip or [])
        # Update parent's tool_defs attribute
        self.tool_defs = self._rebuild_tool_defs() or None

    def remove_tool(self, tool: Callable) -> None:
        """Remove a tool from the environment."""
        tool_name = getattr(tool, "__name__", tool.__class__.__name__)

        if tool in self.tools:
            self.tools.remove(tool)
        if tool_name in self.tool_map:
            del self.tool_map[tool_name]
        if tool_name in self.skipped_args:
            del self.skipped_args[tool_name]

        # Update parent's tool_defs attribute
        self.tool_defs = self._rebuild_tool_defs() or None

    # =========================================================================
    # Hidden Arg Injection (subclass must implement)
    # =========================================================================

    @abstractmethod
    def update_tool_args(
        self,
        tool_name: str,
        tool_args: dict,
        messages: Messages,
        state: State,
    ) -> dict:
        """
        Inject hidden state into tool args before execution.

        Subclass must implement this to provide hidden arguments
        that were excluded from the tool schema via args_to_skip.

        Args:
            tool_name: Name of the tool being called
            tool_args: Arguments parsed from model's tool call
            messages: Full message history
            state: Current rollout state

        Returns:
            Updated tool_args dict with hidden args injected

        Example:
            def update_tool_args(self, tool_name, tool_args, messages, state):
                if tool_name == "python_repl":
                    return {**tool_args, "state": state, "session_id": state["session_id"]}
                return tool_args
        """
        pass

    # =========================================================================
    # Tool Execution (env_response override)
    # =========================================================================

    async def call_tool(
        self,
        tool_name: str,
        tool_args: dict,
        tool_call_id: str,
    ) -> ToolMessage:
        """Execute tool and return result message."""
        tool_func = self.tool_map[tool_name]
        result = await maybe_await(tool_func, **tool_args)
        return ToolMessage(
            role="tool",
            content=str(result),
            tool_call_id=tool_call_id,
        )

    async def env_response(
        self, messages: Messages, state: State, **kwargs
    ) -> Messages:
        """
        Execute tool calls from the last assistant message.

        Called by MultiAgentEnv's tool loop when the model makes tool calls.
        Parses tool calls, injects hidden args via update_tool_args(),
        executes each tool, and returns result messages.
        """
        if not messages:
            return []

        last_msg = messages[-1]
        if not hasattr(last_msg, "tool_calls") or not last_msg.tool_calls:
            return []

        tool_messages: list[ToolMessage] = []

        for tool_call in last_msg.tool_calls:
            tool_call_id = tool_call.id

            # Parse tool call
            try:
                tool_name = tool_call.name
                tool_args = json.loads(tool_call.arguments)
            except Exception as e:
                tool_messages.append(ToolMessage(
                    role="tool",
                    content=f"Error parsing tool call: {e}",
                    tool_call_id=tool_call_id,
                ))
                continue

            # Check tool exists
            if tool_name not in self.tool_map:
                tool_messages.append(ToolMessage(
                    role="tool",
                    content=f"Unknown tool: {tool_name}",
                    tool_call_id=tool_call_id,
                ))
                continue

            # Inject hidden args
            tool_args = self.update_tool_args(tool_name, tool_args, messages, state)

            # Execute
            try:
                tool_msg = await self.call_tool(tool_name, tool_args, tool_call_id)
                tool_messages.append(tool_msg)
            except Exception as e:
                tool_messages.append(ToolMessage(
                    role="tool",
                    content=f"Error executing {tool_name}: {e}",
                    tool_call_id=tool_call_id,
                ))

        return tool_messages


# =============================================================================
# Export
# =============================================================================

__all__ = ["MultiAgentStatefulToolEnv"]
