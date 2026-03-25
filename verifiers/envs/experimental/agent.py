"""Agent protocol — HOW one entity solves a task.

An Agent owns tools, context management, and the execution loop.
It receives a prompt and a state (which carries the LLM client, model,
and sampling args) and produces a list of trajectory steps.

Concrete implementations:

* ``SingleTurnAgent`` — one-shot prompt → response (e.g. user simulator).
* ``ReActAgent`` — tool-calling loop (outside the sandbox).  Formerly ``LLMAgent``.
"""

from __future__ import annotations

import inspect
import json
import logging
import uuid
from typing import Any, Callable, Protocol, runtime_checkable

from verifiers.types import (
    AssistantMessage,
    Messages,
    Response,
    State,
    SystemMessage,
    Tool,
    ToolCall,
    ToolMessage,
    TrajectoryStep,
)
from verifiers.utils.async_utils import maybe_await
from verifiers.utils.message_utils import concat_messages
from verifiers.utils.response_utils import parse_response_message
from verifiers.utils.tool_utils import convert_func_to_tool_def

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class Agent(Protocol):
    """Protocol describing HOW one entity solves a task.

    ``setup()`` is called once per rollout to bind sandbox-dependent state
    (upload scripts, set working directories, etc.).

    ``run()`` drives the agent loop and returns the trajectory steps it
    produced.  The *state* dict carries the LLM ``client``, ``model``, and
    ``sampling_args`` so the agent can make inference calls without a
    back-reference to the environment.
    """

    async def setup(
        self,
        sandbox_client: Any,
        sandbox_id: str,
        state: State,
    ) -> None:
        """One-time per-rollout setup (upload scripts, bind sandbox, etc.)."""
        ...

    async def run(
        self,
        prompt: Messages,
        state: State,
    ) -> list[TrajectoryStep]:
        """Execute the agent loop and return trajectory steps."""
        ...


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _filter_signature(func: Callable, args_to_skip: list[str]) -> Callable:
    """Return a wrapper whose signature hides *args_to_skip*.

    Same pattern as ``StatefulToolEnv.filter_signature`` — the wrapper is
    only used for schema generation; the original callable is invoked at
    call time so skipped args can still be injected.
    """
    if not args_to_skip:
        return func
    sig = inspect.signature(func)
    filtered_sig = sig.replace(
        parameters=[
            p
            for n, p in sig.parameters.items()
            if n not in args_to_skip and n != "self"
        ]
    )
    filtered_annotations = {
        k: v
        for k, v in getattr(func, "__annotations__", {}).items()
        if k not in args_to_skip
    }

    def wrapper(*args: Any, **kwargs: Any) -> Any:
        return func(*args, **kwargs)

    wrapper.__name__ = func.__name__
    wrapper.__doc__ = func.__doc__
    wrapper.__signature__ = filtered_sig  # type: ignore[attr-defined]
    wrapper.__annotations__ = filtered_annotations
    return wrapper


async def _get_response(
    state: State,
    prompt: Messages,
    tool_defs: list[Tool] | None = None,
) -> Response:
    """Call the LLM through the client stored in *state*."""
    client = state["client"]
    model = state["model"]
    sampling_args = state.get("sampling_args") or {}
    return await client.get_response(
        prompt=prompt,
        model=model,
        tools=tool_defs,
        sampling_args=sampling_args,
        state=state,
    )


def _make_trajectory_step(
    prompt: Messages,
    completion: Messages,
    response: Response,
    trajectory_id: str,
    extras: dict[str, Any] | None = None,
) -> TrajectoryStep:
    """Build a ``TrajectoryStep`` from a prompt/response pair."""
    is_truncated = (response.message.is_truncated or False) if response.message else False
    return TrajectoryStep(
        prompt=prompt,
        completion=completion,
        response=response,
        tokens=None,  # token-level data filled by trainer if needed
        reward=None,
        advantage=None,
        is_truncated=is_truncated,
        trajectory_id=trajectory_id,
        extras=extras or {},
    )


# ---------------------------------------------------------------------------
# SingleTurnAgent
# ---------------------------------------------------------------------------


class SingleTurnAgent:
    """One-shot agent: prompt in → response out.  No tools, no loop.

    Useful as a user-simulator, judge, or any role that just responds once.
    Accepts an optional *system_prompt* that is prepended to every call.
    """

    def __init__(
        self,
        system_prompt: str | None = None,
        agent_id: str = "single_turn",
    ):
        self.system_prompt = system_prompt
        self.agent_id = agent_id

    async def setup(
        self,
        sandbox_client: Any,
        sandbox_id: str,
        state: State,
    ) -> None:
        pass  # nothing to set up

    async def run(
        self,
        prompt: Messages,
        state: State,
    ) -> list[TrajectoryStep]:
        messages: Messages = list(prompt)
        if self.system_prompt:
            messages = [SystemMessage(content=self.system_prompt)] + messages

        response = await _get_response(state, messages)
        completion = await parse_response_message(response)
        trajectory_id = uuid.uuid4().hex

        step = _make_trajectory_step(
            prompt=messages,
            completion=completion,
            response=response,
            trajectory_id=trajectory_id,
            extras={"agent_id": self.agent_id},
        )
        return [step]


# ---------------------------------------------------------------------------
# LLMAgent
# ---------------------------------------------------------------------------


class ReActAgent:
    """Tool-calling agent that runs *outside* the sandbox.

    The LLM runs in the verifiers process; tool calls are dispatched by
    calling ``sandbox_client.execute_command()`` (or any callable) and the
    results are fed back into the next prompt.  This is the pattern used by
    mini_swe_agent_plus, lean_code, etc.

    Tools are plain callables whose signatures are inspected to build
    JSON-schema tool definitions (same as ``ToolEnv``).  Tools that need
    sandbox access should accept ``sandbox_client``/``sandbox_id``/``state``
    params and use ``args_to_skip`` to hide them from the schema.
    """

    def __init__(
        self,
        tools: list[Callable] | None = None,
        max_turns: int = 200,
        system_prompt: str | None = None,
        agent_id: str = "react_agent",
        error_formatter: Callable[[Exception], str] = lambda e: str(e),
    ):
        self.max_turns = max_turns
        self.system_prompt = system_prompt
        self.agent_id = agent_id
        self.error_formatter = error_formatter

        self._tools: list[Callable] = []
        self._tool_defs: list[Tool] = []
        self._tool_map: dict[str, Callable] = {}
        self._skipped_args: dict[str, list[str]] = {}

        for t in (tools or []):
            self.add_tool(t)

        # Populated during setup()
        self._sandbox_client: Any = None
        self._sandbox_id: str | None = None

        # Populated per-rollout via inject_tool_args()
        self._injected_args: dict[str, Any] = {}

    # -- tool management ----------------------------------------------------

    def add_tool(self, tool: Callable, args_to_skip: list[str] | None = None) -> None:
        """Register a tool, optionally hiding args from the schema.

        Skipped args are removed from the JSON schema shown to the model but
        the original callable is stored, so hidden args can be injected at
        call time (e.g. ``state``, ``sandbox_client``).
        """
        name = getattr(tool, "__name__", tool.__class__.__name__)
        schema_func = _filter_signature(tool, args_to_skip or [])
        tool_def = convert_func_to_tool_def(schema_func)
        # Clean up $defs references for skipped args
        params = tool_def.parameters
        for arg in args_to_skip or []:
            if "properties" in params and isinstance(params["properties"], dict):
                params["properties"].pop(arg, None)
            if "required" in params and isinstance(params["required"], list):
                if arg in params["required"]:
                    params["required"].remove(arg)
        if "$defs" in params and not params.get("properties"):
            params.pop("$defs", None)

        self._tools.append(tool)
        self._tool_defs.append(tool_def)
        self._tool_map[name] = tool
        self._skipped_args[name] = args_to_skip or []

    def remove_tool(self, name: str) -> None:
        """Remove a previously registered tool by name."""
        if name not in self._tool_map:
            return
        tool = self._tool_map.pop(name)
        self._tools.remove(tool)
        self._tool_defs = [td for td in self._tool_defs if td.name != name]

    @property
    def tool_defs(self) -> list[Tool] | None:
        return self._tool_defs if self._tool_defs else None

    def inject_tool_args(self, **kwargs: Any) -> None:
        """Set args to inject into all tool calls at dispatch time.

        Use this to pass ``state``, ``sandbox_client``, etc. to tools
        whose signatures were hidden via ``args_to_skip``.  Called by
        ``ComposableEnv`` before ``run()``.
        """
        self._injected_args.update(kwargs)

    # -- lifecycle ----------------------------------------------------------

    async def setup(
        self,
        sandbox_client: Any,
        sandbox_id: str,
        state: State,
    ) -> None:
        """Store sandbox references so tools added later can use them."""
        self._sandbox_client = sandbox_client
        self._sandbox_id = sandbox_id

    # -- tool dispatch ------------------------------------------------------

    async def _call_tool(
        self,
        tool_call: ToolCall,
    ) -> ToolMessage:
        """Dispatch a single tool call, returning a ``ToolMessage``."""
        tool_name = tool_call.name
        tool_call_id = tool_call.id

        try:
            tool_args: dict = json.loads(tool_call.arguments)
        except Exception as e:
            return ToolMessage(
                role="tool",
                content=self.error_formatter(e),
                tool_call_id=tool_call_id,
            )

        tool_func = self._tool_map.get(tool_name)
        if tool_func is None:
            return ToolMessage(
                role="tool",
                content=f"Unknown tool: {tool_name}",
                tool_call_id=tool_call_id,
            )

        # Inject hidden args (state, sandbox_client, etc.)
        skipped = self._skipped_args.get(tool_name, [])
        for arg_name in skipped:
            if arg_name in self._injected_args and arg_name not in tool_args:
                tool_args[arg_name] = self._injected_args[arg_name]

        try:
            result = await maybe_await(tool_func, **tool_args)
            content = str(result)
        except Exception as e:
            content = self.error_formatter(e)

        return ToolMessage(
            role="tool",
            content=content,
            tool_call_id=tool_call_id,
        )

    # -- main loop ----------------------------------------------------------

    async def run(
        self,
        prompt: Messages,
        state: State,
    ) -> list[TrajectoryStep]:
        """Run the tool-calling loop until the model stops or max_turns."""
        trajectory_id = uuid.uuid4().hex
        steps: list[TrajectoryStep] = []
        messages: Messages = list(prompt)

        first_role = getattr(messages[0], "role", None) if messages else None
        if self.system_prompt and first_role != "system":
            messages = [SystemMessage(content=self.system_prompt)] + messages

        for turn in range(self.max_turns):
            response = await _get_response(state, messages, tool_defs=self.tool_defs)
            completion = await parse_response_message(response)

            step = _make_trajectory_step(
                prompt=messages,
                completion=completion,
                response=response,
                trajectory_id=trajectory_id,
                extras={"agent_id": self.agent_id, "turn": turn},
            )
            steps.append(step)

            # Check if model wants to call tools
            assistant_msg = completion[-1] if completion else None
            has_tool_calls = (
                isinstance(assistant_msg, AssistantMessage)
                and assistant_msg.tool_calls
                and len(assistant_msg.tool_calls) > 0
            )

            if not has_tool_calls:
                break  # model is done

            # Dispatch tool calls
            tool_messages: Messages = []
            for tc in assistant_msg.tool_calls:  # type: ignore[union-attr]
                if isinstance(tc, ToolCall):
                    tool_msg = await self._call_tool(tc)
                    tool_messages.append(tool_msg)

            # Build next prompt: full history + tool results
            messages = concat_messages([messages, completion, tool_messages])

        return steps


# Backwards-compatible alias
LLMAgent = ReActAgent
