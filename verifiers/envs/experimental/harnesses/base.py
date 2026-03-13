from __future__ import annotations

import time
from typing import Any

import verifiers as vf
from verifiers.clients import Client
from verifiers.types import Messages, Response, SamplingArgs, State, Tool


class HarnessMonitorRubric(vf.Rubric):
    """Monitor rubric for agent-harness execution state."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_metric(self.agent_timeout)
        self.add_metric(self.agent_error)

    async def agent_timeout(self, state: State) -> float:
        return float(bool(state.get("agent_timed_out")))

    async def agent_error(self, state: State) -> float:
        agent_exit_code = state.get("agent_exit_code")
        if agent_exit_code is None:
            return 0.0
        return float(agent_exit_code != 0)


class Harness:
    """Base class for agent-side harnesses.

    Harnesses own the agent-facing interaction model: how prompts are received,
    how model calls are normalized, when the agent is considered finished, and
    any harness-specific lifecycle work such as starting background jobs.
    """

    def __init__(
        self,
        tools: list[str] | None = None,
        system_prompt: str = "",
        timeout_seconds: float | None = None,
    ):
        self.tools = list(tools or [])
        self.system_prompt = system_prompt
        self.timeout_seconds = timeout_seconds

    def build_monitor_rubric(self) -> vf.Rubric | None:
        return HarnessMonitorRubric()

    async def setup(self, env: Any, state: State) -> None:
        """Initialize harness-specific state before the rollout starts."""
        state.setdefault("agent_completed", False)
        state.setdefault("agent_timed_out", False)

    async def start(self, env: Any, state: State) -> None:
        """Start the harness once any task/environment setup is complete."""

    async def build_env_vars(self, env: Any, state: State) -> dict[str, str]:
        """Return harness-specific sandbox environment variables."""
        return {}

    async def get_prompt_messages(self, env: Any, state: State) -> Messages:
        """Return the next prompt to send to the model."""
        return await env.default_get_prompt_messages(state)

    async def get_model_response(
        self,
        env: Any,
        state: State,
        prompt: Messages | str,
        client: Client | None = None,
        model: str | None = None,
        tool_defs: list[Tool] | None = None,
        sampling_args: SamplingArgs | None = None,
    ) -> Response:
        return await env.request_model_response(
            state=state,
            prompt=prompt,
            client=client,
            model=model,
            tool_defs=tool_defs,
            sampling_args=sampling_args,
        )

    async def add_model_response(
        self,
        env: Any,
        state: State,
        prompt_messages: Messages,
        response: Response,
    ) -> None:
        if not prompt_messages:
            return
        await env.record_model_response(
            state,
            prompt_messages,
            env.normalize_response(response),
        )

    def normalize_tools(
        self,
        env: Any,
        tools: list[Tool] | list[dict[str, Any]] | None,
    ) -> list[Tool] | None:
        return env._normalize_tool_defs(tools)

    def normalize_response(self, env: Any, response: Response) -> Response:
        return response

    async def cleanup(self, env: Any, state: State) -> None:
        """Cleanup per-rollout harness resources."""

    async def teardown(self, env: Any) -> None:
        """Cleanup shared harness resources."""

    async def agent_completed(self, env: Any, state: State) -> bool:
        return state.get("agent_completed", False)

    async def timeout_reached(self, env: Any, state: State) -> bool:
        if self.timeout_seconds is None:
            return False
        timing = state.get("timing")
        if not isinstance(timing, dict) or "start_time" not in timing:
            return False
        return time.time() - float(timing["start_time"]) > self.timeout_seconds
