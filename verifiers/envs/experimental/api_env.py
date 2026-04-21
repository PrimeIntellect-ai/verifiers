import asyncio
import logging
import os
import time
import uuid
from collections import Counter
from typing import Any, Callable, cast

import verifiers as vf
from verifiers.clients import Client
from verifiers.types import (
    AssistantMessage,
    Messages,
    MessageType,
    Response,
    SamplingArgs,
    State,
    Tool,
    ToolCall,
)
from verifiers.utils.interception_utils import (
    InterceptionServer,
    deliver_response,
    synthesize_stream,
)
from verifiers.utils.logging_utils import print_time, truncate
from verifiers.utils.message_utils import normalize_messages
from verifiers.utils.serve_utils import get_free_port

from prime_tunnel import Tunnel

logger = logging.getLogger(__name__)


class ApiEnvMonitorRubric(vf.Rubric):
    """Monitor rubric that tracks ApiEnv execution state."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_metric(self.agent_timeout)

    async def agent_timeout(self, state: vf.State) -> float:
        """Whether the agent timed out."""
        return float(bool(state.get("agent_timed_out")))


class ApiEnv(vf.MultiTurnEnv):
    """
    Base environment for running agent code that makes API calls through an
    interception proxy. All LLM calls are intercepted, forwarded to the real
    model, and recorded through the verifiers pipeline.
    """

    def __init__(
        self,
        agent_fn: Callable[[str, State], Any] | None = None,
        *,
        interception_port: int | None = None,
        interception_url: str | None = None,
        use_tunnel: bool = False,
        max_turns: int = -1,
        timeout_seconds: float = 3600.0,
        poll_interval: float = 1.0,
        **kwargs,
    ):
        super().__init__(max_turns=max_turns, message_type="chat", **kwargs)
        self.agent_fn = agent_fn
        self.poll_interval = poll_interval
        self.timeout_seconds = timeout_seconds
        self.use_tunnel = use_tunnel

        interception_port = (
            get_free_port() if interception_port is None else interception_port
        )
        self.init_interception(interception_port, interception_url)
        self.add_rubric(ApiEnvMonitorRubric())

    TUNNEL_CHECK_INTERVAL = 60.0  # seconds between server-side liveness checks

    def init_interception(
        self,
        interception_port: int = 8765,
        interception_url: str | None = None,
    ):
        """Initialize interception server and optional tunnel resources."""
        self.interception_port = interception_port
        self.interception_url = interception_url
        self._tunnel: Tunnel | None = None
        self._tunnel_lock = asyncio.Lock()
        self._tunnel_last_checked: float = 0.0
        self._interception_server = InterceptionServer(
            port=interception_port, secret=os.environ.get("INTERCEPTION_SECRET")
        )

    def _require_interception_server(self) -> InterceptionServer:
        if self._interception_server is None:
            raise RuntimeError("Interception server is not initialized.")
        return self._interception_server

    async def get_tunnel_url(self) -> str:
        """Get tunnel URL, starting the tunnel if needed. Recreates dead tunnels."""
        async with self._tunnel_lock:
            if self._tunnel is not None and not self._tunnel.is_running:
                frpc_output = "\n".join(self._tunnel.recent_output)
                self.logger.warning(
                    f"Tunnel process died, recreating. frpc output:\n{frpc_output}"
                )
                self._tunnel.sync_stop()
                self._tunnel = None

            if self._tunnel is not None:
                now = time.time()
                if now - self._tunnel_last_checked > self.TUNNEL_CHECK_INTERVAL:
                    self._tunnel_last_checked = now
                    try:
                        registered = await self._tunnel.check_registered()
                        if not registered:
                            self.logger.warning(
                                "Tunnel registration expired server-side, recreating."
                            )
                            self._tunnel.sync_stop()
                            self._tunnel = None
                    except Exception as e:
                        self.logger.warning(
                            f"Tunnel health check failed (will retry): {e}"
                        )

            if self._tunnel is None:
                interception_server = self._require_interception_server()
                port = interception_server.port
                if self.logger.isEnabledFor(logging.DEBUG):
                    self._tunnel = Tunnel(local_port=port, log_level="debug")
                else:
                    self._tunnel = Tunnel(local_port=port)
                url = await self._tunnel.start()
                self._tunnel_last_checked = time.time()
                self.logger.debug(f"Prime Tunnel started: {url}")
                return url
            else:
                assert self._tunnel.url is not None, "Tunnel started but URL is None"
                return self._tunnel.url

    async def compute_base_url(self, state: State, rollout_id: str) -> str:
        """Compute the interception proxy base URL for this rollout.

        Override in subclasses to change URL routing (e.g. always use tunnel).
        """
        interception_server = self._require_interception_server()
        if self.interception_url is not None:
            return f"{self.interception_url.rstrip('/')}/rollout/{rollout_id}/v1"
        elif self.use_tunnel:
            tunnel_url = await self.get_tunnel_url()
            return f"{tunnel_url}/rollout/{rollout_id}/v1"
        else:
            return (
                f"http://localhost:{interception_server.port}/rollout/{rollout_id}/v1"
            )

    async def setup_state(self, state: State) -> State:
        """Start interception, register rollout, launch agent."""
        state = await super().setup_state(state)

        rollout_id = f"rollout_{uuid.uuid4().hex[:8]}"
        state["rollout_id"] = rollout_id

        interception_server = self._require_interception_server()
        await interception_server.start()

        state["interception_base_url"] = await self.compute_base_url(state, rollout_id)

        # Pass state so the server can surface stream-interruption errors
        # (e.g. tunnel dies mid-SSE) back onto the rollout; without this the
        # agent sees a truncated stream and often exits with code 0 and an
        # empty trajectory.
        request_id_queue = interception_server.register_rollout(
            rollout_id, state=state
        )
        state["request_id_queue"] = request_id_queue
        state["agent_completed"] = False

        await self.launch_agent(state)

        self.logger.info(
            f"Started  rollout_id={rollout_id} | example_id={state['example_id']}"
        )
        return state

    async def launch_agent(self, state: State) -> None:
        """Start the agent. Override in subclasses for different execution models."""
        if self.agent_fn is None:
            raise vf.InfraError(
                "ApiEnv requires agent_fn to be set, or launch_agent to be overridden."
            )
        base_url = state["interception_base_url"]
        state["agent_task"] = asyncio.create_task(self._run_agent_fn(state, base_url))
        state["agent_start_time"] = time.time()

    async def _run_agent_fn(self, state: State, base_url: str) -> None:
        """Execute the user's agent_fn, handling sync/async and errors."""
        assert self.agent_fn is not None
        try:
            if asyncio.iscoroutinefunction(self.agent_fn):
                result = await self.agent_fn(base_url, state)
            else:
                result = await asyncio.to_thread(self.agent_fn, base_url, state)
            if asyncio.iscoroutine(result):
                result = await result
            state["agent_result"] = result
        except asyncio.CancelledError:
            self.logger.debug("Agent task cancelled")
            raise
        except Exception as e:
            state["agent_error"] = str(e)
            state["error"] = vf.InfraError(f"Agent function failed: {e}")
            self.logger.warning(f"Agent function raised: {type(e).__name__}: {e}")
        finally:
            state["agent_completed"] = True

    async def check_agent_completed(self, state: State) -> bool:
        """Check if the agent has completed. Override for custom completion logic."""
        return state.get("agent_completed", False)

    async def cleanup_agent(self, state: State) -> None:
        """Clean up agent-specific resources."""
        agent_task = state.get("agent_task")
        if agent_task and not agent_task.done():
            agent_task.cancel()
            try:
                await agent_task
            except asyncio.CancelledError:
                pass

    def normalize_intercepted_tools(self, intercept_tools: object) -> list[Tool] | None:
        """Normalize intercepted request tools for the provider-agnostic runtime.

        Assumes that agent requests arrives in OpenAI-tool format.
        TODO: Support other tool formats.
        """
        if not isinstance(intercept_tools, list):
            raise TypeError("Intercepted tools must be provided as a list.")

        normalized: list[Tool] = []
        for raw_tool in intercept_tools:
            if isinstance(raw_tool, Tool):
                normalized.append(raw_tool)
                continue
            if not isinstance(raw_tool, dict):
                raise TypeError(
                    "Intercepted tools must be vf.Tool objects or dict tool definitions."
                )
            raw_tool_dict = cast(dict[str, Any], raw_tool)

            function_payload = raw_tool_dict.get("function")
            if raw_tool_dict.get("type") == "function" and isinstance(
                function_payload, dict
            ):
                parameters = function_payload.get("parameters", {})
                if not isinstance(parameters, dict):
                    raise TypeError(
                        "Intercepted function tool parameters must be a JSON object."
                    )
                normalized.append(
                    Tool(
                        name=function_payload.get("name", ""),
                        description=function_payload.get("description", ""),
                        parameters=parameters,
                        strict=function_payload.get("strict"),
                    )
                )
                continue

            normalized.append(Tool.model_validate(raw_tool_dict))

        return normalized

    async def normalize_intercepted_messages(
        self, intercepted_messages: object
    ) -> Messages:
        """Normalize messages received from the agent before model inference."""
        return await asyncio.to_thread(normalize_messages, intercepted_messages)  # type: ignore

    async def normalize_response(self, response: Response) -> Response:
        """Hook to normalize the model response before it is stored in the trajectory.

        Override in subclasses to align the stored step format with the agent's
        own message history conventions, enabling TITO prefix cache hits.
        """
        return response

    async def _poll_next_request(self, state: State) -> str | None:
        """Poll for the next intercepted request, checking liveness in between.

        Returns a request_id when a request arrives, or None when the agent
        has completed or the rollout has timed out.
        """
        request_id_queue = state["request_id_queue"]
        while True:
            try:
                return await asyncio.wait_for(
                    request_id_queue.get(), timeout=self.poll_interval
                )
            except asyncio.TimeoutError:
                if self._tunnel is not None and not self._tunnel.is_running:
                    frpc_output = "\n".join(self._tunnel.recent_output)
                    raise vf.TunnelError(
                        f"Tunnel process died during rollout. "
                        f"frpc output:\n{frpc_output}"
                    )
                if await self.check_agent_completed(state):
                    state["agent_completed"] = True
                    return None
                if time.time() - state["timing"]["start_time"] > self.timeout_seconds:
                    state["agent_timed_out"] = True
                    return None

    async def get_prompt_messages(self, state: State) -> Messages:
        """Wait for agent to make an API request OR agent completion."""
        interception_server = self._require_interception_server()

        request_id = await self._poll_next_request(state)
        if request_id is None:
            return []

        state["current_request_id"] = request_id
        intercept = interception_server.intercepts[request_id]
        return await self.normalize_intercepted_messages(intercept["messages"])

    async def get_model_response(
        self,
        state: State,
        prompt: Messages,
        client: Client | None = None,
        model: str | None = None,
        tool_defs: list[Tool] | None = None,
        sampling_args: SamplingArgs | None = None,
        message_type: MessageType | None = None,
    ) -> Response:
        """Get model response and unblock the waiting HTTP handler."""
        # Handle agent completion case (empty prompt)
        if not prompt:
            resolved_model = model or state["model"]
            return Response(
                id="agent-completed",
                created=int(time.time()),
                model=resolved_model,
                usage=None,
                message=vf.ResponseMessage(
                    content="",
                    reasoning_content=None,
                    tool_calls=None,
                    finish_reason="stop",
                    is_truncated=False,
                    tokens=None,
                ),
            )

        request_id = state.get("current_request_id")
        intercept = None
        if request_id:
            intercept = self._require_interception_server().intercepts.get(request_id)

        if intercept:
            # Always use the configured model from state, not the intercepted model
            model = state.get("model") or model
            intercept_tools = intercept.get("tools")
            if intercept_tools:

                def _tool_name(t: object) -> str:
                    if isinstance(t, Tool):
                        return t.name
                    if isinstance(t, dict):
                        td = cast(dict[str, Any], t)
                        fn = td.get("function") or {}
                        return fn.get("name", "")
                    return ""

                cache_key = tuple(sorted(_tool_name(t) for t in intercept_tools))
                cached_key, cached_defs = state.get("_cached_tool_defs", (None, None))
                if cached_key == cache_key and cached_defs is not None:
                    tool_defs = cached_defs
                else:
                    tool_defs = (
                        self.normalize_intercepted_tools(intercept_tools) or tool_defs
                    )
                    state["_cached_tool_defs"] = (cache_key, tool_defs)

        response: Response | None = None
        error: BaseException | None = None

        try:
            response = await super().get_model_response(
                state=state,
                prompt=prompt,
                client=client,
                model=model,
                tool_defs=tool_defs,
                sampling_args=sampling_args,
            )
        except BaseException as e:
            error = e
            raise
        finally:
            # Always unblock HTTP handler, even on exception
            if intercept:
                if intercept.get("stream"):
                    await synthesize_stream(intercept, response, error)
                else:
                    deliver_response(intercept, response, error)
                state["current_request_id"] = None

        assert response is not None
        return response

    async def add_model_response(
        self,
        state: State,
        prompt_messages: Messages,
        response: Response,
    ):
        """Add model response and update top-level prompt on first turn."""
        # Skip adding empty "agent completed" step
        if not prompt_messages:
            return
        # On first turn, update state["prompt"] to match the agent's actual prompt
        if len(state["trajectory"]) == 0:
            state["prompt"] = prompt_messages
        await super().add_model_response(
            state, prompt_messages, await self.normalize_response(response)
        )

    async def env_response(
        self, messages: Messages, state: State, **kwargs
    ) -> Messages:
        """No environment response -- the agent drives the conversation."""
        return []

    async def post_rollout(self, state: State):
        """Log rollout summary after completion."""
        tool_counts: Counter[str] = Counter()
        for step in state.get("trajectory", []):
            for msg in step.get("completion", []):
                if isinstance(msg, AssistantMessage) and isinstance(
                    msg.tool_calls, list
                ):
                    for tc in msg.tool_calls:
                        if isinstance(tc, ToolCall):
                            tool_counts[tc.name] += 1

        example_id = state.get("example_id")
        num_turns = len(state.get("trajectory", []))
        stop_condition = state.get("stop_condition", "unknown")
        error = state.get("error")
        error_info = (
            f"{type(error).__name__}: {truncate(str(error), 80)}" if error else None
        )
        agent_error = state.get("agent_error")
        exit_code = state.get("agent_exit_code")
        timed_out = state.get("agent_timed_out", False)
        duration_s = state["timing"].get("total_ms", 0) / 1000
        tools_str = ",".join(f"{k}:{v}" for k, v in tool_counts.most_common())
        parts = [
            f"Finished rollout_id={state.get('rollout_id')}",
            f"example_id={example_id}",
            f"turns={num_turns}",
            f"tools=[{tools_str}]",
            f"stop={stop_condition}",
            f"duration={print_time(duration_s)}",
        ]
        if exit_code is not None:
            parts.append(f"exit_code={exit_code}")
        if timed_out:
            parts.append("timed_out=True")
        if agent_error:
            parts.append(f"agent_error={truncate(str(agent_error), 80)}")
        if error_info:
            parts.append(f"error={error_info}")
        self.logger.info(" | ".join(parts))

    @vf.stop
    async def agent_completed(self, state: State) -> bool:
        """Check if agent has completed."""
        return state.get("agent_completed", False)

    @vf.stop
    async def timeout_reached(self, state: State) -> bool:
        """Check rollout timeout."""
        elapsed = time.time() - state["timing"]["start_time"]
        if elapsed > self.timeout_seconds:
            state["agent_timed_out"] = True
            return True
        return False

    @vf.cleanup(priority=10)
    async def cleanup_rollout(self, state: State):
        """Post-rollout logging, agent cleanup, and interception unregistration.

        Priority 10 ensures this runs before lower-priority cleanup handlers
        (e.g. sandbox destruction) that may remove resources needed by
        post_rollout.
        """
        if state.get("is_completed", False):
            await self.post_rollout(state)

        await self.cleanup_agent(state)

        rollout_id = state.get("rollout_id")
        if rollout_id and self._interception_server is not None:
            self._interception_server.unregister_rollout(rollout_id)

    @vf.teardown
    async def teardown_resources(self):
        """Stop Prime Tunnel and HTTP interception server."""
        async with self._tunnel_lock:
            if self._tunnel is not None:
                try:
                    self._tunnel.sync_stop()
                    self.logger.debug("Prime Tunnel stopped")
                except Exception as e:
                    self.logger.warning(f"Error stopping Prime Tunnel: {e}")
                finally:
                    self._tunnel = None
        if self._interception_server is not None:
            await self._interception_server.stop()
