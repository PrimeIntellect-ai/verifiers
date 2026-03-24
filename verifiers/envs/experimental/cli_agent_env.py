import asyncio
import logging
import math
import time
import uuid
from collections import Counter
from typing import Any, cast

from prime_sandboxes import (
    AdvancedConfigs,
    BackgroundJob,
    BackgroundJobStatus,
    CreateSandboxRequest,
)
from prime_tunnel import Tunnel

import verifiers as vf
from verifiers.clients import Client
from verifiers.envs.experimental.sandbox_mixin import SandboxMixin, SandboxMonitorRubric
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

logger = logging.getLogger(__name__)


class AgentError(vf.InfraError):
    """Raised when the agent process fails or exits unexpectedly."""


class CliAgentMonitorRubric(vf.Rubric):
    """Monitor rubric that tracks CLI agent execution state."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_metric(self.agent_timeout)
        self.add_metric(self.agent_error)

    async def agent_timeout(self, state: vf.State) -> float:
        """Whether the agent timed out."""
        return float(bool(state.get("agent_timed_out")))

    async def agent_error(self, state: vf.State) -> float:
        """Whether the agent errored (non-zero exit_code)."""
        agent_exit_code = state.get("agent_exit_code")
        if agent_exit_code is None:
            return 0.0
        return float(agent_exit_code != 0)


class CliAgentEnv(SandboxMixin, vf.MultiTurnEnv):
    """
    Environment for running full agent code inside sandboxes.
    Extends MultiTurnEnv to reuse rollout loop, but intercepts agent's
    API requests via HTTP proxy server. Each agent request triggers one
    rollout step.
    """

    def __init__(
        self,
        run_command: str,
        interception_port: int | None = None,
        interception_url: str | None = None,
        max_turns: int = -1,
        timeout_seconds: float = 3600.0,
        poll_interval: float = 1.0,
        docker_image: str = "python:3.11-slim",
        start_command: str = "tail -f /dev/null",
        cpu_cores: int = 1,
        memory_gb: int = 2,
        disk_size_gb: int = 5,
        gpu_count: int = 0,
        environment_vars: dict[str, str] | None = None,
        team_id: str | None = None,
        advanced_configs: AdvancedConfigs | None = None,
        labels: list[str] | None = None,
        max_retries: int = 5,
        base_delay: float = 0.5,
        backoff_factor: float = 2.0,
        max_backoff_seconds: float = 30.0,
        jitter: float = 1e-3,
        sandbox_client_max_workers: int = 50,
        sandbox_client_max_connections: int = 1000,
        sandbox_client_max_keepalive_connections: int = 200,
        sandbox_wait_for_creation_max_attempts: int = 120,
        sandbox_creations_per_minute: float | None = 128,
        keep_sandbox_for_scoring: bool = False,
        **kwargs,
    ):
        super().__init__(max_turns=max_turns, message_type="chat", **kwargs)
        self.init_sandbox_client(
            max_retries=max_retries,
            base_delay=base_delay,
            backoff_factor=backoff_factor,
            max_backoff_seconds=max_backoff_seconds,
            jitter=jitter,
            sandbox_client_max_workers=sandbox_client_max_workers,
            sandbox_client_max_connections=sandbox_client_max_connections,
            sandbox_client_max_keepalive_connections=sandbox_client_max_keepalive_connections,
            sandbox_wait_for_creation_max_attempts=sandbox_wait_for_creation_max_attempts,
            sandbox_creations_per_minute=sandbox_creations_per_minute,
        )
        self.keep_sandbox_for_scoring = keep_sandbox_for_scoring
        self.run_command = run_command
        self.poll_interval = poll_interval
        self.timeout_seconds = timeout_seconds
        self.docker_image = docker_image
        self.start_command = start_command
        self.cpu_cores = cpu_cores
        self.memory_gb = memory_gb
        self.disk_size_gb = disk_size_gb
        self.gpu_count = gpu_count
        self.environment_vars = environment_vars
        self.team_id = team_id
        self.advanced_configs = advanced_configs
        self.labels = labels

        interception_port = (
            get_free_port() if interception_port is None else interception_port
        )
        self.init_interception(interception_port, interception_url)
        self.add_rubric(SandboxMonitorRubric())
        self.add_rubric(CliAgentMonitorRubric())

    def init_interception(
        self,
        interception_port: int = 8765,
        interception_url: str | None = None,
    ):
        """Initialize interception server and tunnel resources. Call from __init__."""
        self.interception_port = interception_port
        self.interception_url = interception_url
        self._tunnel: Tunnel | None = None
        self._tunnel_lock = asyncio.Lock()
        self._interception_server = InterceptionServer(port=interception_port)

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

            if self._tunnel is None:
                interception_server = self._require_interception_server()
                port = interception_server.port
                if self.logger.isEnabledFor(logging.DEBUG):
                    self._tunnel = Tunnel(
                        local_port=port,
                        log_level="debug",
                    )
                else:
                    self._tunnel = Tunnel(local_port=port)
                url = await self._tunnel.start()
                self.logger.debug(f"Prime Tunnel started: {url}")
                return url
            else:
                assert self._tunnel.url is not None, "Tunnel started but URL is None"
                return self._tunnel.url

    async def setup_state(self, state: State) -> State:
        """Setup sandbox + interception for this rollout"""
        state = await super().setup_state(state)

        rollout_id = f"rollout_{uuid.uuid4().hex[:8]}"
        state["rollout_id"] = rollout_id

        interception_server = self._require_interception_server()
        await interception_server.start()

        if self.interception_url is None:
            tunnel_url = await self.get_tunnel_url()
            state["interception_base_url"] = f"{tunnel_url}/rollout/{rollout_id}/v1"
        else:
            state["interception_base_url"] = (
                f"{self.interception_url.rstrip('/')}/rollout/{rollout_id}/v1"
            )

        env_vars = await self.build_env_vars(state)
        docker_image = await self.get_docker_image(state)

        sandbox_request = CreateSandboxRequest(
            name=rollout_id,
            docker_image=docker_image,
            start_command=self.start_command,
            cpu_cores=self.cpu_cores,
            memory_gb=self.memory_gb,
            disk_size_gb=self.disk_size_gb,
            gpu_count=self.gpu_count,
            timeout_minutes=math.ceil(self.timeout_seconds / 60),
            environment_vars=env_vars,
            team_id=self.team_id,
            advanced_configs=self.advanced_configs,
            labels=self.labels if self.labels else [],
        )
        self.logger.debug(
            f"Creating sandbox with OPENAI_BASE_URL={env_vars.get('OPENAI_BASE_URL')} "
            f"docker_image={docker_image}"
        )
        await self.create_sandbox(state, sandbox_request)

        # Register rollout for interception
        request_id_queue = interception_server.register_rollout(rollout_id)
        state["request_id_queue"] = request_id_queue
        state["agent_completed"] = False

        await self.start_agent(state)

        parts = [
            f"Started  rollout_id={state['rollout_id']}",
            f"example_id={state['example_id']}",
        ]
        self.logger.info(" | ".join(parts))

        return state

    async def get_docker_image(self, state: State) -> str:
        """Get the Docker image for the sandbox. Override for per-task images."""
        return self.docker_image

    async def build_env_vars(self, state: State) -> dict[str, str]:
        """Build environment variables for the sandbox. Override to add custom vars."""
        env_vars = dict(self.environment_vars) if self.environment_vars else {}
        env_vars["OPENAI_BASE_URL"] = state["interception_base_url"]
        env_vars.setdefault("OPENAI_TIMEOUT", "3600")
        env_vars.setdefault("OPENAI_REQUEST_TIMEOUT", "3600")
        env_vars.setdefault("HTTPX_TIMEOUT", "3600")
        model = state.get("model")
        if model:
            env_vars["OPENAI_MODEL"] = model
        return env_vars

    async def post_sandbox_setup(self, state: State) -> None:
        """Hook for post-sandbox setup. Override to upload files, run commands, etc."""
        pass

    async def start_agent(self, state: State) -> None:
        """Start the agent command using background job."""
        sandbox_id = state["sandbox_id"]

        self.logger.debug(f"Starting agent in sandbox {sandbox_id}")
        try:
            background_job: BackgroundJob = (
                await self.sandbox_client.start_background_job(
                    sandbox_id,
                    self.run_command,
                )
            )
        except Exception as e:
            raise vf.SandboxError(f"Failed to start agent: {e}") from e
        state["background_job"] = background_job
        state["agent_start_time"] = time.time()

        state["completion_wait_task"] = asyncio.create_task(
            self.wait_for_completion(state)
        )

    async def wait_for_completion(self, state: State) -> None:
        """Poll for agent completion using background job API."""
        sandbox_id = state.get("sandbox_id")
        background_job: BackgroundJob | None = state.get("background_job")

        if not sandbox_id or not background_job:
            state["agent_completed"] = True
            return

        try:
            await asyncio.wait_for(
                self.poll_job_completion(state, sandbox_id, background_job),
                timeout=self.timeout_seconds,
            )
        except asyncio.TimeoutError:
            self.logger.warning(f"Agent timed out after {self.timeout_seconds}s")
            state["agent_timed_out"] = True
        except asyncio.CancelledError:
            self.logger.debug("Completion wait task cancelled")
            raise
        except Exception as e:
            error = AgentError(f"Agent polling failed: {e}")
            state["error"] = error
            self.logger.error(f"Agent polling failed: {e}")
        finally:
            state["agent_completed"] = True

    async def poll_job_completion(
        self, state: State, sandbox_id: str, background_job: BackgroundJob
    ) -> None:
        """Poll until background job completes, capturing output."""
        while True:
            status: BackgroundJobStatus = await self.sandbox_client.get_background_job(
                sandbox_id, background_job
            )
            if status.completed:
                state["agent_exit_code"] = status.exit_code
                state["agent_stdout"] = status.stdout
                state["agent_stderr"] = status.stderr
                if status.exit_code == 0:
                    self.logger.debug(
                        f"Agent completed successfully (exit_code={status.exit_code})"
                    )
                else:
                    self.logger.warning(
                        f"Agent failed (exit_code={status.exit_code}) stdout={status.stdout}, stderr={status.stderr}"
                    )
                return
            await asyncio.sleep(1)

    async def check_agent_completed(self, state: State) -> bool:
        """Check if agent process has completed."""
        return state.get("agent_completed", False)

    def normalize_intercepted_tools(self, intercept_tools: object) -> list[Tool] | None:
        """Normalize intercepted request tools for the provider-agnostic runtime.

        Assumes that agent requests arrive in OpenAI-tool format.
        Avoids redundant Pydantic round-trips for already-validated Tool objects.
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
        """Hook to normalize messages received from the agent before model inference.

        Assumes that agent requests arrive in OpenAI-format.
        """
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
                    return None

    async def get_prompt_messages(self, state: State) -> Messages:
        """Wait for agent to make an API request OR agent completion, whichever comes first."""
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
            # (agent may send a placeholder like "model" from its config)
            model = state.get("model") or model
            intercept_tools = intercept.get("tools")
            if intercept_tools:
                # Cache normalized tools per rollout — agents typically send
                # the same tool definitions on every request. Key on tool
                # names so swapping tools with the same count invalidates
                # the cache; normalize_intercepted_tools is idempotent so
                # a false miss just re-normalizes.
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
            # Always use base class path (non-streaming, supports TITO)
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
        # Skip adding empty "agent completed" step - keeps trajectory clean
        if not prompt_messages:
            return
        # On first turn, update state["prompt"] to match the agent's actual prompt
        if len(state["trajectory"]) == 0:
            state["prompt"] = prompt_messages
        await super().add_model_response(
            state, prompt_messages, await self.normalize_response(response)
        )

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

    @vf.cleanup
    async def cleanup_interception_context(self, state: State):
        """Cleanup interception context for rollout"""
        # Cancel completion wait task if still running
        task = state.get("completion_wait_task")
        if task and not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        state.pop("background_job", None)

        rollout_id = state.get("rollout_id")
        if rollout_id and self._interception_server is not None:
            self._interception_server.unregister_rollout(rollout_id)

    @vf.stop
    async def agent_completed(self, state: State) -> bool:
        """Check if agent has completed."""
        return state.get("agent_completed", False)

    @vf.stop
    async def timeout_reached(self, state: State) -> bool:
        """Check rollout timeout"""
        elapsed = time.time() - state["timing"]["start_time"]
        return elapsed > self.timeout_seconds

    async def post_rollout(self, state: State):
        """
        Override for custom post-rollout logic. For example, if sandbox state is needed for reward functions,
        run computation here and cache the result in state before sandbox is destroyed.
        """
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
            f"exit_code={exit_code}",
            f"duration={print_time(duration_s)}",
        ]
        if timed_out:
            parts.append("timed_out=True")
        if error_info:
            parts.append(f"error={error_info}")
        self.logger.info(" | ".join(parts))

    @vf.cleanup
    async def destroy_sandbox(self, state: State):
        """Cleanup sandbox after rollout.

        When `keep_sandbox_for_scoring` is True, sandbox deletion is deferred
        (e.g. when the rubric needs sandbox access during scoring).
        The sandbox is still deregistered from active tracking so the
        environment teardown does not attempt a redundant bulk-delete.

        If the rollout was not completed (e.g. cancelled during shutdown),
        the sandbox is always deleted since scoring will not happen.
        """
        completed = state.get("is_completed", False)
        if completed:
            await self.post_rollout(state)
        sandbox_id = state.get("sandbox_id")
        if sandbox_id:
            if self.keep_sandbox_for_scoring and completed:
                self.deregister_sandbox(sandbox_id)
            else:
                await self.delete_sandbox(sandbox_id)

    async def env_response(
        self, messages: Messages, state: State, **kwargs
    ) -> Messages:
        """
        Generate a response from the environment.
        For CliAgentEnv, there is no environment response - the agent
        controls the conversation flow via its requests.
        """
        return []
