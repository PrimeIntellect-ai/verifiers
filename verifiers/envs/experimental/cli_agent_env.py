import asyncio
import logging
import os
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
from verifiers.envs.experimental.sandbox_mixin import (
    SandboxMixin,
    SandboxMonitorRubric,
    SandboxTimeouts,
    format_rollout_log_event,
    log_rollout_event,
)
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
from verifiers.utils.logging_utils import truncate
from verifiers.utils.message_utils import normalize_messages

logger = logging.getLogger(__name__)


class AgentError(vf.InfraError):
    """Raised when the agent process fails or exits unexpectedly."""


def make_agent_error(state: State, message: str) -> AgentError:
    """Create an AgentError with rollout-specific sandbox context when available."""
    context_parts = [
        f"sandbox_id={state['sandbox_id']}",
        f"rollout_id={state['rollout_id']}",
        f"example_id={state['example_id']}",
    ]
    state_info = state["input"].get("info", {})
    instance_id = state_info.get("instance_id")
    if instance_id:
        context_parts.append(f"instance_id={instance_id}")
    return AgentError(f"{message} ({', '.join(context_parts)})")


def _collect_tool_counts(state: State) -> Counter[str]:
    tool_counts: Counter[str] = Counter()
    for step in state.get("trajectory", []):
        for msg in step.get("completion", []):
            if isinstance(msg, AssistantMessage) and isinstance(msg.tool_calls, list):
                for tc in msg.tool_calls:
                    if isinstance(tc, ToolCall):
                        tool_counts[tc.name] += 1
    return tool_counts


def _phase(timing: Any, name: str) -> Any:
    if timing is None:
        return None
    if isinstance(timing, dict):
        return timing.get(name)
    return getattr(timing, name, None)


def _span_start(span: Any) -> float:
    if span is None:
        return 0.0
    if isinstance(span, dict):
        return float(span.get("start", 0.0) or 0.0)
    return float(getattr(span, "start", 0.0) or 0.0)


def _span_end(span: Any) -> float:
    if span is None:
        return 0.0
    if isinstance(span, dict):
        return float(span.get("end", 0.0) or 0.0)
    return float(getattr(span, "end", 0.0) or 0.0)


def _span_duration(span: Any) -> float:
    if span is None:
        return 0.0
    duration = (
        span.get("duration")
        if isinstance(span, dict)
        else getattr(span, "duration", None)
    )
    if duration is not None:
        return float(duration or 0.0)
    end = _span_end(span)
    start = _span_start(span)
    return end - start if end > 0 and start > 0 else 0.0


def _time_spans(timing: Any, name: str) -> list[Any]:
    phase = _phase(timing, name)
    if phase is None:
        return []
    if isinstance(phase, dict):
        spans = phase.get("spans", [])
    else:
        spans = getattr(phase, "spans", [])
    return list(spans or [])


def _first_call_latency_s(state: State) -> float | None:
    timing = state.get("timing")
    spans = _time_spans(timing, "model")
    if not spans:
        return None
    first_model_start = _span_start(spans[0])
    agent_start = float(state.get("agent_start_time") or 0.0)
    if agent_start <= 0:
        agent_start = _span_end(_phase(timing, "setup"))
    if first_model_start <= 0 or agent_start <= 0:
        return None
    return max(0.0, first_model_start - agent_start)


def _agent_duration_s(state: State) -> float:
    agent_start = float(state.get("agent_start_time") or 0.0)
    if agent_start <= 0:
        return 0.0
    agent_end = float(state.get("agent_end_time") or 0.0)
    if agent_end <= 0:
        agent_end = _span_end(_phase(state.get("timing"), "generation"))
    return max(0.0, agent_end - agent_start) if agent_end > 0 else 0.0


def _rollout_total_s(state: State) -> float:
    timing = state.get("timing")
    scoring = _phase(timing, "scoring")
    generation = _phase(timing, "generation")
    scoring_end = _span_end(scoring)
    generation_start = _span_start(generation)
    if scoring_end > 0 and generation_start > 0:
        return max(0.0, scoring_end - generation_start)
    return _span_duration(generation)


def _format_optional_s(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}"


class CliAgentMonitorRubric(vf.Rubric):
    """Monitor rubric that tracks CLI agent execution state."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_metric(self.agent_timeout)
        self.add_metric(self.agent_error)

    async def agent_timeout(self, state: vf.State) -> float:
        """Whether the agent timed out."""
        return float(bool(state.get("timed_out")))

    async def agent_error(self, state: vf.State) -> float:
        """Whether the agent errored (non-zero exit_code)."""
        agent_exit_code = state.get("agent_exit_code")
        if agent_exit_code is None:
            return 0.0
        return float(agent_exit_code != 0)

    @vf.cleanup
    async def log_rollout_finished(self, state: vf.State) -> None:
        """Log final rollout lifecycle details after scoring has completed."""
        if state.get("_cli_agent_rollout_finished_logged"):
            return
        state["_cli_agent_rollout_finished_logged"] = True

        tool_counts = Counter(state.get("_cli_agent_tool_counts") or {})
        if not tool_counts:
            tool_counts = _collect_tool_counts(state)
        tools_str = ",".join(f"{k}:{v}" for k, v in tool_counts.most_common())
        error = state.get("error")
        error_info = (
            f"{type(error).__name__}: {truncate(str(error), 80)}" if error else None
        )
        timed_out = state.get("timed_out", False)
        stop_condition = state.get("stop_condition", "unknown")
        exit_code = state.get("agent_exit_code")

        if error_info or timed_out:
            self.logger.info(
                format_rollout_log_event(
                    "rollout_aborted",
                    state,
                    stop=stop_condition,
                    exit_code=exit_code,
                    timed_out=timed_out if timed_out else None,
                    error=error_info,
                )
            )

        timing = state.get("timing")
        self.logger.info(
            format_rollout_log_event(
                "rollout_finished",
                state,
                turns=len(state.get("trajectory", [])),
                tools=f"[{tools_str}]",
                stop=stop_condition,
                exit_code=exit_code,
                setup_s=_span_duration(_phase(timing, "setup")),
                first_call_latency_s=_format_optional_s(_first_call_latency_s(state)),
                agent_s=_agent_duration_s(state),
                scoring_s=_span_duration(_phase(timing, "scoring")),
                duration_s=_rollout_total_s(state),
            )
        )


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
        poll_interval: float = 5.0,
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
        sandbox_client_max_workers: int | None = None,
        sandbox_client_max_connections: int = 1000,
        sandbox_client_max_keepalive_connections: int = 200,
        sandbox_wait_for_creation_max_attempts: int = 120,
        sandbox_creations_per_minute: float | None = 128,
        timeouts: SandboxTimeouts = SandboxTimeouts(),
        keep_sandbox_for_scoring: bool = False,
        **kwargs,
    ):
        super().__init__(
            max_turns=max_turns,
            message_type="chat",
            **kwargs,
        )
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
            timeouts=timeouts,
        )
        self.keep_sandbox_for_scoring = keep_sandbox_for_scoring
        self.run_command = run_command
        self.poll_interval = poll_interval
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

        interception_port = 0 if interception_port is None else interception_port
        self.init_interception(interception_port, interception_url)
        self.add_rubric(SandboxMonitorRubric())
        self.add_rubric(CliAgentMonitorRubric())

    TUNNEL_CHECK_INTERVAL = 60.0  # seconds between server-side liveness checks

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

            # Periodic server-side liveness check
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
                    self._tunnel = Tunnel(
                        local_port=port,
                        log_level="debug",
                    )
                else:
                    self._tunnel = Tunnel(local_port=port)
                url = await self._tunnel.start()
                self._tunnel_last_checked = time.time()
                self.logger.debug(f"Prime Tunnel started: {url}")
                return url
            else:
                assert self._tunnel.url is not None, "Tunnel started but URL is None"
                return self._tunnel.url

    async def setup_state(self, state: State) -> State:
        """Setup sandbox + interception for this rollout"""
        setup_state = await super().setup_state(state)
        if setup_state is not None:
            state = setup_state

        rollout_id = f"rollout_{uuid.uuid4().hex[:8]}"
        state["rollout_id"] = rollout_id
        setup_start = time.perf_counter()
        setup_succeeded = False
        log_rollout_event(self.logger, "setup_started", state)
        try:
            await self._setup_sandbox_and_agent(state, rollout_id)
            setup_succeeded = True
        finally:
            log_rollout_event(
                self.logger,
                "setup_finished",
                state,
                status="ok" if setup_succeeded else "error",
                elapsed_s=time.perf_counter() - setup_start,
            )

    async def _setup_sandbox_and_agent(self, state: State, rollout_id: str) -> None:
        """Create sandbox resources and launch the CLI agent."""

        interception_server = self._require_interception_server()
        await interception_server.start()
        self.interception_port = interception_server.port

        if self.interception_url is None:
            tunnel_url = await self.get_tunnel_url()
            state["interception_base_url"] = f"{tunnel_url}/rollout/{rollout_id}/v1"
        else:
            state["interception_base_url"] = (
                f"{self.interception_url.rstrip('/')}/rollout/{rollout_id}/v1"
            )

        env_vars = await self.build_env_vars(state)
        docker_image = await self.get_docker_image(state)
        resources = self.get_sandbox_resources(state)

        sandbox_request = CreateSandboxRequest(
            name=rollout_id,
            docker_image=docker_image,
            start_command=self.start_command,
            cpu_cores=resources["cpu_cores"],
            memory_gb=resources["memory_gb"],
            disk_size_gb=resources["disk_size_gb"],
            gpu_count=resources["gpu_count"],
            gpu_type=resources.get("gpu_type"),
            vm=resources.get("vm", resources["gpu_count"] > 0),
            timeout_minutes=resources["timeout_minutes"],
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

        # Register rollout for interception. Pass state so the server can
        # surface stream-interruption errors (e.g. tunnel dies mid-SSE) back
        # onto the rollout; without this the agent sees a truncated stream
        # and often exits with code 0 and an empty trajectory.
        request_id_queue = interception_server.register_rollout(rollout_id, state=state)
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

    def get_sandbox_resources(self, state: State) -> dict[str, Any]:
        """Get sandbox resource allocation. Override for per-instance resources."""
        return {
            "cpu_cores": self.cpu_cores,
            "memory_gb": self.memory_gb,
            "disk_size_gb": self.disk_size_gb,
            "gpu_count": self.gpu_count,
            "gpu_type": None,
            "vm": self.gpu_count > 0,
            "timeout_minutes": self.compute_sandbox_timeout_minutes(),
        }

    # Keys set by build_env_vars that subclasses must not override.
    PROTECTED_ENV_VARS = frozenset(
        {
            "OPENAI_BASE_URL",
            "OPENAI_TIMEOUT",
            "OPENAI_REQUEST_TIMEOUT",
            "HTTPX_TIMEOUT",
            "OPENAI_MODEL",
            "OPENAI_API_KEY",
            "ANTHROPIC_BASE_URL",
            "ANTHROPIC_API_KEY",
        }
    )

    async def build_env_vars(self, state: State) -> dict[str, str]:
        """Build environment variables for the sandbox. Override to add custom vars."""
        env_vars = dict(self.environment_vars) if self.environment_vars else {}
        interception_base_url = str(state["interception_base_url"]).rstrip("/")
        env_vars["OPENAI_BASE_URL"] = interception_base_url
        env_vars["ANTHROPIC_BASE_URL"] = interception_base_url.removesuffix("/v1")
        env_vars.setdefault("OPENAI_TIMEOUT", "3600")
        env_vars.setdefault("OPENAI_REQUEST_TIMEOUT", "3600")
        env_vars.setdefault("HTTPX_TIMEOUT", "3600")
        secret = self._require_interception_server().secret
        env_vars["OPENAI_API_KEY"] = secret
        env_vars["ANTHROPIC_API_KEY"] = secret
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
        launch_start = time.perf_counter()
        log_rollout_event(self.logger, "agent_launch_started", state)
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
        log_rollout_event(
            self.logger,
            "agent_launched",
            state,
            elapsed_s=time.perf_counter() - launch_start,
        )

    async def wait_for_completion(self, state: State) -> None:
        """Poll for agent completion using background job API."""
        sandbox_id = state.get("sandbox_id")
        background_job: BackgroundJob | None = state.get("background_job")

        if not sandbox_id or not background_job:
            state["agent_completed"] = True
            return

        try:
            await self.poll_job_completion(state, sandbox_id, background_job)
        except asyncio.CancelledError:
            self.logger.debug("Completion wait task cancelled")
            raise
        except Exception as e:
            error = make_agent_error(state, f"Agent polling failed: {e}")
            state["error"] = error
            self.logger.error(str(error))
        finally:
            state["agent_completed"] = True

    async def poll_job_completion(
        self, state: State, sandbox_id: str, background_job: BackgroundJob
    ) -> None:
        """Poll until background job completes, capturing output."""
        while True:
            status: BackgroundJobStatus = await self.sandbox_client.get_background_job(
                sandbox_id, background_job, timeout=self.timeouts.poll
            )
            if status.completed:
                state["agent_end_time"] = time.time()
                state["agent_exit_code"] = status.exit_code
                state["agent_stdout"] = status.stdout
                state["agent_stderr"] = status.stderr
                if status.exit_code == 0:
                    self.logger.debug(
                        f"Agent completed successfully (exit_code={status.exit_code})"
                    )
                else:
                    stderr_full = status.stderr or ""
                    num_turns = len(state.get("trajectory", []))
                    if num_turns == 0:
                        error = make_agent_error(
                            state,
                            f"Agent crashed before any LLM call "
                            f"(exit_code={status.exit_code}): {stderr_full}",
                        )
                    else:
                        error = make_agent_error(
                            state,
                            f"Agent crashed after {num_turns} turn(s) "
                            f"(exit_code={status.exit_code}): {stderr_full}",
                        )
                    state["error"] = error
                    self.logger.error(str(error))
                return
            await asyncio.sleep(self.poll_interval)

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
        own message history conventions.
        """
        return response

    async def _poll_next_request(self, state: State) -> str | None:
        """Poll for the next intercepted request, checking liveness in between.

        Returns a request_id when a request arrives, or None when the agent
        has completed.
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
            # Always use the base-class path; streaming is synthesized afterward.
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
                # Stash headers on state before clearing current_request_id —
                # add_trajectory_step runs after this finally (via
                # add_model_response) and needs to inspect the originating
                # request's headers (e.g. ComposableEnv reads X-RLM-Depth
                # to drop sub-agent steps from the trajectory).
                raw_headers = intercept.get("headers")
                state["_last_request_headers"] = (
                    raw_headers if isinstance(raw_headers, dict) else {}
                )
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

    async def post_rollout(self, state: State):
        """
        Override for custom post-rollout logic. For example, if sandbox state is needed for reward functions,
        run computation here and cache the result in state before sandbox is destroyed.
        """
        state["_cli_agent_tool_counts"] = dict(_collect_tool_counts(state))

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
