import asyncio
import logging
import time
import uuid

from openai import AsyncOpenAI
from prime_sandboxes import (
    AdvancedConfigs,
    BackgroundJob,
    BackgroundJobStatus,
    CreateSandboxRequest,
)
from prime_tunnel import Tunnel

import verifiers as vf
from verifiers.envs.experimental.sandbox_mixin import SandboxMixin
from verifiers.types import (
    ChatCompletionToolParam,
    Messages,
    MessageType,
    ModelResponse,
    SamplingArgs,
    State,
)
from verifiers.utils.interception_utils import (
    InterceptionServer,
    create_empty_completion,
    deliver_response,
    get_streaming_model_response,
)

logger = logging.getLogger(__name__)


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
        interception_port: int = 8765,
        interception_url: str | None = None,
        max_turns: int = -1,
        timeout_seconds: float = 3600.0,
        poll_interval: float = 2.0,
        docker_image: str = "python:3.11-slim",
        start_command: str = "tail -f /dev/null",
        cpu_cores: int = 1,
        memory_gb: int = 2,
        disk_size_gb: int = 5,
        gpu_count: int = 0,
        timeout_minutes: int = 60,
        environment_vars: dict[str, str] | None = None,
        team_id: str | None = None,
        advanced_configs: AdvancedConfigs | None = None,
        labels: list[str] | None = None,
        max_retries: int = 5,
        base_delay: float = 0.5,
        backoff_factor: float = 2.0,
        max_backoff_seconds: float = 30.0,
        jitter: float = 1e-3,
        sandbox_client_max_workers: int = 10,
        sandbox_client_max_connections: int = 100,
        sandbox_client_max_keepalive_connections: int = 50,
        sandbox_wait_for_creation_max_attempts: int = 120,
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
        )
        self.run_command = run_command
        self.poll_interval = poll_interval
        self.interception_port = interception_port
        self.interception_url = interception_url
        self.timeout_seconds = timeout_seconds
        self.docker_image = docker_image
        self.start_command = start_command
        self.cpu_cores = cpu_cores
        self.memory_gb = memory_gb
        self.disk_size_gb = disk_size_gb
        self.gpu_count = gpu_count
        self.timeout_minutes = timeout_minutes
        self.environment_vars = environment_vars
        self.team_id = team_id
        self.advanced_configs = advanced_configs
        self.labels = labels

        # Tunnel and interception server
        self._tunnel: Tunnel | None = None
        self._tunnel_lock = asyncio.Lock()
        self._interception_server = InterceptionServer(port=interception_port)

    async def get_tunnel_url(self) -> str:
        """Get tunnel URL, starting the tunnel if needed."""
        async with self._tunnel_lock:
            if self._tunnel is None:
                port = self._interception_server.port
                if logger.isEnabledFor(logging.DEBUG):
                    self._tunnel = Tunnel(
                        local_port=port,
                        log_level="debug",
                    )
                else:
                    self._tunnel = Tunnel(local_port=port)
                url = await self._tunnel.start()
                logger.debug(f"Prime Tunnel started: {url}")
                return url
            else:
                assert self._tunnel.url is not None, "Tunnel started but URL is None"
                return self._tunnel.url

    async def setup_state(self, state: State) -> State:
        """Setup sandbox + interception for this rollout"""
        state = await super().setup_state(state)

        rollout_id = f"rollout_{uuid.uuid4().hex[:8]}"
        state["rollout_id"] = rollout_id

        await self._interception_server.start()

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
            timeout_minutes=self.timeout_minutes,
            environment_vars=env_vars,
            team_id=self.team_id,
            advanced_configs=self.advanced_configs,
            labels=self.labels if self.labels else [],
        )
        logger.debug(
            f"Creating sandbox with OPENAI_BASE_URL={env_vars.get('OPENAI_BASE_URL')} "
            f"docker_image={docker_image}"
        )
        await self.create_sandbox(state, sandbox_request)

        # Register rollout for interception
        request_id_queue = self._interception_server.register_rollout(rollout_id)
        state["request_id_queue"] = request_id_queue
        state["agent_completed"] = False

        await self.start_agent(state)

        return state

    async def get_docker_image(self, state: State) -> str:
        """Get the Docker image for the sandbox. Override for per-task images."""
        return self.docker_image

    async def build_env_vars(self, state: State) -> dict[str, str]:
        """Build environment variables for the sandbox. Override to add custom vars."""
        env_vars = dict(self.environment_vars) if self.environment_vars else {}
        env_vars["OPENAI_BASE_URL"] = state["interception_base_url"]
        env_vars.setdefault("OPENAI_TIMEOUT", "600")
        env_vars.setdefault("OPENAI_REQUEST_TIMEOUT", "600")
        env_vars.setdefault("HTTPX_TIMEOUT", "600")
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

        background_job: BackgroundJob = await self.sandbox_client.start_background_job(
            sandbox_id,
            self.run_command,
        )
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
            logger.warning(f"Agent timed out after {self.timeout_seconds}s")
            state["agent_timed_out"] = True
        except asyncio.CancelledError:
            logger.debug("Completion wait task cancelled")
            raise
        except Exception as e:
            logger.debug(f"Completion wait ended: {e}")
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
                logger.debug(f"Agent completed with exit_code={status.exit_code}")
                return
            await asyncio.sleep(1)

    async def check_agent_completed(self, state: State) -> bool:
        """Check if agent process has completed."""
        return state.get("agent_completed", False)

    async def get_prompt_messages(self, state: State) -> Messages:
        """Wait for agent to make an API request OR agent completion, whichever comes first."""
        request_id_queue = state["request_id_queue"]

        while True:
            try:
                # Short timeout so we can check completion frequently
                request_id = await asyncio.wait_for(
                    request_id_queue.get(),
                    timeout=self.poll_interval,
                )
                # Got a request, proceed normally
                state["current_request_id"] = request_id
                intercept = self._interception_server.intercepts[request_id]
                return intercept["messages"]

            except asyncio.TimeoutError:
                # No request yet, check if agent finished or timed out
                if await self.check_agent_completed(state):
                    state["agent_completed"] = True
                    return []
                if time.time() - state["timing"]["start_time"] > self.timeout_seconds:
                    return []

    async def get_model_response(
        self,
        state: State,
        prompt: Messages,
        client: AsyncOpenAI | None = None,
        model: str | None = None,
        oai_tools: list[ChatCompletionToolParam] | None = None,
        sampling_args: SamplingArgs | None = None,
        message_type: MessageType | None = None,
    ) -> ModelResponse:
        """Get model response and unblock the waiting HTTP handler."""
        # Handle agent completion case (empty prompt)
        if not prompt:
            return create_empty_completion(model or state["model"])

        request_id = state.get("current_request_id")
        intercept = (
            self._interception_server.intercepts.get(request_id) if request_id else None
        )

        if intercept:
            # Always use the configured model from state, not the intercepted model
            # (agent may send a placeholder like "model" from its config)
            model = state.get("model") or model
            oai_tools = intercept.get("tools") or oai_tools

        response: ModelResponse | None = None
        error: BaseException | None = None

        try:
            # Handle streaming requests
            if intercept and intercept.get("stream"):
                response = await get_streaming_model_response(
                    state=state,
                    prompt=prompt,
                    intercept=intercept,
                    client=client,
                    model=model,
                    oai_tools=oai_tools,
                    sampling_args=sampling_args,
                )
            else:
                response = await super().get_model_response(
                    state=state,
                    prompt=prompt,
                    client=client,
                    model=model,
                    oai_tools=oai_tools,
                    sampling_args=sampling_args,
                    message_type=message_type,
                )
        except BaseException as e:
            error = e
            raise
        finally:
            # Always unblock HTTP handler, even on exception
            if intercept:
                deliver_response(intercept, response, error)
                state["current_request_id"] = None

        return response

    async def add_model_response(
        self,
        state: State,
        prompt_messages: Messages,
        response: ModelResponse,
    ):
        """Add model response and update top-level prompt on first turn."""
        # Skip adding empty "agent completed" step - keeps trajectory clean
        if not prompt_messages:
            return
        # On first turn, update state["prompt"] to match the agent's actual prompt
        if len(state["trajectory"]) == 0:
            state["prompt"] = prompt_messages
        await super().add_model_response(state, prompt_messages, response)

    @vf.teardown
    async def teardown_resources(self):
        """Stop Prime Tunnel and HTTP interception server."""
        async with self._tunnel_lock:
            if self._tunnel is not None:
                try:
                    await self._tunnel.stop()
                    logger.debug("Prime Tunnel stopped")
                except Exception as e:
                    logger.warning(f"Error stopping Prime Tunnel: {e}")
                finally:
                    self._tunnel = None
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
        if rollout_id:
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
        pass

    @vf.cleanup
    async def destroy_sandbox(self, state: State):
        """Cleanup sandbox after rollout."""
        await self.post_rollout(state)
        sandbox_id = state.get("sandbox_id")
        if sandbox_id:
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
