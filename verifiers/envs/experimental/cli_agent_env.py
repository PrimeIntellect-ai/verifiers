import asyncio
import logging
import time
import uuid
from typing import Any, cast
from urllib.parse import urlparse

import httpx
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
from verifiers.envs.multiturn_env import MultiTurnMonitorRubric
from verifiers.types import RolloutInput, SamplingArgs, State, TrajectoryStep

logger = logging.getLogger(__name__)


class CliAgentEnv(SandboxMixin, vf.Environment):
    """
    Environment for running full agent code inside sandboxes.

    The sandboxed agent talks directly to the rollout gateway running in the vLLM
    server through a prime tunnel URL. The environment only handles sandbox
    lifecycle, rollout registration, trajectory fetch, and reward computation.
    """

    def __init__(
        self,
        run_command: str,
        gateway_port: int = 8000,
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
        super().__init__(message_type="chat", **kwargs)
        self.add_rubric(MultiTurnMonitorRubric())

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
        self.gateway_port = gateway_port
        self.max_turns = max_turns
        self.poll_interval = poll_interval
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

        self._tunnel: Tunnel | None = None
        self._tunnel_lock = asyncio.Lock()

    def _resolve_tunnel_local_addr(self, state: State) -> str:
        gateway_url = cast(str, state["gateway_url"])
        parsed = urlparse(gateway_url)
        host = parsed.hostname
        if host is None:
            raise ValueError(f"Invalid gateway URL; missing hostname: {gateway_url}")
        return host

    async def get_tunnel_url(self, local_addr: str | None = None) -> str:
        """Get tunnel URL, starting the tunnel if needed."""
        async with self._tunnel_lock:
            if self._tunnel is None:
                if local_addr is None:
                    raise ValueError("local_addr is required when starting tunnel")
                if logger.isEnabledFor(logging.DEBUG):
                    self._tunnel = Tunnel(
                        local_port=self.gateway_port,
                        local_addr=local_addr,
                        log_level="debug",
                    )
                else:
                    self._tunnel = Tunnel(
                        local_port=self.gateway_port,
                        local_addr=local_addr,
                    )
                url = await self._tunnel.start()
                logger.debug(f"Prime Tunnel started: {url}")
                return url

            assert self._tunnel.url is not None, "Tunnel started but URL is None"
            return self._tunnel.url

    def _resolve_gateway_url(self, state: State) -> str:
        client = cast(AsyncOpenAI, state["client"])
        gateway_url = str(client.base_url).rstrip("/")
        if gateway_url.endswith("/v1"):
            gateway_url = gateway_url[:-3]
        return gateway_url

    def _rollout_endpoint(self, state: State, suffix: str) -> str:
        gateway_url = cast(str, state["gateway_url"])
        rollout_id = cast(str, state["rollout_id"])
        return f"{gateway_url}/v1/rollouts/{rollout_id}/{suffix.lstrip('/')}"

    async def _gateway_post(
        self,
        state: State,
        suffix: str,
        payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        timeout = httpx.Timeout(self.timeout_seconds)
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(
                self._rollout_endpoint(state, suffix),
                json=payload,
            )
            response.raise_for_status()
            if not response.content:
                return {}
            return cast(dict[str, Any], response.json())

    async def _gateway_get(self, state: State, suffix: str) -> dict[str, Any]:
        timeout = httpx.Timeout(self.timeout_seconds)
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.get(self._rollout_endpoint(state, suffix))
            response.raise_for_status()
            return cast(dict[str, Any], response.json())

    async def register_rollout(self, state: State) -> None:
        sampling_params = dict(state.get("sampling_args") or {})
        payload = {
            "model": state["model"],
            "sampling_params": sampling_params,
            "max_turns": self.max_turns,
            "max_seq_len": self.max_seq_len,
        }
        await self._gateway_post(state, "register", payload)

    async def unregister_rollout(self, state: State) -> None:
        await self._gateway_post(state, "unregister")

    async def fetch_trajectory(self, state: State) -> None:
        data = await self._gateway_get(state, "trajectory")
        raw_trajectory = cast(list[dict[str, Any]], data.get("trajectory", []))

        trajectory: list[TrajectoryStep] = []
        for raw_step in raw_trajectory:
            step = dict(raw_step)
            step.setdefault("response", None)
            step.setdefault("reward", None)
            step.setdefault("advantage", None)
            step.setdefault("is_truncated", False)
            step.setdefault("trajectory_id", state.get("trajectory_id", ""))
            step.setdefault("extras", {})
            trajectory.append(cast(TrajectoryStep, step))

        state["trajectory"] = trajectory
        state["prompt"] = data.get("prompt")
        state["completion"] = data.get("completion")
        state["is_truncated"] = bool(
            data.get("is_truncated", state.get("is_truncated", False))
        )

    async def get_docker_image(self, state: State) -> str:
        """Get the Docker image for the sandbox. Override for per-task images."""
        return self.docker_image

    async def build_env_vars(self, state: State) -> dict[str, str]:
        """Build environment variables for the sandbox. Override to add custom vars."""
        env_vars = dict(self.environment_vars) if self.environment_vars else {}
        env_vars["OPENAI_BASE_URL"] = cast(str, state["rollout_base_url"])
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
        sandbox_id = cast(str, state["sandbox_id"])
        background_job: BackgroundJob = await self.sandbox_client.start_background_job(
            sandbox_id,
            self.run_command,
        )
        state["background_job"] = background_job
        state["agent_start_time"] = time.time()
        state["agent_completed"] = False

    async def wait_for_agent_completion(self, state: State) -> None:
        """Poll for agent completion using background job API."""
        sandbox_id = state.get("sandbox_id")
        background_job = state.get("background_job")
        if not sandbox_id or not background_job:
            state["agent_completed"] = True
            return

        try:
            await asyncio.wait_for(
                self.poll_job_completion(
                    state,
                    cast(str, sandbox_id),
                    cast(BackgroundJob, background_job),
                ),
                timeout=self.timeout_seconds,
            )
        except asyncio.TimeoutError:
            logger.warning(f"Agent timed out after {self.timeout_seconds}s")
            state["agent_timed_out"] = True
        finally:
            state["agent_completed"] = True

    async def poll_job_completion(
        self,
        state: State,
        sandbox_id: str,
        background_job: BackgroundJob,
    ) -> None:
        """Poll until background job completes, capturing output."""
        while True:
            status: BackgroundJobStatus = await self.sandbox_client.get_background_job(
                sandbox_id,
                background_job,
            )
            if status.completed:
                state["agent_exit_code"] = status.exit_code
                state["agent_stdout"] = status.stdout
                state["agent_stderr"] = status.stderr
                logger.debug(f"Agent completed with exit_code={status.exit_code}")
                return
            await asyncio.sleep(1)

    def _render_timing(self, state: State) -> None:
        start_time = cast(float, state["timing"]["start_time"])
        end_time = time.time()
        generation_ms = (end_time - start_time) * 1000
        state["timing"]["generation_ms"] = generation_ms
        state["timing"]["total_ms"] = generation_ms

    async def rollout(
        self,
        input: RolloutInput,
        client: AsyncOpenAI,
        model: str,
        sampling_args: SamplingArgs | None = None,
    ) -> State:
        state = await self.init_state(input, client, model, sampling_args)
        state["rollout_id"] = f"rollout_{uuid.uuid4().hex[:8]}"
        state["gateway_url"] = self._resolve_gateway_url(state)

        rollout_registered = False
        try:
            await self.register_rollout(state)
            rollout_registered = True

            tunnel_local_addr = self._resolve_tunnel_local_addr(state)
            state["tunnel_local_addr"] = tunnel_local_addr
            tunnel_url = await self.get_tunnel_url(local_addr=tunnel_local_addr)
            state["tunnel_url"] = tunnel_url
            state["rollout_base_url"] = (
                f"{tunnel_url.rstrip('/')}/v1/rollouts/{state['rollout_id']}"
            )

            env_vars = await self.build_env_vars(state)
            docker_image = await self.get_docker_image(state)
            sandbox_request = CreateSandboxRequest(
                name=cast(str, state["rollout_id"]),
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

            await self.start_agent(state)
            await self.wait_for_agent_completion(state)
            await self.fetch_trajectory(state)
        except vf.Error as e:
            state["error"] = e
        except Exception as e:
            state["error"] = vf.InfraError(str(e))
        finally:
            if rollout_registered:
                try:
                    await self.unregister_rollout(state)
                except Exception as e:
                    logger.warning(
                        f"Failed to unregister rollout {state['rollout_id']}: {e}"
                    )
                    if state.get("error") is None:
                        state["error"] = vf.InfraError(str(e))

            if state.get("sandbox_id"):
                try:
                    await self.destroy_sandbox(state)
                except Exception as e:
                    logger.warning(
                        f"Failed to destroy sandbox {state.get('sandbox_id')}: {e}"
                    )
                    if state.get("error") is None:
                        state["error"] = vf.InfraError(str(e))

            state["is_completed"] = True
            self._render_timing(state)

        return state

    @vf.teardown
    async def teardown_resources(self):
        """Stop Prime Tunnel."""
        async with self._tunnel_lock:
            if self._tunnel is not None:
                try:
                    self._tunnel.sync_stop()
                    logger.debug("Prime Tunnel stopped")
                except Exception as e:
                    logger.warning(f"Error stopping Prime Tunnel: {e}")
                finally:
                    self._tunnel = None

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
            await self.delete_sandbox(cast(str, sandbox_id))
