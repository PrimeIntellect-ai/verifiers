import asyncio
import logging
import time
import uuid
from typing import Any
from urllib.parse import urlparse

import httpx
from prime_sandboxes import CreateSandboxRequest
from prime_tunnel import Tunnel

import verifiers as vf
from verifiers.clients import Client
from verifiers.types import ClientConfig, RolloutInput, SamplingArgs, State

logger = logging.getLogger(__name__)


def _tail_text(value: Any, max_chars: int = 1200) -> str:
    if value is None:
        return ""
    text = str(value)
    return text[-max_chars:] if len(text) > max_chars else text


class RolloutGatewayMixin:
    """Opt-in mixin that replaces MultiTurnEnv's interception-based rollout
    with a server-side gateway path. Toggle via ``use_gateway`` attribute.

    When gateway is active, the agent talks directly to vLLM's rollout
    gateway through a prime tunnel. The env only manages sandbox lifecycle.
    When inactive, falls through to CliAgentEnv's interception path.

    MRO: ``MyEnv → RolloutGatewayMixin → CliAgentEnv → SandboxMixin → MultiTurnEnv → Environment``
    """

    use_gateway: bool = True

    def init_gateway(
        self,
        gateway_port: int = 8000,
        timeout_seconds: float = 3600.0,
    ):
        """Initialize gateway resources. Call in __init__ when use_gateway=True."""
        self.gateway_port = gateway_port
        self._gw_timeout_seconds = timeout_seconds
        self._gw_http_client = httpx.AsyncClient(timeout=httpx.Timeout(timeout_seconds))
        self._gw_tunnels: dict[str, Tunnel] = {}
        self._gw_tunnel_lock = asyncio.Lock()

    # ------------------------------------------------------------------
    # Gateway URL resolution
    # ------------------------------------------------------------------

    def _resolve_gateway_url(self, state: State) -> str:
        client = getattr(state["client"], "client", state["client"])
        gateway_url = str(client.base_url).rstrip("/")
        if gateway_url.endswith("/v1"):
            gateway_url = gateway_url[:-3]
        return gateway_url

    def _resolve_tunnel_local_addr(self, state: State) -> str:
        gateway_url = state["gateway_url"]
        parsed = urlparse(gateway_url)
        host = parsed.hostname
        if host is None:
            raise ValueError(f"Invalid gateway URL; missing hostname: {gateway_url}")
        return host

    def _rollout_endpoint(self, state: State, suffix: str) -> str:
        return f"{state['gateway_url']}/v1/rollouts/{state['rollout_id']}/{suffix.lstrip('/')}"

    # ------------------------------------------------------------------
    # Gateway HTTP helpers
    # ------------------------------------------------------------------

    async def _gateway_post(
        self,
        state: State,
        suffix: str,
        payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        response = await self._gw_http_client.post(
            self._rollout_endpoint(state, suffix),
            json=payload,
        )
        response.raise_for_status()
        if not response.content:
            return {}
        return response.json()

    async def _gateway_get(self, state: State, suffix: str) -> dict[str, Any]:
        response = await self._gw_http_client.get(self._rollout_endpoint(state, suffix))
        response.raise_for_status()
        return response.json()

    # ------------------------------------------------------------------
    # Env var override for gateway path
    # ------------------------------------------------------------------

    async def build_env_vars(self, state: State) -> dict[str, str]:
        """Override to set OPENAI_BASE_URL from rollout_base_url in gateway mode."""
        if not self.use_gateway:
            return await super().build_env_vars(state)
        env_vars = dict(self.environment_vars) if self.environment_vars else {}
        env_vars["OPENAI_BASE_URL"] = state["rollout_base_url"]
        env_vars.setdefault("OPENAI_TIMEOUT", "600")
        env_vars.setdefault("OPENAI_REQUEST_TIMEOUT", "600")
        env_vars.setdefault("HTTPX_TIMEOUT", "600")
        model = state.get("model")
        if model:
            env_vars["OPENAI_MODEL"] = model
        return env_vars

    # ------------------------------------------------------------------
    # Rollout registration & trajectory
    # ------------------------------------------------------------------

    async def register_rollout(self, state: State) -> None:
        sampling_params = state.get("sampling_args") or {}
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
        state["trajectory"] = data.get("trajectory", [])
        state["prompt"] = data.get("prompt")
        state["completion"] = data.get("completion")
        state["is_truncated"] = bool(
            data.get("is_truncated", state.get("is_truncated", False))
        )

    # ------------------------------------------------------------------
    # Gateway tunnel management
    # ------------------------------------------------------------------

    async def get_gateway_tunnel_url(self, local_addr: str | None = None) -> str:
        """Get gateway tunnel URL, starting the tunnel if needed."""
        async with self._gw_tunnel_lock:
            if local_addr is None:
                if len(self._gw_tunnels) == 1:
                    tunnel = next(iter(self._gw_tunnels.values()))
                    assert tunnel.url is not None, "Tunnel started but URL is None"
                    return tunnel.url
                if len(self._gw_tunnels) == 0:
                    raise ValueError("local_addr is required when starting tunnel")
                raise ValueError(
                    "local_addr is required when multiple tunnels are active"
                )

            tunnel = self._gw_tunnels.get(local_addr)
            if tunnel is None:
                tunnel = Tunnel(
                    local_port=self.gateway_port,
                    local_addr=local_addr,
                    log_level="debug" if logger.isEnabledFor(logging.DEBUG) else "info",
                )
                url = await tunnel.start()
                self._gw_tunnels[local_addr] = tunnel
                logger.debug(f"Prime Tunnel started local_addr={local_addr} url={url}")
                return url

            assert tunnel.url is not None, "Tunnel started but URL is None"
            return tunnel.url

    # ------------------------------------------------------------------
    # Agent start & completion polling
    # ------------------------------------------------------------------

    async def start_agent(self, state: State) -> None:
        """Start the agent command. In gateway mode, skip background completion task."""
        if not self.use_gateway:
            return await super().start_agent(state)
        sandbox_id = state["sandbox_id"]
        background_job = await self.sandbox_client.start_background_job(
            sandbox_id,
            self.run_command,
        )
        state["background_job"] = background_job
        state["agent_start_time"] = time.time()
        state["agent_completed"] = False

    async def poll_job_completion(
        self,
        state: State,
        sandbox_id: str,
        background_job,
    ) -> None:
        """Poll until background job completes, capturing output."""
        if not self.use_gateway:
            return await super().poll_job_completion(state, sandbox_id, background_job)
        while True:
            status = await self.sandbox_client.get_background_job(
                sandbox_id,
                background_job,
            )
            if status.completed:
                state["agent_exit_code"] = status.exit_code
                state["agent_stdout"] = status.stdout
                state["agent_stderr"] = status.stderr
                if status.exit_code not in (None, 0):
                    logger.warning(
                        f"rollout={state.get('rollout_id')} sandbox={sandbox_id} "
                        f"stage=agent_completed exit_code={status.exit_code} "
                        f"stdout_tail={_tail_text(status.stdout)!r} "
                        f"stderr_tail={_tail_text(status.stderr)!r}"
                    )
                else:
                    logger.debug(
                        f"rollout={state.get('rollout_id')} sandbox={sandbox_id} "
                        f"stage=agent_completed exit_code={status.exit_code}"
                    )
                return
            await asyncio.sleep(self.poll_interval)

    async def wait_for_agent_completion(self, state: State) -> None:
        """Poll for agent completion using background job API."""
        sandbox_id = state.get("sandbox_id")
        background_job = state.get("background_job")
        if not sandbox_id or not background_job:
            state["agent_completed"] = True
            return

        try:
            await asyncio.wait_for(
                self.poll_job_completion(state, sandbox_id, background_job),
                timeout=self._gw_timeout_seconds,
            )
        except asyncio.TimeoutError:
            logger.warning(
                f"rollout={state.get('rollout_id')} sandbox={state.get('sandbox_id')} "
                f"stage=wait_for_agent_completion timed out after {self._gw_timeout_seconds:.1f}s"
            )
            state["agent_timed_out"] = True
        finally:
            state["agent_completed"] = True

    # ------------------------------------------------------------------
    # Timing
    # ------------------------------------------------------------------

    async def _render_timing(self, state: State) -> None:
        start_time = state["timing"]["start_time"]
        end_time = time.time()
        generation_ms = (end_time - start_time) * 1000
        state["timing"]["generation_ms"] = generation_ms
        state["timing"]["total_ms"] = generation_ms

    # ------------------------------------------------------------------
    # rollout() — gateway path with fallback
    # ------------------------------------------------------------------

    async def rollout(
        self,
        input: RolloutInput,
        client: Client | ClientConfig,
        model: str,
        sampling_args: SamplingArgs | None = None,
    ) -> State:
        if not self.use_gateway:
            return await super().rollout(input, client, model, sampling_args)

        state = await self.init_state(input, client, model, sampling_args)
        state["rollout_id"] = f"rollout_{uuid.uuid4().hex[:8]}"
        state["gateway_url"] = self._resolve_gateway_url(state)
        rollout_id = state["rollout_id"]
        info = state.get("info") or {}
        logger.info(
            f"rollout={rollout_id} stage=start model={model} "
            f"example_id={info.get('instance_id') or info.get('example_id')} "
            f"repo={info.get('repo_name')}"
        )

        rollout_registered = False
        try:
            await self.register_rollout(state)
            rollout_registered = True
            logger.debug(f"rollout={rollout_id} stage=register_rollout ok")

            tunnel_local_addr = self._resolve_tunnel_local_addr(state)
            state["tunnel_local_addr"] = tunnel_local_addr
            logger.debug(
                f"rollout={rollout_id} stage=resolve_tunnel_local_addr addr={tunnel_local_addr}"
            )

            tunnel_url = await self.get_gateway_tunnel_url(local_addr=tunnel_local_addr)
            state["tunnel_url"] = tunnel_url
            state["rollout_base_url"] = (
                f"{tunnel_url.rstrip('/')}/v1/rollouts/{state['rollout_id']}"
            )
            logger.debug(f"rollout={rollout_id} stage=start_tunnel url={tunnel_url}")

            env_vars = await self.build_env_vars(state)
            docker_image = await self.get_docker_image(state)
            sandbox_request = CreateSandboxRequest(
                name=state["rollout_id"],
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
                labels=self.labels or [],
            )
            logger.debug(
                f"Creating sandbox with OPENAI_BASE_URL={env_vars.get('OPENAI_BASE_URL')} "
                f"docker_image={docker_image}"
            )
            await self.create_sandbox(state, sandbox_request)
            logger.info(
                f"rollout={rollout_id} stage=create_sandbox ok "
                f"sandbox_id={state.get('sandbox_id')} docker_image={docker_image}"
            )

            await self.start_agent(state)
            logger.debug(
                f"rollout={rollout_id} stage=start_agent ok "
                f"sandbox_id={state.get('sandbox_id')}"
            )
            await self.wait_for_agent_completion(state)
            logger.debug(
                f"rollout={rollout_id} stage=wait_for_agent_completion ok "
                f"exit_code={state.get('agent_exit_code')}"
            )
            await self.fetch_trajectory(state)
            trajectory = state.get("trajectory") or []
            logger.info(
                f"rollout={rollout_id} stage=fetch_trajectory ok "
                f"turns={len(trajectory)} truncated={state.get('is_truncated', False)}"
            )
            if len(trajectory) == 0:
                logger.warning(
                    f"rollout={rollout_id} stage=fetch_trajectory empty_trajectory "
                    f"agent_exit_code={state.get('agent_exit_code')} "
                    f"stdout_tail={_tail_text(state.get('agent_stdout'))!r} "
                    f"stderr_tail={_tail_text(state.get('agent_stderr'))!r}"
                )
        except asyncio.CancelledError:
            if rollout_registered:
                try:
                    await self._gateway_post(state, "cancel")
                except Exception:
                    pass
            raise
        except vf.Error as e:
            state["error"] = e
            logger.exception(
                f"rollout={rollout_id} stage={type(e).__name__} vf_error={e}"
            )
        except Exception as e:
            state["error"] = vf.InfraError(str(e))
            logger.exception(
                f"rollout={rollout_id} stage={type(e).__name__} unhandled_error={e}"
            )
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
                    await self._cleanup(state)
                except Exception as e:
                    logger.warning(
                        f"Failed to destroy sandbox {state.get('sandbox_id')}: {e}"
                    )
                    if state.get("error") is None:
                        state["error"] = vf.InfraError(str(e))

            if state.get("completion") is None:
                state["completion"] = []
            if state.get("stop_condition") is None:
                if state.get("error") is not None:
                    state["stop_condition"] = "has_error"
                elif state.get("agent_timed_out", False):
                    state["stop_condition"] = "agent_timeout"
                else:
                    state["stop_condition"] = "completed"
            state["is_completed"] = True
            await self._render_timing(state)
            error_name = type(state["error"]).__name__ if state.get("error") else None
            logger.info(
                f"rollout={rollout_id} stage=finish stop={state.get('stop_condition')} "
                f"sandbox_id={state.get('sandbox_id')} "
                f"turns={len(state.get('trajectory', []))} "
                f"agent_exit_code={state.get('agent_exit_code')} error={error_name}"
            )

        return state

    # ------------------------------------------------------------------
    # Teardown
    # ------------------------------------------------------------------

    @vf.teardown
    async def teardown_gateway(self):
        """Close gateway HTTP client and stop gateway tunnels."""
        if not hasattr(self, "_gw_http_client"):
            return
        await self._gw_http_client.aclose()
        async with self._gw_tunnel_lock:
            tunnels = list(self._gw_tunnels.items())
            self._gw_tunnels = {}
            for local_addr, tunnel in tunnels:
                try:
                    tunnel.sync_stop()
                    logger.debug(f"Prime Tunnel stopped local_addr={local_addr}")
                except Exception as e:
                    logger.warning(
                        f"Error stopping Prime Tunnel local_addr={local_addr}: {e}"
                    )
