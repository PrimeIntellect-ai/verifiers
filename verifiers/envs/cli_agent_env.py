import asyncio
import logging
import time
import uuid
from typing import Any

from aiohttp import web
from openai import AsyncOpenAI
from prime_sandboxes import (
    AdvancedConfigs,
    AsyncSandboxClient,
    CreateSandboxRequest,
)

import verifiers as vf
from verifiers.types import (
    ChatCompletionToolParam,
    Messages,
    ModelResponse,
    SamplingArgs,
    State,
)
from verifiers.utils.tunnel import TunnelPool

logger = logging.getLogger(__name__)


class CliAgentEnv(vf.MultiTurnEnv):
    """
    Environment for running full agent code inside sandboxes.
    Extends MultiTurnEnv to reuse rollout loop, but intercepts agent's
    API requests via HTTP proxy server. Each agent request triggers one
    rollout step.
    Cloudflare Tunnel is automatically installed and started to expose the
    interception server. Supports high concurrency for parallel requests.
    Fully automated with no configuration or authentication required.
    """

    def __init__(
        self,
        interception_port: int = 8765,
        interception_host: str | None = None,
        max_turns: int = -1,
        timeout_seconds: float = 3600.0,
        request_timeout: float = 300.0,
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
        **kwargs,
    ):
        super().__init__(max_turns=max_turns, message_type="chat", **kwargs)
        self.interception_port = interception_port
        self.interception_host = interception_host
        self._tunnel_pool: TunnelPool | None = (
            TunnelPool(port=interception_port) if interception_host is None else None
        )
        self.timeout_seconds = timeout_seconds
        self.request_timeout = request_timeout
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
        self.active_rollouts: dict[str, dict[str, Any]] = {}
        self.intercepts: dict[str, dict[str, Any]] = {}  # request_id -> intercept data
        self.interception_server: Any = None
        self._server_lock = asyncio.Lock()
        self._server_runner: Any = None
        self._server_site: Any = None

    async def setup_state(self, state: State) -> State:
        """Setup sandbox + interception for this rollout"""
        state = await super().setup_state(state)

        rollout_id = f"rollout_{uuid.uuid4().hex[:8]}"
        state["rollout_id"] = rollout_id

        # Start interception server first (tunnel needs it to be running)
        await self._ensure_interception_server()

        # Get tunnel URL for sandbox to call back
        tunnel_url: str | None = None
        if self._tunnel_pool:
            tunnel_url = await self._tunnel_pool.get_tunnel_url(
                len(self.active_rollouts)
            )
            # Use full HTTPS URL from tunnel
            state["interception_base_url"] = f"{tunnel_url}/rollout/{rollout_id}/v1"
        else:
            # Manual hostname/IP provided - use with configured port
            state["interception_base_url"] = (
                f"http://{self.interception_host}:{self.interception_port}/rollout/{rollout_id}/v1"
            )

        env_vars = dict(self.environment_vars) if self.environment_vars else {}
        env_vars["OPENAI_BASE_URL"] = state["interception_base_url"]
        model = state.get("model")
        if model:
            env_vars["OPENAI_MODEL"] = model

        sandbox_client = AsyncSandboxClient()
        sandbox_request = CreateSandboxRequest(
            name=f"cli-agent-{rollout_id}",
            docker_image=self.docker_image,
            start_command=self.start_command,
            cpu_cores=self.cpu_cores,
            memory_gb=self.memory_gb,
            disk_size_gb=self.disk_size_gb,
            gpu_count=self.gpu_count,
            timeout_minutes=self.timeout_minutes,
            environment_vars=env_vars,
            team_id=self.team_id,
            advanced_configs=self.advanced_configs,
        )
        logger.debug(
            f"Creating sandbox with OPENAI_BASE_URL={env_vars.get('OPENAI_BASE_URL')}"
        )
        sandbox = await sandbox_client.create(sandbox_request)
        state["sandbox_id"] = sandbox.id
        logger.debug(f"Created sandbox {sandbox.id}")
        await sandbox_client.wait_for_creation(sandbox.id)

        request_id_queue: asyncio.Queue = asyncio.Queue()
        state["request_id_queue"] = request_id_queue
        state["current_request_id"] = None
        state["tunnel_url"] = tunnel_url
        self.active_rollouts[rollout_id] = {
            "request_id_queue": request_id_queue,
            "current_request_id": None,
        }

        return state

    async def get_prompt_messages(self, state: State) -> Messages:
        """
        ALL interception logic happens here.
        Pull request_id from queue (blocks until agent makes request), look up intercept,
        process request immediately with injected sampling_args, store response in intercept,
        return messages.
        """
        request_id_queue = state["request_id_queue"]

        request_id = await asyncio.wait_for(
            request_id_queue.get(),
            timeout=self.request_timeout,
        )

        intercept = self.intercepts[request_id]
        messages = intercept["messages"]
        request_model = intercept.get("model") or state.get("model")
        if request_model is None:
            raise RuntimeError("Model not set in state")
        request_tools = intercept.get("tools")
        effective_sampling_args = state.get("sampling_args") or {}

        client = state.get("client")
        if client is None:
            raise RuntimeError("Client not set in state")

        # call get_model_response early and store response in intercept
        response = await super().get_model_response(
            client=client,
            model=request_model,
            prompt=messages,
            oai_tools=request_tools,
            sampling_args=effective_sampling_args,
            message_type=None,
        )

        intercept["response_future"].set_result(response)
        intercept["response"] = response
        state["current_request_id"] = request_id
        self.active_rollouts[state["rollout_id"]]["current_request_id"] = request_id

        return messages

    async def get_model_response(
        self,
        client: AsyncOpenAI,
        model: str,
        prompt: Messages,
        oai_tools: list[ChatCompletionToolParam] | None = None,
        sampling_args: SamplingArgs | None = None,
        message_type: str | None = None,
    ) -> ModelResponse:
        """
        Return cached response if available (set by get_prompt_messages).
        Otherwise fall back to parent implementation.
        """
        for request_id, intercept in list(self.intercepts.items()):
            rollout_id = intercept.get("rollout_id")
            if rollout_id and rollout_id in self.active_rollouts:
                context = self.active_rollouts[rollout_id]
                if context.get("current_request_id") == request_id:
                    if "response" in intercept:
                        response = intercept.pop("response")
                        context["current_request_id"] = None
                        return response

        return await super().get_model_response(
            client=client,
            model=model,
            prompt=prompt,
            oai_tools=oai_tools,
            sampling_args=sampling_args,
            message_type=None,
        )

    async def _ensure_interception_server(self):
        """Start shared HTTP server if needed"""
        async with self._server_lock:
            if self.interception_server is not None:
                return

            app = web.Application()  # type: ignore
            app.router.add_post(
                "/rollout/{rollout_id}/v1/chat/completions",
                self._handle_intercepted_request,
            )

            runner = web.AppRunner(app)  # type: ignore
            await runner.setup()
            site = web.TCPSite(runner, "0.0.0.0", self.interception_port)  # type: ignore
            await site.start()

            self.interception_server = app
            self._server_runner = runner
            self._server_site = site

            logger.debug(
                f"Started interception server on port {self.interception_port}"
            )

    async def _handle_intercepted_request(self, request: Any) -> Any:
        """HTTP handler: queue request, wait for response, return"""
        rollout_id = request.match_info["rollout_id"]
        context = self.active_rollouts.get(rollout_id)
        if not context:
            return web.json_response(  # type: ignore
                {"error": "Rollout not found"}, status=404
            )

        try:
            request_body = await request.json()
        except Exception as e:
            return web.json_response(  # type: ignore
                {"error": f"Invalid JSON: {e}"}, status=400
            )

        request_id = f"req_{uuid.uuid4().hex[:8]}"
        intercept = {
            "request_id": request_id,
            "rollout_id": rollout_id,
            "messages": request_body["messages"],
            "model": request_body.get("model"),
            "tools": request_body.get("tools"),
            "response_future": asyncio.Future(),
        }

        self.intercepts[request_id] = intercept
        await context["request_id_queue"].put(request_id)

        try:
            response = await intercept["response_future"]
        except Exception as e:
            logger.error(f"Error processing intercepted request: {e}")
            return web.json_response(  # type: ignore
                {"error": str(e)}, status=500
            )

        response_dict = (
            response.model_dump() if hasattr(response, "model_dump") else dict(response)
        )
        return web.json_response(response_dict)  # type: ignore

    @vf.teardown
    async def teardown_tunnel(self):
        """Stop all cloudflared tunnel processes"""
        if self._tunnel_pool:
            self._tunnel_pool.teardown()

    @vf.cleanup
    async def cleanup_interception_context(self, state: State):
        """Cleanup interception context for rollout"""
        rollout_id = state.get("rollout_id")
        if rollout_id and rollout_id in self.active_rollouts:
            context = self.active_rollouts[rollout_id]
            request_id = context.get("current_request_id")
            if request_id and request_id in self.intercepts:
                del self.intercepts[request_id]
            del self.active_rollouts[rollout_id]

        # Release tunnel
        tunnel_url = state.get("tunnel_url")
        if tunnel_url and self._tunnel_pool:
            await self._tunnel_pool.release_tunnel(tunnel_url)

    @vf.stop
    async def agent_signaled_completion(self, state: State) -> bool:
        """Check for /tmp/vf_complete marker file"""
        sandbox_id = state.get("sandbox_id")
        if not sandbox_id:
            return False

        try:
            sandbox_client = AsyncSandboxClient()
            result = await sandbox_client.execute_command(
                sandbox_id,
                "test -f /tmp/vf_complete && echo 'done' || echo 'not_done'",
            )
            return result.stdout.strip() == "done"
        except Exception as e:
            logger.debug(f"Error checking completion signal: {e}")
            return False

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
        """Cleanup sandbox after rollout"""
        await self.post_rollout(state)
        sandbox_id = state.get("sandbox_id")
        if sandbox_id:
            try:
                sandbox_client = AsyncSandboxClient()
                await sandbox_client.delete(sandbox_id)
                logger.debug(f"Deleted sandbox {sandbox_id}")
            except Exception as e:
                logger.warning(f"Failed to delete sandbox {sandbox_id}: {e}")

    async def env_response(
        self, messages: Messages, state: State, **kwargs
    ) -> Messages:
        """
        Generate a response from the environment.
        For CliAgentEnv, there is no environment response - the agent
        controls the conversation flow via its requests.
        """
        return []
