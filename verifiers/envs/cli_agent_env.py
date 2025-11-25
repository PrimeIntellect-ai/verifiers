import asyncio
import logging
import subprocess
import time
import uuid
import json
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

logger = logging.getLogger(__name__)


def _truncate(text: str, max_len: int = 500) -> str:
    """Truncate text for logging."""
    if len(text) <= max_len:
        return text
    return text[:max_len] + f"... ({len(text) - max_len} more chars)"


def _format_message(msg: dict) -> str:
    """Format a message for logging."""
    role = msg.get("role", "unknown")
    content = msg.get("content", "")

    # Handle tool calls
    tool_calls = msg.get("tool_calls", [])
    if tool_calls:
        tools_summary = []
        for tc in tool_calls:
            func = tc.get("function", {})
            name = func.get("name", "unknown")
            args = func.get("arguments", "")
            tools_summary.append(f"{name}({_truncate(args, 200)})")
        return f"[{role}] tool_calls: {', '.join(tools_summary)}"

    # Handle tool results
    if role == "tool":
        tool_call_id = msg.get("tool_call_id", "")
        return f"[{role}:{tool_call_id}] {_truncate(str(content), 300)}"

    return f"[{role}] {_truncate(str(content), 500)}"


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
        log_requests: bool = True,
        **kwargs,
    ):
        super().__init__(max_turns=max_turns, message_type="chat", **kwargs)
        self.interception_port = interception_port
        self.interception_host = interception_host
        self._tunnels: list[
            dict[str, Any]
        ] = []  # List of {url, process, active_rollouts}
        self._tunnel_lock = asyncio.Lock()
        self._tunnel_round_robin_index = 0
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
        self.log_requests = log_requests
        self.active_rollouts: dict[str, dict[str, Any]] = {}
        self.intercepts: dict[str, dict[str, Any]] = {}  # request_id -> intercept data
        self.interception_server: Any = None
        self._server_lock = asyncio.Lock()
        self._server_runner: Any = None
        self._server_site: Any = None
        self._request_counts: dict[str, int] = {}  # rollout_id -> request count

    def _ensure_cloudflared_installed(self) -> str:
        """Install cloudflared if not already installed. Returns path to cloudflared binary."""
        import platform
        import shutil

        cloudflared_path = shutil.which("cloudflared")
        if cloudflared_path:
            return cloudflared_path

        logger.info("Installing cloudflared...")
        system = platform.system()

        if system == "Darwin":  # macOS
            result = subprocess.run(
                ["brew", "install", "cloudflare/cloudflare/cloudflared"],
                capture_output=True,
                text=True,
                timeout=300,
            )
            if result.returncode != 0:
                raise RuntimeError(
                    f"Failed to install cloudflared via Homebrew: {result.stderr}"
                )
            cloudflared_path = shutil.which("cloudflared")
            if not cloudflared_path:
                raise RuntimeError("cloudflared installed but not found in PATH")
            return cloudflared_path
        elif system == "Linux":
            # Official cloudflared installation script
            install_script = "curl -L --output cloudflared.deb https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb && sudo dpkg -i cloudflared.deb && rm cloudflared.deb"
            result = subprocess.run(
                ["bash", "-c", install_script],
                capture_output=True,
                text=True,
                timeout=300,
            )
            if result.returncode != 0:
                raise RuntimeError(f"Failed to install cloudflared: {result.stderr}")
            cloudflared_path = shutil.which("cloudflared")
            if not cloudflared_path:
                raise RuntimeError("cloudflared installed but not found in PATH")
            return cloudflared_path
        else:
            raise RuntimeError(
                f"Unsupported platform: {system}. "
                "Please install cloudflared manually: https://developers.cloudflare.com/cloudflare-one/connections/connect-apps/install-and-setup/installation/"
            )

    def _extract_tunnel_url_from_line(self, line: str) -> str | None:
        """Extract tunnel URL from a line of cloudflared output."""
        if ".trycloudflare.com" not in line:
            return None

        # Find the start of the URL
        start_idx = line.find("https://")
        if start_idx == -1:
            return None

        # Extract URL up to the next whitespace or end of line
        url_start = start_idx
        url_end = url_start + 8  # Skip "https://"
        while url_end < len(line) and not line[url_end].isspace():
            url_end += 1

        url = line[url_start:url_end].rstrip("/")
        if ".trycloudflare.com" in url:
            return url
        return None

    def _start_cloudflared_tunnel(self) -> tuple[str, subprocess.Popen]:
        """Start cloudflared tunnel and return (URL, process)."""
        cloudflared_path = self._ensure_cloudflared_installed()

        # Start cloudflared tunnel process
        tunnel_process = subprocess.Popen(
            [
                cloudflared_path,
                "tunnel",
                "--url",
                f"http://localhost:{self.interception_port}",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )

        # Read stderr line by line until we find the tunnel URL
        stderr_lines = []
        max_wait_seconds = 30
        check_interval = 0.5
        max_iterations = int(max_wait_seconds / check_interval)

        for _ in range(max_iterations):
            # Check if process died
            if tunnel_process.poll() is not None:
                if tunnel_process.stderr:
                    remaining = tunnel_process.stderr.read()
                    stderr_lines.append(remaining)
                error_output = "".join(stderr_lines)
                raise RuntimeError(
                    f"cloudflared tunnel failed to start: {error_output}"
                )

            # Try to read a line from stderr
            if tunnel_process.stderr:
                line = tunnel_process.stderr.readline()
                if line:
                    stderr_lines.append(line)
                    url = self._extract_tunnel_url_from_line(line)
                    if url:
                        logger.info(f"Cloudflare tunnel started: {url}")
                        return url, tunnel_process

            time.sleep(check_interval)

        # Final check: search all collected lines
        all_output = "".join(stderr_lines)
        for line in stderr_lines:
            url = self._extract_tunnel_url_from_line(line)
            if url:
                logger.info(f"Cloudflare tunnel started: {url}")
                return url, tunnel_process

        raise RuntimeError(
            f"Failed to get tunnel URL from cloudflared after {max_wait_seconds} seconds. "
            f"Output: {all_output[:500]}"
        )

    async def _get_tunnel_url(self) -> str:
        """Get tunnel URL from pool, creating new tunnels as needed (1 per 50 active rollouts)."""
        async with self._tunnel_lock:
            # Count total active rollouts
            total_active_rollouts = len(self.active_rollouts)

            # Calculate required tunnels (at least 1 per 50 rollouts, minimum 1)
            required_tunnels = max(1, (total_active_rollouts + 49) // 50)

            # Ensure we have enough tunnels
            while len(self._tunnels) < required_tunnels:
                try:
                    url, process = self._start_cloudflared_tunnel()
                    self._tunnels.append(
                        {
                            "url": url,
                            "process": process,
                            "active_rollouts": 0,
                        }
                    )
                    logger.debug(
                        f"Created tunnel {len(self._tunnels)}/{required_tunnels}: {url}"
                    )
                except Exception as e:
                    logger.error(f"Failed to create tunnel: {e}")
                    raise

            # Round-robin selection
            tunnel = self._tunnels[self._tunnel_round_robin_index % len(self._tunnels)]
            self._tunnel_round_robin_index += 1

            # Increment active rollouts for this tunnel
            tunnel["active_rollouts"] += 1

            return tunnel["url"]

    async def setup_state(self, state: State) -> State:
        """Setup sandbox + interception for this rollout"""
        state = await super().setup_state(state)

        rollout_id = f"rollout_{uuid.uuid4().hex[:8]}"
        state["rollout_id"] = rollout_id

        # Start interception server first (tunnel needs it to be running)
        await self._ensure_interception_server()

        # Auto-start Cloudflare tunnel if not provided
        tunnel_url: str | None = None
        if self.interception_host is None:
            tunnel_url = await self._get_tunnel_url()
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
        state["tunnel_url"] = tunnel_url if self.interception_host is None else None
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
        rollout_id = state.get("rollout_id", "unknown")
        request_id_queue = state["request_id_queue"]

        # Track request count for logging
        if rollout_id not in self._request_counts:
            self._request_counts[rollout_id] = 0
        self._request_counts[rollout_id] += 1
        req_num = self._request_counts[rollout_id]

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

        # Log the intercepted request
        if self.log_requests:
            logger.info(
                f"[Request #{req_num}] model={request_model}, "
                f"messages={len(messages)}, tools={len(request_tools or [])}"
            )
            # Log the last few messages (most relevant context)
            for msg in messages[-3:]:
                logger.info(f"  {_format_message(msg)}")

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

        # Log the response
        if self.log_requests and response.choices:
            choice = response.choices[0]
            msg = choice.message
            if msg.tool_calls:
                tools = [f"{tc.function.name}(...)" for tc in msg.tool_calls]
                logger.info(f"[Response #{req_num}] tool_calls: {', '.join(tools)}")
            elif msg.content:
                logger.info(f"[Response #{req_num}] {_truncate(msg.content, 200)}")
            else:
                logger.info(f"[Response #{req_num}] (empty)")

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

        # Log request details including stream parameter
        stream_requested = request_body.get("stream", False)
        logger.info(
            f"Intercepted request: stream={stream_requested}, "
            f"model={request_body.get('model')}, "
            f"messages={len(request_body.get('messages', []))}"
        )

        # Force non-streaming - we don't support SSE streaming yet
        # The response will be converted to SSE format if stream was requested
        request_body["stream"] = False

        request_id = f"req_{uuid.uuid4().hex[:8]}"
        intercept = {
            "request_id": request_id,
            "rollout_id": rollout_id,
            "messages": request_body["messages"],
            "model": request_body.get("model"),
            "tools": request_body.get("tools"),
            "stream_requested": stream_requested,  # Remember if client wanted streaming
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

        logger.info(
            f"Response to agent: {json.dumps(response_dict, indent=2, default=str)[:2000]}"
        )

        # If client requested streaming, convert to SSE format
        if intercept.get("stream_requested", False):
            return self._create_sse_response(response_dict)

        return web.json_response(response_dict)  # type: ignore

    def _create_sse_response(self, response_dict: dict) -> web.Response:
        """Convert a chat completion response to SSE streaming format."""
        response_id = response_dict.get("id", "chatcmpl-unknown")
        created = response_dict.get("created", 0)
        model = response_dict.get("model", "unknown")

        chunks = []

        for choice in response_dict.get("choices", []):
            message = choice.get("message", {})
            finish_reason = choice.get("finish_reason")
            index = choice.get("index", 0)

            # First chunk: role
            first_chunk = {
                "id": response_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [
                    {
                        "index": index,
                        "delta": {"role": message.get("role", "assistant")},
                        "finish_reason": None,
                    }
                ],
            }
            chunks.append(f"data: {json.dumps(first_chunk)}\n\n")

            # Content chunk (if any)
            content = message.get("content")
            if content:
                content_chunk = {
                    "id": response_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model,
                    "choices": [
                        {
                            "index": index,
                            "delta": {"content": content},
                            "finish_reason": None,
                        }
                    ],
                }
                chunks.append(f"data: {json.dumps(content_chunk)}\n\n")

            # Tool calls chunks (if any)
            tool_calls = message.get("tool_calls", [])
            if tool_calls:
                # Send tool call with full info in one chunk
                tc_delta = []
                for i, tc in enumerate(tool_calls):
                    tc_delta.append(
                        {
                            "index": i,
                            "id": tc.get("id"),
                            "type": tc.get("type", "function"),
                            "function": {
                                "name": tc.get("function", {}).get("name", ""),
                                "arguments": tc.get("function", {}).get(
                                    "arguments", ""
                                ),
                            },
                        }
                    )

                tool_chunk = {
                    "id": response_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model,
                    "choices": [
                        {
                            "index": index,
                            "delta": {"tool_calls": tc_delta},
                            "finish_reason": None,
                        }
                    ],
                }
                chunks.append(f"data: {json.dumps(tool_chunk)}\n\n")

            # Final chunk with finish_reason
            final_chunk = {
                "id": response_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [
                    {
                        "index": index,
                        "delta": {},
                        "finish_reason": finish_reason,
                    }
                ],
            }
            chunks.append(f"data: {json.dumps(final_chunk)}\n\n")

        # End of stream
        chunks.append("data: [DONE]\n\n")

        body = "".join(chunks)
        logger.debug(f"SSE response body:\n{body[:1500]}")
        return web.Response(
            body=body,
            status=200,
            content_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            },
        )

    @vf.teardown
    async def teardown_tunnel(self):
        """Stop all cloudflared tunnel processes"""
        async with self._tunnel_lock:
            for tunnel in self._tunnels:
                process = tunnel.get("process")
                if process:
                    try:
                        process.terminate()
                        process.wait(timeout=5)
                    except Exception as e:
                        logger.warning(f"Error stopping cloudflared tunnel: {e}")
                        try:
                            process.kill()
                        except Exception:
                            pass
            self._tunnels.clear()

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

        # Clean up request count
        if rollout_id and rollout_id in self._request_counts:
            del self._request_counts[rollout_id]

        # Decrement active rollouts for the tunnel used by this rollout
        tunnel_url = state.get("tunnel_url")
        if tunnel_url:
            async with self._tunnel_lock:
                for tunnel in self._tunnels:
                    if tunnel["url"] == tunnel_url:
                        tunnel["active_rollouts"] = max(
                            0, tunnel["active_rollouts"] - 1
                        )
                        break

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
