import asyncio
import json
import logging
import time
import uuid
from typing import Any, cast

from aiohttp import web
from openai import AsyncOpenAI
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionMessage,
    ChatCompletionMessageToolCall,
)
from openai.types.chat.chat_completion import Choice
from openai.types.chat.chat_completion_message_tool_call import Function
from prime_sandboxes import (
    AdvancedConfigs,
    AsyncSandboxClient,
    BackgroundJob,
    BackgroundJobStatus,
    CreateSandboxRequest,
    SandboxClient,
)
from prime_sandboxes.core import APIClient
from prime_tunnel import Tunnel

import verifiers as vf
from verifiers.types import (
    ChatCompletionToolParam,
    Messages,
    MessageType,
    ModelResponse,
    SamplingArgs,
    State,
)

logger = logging.getLogger(__name__)


class CliAgentEnv(vf.MultiTurnEnv):
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
        **kwargs,
    ):
        super().__init__(max_turns=max_turns, message_type="chat", **kwargs)
        self.run_command = run_command
        self.poll_interval = poll_interval
        self.interception_port = interception_port
        self.interception_url = interception_url
        self.tunnel: Tunnel | None = None
        self.tunnel_lock = asyncio.Lock()
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
        self.active_rollouts: dict[str, dict[str, Any]] = {}
        self.intercepts: dict[str, dict[str, Any]] = {}  # request_id -> intercept data
        self.active_sandboxes: set[str] = set()
        self.interception_server: Any = None
        self.server_lock = asyncio.Lock()
        self.server_runner: Any = None
        self.server_site: Any = None

    async def get_tunnel_url(self) -> str:
        """Get tunnel URL, starting the tunnel if needed."""
        async with self.tunnel_lock:
            if self.tunnel is None:
                if logger.isEnabledFor(logging.DEBUG):
                    self.tunnel = Tunnel(
                        local_port=self.interception_port,
                        log_level="debug",
                    )
                else:
                    self.tunnel = Tunnel(local_port=self.interception_port)
                url = await self.tunnel.start()
                logger.debug(f"Prime Tunnel started: {url}")
                return url
            else:
                assert self.tunnel.url is not None, "Tunnel started but URL is None"
                return self.tunnel.url

    async def setup_state(self, state: State) -> State:
        """Setup sandbox + interception for this rollout"""
        state = await super().setup_state(state)

        rollout_id = f"rollout_{uuid.uuid4().hex[:8]}"
        state["rollout_id"] = rollout_id

        await self.ensure_interception_server()

        # Auto-start Prime Tunnel if no interception URL provided
        if self.interception_url is None:
            tunnel_url = await self.get_tunnel_url()
            state["interception_base_url"] = f"{tunnel_url}/rollout/{rollout_id}/v1"
        else:
            state["interception_base_url"] = (
                f"{self.interception_url.rstrip('/')}/rollout/{rollout_id}/v1"
            )

        env_vars = await self.build_env_vars(state)
        docker_image = await self.get_docker_image(state)

        sandbox_client = AsyncSandboxClient(
            max_connections=100,
            max_keepalive_connections=50,
        )
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
        sandbox = await sandbox_client.create(sandbox_request)
        state["sandbox_id"] = sandbox.id
        self.active_sandboxes.add(sandbox.id)
        logger.debug(f"Created sandbox {sandbox.id}")
        await sandbox_client.wait_for_creation(sandbox.id, max_attempts=120)

        await self.post_sandbox_setup(state, sandbox_client)

        request_id_queue: asyncio.Queue = asyncio.Queue()
        state["request_id_queue"] = request_id_queue
        state["agent_completed"] = False
        self.active_rollouts[rollout_id] = {
            "request_id_queue": request_id_queue,
        }

        await self.start_agent(state, sandbox_client)

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

    async def post_sandbox_setup(
        self, state: State, sandbox_client: AsyncSandboxClient
    ) -> None:
        """Hook for post-sandbox setup. Override to upload files, run commands, etc."""
        pass

    async def start_agent(
        self, state: State, sandbox_client: AsyncSandboxClient
    ) -> None:
        """Start the agent command using background job."""
        sandbox_id = state["sandbox_id"]

        # Start the agent as a background job
        background_job: BackgroundJob = await sandbox_client.start_background_job(
            sandbox_id,
            self.run_command,
        )
        state["background_job"] = background_job
        state["agent_start_time"] = time.time()

        # Start the polling task
        state["completion_wait_task"] = asyncio.create_task(
            self.wait_for_completion(state, sandbox_client)
        )

    async def wait_for_completion(
        self, state: State, sandbox_client: AsyncSandboxClient
    ) -> None:
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
        sandbox_client = AsyncSandboxClient(
            max_connections=100,
            max_keepalive_connections=50,
        )
        while True:
            status: BackgroundJobStatus = await sandbox_client.get_background_job(
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
                intercept = self.intercepts[request_id]
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
            return ChatCompletion(
                id="agent-completed",
                choices=[
                    Choice(
                        finish_reason="stop",
                        index=0,
                        message=ChatCompletionMessage(role="assistant", content=""),
                    )
                ],
                created=int(time.time()),
                model=model or state["model"],
                object="chat.completion",
            )

        request_id = state.get("current_request_id")
        intercept = self.intercepts.get(request_id) if request_id else None

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
                response = await self._get_streaming_model_response(
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
                future = intercept.get("response_future")
                if future and not future.done():
                    if error is not None:
                        future.set_exception(error)
                    elif response is not None:
                        future.set_result(response)
                state["current_request_id"] = None

        return response

    async def _get_streaming_model_response(
        self,
        state: State,
        prompt: Messages,
        intercept: dict,
        client: AsyncOpenAI | None = None,
        model: str | None = None,
        oai_tools: list[ChatCompletionToolParam] | None = None,
        sampling_args: SamplingArgs | None = None,
    ) -> ChatCompletion:
        """Handle streaming API call, forwarding chunks and accumulating response."""
        chunk_queue = cast(asyncio.Queue, intercept["chunk_queue"])

        # Resolve client and model
        client = client or state["client"]
        model = model or state["model"]
        sampling_args = sampling_args or state.get("sampling_args") or {}

        # Remove max_tokens and use max_completion_tokens for chat
        if "max_tokens" in sampling_args:
            sampling_args = dict(sampling_args)
            max_tokens = sampling_args.pop("max_tokens")
            if "max_completion_tokens" not in sampling_args:
                sampling_args["max_completion_tokens"] = max_tokens

        # Make streaming API call
        create_kwargs: dict[str, Any] = {
            "model": model,
            "messages": prompt,
            "stream": True,
        }
        if oai_tools:
            create_kwargs["tools"] = oai_tools
        create_kwargs.update(sampling_args)

        stream = await client.chat.completions.create(**create_kwargs)

        # Accumulate response while streaming chunks
        accumulated_content = ""
        accumulated_tool_calls: dict[int, dict] = {}  # index -> {id, type, function}
        finish_reason = None
        completion_id = None
        created_time = int(time.time())
        stream_ended = False

        try:
            async for chunk in stream:
                # Forward chunk to HTTP handler
                await chunk_queue.put(chunk)

                # Accumulate data
                if not completion_id and chunk.id:
                    completion_id = chunk.id
                if chunk.created:
                    created_time = chunk.created

                if chunk.choices:
                    choice = chunk.choices[0]
                    if choice.finish_reason:
                        finish_reason = choice.finish_reason

                    delta = choice.delta
                    if delta:
                        if delta.content:
                            accumulated_content += delta.content

                        # Accumulate tool calls
                        if delta.tool_calls:
                            for tc in delta.tool_calls:
                                idx = tc.index
                                if idx not in accumulated_tool_calls:
                                    accumulated_tool_calls[idx] = {
                                        "id": tc.id or "",
                                        "type": tc.type or "function",
                                        "function": {"name": "", "arguments": ""},
                                    }
                                if tc.id:
                                    accumulated_tool_calls[idx]["id"] = tc.id
                                if tc.function:
                                    if tc.function.name:
                                        accumulated_tool_calls[idx]["function"][
                                            "name"
                                        ] = tc.function.name
                                    if tc.function.arguments:
                                        accumulated_tool_calls[idx]["function"][
                                            "arguments"
                                        ] += tc.function.arguments

            # Signal end of stream
            await chunk_queue.put(None)
            stream_ended = True
        finally:
            # Always signal end of stream to unblock HTTP handler
            if not stream_ended:
                try:
                    chunk_queue.put_nowait(None)
                except asyncio.QueueFull:
                    pass

        # Build accumulated ChatCompletion
        tool_calls_list = None
        if accumulated_tool_calls:
            tool_calls_list = [
                ChatCompletionMessageToolCall(
                    id=tc_data["id"],
                    type="function",
                    function=Function(
                        name=tc_data["function"]["name"],
                        arguments=tc_data["function"]["arguments"],
                    ),
                )
                for idx, tc_data in sorted(accumulated_tool_calls.items())
            ]

        message = ChatCompletionMessage(
            role="assistant",
            content=accumulated_content if accumulated_content else None,
            tool_calls=tool_calls_list,
        )

        result = ChatCompletion(
            id=completion_id or f"chatcmpl-{uuid.uuid4().hex[:8]}",
            choices=[
                Choice(
                    finish_reason=finish_reason or "stop",
                    index=0,
                    message=message,
                )
            ],
            created=created_time,
            model=model,
            object="chat.completion",
        )

        # Log the accumulated response
        rollout_id = intercept.get("rollout_id", "?")
        self.log_response(rollout_id, result.model_dump())

        return result

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

    def truncate(self, s: str, limit: int = 200) -> str:
        return (s[:limit] + "...") if len(s) > limit else s

    def log_request(self, rollout_id: str, body: dict) -> None:
        logger.debug(f"[{rollout_id}] <- INTERCEPTED REQUEST")
        for msg in body.get("messages", []):
            logger.debug(
                f"  [{msg.get('role', '?')}] {self.truncate(msg.get('content', ''))}"
            )
        if body.get("tools"):
            logger.debug(f"  [tools] {len(body['tools'])} tool(s)")

    def log_response(self, rollout_id: str, response: dict) -> None:
        logger.debug(f"[{rollout_id}] -> RESPONSE")
        msg = response.get("choices", [{}])[0].get("message", {})
        if msg.get("content"):
            logger.debug(f"  [assistant] {self.truncate(msg['content'])}")
        for tc in msg.get("tool_calls") or []:
            func = tc.get("function", {})
            logger.debug(
                f"  [tool_call] {func.get('name')}({self.truncate(func.get('arguments', ''), 100)})"
            )

    async def ensure_interception_server(self):
        """Start shared HTTP server if needed"""
        async with self.server_lock:
            if self.interception_server is not None:
                return

            app = web.Application()
            app.router.add_post(
                "/rollout/{rollout_id}/v1/chat/completions",
                self.handle_intercepted_request,
            )

            app.router.add_get(
                "/health",
                lambda _: web.json_response({"status": "ok"}),
            )

            runner = web.AppRunner(app)
            await runner.setup()
            site = web.TCPSite(runner, "0.0.0.0", self.interception_port)
            await site.start()

            self.interception_server = app
            self.server_runner = runner
            self.server_site = site

            logger.debug(
                f"Started interception server on port {self.interception_port}"
            )

    async def handle_intercepted_request(self, request: Any) -> Any:
        """HTTP handler: queue request, wait for response, return"""
        rollout_id = request.match_info["rollout_id"]
        context = self.active_rollouts.get(rollout_id)
        if not context:
            return web.json_response({"error": "Rollout not found"}, status=404)

        try:
            request_body = await request.json()
        except Exception as e:
            return web.json_response({"error": f"Invalid JSON: {e}"}, status=400)

        self.log_request(rollout_id, request_body)

        is_streaming = request_body.get("stream", False)
        request_id = f"req_{uuid.uuid4().hex[:8]}"

        # For streaming, we use a queue to pass chunks from get_model_response
        chunk_queue: asyncio.Queue | None = asyncio.Queue() if is_streaming else None

        intercept = {
            "request_id": request_id,
            "rollout_id": rollout_id,
            "messages": request_body["messages"],
            "model": request_body.get("model"),
            "tools": request_body.get("tools"),
            "stream": is_streaming,
            "chunk_queue": chunk_queue,
            "response_future": asyncio.Future(),
        }

        self.intercepts[request_id] = intercept
        await context["request_id_queue"].put(request_id)

        if is_streaming:
            return await self._handle_streaming_response(request, rollout_id, intercept)
        else:
            try:
                response_future = cast(
                    asyncio.Future[Any], intercept["response_future"]
                )
                response = await response_future
            except asyncio.CancelledError:
                return web.json_response({"error": "Rollout cancelled"}, status=499)
            except Exception as e:
                logger.error(f"Error processing intercepted request: {e}")
                return web.json_response({"error": str(e)}, status=500)

            response_dict = (
                response.model_dump()
                if hasattr(response, "model_dump")
                else dict(response)
            )

            self.log_response(rollout_id, response_dict)
            return web.json_response(response_dict)

    async def _handle_streaming_response(
        self, http_request: Any, rollout_id: str, intercept: dict
    ) -> Any:
        """Handle streaming SSE response to the agent."""
        chunk_queue = cast(asyncio.Queue, intercept["chunk_queue"])
        response_future = cast(asyncio.Future[Any], intercept["response_future"])

        response = web.StreamResponse(
            status=200,
            headers={
                "Content-Type": "text/event-stream",
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            },
        )
        await response.prepare(http_request)

        try:
            while True:
                # Wait for chunks from get_model_response
                chunk = await chunk_queue.get()

                if chunk is None:
                    # End of stream signal
                    await response.write(b"data: [DONE]\n\n")
                    break

                # Convert chunk to SSE format
                chunk_dict = (
                    chunk.model_dump() if hasattr(chunk, "model_dump") else dict(chunk)
                )
                chunk_json = json.dumps(chunk_dict)
                await response.write(f"data: {chunk_json}\n\n".encode())

            # Wait for the accumulated response
            await response_future

        except asyncio.CancelledError:
            logger.debug(f"[{rollout_id}] Streaming cancelled")
        except Exception as e:
            logger.error(f"[{rollout_id}] Streaming error: {e}")

        await response.write_eof()
        return response

    @vf.teardown
    async def teardown_tunnel(self):
        """Stop Prime Tunnel and HTTP interception server"""
        # Stop Prime Tunnel
        async with self.tunnel_lock:
            if self.tunnel is not None:
                try:
                    await self.tunnel.stop()
                    logger.debug("Prime Tunnel stopped")
                except Exception as e:
                    logger.warning(f"Error stopping Prime Tunnel: {e}")
                finally:
                    self.tunnel = None

        # Stop HTTP interception server
        async with self.server_lock:
            if self.server_runner is not None:
                try:
                    await self.server_runner.cleanup()
                    logger.debug("Stopped HTTP interception server")
                except RuntimeError as e:
                    if "Event loop is closed" not in str(e):
                        raise
                    logger.debug("HTTP server cleanup skipped (event loop closed)")
                finally:
                    self.server_runner = None
                    self.server_site = None
                    self.interception_server = None

    @vf.teardown
    async def teardown_sandboxes(self):
        """Delete all active sandboxes on teardown.

        Uses the synchronous SandboxClient for teardown to avoid event loop issues
        during signal handling and interpreter shutdown.
        """
        if len(self.active_sandboxes) == 0:
            return
        logger.info(f"Deleting {len(self.active_sandboxes)} remaining sandboxes")

        sync_client = SandboxClient(APIClient())
        sandbox_ids = list(self.active_sandboxes)

        batch_size = 100
        for i in range(0, len(sandbox_ids), batch_size):
            batch = sandbox_ids[i : i + batch_size]
            try:
                sync_client.bulk_delete(sandbox_ids=batch)
                for sandbox_id in batch:
                    self.active_sandboxes.discard(sandbox_id)
                logger.debug(f"Bulk deleted batch of {len(batch)} sandboxes")
            except Exception as e:
                logger.warning(f"Bulk delete failed for batch: {e}")

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
            for request_id in list(self.intercepts.keys()):
                intercept = self.intercepts.get(request_id)
                if intercept and intercept.get("rollout_id") == rollout_id:
                    # For streaming requests, signal the chunk queue to exit
                    chunk_queue = intercept.get("chunk_queue")
                    if chunk_queue is not None:
                        try:
                            chunk_queue.put_nowait(None)
                        except asyncio.QueueFull:
                            pass
                    # Cancel pending future to unblock HTTP handler
                    future = intercept.get("response_future")
                    if future and not future.done():
                        future.cancel()
                    del self.intercepts[request_id]

            if rollout_id in self.active_rollouts:
                del self.active_rollouts[rollout_id]

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
        """Cleanup sandbox after rollout"""
        await self.post_rollout(state)
        sandbox_id = state.get("sandbox_id")
        if sandbox_id:
            try:
                sandbox_client = AsyncSandboxClient(
                    max_connections=100,
                    max_keepalive_connections=50,
                )
                await sandbox_client.delete(sandbox_id)
                self.active_sandboxes.discard(sandbox_id)
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
