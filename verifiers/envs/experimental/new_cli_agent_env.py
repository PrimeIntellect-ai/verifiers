"""CLI Agent environment using SandboxManager for resource lifecycle."""

import asyncio
import json
import logging
import time
import uuid
from typing import Any, cast

from aiohttp import web
from aiohttp.client_exceptions import ClientConnectionResetError
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
)
from prime_tunnel import Tunnel

import verifiers as vf
from verifiers.envs.experimental.resource_managers import SandboxManager
from verifiers.types import (
    ChatCompletionToolParam,
    Messages,
    MessageType,
    ModelResponse,
    SamplingArgs,
    State,
)

logger = logging.getLogger(__name__)


class NewCliAgentEnv(vf.MultiTurnEnv):
    """CLI agent environment with managed sandbox lifecycle.

    Runs agent code in sandboxes and intercepts API calls.
    Uses SandboxManager for sandbox lifecycle instead of direct client calls.
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
        **kwargs: Any,
    ):
        super().__init__(max_turns=max_turns, message_type="chat", **kwargs)
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

        # Interception state
        self.tunnel: Tunnel | None = None
        self.tunnel_lock = asyncio.Lock()
        self.active_rollouts: dict[str, dict[str, Any]] = {}
        self.intercepts: dict[str, dict[str, Any]] = {}
        self.interception_server: Any = None
        self.server_lock = asyncio.Lock()
        self.server_runner: Any = None
        self.server_site: Any = None

        # Sandbox manager
        request = CreateSandboxRequest(
            name="cli-agent",
            docker_image=docker_image,
            start_command=start_command,
            cpu_cores=cpu_cores,
            memory_gb=memory_gb,
            disk_size_gb=disk_size_gb,
            gpu_count=gpu_count,
            timeout_minutes=timeout_minutes,
            environment_vars=environment_vars,
            team_id=team_id,
            advanced_configs=advanced_configs,
            labels=labels or [],
        )
        self.sandbox_manager = SandboxManager(
            default_request=request,
            timeout_per_command=300,
        )

    # =========================================================================
    # Sandbox lifecycle (uses manager)
    # =========================================================================

    async def setup_state(self, state: State) -> State:
        state = await super().setup_state(state)

        rollout_id = f"rollout_{uuid.uuid4().hex[:8]}"
        state["rollout_id"] = rollout_id

        await self.ensure_interception_server()

        if self.interception_url is None:
            tunnel_url = await self.get_tunnel_url()
            state["interception_base_url"] = f"{tunnel_url}/rollout/{rollout_id}/v1"
        else:
            state["interception_base_url"] = f"{self.interception_url.rstrip('/')}/rollout/{rollout_id}/v1"

        env_vars = await self.build_env_vars(state)
        docker_image = await self.get_docker_image(state)

        # Use manager for sandbox lifecycle
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
            labels=self.labels or [],
        )

        sandbox = await self.sandbox_manager.acquire(rollout_id=rollout_id, request=sandbox_request)
        state["sandbox_id"] = sandbox.id
        await self.sandbox_manager.wait_for_ready(sandbox.id)

        sandbox_client = AsyncSandboxClient()
        await self.post_sandbox_setup(state, sandbox_client)

        request_id_queue: asyncio.Queue = asyncio.Queue()
        state["request_id_queue"] = request_id_queue
        state["agent_completed"] = False
        self.active_rollouts[rollout_id] = {"request_id_queue": request_id_queue}

        await self.start_agent(state, sandbox_client)
        return state

    @vf.cleanup
    async def destroy_sandbox(self, state: State):
        await self.post_rollout(state)
        sandbox_id = state.get("sandbox_id")
        if sandbox_id:
            await self.sandbox_manager.release(sandbox_id)

    @vf.teardown
    async def teardown_sandboxes(self):
        self.sandbox_manager.print_summary()
        await self.sandbox_manager.release_all()
        self.sandbox_manager.teardown()

    # =========================================================================
    # Hooks for subclasses
    # =========================================================================

    async def get_docker_image(self, state: State) -> str:
        return self.docker_image

    async def build_env_vars(self, state: State) -> dict[str, str]:
        env_vars = dict(self.environment_vars) if self.environment_vars else {}
        env_vars["OPENAI_BASE_URL"] = state["interception_base_url"]
        env_vars.setdefault("OPENAI_TIMEOUT", "600")
        model = state.get("model")
        if model:
            env_vars["OPENAI_MODEL"] = model
        return env_vars

    async def post_sandbox_setup(self, state: State, sandbox_client: AsyncSandboxClient) -> None:
        pass

    async def post_rollout(self, state: State):
        pass

    # =========================================================================
    # Agent execution
    # =========================================================================

    async def start_agent(self, state: State, sandbox_client: AsyncSandboxClient) -> None:
        sandbox_id = state["sandbox_id"]
        background_job: BackgroundJob = await sandbox_client.start_background_job(sandbox_id, self.run_command)
        state["background_job"] = background_job
        state["agent_start_time"] = time.time()
        state["completion_wait_task"] = asyncio.create_task(self.wait_for_completion(state, sandbox_client))

    async def wait_for_completion(self, state: State, sandbox_client: AsyncSandboxClient) -> None:
        sandbox_id = state.get("sandbox_id")
        background_job = state.get("background_job")
        if not sandbox_id or not background_job:
            state["agent_completed"] = True
            return

        try:
            await asyncio.wait_for(
                self.poll_job_completion(state, sandbox_id, background_job),
                timeout=self.timeout_seconds,
            )
        except asyncio.TimeoutError:
            state["agent_timed_out"] = True
        except asyncio.CancelledError:
            raise
        except Exception:
            pass
        finally:
            state["agent_completed"] = True

    async def poll_job_completion(self, state: State, sandbox_id: str, background_job: BackgroundJob) -> None:
        sandbox_client = AsyncSandboxClient()
        while True:
            status: BackgroundJobStatus = await sandbox_client.get_background_job(sandbox_id, background_job)
            if status.completed:
                state["agent_exit_code"] = status.exit_code
                state["agent_stdout"] = status.stdout
                state["agent_stderr"] = status.stderr
                return
            await asyncio.sleep(1)

    @vf.stop
    async def agent_completed(self, state: State) -> bool:
        return state.get("agent_completed", False)

    @vf.stop
    async def timeout_reached(self, state: State) -> bool:
        return time.time() - state["timing"]["start_time"] > self.timeout_seconds

    # =========================================================================
    # API interception
    # =========================================================================

    async def get_tunnel_url(self) -> str:
        async with self.tunnel_lock:
            if self.tunnel is None:
                self.tunnel = Tunnel(local_port=self.interception_port)
                return await self.tunnel.start()
            assert self.tunnel.url is not None
            return self.tunnel.url

    async def ensure_interception_server(self):
        async with self.server_lock:
            if self.interception_server is not None:
                return

            app = web.Application()
            app.router.add_post("/rollout/{rollout_id}/v1/chat/completions", self.handle_intercepted_request)
            app.router.add_get("/health", lambda _: web.json_response({"status": "ok"}))

            runner = web.AppRunner(app)
            await runner.setup()
            site = web.TCPSite(runner, "0.0.0.0", self.interception_port)
            await site.start()

            self.interception_server = app
            self.server_runner = runner
            self.server_site = site

    async def handle_intercepted_request(self, request: Any) -> Any:
        rollout_id = request.match_info["rollout_id"]
        context = self.active_rollouts.get(rollout_id)
        if not context:
            return web.json_response({"error": "Rollout not found"}, status=404)

        try:
            request_body = await request.json()
        except Exception as e:
            return web.json_response({"error": f"Invalid JSON: {e}"}, status=400)

        is_streaming = request_body.get("stream", False)
        request_id = f"req_{uuid.uuid4().hex[:8]}"
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
                response = await cast(asyncio.Future[Any], intercept["response_future"])
            except asyncio.CancelledError:
                return web.json_response({"error": "Rollout cancelled"}, status=499)
            except Exception as e:
                return web.json_response({"error": str(e)}, status=500)

            response_dict = response.model_dump() if hasattr(response, "model_dump") else dict(response)
            return web.json_response(response_dict)

    async def _handle_streaming_response(self, http_request: Any, rollout_id: str, intercept: dict) -> Any:
        chunk_queue = cast(asyncio.Queue, intercept["chunk_queue"])
        response_future = cast(asyncio.Future[Any], intercept["response_future"])

        response = web.StreamResponse(
            status=200,
            headers={"Content-Type": "text/event-stream", "Cache-Control": "no-cache", "Connection": "keep-alive"},
        )
        await response.prepare(http_request)

        try:
            while True:
                chunk = await chunk_queue.get()
                if chunk is None:
                    await response.write(b"data: [DONE]\n\n")
                    break
                chunk_dict = chunk.model_dump() if hasattr(chunk, "model_dump") else dict(chunk)
                await response.write(f"data: {json.dumps(chunk_dict)}\n\n".encode())
            await response_future
        except (asyncio.CancelledError, ClientConnectionResetError):
            pass
        except Exception as e:
            logger.error(f"Streaming error: {e}")

        try:
            await response.write_eof()
        except Exception:
            pass
        return response

    # =========================================================================
    # Message handling
    # =========================================================================

    async def get_prompt_messages(self, state: State) -> Messages:
        request_id_queue = state["request_id_queue"]

        while True:
            try:
                request_id = await asyncio.wait_for(request_id_queue.get(), timeout=self.poll_interval)
                state["current_request_id"] = request_id
                return self.intercepts[request_id]["messages"]
            except asyncio.TimeoutError:
                if state.get("agent_completed"):
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
        if not prompt:
            return ChatCompletion(
                id="agent-completed",
                choices=[Choice(finish_reason="stop", index=0, message=ChatCompletionMessage(role="assistant", content=""))],
                created=int(time.time()),
                model=model or state["model"],
                object="chat.completion",
            )

        request_id = state.get("current_request_id")
        intercept = self.intercepts.get(request_id) if request_id else None

        if intercept:
            model = state.get("model") or model
            oai_tools = intercept.get("tools") or oai_tools

        response: ModelResponse | None = None
        error: BaseException | None = None

        try:
            if intercept and intercept.get("stream"):
                response = await self._get_streaming_model_response(state, prompt, intercept, client, model, oai_tools, sampling_args)
            else:
                response = await super().get_model_response(state, prompt, client, model, oai_tools, sampling_args, message_type)
        except BaseException as e:
            error = e
            raise
        finally:
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
        chunk_queue = cast(asyncio.Queue, intercept["chunk_queue"])
        client = client or state["client"]
        model = model or state["model"]
        sampling_args = sampling_args or state.get("sampling_args") or {}

        if "max_tokens" in sampling_args:
            sampling_args = dict(sampling_args)
            max_tokens = sampling_args.pop("max_tokens")
            if "max_completion_tokens" not in sampling_args:
                sampling_args["max_completion_tokens"] = max_tokens

        create_kwargs: dict[str, Any] = {"model": model, "messages": prompt, "stream": True}
        if oai_tools:
            create_kwargs["tools"] = oai_tools
        create_kwargs.update(sampling_args)

        stream = await client.chat.completions.create(**create_kwargs)

        accumulated_content = ""
        accumulated_tool_calls: dict[int, dict] = {}
        finish_reason = None
        completion_id = None
        created_time = int(time.time())

        try:
            async for chunk in stream:
                await chunk_queue.put(chunk)
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
                        if delta.tool_calls:
                            for tc in delta.tool_calls:
                                idx = tc.index
                                if idx not in accumulated_tool_calls:
                                    accumulated_tool_calls[idx] = {"id": tc.id or "", "type": "function", "function": {"name": "", "arguments": ""}}
                                if tc.id:
                                    accumulated_tool_calls[idx]["id"] = tc.id
                                if tc.function:
                                    if tc.function.name:
                                        accumulated_tool_calls[idx]["function"]["name"] = tc.function.name
                                    if tc.function.arguments:
                                        accumulated_tool_calls[idx]["function"]["arguments"] += tc.function.arguments
            await chunk_queue.put(None)
        finally:
            try:
                chunk_queue.put_nowait(None)
            except:
                pass

        tool_calls_list = None
        if accumulated_tool_calls:
            tool_calls_list = [
                ChatCompletionMessageToolCall(
                    id=tc["id"],
                    type="function",
                    function=Function(name=tc["function"]["name"], arguments=tc["function"]["arguments"]),
                )
                for tc in [accumulated_tool_calls[i] for i in sorted(accumulated_tool_calls.keys())]
            ]

        return ChatCompletion(
            id=completion_id or f"chatcmpl-{uuid.uuid4().hex[:8]}",
            choices=[Choice(finish_reason=finish_reason or "stop", index=0, message=ChatCompletionMessage(role="assistant", content=accumulated_content or None, tool_calls=tool_calls_list))],
            created=created_time,
            model=model,
            object="chat.completion",
        )

    async def add_model_response(self, state: State, prompt_messages: Messages, response: ModelResponse):
        if not prompt_messages:
            return
        if len(state["trajectory"]) == 0:
            state["prompt"] = prompt_messages
        await super().add_model_response(state, prompt_messages, response)

    async def env_response(self, messages: Messages, state: State, **kwargs) -> Messages:
        return []

    # =========================================================================
    # Cleanup
    # =========================================================================

    @vf.cleanup
    async def cleanup_interception_context(self, state: State):
        task = state.get("completion_wait_task")
        if task and not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        rollout_id = state.get("rollout_id")
        if rollout_id:
            for request_id in list(self.intercepts.keys()):
                intercept = self.intercepts.get(request_id)
                if intercept and intercept.get("rollout_id") == rollout_id:
                    chunk_queue = intercept.get("chunk_queue")
                    if chunk_queue:
                        try:
                            chunk_queue.put_nowait(None)
                        except:
                            pass
                    future = intercept.get("response_future")
                    if future and not future.done():
                        future.cancel()
                    del self.intercepts[request_id]
            self.active_rollouts.pop(rollout_id, None)

    @vf.teardown
    async def teardown_tunnel(self):
        async with self.tunnel_lock:
            if self.tunnel:
                try:
                    await self.tunnel.stop()
                except:
                    pass
                self.tunnel = None

        async with self.server_lock:
            if self.server_runner:
                try:
                    await self.server_runner.cleanup()
                except:
                    pass
                self.server_runner = None
                self.server_site = None
                self.interception_server = None
