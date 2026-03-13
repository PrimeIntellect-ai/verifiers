from __future__ import annotations

import asyncio
import logging
import time
import uuid
from typing import Any, cast

from prime_sandboxes import BackgroundJob, BackgroundJobStatus
from prime_tunnel import Tunnel

import verifiers as vf
from verifiers.envs.experimental.harnesses.base import Harness
from verifiers.types import Messages, Response, SamplingArgs, State, Tool
from verifiers.utils.interception_utils import (
    InterceptionServer,
    deliver_response,
    synthesize_stream,
)
from verifiers.utils.message_utils import normalize_messages
from verifiers.utils.worker_utils import get_free_port

logger = logging.getLogger(__name__)


class CliHarness(Harness):
    """Interceptor-based harness for CLI agents running inside sandboxes."""

    def __init__(
        self,
        run_command: str,
        interception_port: int | None = None,
        interception_url: str | None = None,
        poll_interval: float = 1.0,
        timeout_seconds: float = 3600.0,
    ):
        super().__init__(timeout_seconds=timeout_seconds)
        self.run_command = run_command
        self.poll_interval = poll_interval
        self.init_interception(
            interception_port=get_free_port()
            if interception_port is None
            else interception_port,
            interception_url=interception_url,
        )

    def init_interception(
        self,
        interception_port: int = 8765,
        interception_url: str | None = None,
    ) -> None:
        self.interception_port = interception_port
        self.interception_url = interception_url
        self._tunnel: Tunnel | None = None
        self._tunnel_lock = asyncio.Lock()
        self._tunnel_monitor_task: asyncio.Task | None = None
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
                logger.warning(
                    "Tunnel process died, recreating. frpc output:\n%s", frpc_output
                )
                self._tunnel.sync_stop()
                self._tunnel = None

            if self._tunnel is None:
                interception_server = self._require_interception_server()
                port = interception_server.port
                if logger.isEnabledFor(logging.DEBUG):
                    self._tunnel = Tunnel(local_port=port, log_level="debug")
                else:
                    self._tunnel = Tunnel(local_port=port)
                url = await self._tunnel.start()
                logger.debug("Prime Tunnel started: %s", url)

                if (
                    self._tunnel_monitor_task is None
                    or self._tunnel_monitor_task.done()
                ):
                    self._tunnel_monitor_task = asyncio.create_task(
                        self._tunnel_health_monitor()
                    )

                return url

            assert self._tunnel.url is not None, "Tunnel started but URL is None"
            return self._tunnel.url

    async def _tunnel_health_monitor(self, interval: float = 30.0) -> None:
        try:
            while True:
                await asyncio.sleep(interval)
                async with self._tunnel_lock:
                    if self._tunnel is not None and not self._tunnel.is_running:
                        frpc_output = "\n".join(self._tunnel.recent_output)
                        logger.warning(
                            "Health monitor: tunnel dead. frpc output:\n%s",
                            frpc_output,
                        )
                        self._tunnel.sync_stop()
                        port = self._require_interception_server().port
                        if logger.isEnabledFor(logging.DEBUG):
                            self._tunnel = Tunnel(local_port=port, log_level="debug")
                        else:
                            self._tunnel = Tunnel(local_port=port)
                        url = await self._tunnel.start()
                        logger.info("Health monitor: restarted tunnel url=%s", url)
        except asyncio.CancelledError:
            return

    async def setup(self, env: Any, state: State) -> None:
        await super().setup(env, state)

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

        state["request_id_queue"] = interception_server.register_rollout(rollout_id)

    async def build_env_vars(self, env: Any, state: State) -> dict[str, str]:
        env_vars = {
            "OPENAI_BASE_URL": state["interception_base_url"],
            "OPENAI_TIMEOUT": "600",
            "OPENAI_REQUEST_TIMEOUT": "600",
            "HTTPX_TIMEOUT": "600",
        }
        model = state.get("model")
        if model:
            env_vars["OPENAI_MODEL"] = model
        return env_vars

    async def start(self, env: Any, state: State) -> None:
        sandbox_id = state["sandbox_id"]
        background_job: BackgroundJob = await env.sandbox_client.start_background_job(
            sandbox_id,
            self.run_command,
        )
        state["background_job"] = background_job
        state["agent_start_time"] = time.time()
        state["completion_wait_task"] = asyncio.create_task(
            self.wait_for_completion(env, state)
        )

    async def wait_for_completion(self, env: Any, state: State) -> None:
        sandbox_id = state.get("sandbox_id")
        background_job: BackgroundJob | None = state.get("background_job")

        if not sandbox_id or not background_job:
            state["agent_completed"] = True
            return

        try:
            await asyncio.wait_for(
                self.poll_job_completion(env, state, sandbox_id, background_job),
                timeout=self.timeout_seconds,
            )
        except asyncio.TimeoutError:
            logger.warning("Agent timed out after %ss", self.timeout_seconds)
            state["agent_timed_out"] = True
        except asyncio.CancelledError:
            logger.debug("Completion wait task cancelled")
            raise
        except Exception as exc:
            logger.debug("Completion wait ended: %s", exc)
        finally:
            state["agent_completed"] = True

    async def poll_job_completion(
        self,
        env: Any,
        state: State,
        sandbox_id: str,
        background_job: BackgroundJob,
    ) -> None:
        while True:
            status: BackgroundJobStatus = await env.sandbox_client.get_background_job(
                sandbox_id,
                background_job,
            )
            if status.completed:
                state["agent_exit_code"] = status.exit_code
                state["agent_stdout"] = status.stdout
                state["agent_stderr"] = status.stderr
                if status.exit_code == 0:
                    logger.debug(
                        "Agent completed successfully (exit_code=%s)",
                        status.exit_code,
                    )
                else:
                    logger.warning(
                        "Agent failed (exit_code=%s) stdout=%s, stderr=%s",
                        status.exit_code,
                        status.stdout,
                        status.stderr,
                    )
                return
            await asyncio.sleep(1)

    async def check_agent_completed(self, env: Any, state: State) -> bool:
        return state.get("agent_completed", False)

    def normalize_intercepted_tools(
        self,
        env: Any,
        intercept_tools: object,
    ) -> list[Tool] | None:
        if not isinstance(intercept_tools, list):
            raise TypeError("Intercepted tools must be provided as a list.")

        normalized_inputs: list[dict[str, Any]] = []
        for raw_tool in intercept_tools:
            if isinstance(raw_tool, Tool):
                normalized_inputs.append(raw_tool.model_dump(exclude_none=True))
                continue
            if not isinstance(raw_tool, dict):
                raise TypeError(
                    "Intercepted tools must be vf.Tool objects or dict tool definitions."
                )
            raw_tool_dict = cast(dict[str, Any], raw_tool)
            function_payload = raw_tool_dict.get("function")
            if raw_tool_dict.get("type") == "function" and isinstance(
                function_payload,
                dict,
            ):
                parameters = function_payload.get("parameters", {})
                if not isinstance(parameters, dict):
                    raise TypeError(
                        "Intercepted function tool parameters must be a JSON object."
                    )
                normalized_inputs.append(
                    {
                        "name": function_payload.get("name"),
                        "description": function_payload.get("description", ""),
                        "parameters": parameters,
                        "strict": function_payload.get("strict"),
                    }
                )
                continue

            normalized_inputs.append(raw_tool_dict)

        return env._normalize_tool_defs(normalized_inputs)

    def normalize_intercepted_messages(self, intercepted_messages: object) -> Messages:
        return normalize_messages(intercepted_messages)  # type: ignore[arg-type]

    async def get_prompt_messages(self, env: Any, state: State) -> Messages:
        request_id_queue = state["request_id_queue"]
        interception_server = self._require_interception_server()

        while True:
            try:
                request_id = await asyncio.wait_for(
                    request_id_queue.get(),
                    timeout=self.poll_interval,
                )
                state["current_request_id"] = request_id
                intercept = interception_server.intercepts[request_id]
                return self.normalize_intercepted_messages(intercept["messages"])
            except asyncio.TimeoutError:
                if self._tunnel is not None and not self._tunnel.is_running:
                    frpc_output = "\n".join(self._tunnel.recent_output)
                    raise vf.TunnelError(
                        "Tunnel process died during rollout. "
                        f"frpc output:\n{frpc_output}"
                    )
                if await self.check_agent_completed(env, state):
                    state["agent_completed"] = True
                    return []
                if await self.timeout_reached(env, state):
                    return []

    async def get_model_response(
        self,
        env: Any,
        state: State,
        prompt: Messages | str,
        client: vf.Client | None = None,
        model: str | None = None,
        tool_defs: list[Tool] | None = None,
        sampling_args: SamplingArgs | None = None,
    ) -> Response:
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
            model = state.get("model") or model
            intercept_tools = intercept.get("tools")
            if intercept_tools:
                tool_defs = (
                    self.normalize_intercepted_tools(env, intercept_tools) or tool_defs
                )

        response: Response | None = None
        error: BaseException | None = None

        try:
            response = await env.request_model_response(
                state=state,
                prompt=prompt,
                client=client,
                model=model,
                tool_defs=tool_defs,
                sampling_args=sampling_args,
            )
        except BaseException as exc:
            error = exc
            raise
        finally:
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
        env: Any,
        state: State,
        prompt_messages: Messages,
        response: Response,
    ) -> None:
        if not prompt_messages:
            return
        if len(state["trajectory"]) == 0:
            state["prompt"] = prompt_messages
        await env.record_model_response(
            state,
            prompt_messages,
            env.normalize_response(response),
        )

    async def cleanup(self, env: Any, state: State) -> None:
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

    async def teardown(self, env: Any) -> None:
        if (
            self._tunnel_monitor_task is not None
            and not self._tunnel_monitor_task.done()
        ):
            self._tunnel_monitor_task.cancel()
            try:
                await self._tunnel_monitor_task
            except asyncio.CancelledError:
                pass
            self._tunnel_monitor_task = None

        async with self._tunnel_lock:
            if self._tunnel is not None:
                try:
                    self._tunnel.sync_stop()
                    logger.debug("Prime Tunnel stopped")
                except Exception as exc:
                    logger.warning("Error stopping Prime Tunnel: %s", exc)
                finally:
                    self._tunnel = None

        if self._interception_server is not None:
            await self._interception_server.stop()


InterceptorHarness = CliHarness
