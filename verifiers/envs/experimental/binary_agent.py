"""BinaryAgent — runs an agent binary *inside* the sandbox.

The agent binary (OpenCode, Aider, Claude Code, or any CLI tool) runs
inside the same sandbox as the task code.  Its LLM API calls are
intercepted via an HTTP proxy (``InterceptionServer`` + ``Tunnel``),
forwarded to the real model, and the responses are delivered back.

The trajectory is recorded from the intercepted request/response pairs.

Usage::

    agent = BinaryAgent(
        install_script="curl -fsSL https://github.com/.../install.sh | bash",
        run_command="opencode run --prompt /tmp/prompt.md",
    )
    env = ComposableEnv(task=task, agent=agent, dataset=...)
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from typing import Any

from prime_sandboxes import BackgroundJob, BackgroundJobStatus, SandboxOOMError, SandboxTimeoutError

import verifiers as vf
from verifiers.types import (
    Messages,
    Response,
    State,
    Tool,
    TrajectoryStep,
)
from verifiers.utils.interception_utils import (
    InterceptionServer,
    deliver_response,
    synthesize_stream,
)
from verifiers.utils.message_utils import normalize_messages
from verifiers.utils.response_utils import parse_response_message
from verifiers.utils.worker_utils import get_free_port

logger = logging.getLogger(__name__)


class BinaryAgent:
    """Agent that runs a binary *inside* the sandbox.

    The binary's API calls (to ``OPENAI_BASE_URL``) are intercepted via
    an HTTP proxy, forwarded to the real LLM, and recorded as trajectory
    steps.

    Parameters
    ----------
    install_script:
        Shell command(s) to install the agent binary in the sandbox.
        Runs during ``setup()`` after the task has prepared the sandbox.
        Example: ``"curl -fsSL https://example.com/install.sh | bash"``
    run_command:
        Shell command to start the agent.  ``{prompt_file}`` is replaced
        with the path to a file containing the prompt text.
        Runs as a background job during ``run()``.
    prompt_file_path:
        Path inside the sandbox where the prompt is written before
        starting the agent.
    timeout_seconds:
        Maximum time for the agent binary to complete.
    poll_interval:
        How often to check for intercepted requests / agent completion.
    interception_port:
        Local port for the interception HTTP server.  0 = auto.
    interception_url:
        If set, use this URL directly (skip tunnel).
    extra_env_vars:
        Additional environment variables to set in the sandbox for the
        agent binary.
    agent_id:
        Identifier for this agent in trajectory step extras.
    """

    def __init__(
        self,
        install_script: str | None = None,
        run_command: str = "echo 'No run_command specified'",
        prompt_file_path: str = "/tmp/prompt.md",
        timeout_seconds: float = 3600.0,
        poll_interval: float = 1.0,
        interception_port: int = 0,
        interception_url: str | None = None,
        extra_env_vars: dict[str, str] | None = None,
        agent_id: str = "binary_agent",
    ):
        self.install_script = install_script
        self.run_command = run_command
        self.prompt_file_path = prompt_file_path
        self.timeout_seconds = timeout_seconds
        self.poll_interval = poll_interval
        self.interception_url = interception_url
        self.extra_env_vars = extra_env_vars or {}
        self.agent_id = agent_id

        port = interception_port or get_free_port()
        self._interception_server = InterceptionServer(port=port)
        self._tunnel = None
        self._tunnel_lock = asyncio.Lock()

        # Populated during setup
        self._sandbox_client: Any = None
        self._sandbox_id: str | None = None

    # -- lifecycle ----------------------------------------------------------

    async def setup(
        self,
        sandbox_client: Any,
        sandbox_id: str,
        state: State,
    ) -> None:
        """Install the agent binary in the sandbox."""
        self._sandbox_client = sandbox_client
        self._sandbox_id = sandbox_id

        if self.install_script:
            logger.info(f"Installing agent in sandbox {sandbox_id}")
            result = await sandbox_client.execute_command(
                sandbox_id,
                self.install_script,
                timeout=300,
            )
            if result.exit_code != 0:
                output = (result.stdout or "") + (result.stderr or "")
                raise vf.SandboxError(
                    f"Agent install failed (exit={result.exit_code}): {output[:500]}"
                )
            logger.debug(f"Agent installed in sandbox {sandbox_id}")

    def get_env_vars(self, state: State) -> dict[str, str]:
        """Return env vars the sandbox needs for the agent binary.

        Called by ComposableEnv before sandbox creation to inject
        ``OPENAI_BASE_URL`` pointing at the interception proxy.
        """
        env_vars = dict(self.extra_env_vars)
        # The interception URL will be set during run() when we know the rollout_id
        return env_vars

    # -- interception -------------------------------------------------------

    async def _get_tunnel_url(self) -> str:
        """Start tunnel if needed and return the public URL."""
        async with self._tunnel_lock:
            if self._tunnel is not None and not self._tunnel.is_running:
                logger.warning("Tunnel died, recreating")
                self._tunnel.sync_stop()
                self._tunnel = None

            if self._tunnel is None:
                from prime_tunnel import Tunnel

                self._tunnel = Tunnel(local_port=self._interception_server.port)
                url = await self._tunnel.start()
                logger.debug(f"Tunnel started: {url}")
                return url
            else:
                assert self._tunnel.url is not None
                return self._tunnel.url

    async def _poll_next_request(
        self,
        request_id_queue: asyncio.Queue,
        state: State,
    ) -> str | None:
        """Poll for next intercepted request or agent completion."""
        start_time = state["timing"]["start_time"]
        while True:
            try:
                return await asyncio.wait_for(
                    request_id_queue.get(), timeout=self.poll_interval
                )
            except asyncio.TimeoutError:
                # Check tunnel health
                if self._tunnel is not None and not self._tunnel.is_running:
                    raise vf.TunnelError("Tunnel died during rollout")
                # Check agent completion
                if state.get("agent_completed", False):
                    return None
                # Check timeout
                if time.time() - start_time > self.timeout_seconds:
                    state["agent_timed_out"] = True
                    return None

    # -- main loop ----------------------------------------------------------

    async def run(
        self,
        prompt: Messages,
        state: State,
    ) -> list[TrajectoryStep]:
        """Start the agent binary and relay intercepted API calls."""
        sandbox_client = self._sandbox_client
        sandbox_id = self._sandbox_id
        assert sandbox_client is not None and sandbox_id is not None

        rollout_id = state.get("rollout_id", f"rollout_{uuid.uuid4().hex[:8]}")
        trajectory_id = uuid.uuid4().hex
        steps: list[TrajectoryStep] = []

        # Start interception server
        await self._interception_server.start()

        # Determine interception URL
        if self.interception_url:
            base_url = f"{self.interception_url.rstrip('/')}/rollout/{rollout_id}/v1"
        else:
            tunnel_url = await self._get_tunnel_url()
            base_url = f"{tunnel_url}/rollout/{rollout_id}/v1"

        # Set OPENAI_BASE_URL in sandbox
        await sandbox_client.execute_command(
            sandbox_id,
            f'echo "export OPENAI_BASE_URL={base_url}" >> /root/.bashrc && '
            f'echo "export OPENAI_TIMEOUT=3600" >> /root/.bashrc && '
            f'echo "export OPENAI_REQUEST_TIMEOUT=3600" >> /root/.bashrc',
            timeout=10,
        )

        # Write prompt to file in sandbox
        prompt_text = ""
        for msg in prompt:
            content = getattr(msg, "content", "") if not isinstance(msg, dict) else msg.get("content", "")
            if content:
                prompt_text += str(content) + "\n"

        from verifiers.utils.path_utils import write_temp_file
        import os

        local_path = write_temp_file(prompt_text)
        try:
            await sandbox_client.upload_file(sandbox_id, self.prompt_file_path, local_path)
        finally:
            os.unlink(local_path)

        # Register rollout for interception
        request_id_queue = self._interception_server.register_rollout(rollout_id)
        state["agent_completed"] = False

        # Start agent as background job
        run_cmd = self.run_command.replace("{prompt_file}", self.prompt_file_path)
        env_prefix = " ".join(
            f'{k}="{v}"' for k, v in {
                "OPENAI_BASE_URL": base_url,
                "OPENAI_TIMEOUT": "3600",
                **self.extra_env_vars,
            }.items()
        )
        full_cmd = f"{env_prefix} {run_cmd}"

        logger.info(f"Starting agent binary: {run_cmd}")
        try:
            background_job: BackgroundJob = await sandbox_client.start_background_job(
                sandbox_id, full_cmd
            )
        except Exception as e:
            raise vf.SandboxError(f"Failed to start agent: {e}") from e

        # Start polling for agent completion
        completion_task = asyncio.create_task(
            self._poll_agent_completion(sandbox_client, sandbox_id, background_job, state)
        )

        # Main interception loop
        try:
            turn = 0
            while True:
                request_id = await self._poll_next_request(request_id_queue, state)
                if request_id is None:
                    break  # agent completed or timed out

                intercept = self._interception_server.intercepts[request_id]

                # Normalize intercepted messages
                raw_messages = intercept.get("messages", [])
                messages = normalize_messages(raw_messages, field_name="intercepted_messages")

                # Normalize intercepted tools
                intercept_tools = intercept.get("tools")
                tool_defs = self._normalize_tools(intercept_tools) if intercept_tools else None

                # Forward to real LLM
                client = state["client"]
                model = state.get("model", "")

                response: Response | None = None
                error: BaseException | None = None
                try:
                    response = await client.get_response(
                        prompt=messages,
                        model=model,
                        tools=tool_defs,
                        sampling_args=state.get("sampling_args") or {},
                        state=state,
                    )
                except BaseException as e:
                    error = e
                    raise
                finally:
                    # Deliver response back to the agent binary
                    if intercept.get("stream"):
                        await synthesize_stream(intercept, response, error)
                    else:
                        deliver_response(intercept, response, error)

                if response is not None:
                    completion = await parse_response_message(response)

                    # Record on first turn: update state["prompt"]
                    if turn == 0:
                        state["prompt"] = messages

                    step = TrajectoryStep(
                        prompt=messages,
                        completion=completion,
                        response=response,
                        tokens=None,
                        reward=None,
                        advantage=None,
                        is_truncated=response.message.is_truncated or False,
                        trajectory_id=trajectory_id,
                        extras={"agent_id": self.agent_id, "turn": turn},
                    )
                    steps.append(step)
                    turn += 1

        finally:
            # Cleanup
            completion_task.cancel()
            try:
                await completion_task
            except asyncio.CancelledError:
                pass

            self._interception_server.unregister_rollout(rollout_id)

            # Log agent exit info
            exit_code = state.get("agent_exit_code")
            timed_out = state.get("agent_timed_out", False)
            logger.info(
                f"Agent finished: turns={len(steps)} exit_code={exit_code} timed_out={timed_out}"
            )

        return steps

    # -- helpers ------------------------------------------------------------

    async def _poll_agent_completion(
        self,
        sandbox_client: Any,
        sandbox_id: str,
        background_job: BackgroundJob,
        state: State,
    ) -> None:
        """Poll background job until the agent binary exits."""
        try:
            while True:
                try:
                    status: BackgroundJobStatus = await sandbox_client.get_background_job(
                        sandbox_id, background_job
                    )
                except SandboxTimeoutError:
                    logger.warning("Sandbox timed out — marking agent as completed")
                    state["sandbox_timeout"] = True
                    state["agent_completed"] = True
                    return
                except SandboxOOMError:
                    logger.warning("Sandbox OOM — marking agent as completed")
                    state["sandbox_oom"] = True
                    state["agent_completed"] = True
                    return
                if status.completed:
                    state["agent_exit_code"] = status.exit_code
                    state["agent_stdout"] = status.stdout
                    state["agent_stderr"] = status.stderr
                    state["agent_completed"] = True
                    return
                await asyncio.sleep(1)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error(f"Agent polling failed: {e}")
            state["agent_completed"] = True

    def _normalize_tools(self, intercept_tools: list) -> list[Tool] | None:
        """Normalize intercepted tool definitions to vf.Tool objects."""
        normalized: list[Tool] = []
        for raw_tool in intercept_tools:
            if isinstance(raw_tool, Tool):
                normalized.append(raw_tool)
                continue
            if isinstance(raw_tool, dict):
                func_payload = raw_tool.get("function")
                if raw_tool.get("type") == "function" and isinstance(func_payload, dict):
                    normalized.append(
                        Tool(
                            name=func_payload.get("name", ""),
                            description=func_payload.get("description", ""),
                            parameters=func_payload.get("parameters", {}),
                            strict=func_payload.get("strict"),
                        )
                    )
                    continue
                normalized.append(Tool.model_validate(raw_tool))
        return normalized or None

    async def teardown(self) -> None:
        """Stop tunnel and interception server."""
        async with self._tunnel_lock:
            if self._tunnel is not None:
                try:
                    self._tunnel.sync_stop()
                except Exception as e:
                    logger.warning(f"Error stopping tunnel: {e}")
                finally:
                    self._tunnel = None
        if self._interception_server is not None:
            await self._interception_server.stop()
