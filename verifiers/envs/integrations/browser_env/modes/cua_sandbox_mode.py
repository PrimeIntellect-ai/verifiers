"""CUA-based browser mode with automatic sandbox deployment."""

import asyncio
import base64
import copy
import json
import logging
import os
import subprocess
import tarfile
import tempfile
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import tenacity as tc
import verifiers as vf

try:
    from prime_sandboxes import (
        AsyncSandboxClient,
        CreateSandboxRequest,
        SandboxClient,
    )
    from prime_sandboxes.core import APIClient

    SANDBOX_AVAILABLE = True
except ImportError:
    SANDBOX_AVAILABLE = False
    AsyncSandboxClient = None
    CreateSandboxRequest = None
    SandboxClient = None
    APIClient = None


class CUASandboxMode:
    """
    CUA-based browser mode with automatic sandbox deployment.

    By default, uses a pre-built Docker image (deepdream19/cua-server:latest) for
    fastest startup. The server starts automatically via the container's start_command.

    Execution modes:
    1. Pre-built image (default, use_prebuilt_image=True):
       - Uses pre-built Docker image with binary + curl already installed
       - Fastest startup (~5-10s vs ~30-60s)
    2. Binary upload (use_prebuilt_image=False):
       - Creates sandbox, uploads binary, installs curl, starts server
       - Useful for custom server versions

    This mode automatically:
    1. Creates a sandbox container (with pre-built image or base image)
    2. Starts the server (via start_command or manual startup)
    3. Executes browser actions via curl commands inside the sandbox
    4. Cleans up the sandbox when done

    Users don't need to manually start or manage the CUA server.

    Provides vision-based primitives: click, double_click, type_text,
    keypress, scroll, goto, back, forward, wait, screenshot
    """

    def __init__(
        self,
        server_port: int = 3000,
        upload_path: str = "/app/cua-server",
        env: Literal["LOCAL", "BROWSERBASE"] = "BROWSERBASE",
        browserbase_api_key: str | None = None,
        browserbase_project_id: str | None = None,
        viewport_width: int = 1024,
        viewport_height: int = 768,
        max_retries: int = 5,
        base_delay: float = 0.5,
        backoff_factor: float = 2.0,
        max_backoff_seconds: float = 30.0,
        jitter: float = 1e-3,
        screenshot_dir: str | None = None,
        save_screenshots: bool = True,
        keep_recent_screenshots: int | None = 2,
        proxies: bool = False,
        server_ready_timeout: int = 120,
        server_ready_poll_interval: float = 2.0,
        # Sandbox configuration
        docker_image: str = "node:18-slim",
        cpu_cores: int = 2,
        memory_gb: int = 4,
        disk_size_gb: int = 10,
        sandbox_timeout_minutes: int = 60,
        sandbox_timeout_per_command_seconds: int = 60,
        # Binary build configuration (only used when use_prebuilt_image=False)
        use_binary: bool = True,
        # Pre-built image configuration (default - fastest startup)
        use_prebuilt_image: bool = True,
        prebuilt_image: str = "deepdream19/cua-server:latest",
    ):
        if not SANDBOX_AVAILABLE:
            raise ImportError(
                "prime-sandboxes is not installed. "
                "Please install it with `uv pip install prime-sandboxes`."
            )

        self.keep_recent_screenshots = keep_recent_screenshots
        self.logger = None  # Will be set when register_tools is called

        resolved_api_key = browserbase_api_key or os.getenv("BROWSERBASE_API_KEY")
        resolved_project_id = browserbase_project_id or os.getenv(
            "BROWSERBASE_PROJECT_ID"
        )

        self.server_port = server_port
        self.upload_path = upload_path
        self.server_ready_timeout = server_ready_timeout
        self.server_ready_poll_interval = server_ready_poll_interval
        self.sandbox_timeout_per_command_seconds = sandbox_timeout_per_command_seconds

        self.session_config = {
            "env": env,
            "browserbaseApiKey": resolved_api_key,
            "browserbaseProjectId": resolved_project_id,
            "viewport": {"width": viewport_width, "height": viewport_height},
            "proxies": proxies,
        }
        self.session_config = {
            k: v for k, v in self.session_config.items() if v is not None
        }

        self._thread_lock = threading.Lock()
        self._sessions_lock = threading.Lock()
        self._counter_lock = threading.Lock()
        self.active_sessions: set[str] = set()
        self.active_sandboxes: set[str] = set()

        # Sandbox client
        self._sandbox_client: AsyncSandboxClient | None = None

        # Sandbox request template
        self._sandbox_request = CreateSandboxRequest(
            name="cua-server",
            docker_image=docker_image,
            start_command="tail -f /dev/null",
            cpu_cores=cpu_cores,
            memory_gb=memory_gb,
            disk_size_gb=disk_size_gb,
            gpu_count=0,
            timeout_minutes=sandbox_timeout_minutes,
            environment_vars={},
        )

        # Binary build configuration
        self.use_binary = use_binary

        # Pre-built image configuration
        self.use_prebuilt_image = use_prebuilt_image
        self.prebuilt_image = prebuilt_image

        # Reconfigure sandbox request for prebuilt image mode
        if use_prebuilt_image:
            # Build environment variables for the container
            env_vars = {
                "CUA_SERVER_PORT": str(server_port),
                "CUA_SERVER_HOST": "0.0.0.0",
            }
            # Pass OPENAI_API_KEY if available (needed by Stagehand)
            openai_key = os.getenv("OPENAI_API_KEY", "")
            if openai_key:
                env_vars["OPENAI_API_KEY"] = openai_key

            # Update sandbox request to use prebuilt image with start_command
            self._sandbox_request = CreateSandboxRequest(
                name="cua-server",
                docker_image=prebuilt_image,
                start_command="./cua-server-linux-x64",
                cpu_cores=cpu_cores,
                memory_gb=memory_gb,
                disk_size_gb=disk_size_gb,
                gpu_count=0,
                timeout_minutes=sandbox_timeout_minutes,
                environment_vars=env_vars,
            )

        self.save_screenshots = save_screenshots
        self.screenshot_dir = screenshot_dir or os.path.join(os.getcwd(), "screenshots")
        self._screenshot_counters: dict[str, int] = {}

        # Retry configuration
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.backoff_factor = backoff_factor
        self.max_backoff_seconds = max_backoff_seconds
        self.jitter = jitter
        self.with_retry = None  # Will be set in register_tools

        # Get template path
        self._template_path = (
            Path(__file__).parent.parent.parent.parent.parent.parent
            / "assets"
            / "templates"
            / "browserbase"
            / "cua"
        )

    def register_tools(self, env) -> None:
        """Register CUA mode tools with the environment."""
        self.logger = env.logger

        # Set up retry now that we have logger
        self.with_retry = tc.AsyncRetrying(
            stop=tc.stop_after_attempt(self.max_retries),
            wait=tc.wait_exponential_jitter(
                initial=self.base_delay,
                exp_base=self.backoff_factor,
                max=self.max_backoff_seconds,
                jitter=self.jitter,
            ),
            before_sleep=tc.before_sleep_log(self.logger, logging.ERROR),
            reraise=True,
        ).wraps

        # Hide session_id, sandbox_id, and tool_call_id from tool schema
        _skip = ["session_id", "sandbox_id", "tool_call_id"]
        env.add_tool(self.click, args_to_skip=_skip)
        env.add_tool(self.double_click, args_to_skip=_skip)
        env.add_tool(self.type_text, args_to_skip=_skip)
        env.add_tool(self.keypress, args_to_skip=_skip)
        env.add_tool(self.scroll, args_to_skip=_skip)
        env.add_tool(self.goto, args_to_skip=_skip)
        env.add_tool(self.back, args_to_skip=_skip)
        env.add_tool(self.forward, args_to_skip=_skip)
        env.add_tool(self.wait, args_to_skip=_skip)
        env.add_tool(self.screenshot, args_to_skip=_skip)

    # ==================== Sandbox Client Methods ====================

    async def _get_sandbox_client(self) -> AsyncSandboxClient:
        """Get or create the sandbox client."""
        if self._sandbox_client is None:
            self._sandbox_client = AsyncSandboxClient()
        return self._sandbox_client

    async def _create_sandbox(self) -> str:
        """Create a new sandbox and return its ID."""
        client = await self._get_sandbox_client()
        sandbox = await client.create(self._sandbox_request.model_copy())
        self.active_sandboxes.add(sandbox.id)
        if self.logger:
            self.logger.debug(f"Created sandbox {sandbox.id}")
        return sandbox.id

    async def _wait_for_sandbox_ready(self, sandbox_id: str) -> None:
        """Wait for sandbox to be ready."""
        client = await self._get_sandbox_client()
        await client.wait_for_creation(sandbox_id)
        if self.logger:
            self.logger.debug(f"Sandbox {sandbox_id} is ready")

    async def _delete_sandbox(self, sandbox_id: str) -> None:
        """Delete a sandbox."""
        client = await self._get_sandbox_client()
        await client.delete(sandbox_id)
        self.active_sandboxes.discard(sandbox_id)
        if self.logger:
            self.logger.debug(f"Deleted sandbox {sandbox_id}")

    async def _execute_sandbox_command(
        self, sandbox_id: str, command: str, timeout: int | None = None
    ) -> str:
        """Execute a command in the sandbox and return stdout."""
        client = await self._get_sandbox_client()
        result = await client.execute_command(
            sandbox_id,
            command,
            working_dir=None,
            timeout=timeout or self.sandbox_timeout_per_command_seconds,
        )
        return result.stdout if hasattr(result, "stdout") else str(result)

    # ==================== Sandbox Setup Methods ====================

    async def _ensure_binary_exists(self) -> Path:
        """Ensure linux-x64 binary exists, build via Docker if not."""
        binary_path = self._template_path / "dist" / "sea" / "cua-server-linux-x64"

        if binary_path.exists():
            return binary_path

        if self.logger:
            self.logger.info(
                "Building CUA server binary via Docker (first-time setup)..."
            )

        # Run Docker build (force linux/amd64 for sandbox compatibility)
        # Use --no-cache to ensure source changes are always picked up
        result = subprocess.run(
            [
                "docker",
                "build",
                "--no-cache",
                "--platform",
                "linux/amd64",
                "-f",
                "Dockerfile.build",
                "-t",
                "cua-builder",
                ".",
            ],
            cwd=self._template_path,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(f"Docker build failed: {result.stderr}")

        # Extract binary from container
        result = subprocess.run(
            [
                "docker",
                "run",
                "--rm",
                "--platform",
                "linux/amd64",
                "-v",
                f"{self._template_path}/dist:/output",
                "cua-builder",
            ],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(f"Docker run failed: {result.stderr}")

        if not binary_path.exists():
            raise RuntimeError("Binary build completed but binary not found")

        if self.logger:
            self.logger.info("CUA server binary built successfully")

        return binary_path

    async def _upload_server_files(self, sandbox_id: str) -> None:
        """Upload CUA server files to the sandbox using tar archive."""
        if not self._template_path.exists():
            raise RuntimeError(
                f"CUA server template not found at {self._template_path}"
            )

        client = await self._get_sandbox_client()

        # Create a tar archive of the server files
        with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as tmp_file:
            tar_path = Path(tmp_file.name)

        try:
            with tarfile.open(tar_path, "w:gz") as tar:
                if self.use_binary:
                    # Binary mode: only include binary and setup script
                    binary_path = (
                        self._template_path / "dist" / "sea" / "cua-server-linux-x64"
                    )
                    setup_script = self._template_path / "setup-binary.sh"

                    if not binary_path.exists():
                        raise RuntimeError(
                            f"Binary not found at {binary_path}. "
                            "Run _ensure_binary_exists() first."
                        )

                    tar.add(binary_path, arcname="cua-server/cua-server-linux-x64")
                    tar.add(setup_script, arcname="cua-server/setup-binary.sh")
                else:
                    # Source mode: include all files except node_modules and dist
                    exclude_dirs = {"node_modules", "dist", ".git"}
                    for file in self._template_path.glob("**/*"):
                        if file.is_file():
                            relative = file.relative_to(self._template_path)
                            # Skip files in excluded directories
                            if any(part in exclude_dirs for part in relative.parts):
                                continue
                            tar.add(file, arcname=f"cua-server/{relative}")

            remote_tar = "/tmp/cua-server.tar.gz"
            await client.upload_file(sandbox_id, remote_tar, str(tar_path))

            # Extract and set up
            setup_script_name = "setup-binary.sh" if self.use_binary else "setup.sh"
            await self._execute_sandbox_command(
                sandbox_id,
                f"mkdir -p {self.upload_path} && "
                f"tar -xzf {remote_tar} -C /app && "
                f"rm {remote_tar} && "
                f"chmod +x {self.upload_path}/{setup_script_name}",
            )

            if self.logger:
                mode = "binary" if self.use_binary else "source"
                self.logger.debug(
                    f"Uploaded CUA server files ({mode} mode) to sandbox {sandbox_id}"
                )
        finally:
            tar_path.unlink(missing_ok=True)

    async def _start_server(self, sandbox_id: str) -> None:
        """Start the CUA server in the sandbox background."""
        # Pass required environment variables to the server
        env_vars = f"CUA_SERVER_PORT={self.server_port}"

        # Stagehand requires OPENAI_API_KEY for modelApiKey
        openai_key = os.getenv("OPENAI_API_KEY", "")
        if openai_key:
            env_vars += f" OPENAI_API_KEY={openai_key}"

        setup_script = "setup-binary.sh" if self.use_binary else "setup.sh"

        await self._execute_sandbox_command(
            sandbox_id,
            f"cd {self.upload_path} && "
            f"{env_vars} "
            f"nohup bash {setup_script} > /tmp/cua-server.log 2>&1 &",
        )

        if self.logger:
            mode = "binary" if self.use_binary else "source"
            self.logger.debug(
                f"Started CUA server ({mode} mode) in sandbox {sandbox_id}"
            )

    async def _wait_for_server(self, sandbox_id: str) -> None:
        """Wait for the CUA server to be ready by polling the health endpoint."""
        health_url = f"http://localhost:{self.server_port}/health"
        start_time = asyncio.get_event_loop().time()
        attempt = 0
        last_error: str | None = None

        if self.logger:
            self.logger.debug(f"Waiting for CUA server in sandbox {sandbox_id}")

        while True:
            elapsed = asyncio.get_event_loop().time() - start_time
            attempt += 1

            if elapsed > self.server_ready_timeout:
                try:
                    log_content = await self._execute_sandbox_command(
                        sandbox_id,
                        "tail -100 /tmp/cua-server.log 2>/dev/null || echo 'No logs available'",
                    )
                except Exception:
                    log_content = "Could not retrieve logs"

                raise RuntimeError(
                    f"CUA server in sandbox {sandbox_id} did not become ready "
                    f"within {self.server_ready_timeout}s after {attempt} attempts.\n"
                    f"Last error: {last_error or 'unknown'}\n"
                    f"Server logs:\n{log_content}"
                )

            try:
                # Use -w to capture HTTP code, -s for silent, no -f to see error bodies
                stdout = await self._execute_sandbox_command(
                    sandbox_id,
                    f'curl -s -w "\\n%{{http_code}}" {health_url}',
                    timeout=10,
                )
                # Response format: "<body>\n<http_code>"
                lines = stdout.rsplit("\n", 1)
                body = lines[0] if len(lines) > 1 else stdout
                http_code = lines[-1].strip() if len(lines) > 1 else "unknown"

                if "ok" in body.lower() and http_code == "200":
                    if self.logger:
                        self.logger.debug(
                            f"CUA server ready in sandbox {sandbox_id} after {elapsed:.1f}s"
                        )
                    return
                last_error = f"HTTP {http_code}: {body[:100]}"
            except Exception as e:
                last_error = f"{type(e).__name__}: {str(e)[:100]}"
                if self.logger:
                    self.logger.debug(
                        f"Health check attempt {attempt} failed: {last_error}"
                    )

            await asyncio.sleep(self.server_ready_poll_interval)

    async def _create_session_via_curl(self, sandbox_id: str) -> dict:
        """Create a browser session via curl inside the sandbox."""
        payload = json.dumps(self.session_config)
        escaped_payload = payload.replace("'", "'\\''")

        # Use -w to capture HTTP code, -s for silent, no -f to see error bodies
        stdout = await self._execute_sandbox_command(
            sandbox_id,
            f'curl -s -w "\\n%{{http_code}}" -X POST http://localhost:{self.server_port}/sessions '
            f"-H 'Content-Type: application/json' "
            f"-d '{escaped_payload}'",
            timeout=60,
        )

        # Response format: "<body>\n<http_code>"
        lines = stdout.rsplit("\n", 1)
        body = lines[0] if len(lines) > 1 else stdout
        http_code = lines[-1].strip() if len(lines) > 1 else "unknown"

        try:
            return json.loads(body)
        except json.JSONDecodeError as e:
            raise RuntimeError(
                f"Failed to parse session creation response (HTTP {http_code}): {body[:500]}"
            ) from e

    async def _destroy_session_via_curl(self, session_id: str, sandbox_id: str) -> None:
        """Destroy a browser session via curl inside the sandbox."""
        try:
            # Use -s for silent, no -f to see error bodies if needed
            await self._execute_sandbox_command(
                sandbox_id,
                f"curl -s -X DELETE http://localhost:{self.server_port}/sessions/{session_id}",
                timeout=30,
            )
        except Exception as e:
            if self.logger:
                self.logger.warning(f"Failed to destroy session {session_id}: {e}")

    async def _execute_action_via_curl(
        self,
        session_id: str,
        action: dict,
        sandbox_id: str,
        tool_call_id: str | None = None,
    ) -> dict:
        """Execute a browser action via curl inside the sandbox."""
        payload = {**action}
        if tool_call_id:
            payload["tool_call_id"] = tool_call_id

        payload_json = json.dumps(payload)
        escaped_payload = payload_json.replace("'", "'\\''")

        # Use -w to capture HTTP code, -s for silent, no -f to see error bodies
        stdout = await self._execute_sandbox_command(
            sandbox_id,
            f'curl -s -w "\\n%{{http_code}}" -X POST http://localhost:{self.server_port}/sessions/{session_id}/action '
            f"-H 'Content-Type: application/json' "
            f"-d '{escaped_payload}'",
            timeout=60,
        )

        # Response format: "<body>\n<http_code>"
        lines = stdout.rsplit("\n", 1)
        body = lines[0] if len(lines) > 1 else stdout
        http_code = lines[-1].strip() if len(lines) > 1 else "unknown"

        try:
            return json.loads(body)
        except json.JSONDecodeError:
            return {
                "success": False,
                "error": f"Failed to parse response (HTTP {http_code}): {body[:200]}",
                "state": {},
            }

    # ==================== Screenshot Methods ====================

    def _save_screenshot(
        self, session_id: str, screenshot_b64: str, url: str = ""
    ) -> str | None:
        """Save a base64-encoded screenshot to disk."""
        if not self.save_screenshots or not screenshot_b64:
            return None

        try:
            os.makedirs(self.screenshot_dir, exist_ok=True)

            with self._counter_lock:
                if session_id not in self._screenshot_counters:
                    self._screenshot_counters[session_id] = 0
                counter = self._screenshot_counters[session_id]
                self._screenshot_counters[session_id] += 1

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            session_short = session_id[-20:] if len(session_id) > 20 else session_id
            session_short = session_short.replace("/", "_").replace("\\", "_")

            filename = f"{timestamp}_{session_short}_{counter:04d}.png"
            filepath = os.path.join(self.screenshot_dir, filename)

            image_data = base64.b64decode(screenshot_b64)
            with open(filepath, "wb") as f:
                f.write(image_data)

            return filepath

        except Exception as e:
            if self.logger:
                self.logger.warning(f"Failed to save screenshot: {e}")
            return None

    def _format_response(self, response: dict, session_id: str = "") -> list[dict]:
        """Format action response as multipart content with text and image."""
        success = response.get("success", False)
        error = response.get("error")
        state = response.get("state", {})
        screenshot_b64 = state.get("screenshot", "") or ""
        if screenshot_b64.startswith("data:"):
            screenshot_b64 = screenshot_b64.split(",", 1)[-1]
        url = state.get("url", "")
        viewport = state.get("viewport", {})

        status = "Success" if success else "Failed"
        text_parts = [f"Status: {status}"]
        if error:
            text_parts.append(f"Error: {error}")
        if url:
            text_parts.append(f"URL: {url}")
        if viewport:
            text_parts.append(
                f"Viewport: {viewport.get('width', 0)}x{viewport.get('height', 0)}"
            )

        content = [{"type": "text", "text": "\n".join(text_parts)}]

        if screenshot_b64 and session_id:
            self._save_screenshot(session_id, screenshot_b64, url)

        if screenshot_b64:
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{screenshot_b64}"},
                }
            )

        return content

    def filter_screenshots_in_messages(self, messages: list) -> list:
        """Replace older screenshots with text placeholders, keeping only the most recent N."""
        if self.keep_recent_screenshots is None:
            return messages

        screenshot_positions: list[tuple[int, int]] = []
        for msg_idx, msg in enumerate(messages):
            content = msg.get("content")
            if isinstance(content, list):
                for content_idx, item in enumerate(content):
                    if isinstance(item, dict) and item.get("type") == "image_url":
                        screenshot_positions.append((msg_idx, content_idx))

        if len(screenshot_positions) <= self.keep_recent_screenshots:
            return messages

        positions_to_replace = set(
            screenshot_positions
            if self.keep_recent_screenshots == 0
            else screenshot_positions[: -self.keep_recent_screenshots]
        )
        filtered_messages = copy.deepcopy(messages)

        for msg_idx, content_idx in positions_to_replace:
            content_list = filtered_messages[msg_idx]["content"]
            if isinstance(content_list, list) and content_idx < len(content_list):
                content_list[content_idx] = {
                    "type": "text",
                    "text": "[Screenshot removed to save context]",
                }

        return filtered_messages

    # ==================== Lifecycle Methods ====================

    async def setup_state(self, state: vf.State, **kwargs: Any) -> vf.State:
        """Create sandbox, set up CUA server, and create a browser session."""
        if self.use_prebuilt_image:
            # Fast path: prebuilt image with server already configured to start
            if self.logger:
                self.logger.debug(f"Using prebuilt image: {self.prebuilt_image}")

            # Create and wait for sandbox (server starts via start_command)
            sandbox_id = await self.with_retry(self._create_sandbox)()
            await self._wait_for_sandbox_ready(sandbox_id)

            state["cua_sandbox_id"] = sandbox_id

            # Wait for server to be ready (started by start_command)
            await self._wait_for_server(sandbox_id)
        else:
            # Standard path: upload binary and start server manually
            # Ensure binary exists if using binary mode
            if self.use_binary:
                await self._ensure_binary_exists()

            # Create and wait for sandbox
            sandbox_id = await self.with_retry(self._create_sandbox)()
            await self._wait_for_sandbox_ready(sandbox_id)

            state["cua_sandbox_id"] = sandbox_id

            # Upload server files
            await self._upload_server_files(sandbox_id)

            # Start the server
            await self._start_server(sandbox_id)

            # Wait for server to be ready
            await self._wait_for_server(sandbox_id)

        # Create browser session
        result = await self.with_retry(
            lambda: self._create_session_via_curl(sandbox_id)
        )()

        session_id = result.get("sessionId")
        if not session_id:
            raise RuntimeError(
                f"Failed to get session ID from server response. "
                f"Response keys: {list(result.keys())}, Response: {str(result)[:500]}"
            )

        with self._sessions_lock:
            self.active_sessions.add(session_id)

        state["session_id"] = session_id
        state["browser_state"] = result.get("state", {})

        return state

    def update_tool_args(
        self,
        tool_name: str,
        tool_args: dict[str, Any],
        messages: vf.Messages,
        state: vf.State,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Inject session_id and sandbox_id into all browser tool calls."""
        updated_args = dict(tool_args)
        updated_args["session_id"] = state["session_id"]
        updated_args["sandbox_id"] = state["cua_sandbox_id"]
        return updated_args

    async def cleanup_session(self, state: vf.State) -> None:
        """Destroy the browser session and sandbox after rollout completion."""
        session_id = state.get("session_id")
        sandbox_id = state.get("cua_sandbox_id")

        # Clean up browser session
        if session_id and sandbox_id:
            try:
                await self.with_retry(
                    lambda: self._destroy_session_via_curl(session_id, sandbox_id)
                )()
                with self._sessions_lock:
                    self.active_sessions.discard(session_id)
            except Exception as e:
                if self.logger:
                    self.logger.warning(f"Failed to destroy session {session_id}: {e}")

        # Clean up sandbox
        if sandbox_id:
            try:
                await self.with_retry(lambda: self._delete_sandbox(sandbox_id))()
            except Exception as e:
                if self.logger:
                    self.logger.warning(f"Failed to delete sandbox {sandbox_id}: {e}")

    async def teardown(self) -> None:
        """Clean up all remaining sandboxes on environment teardown."""
        if len(self.active_sandboxes) == 0:
            return

        if self.logger:
            self.logger.info(
                f"Deleting {len(self.active_sandboxes)} remaining sandboxes"
            )

        # Use sync client for teardown to avoid event loop issues during shutdown
        sync_client = SandboxClient(APIClient())
        sandbox_ids = list(self.active_sandboxes)

        for sandbox_id in sandbox_ids:
            try:
                sync_client.delete(sandbox_id)
                self.active_sandboxes.discard(sandbox_id)
                if self.logger:
                    self.logger.debug(f"Deleted sandbox {sandbox_id}")
            except Exception as e:
                if self.logger:
                    self.logger.warning(f"Failed to delete sandbox {sandbox_id}: {e}")

    # ==================== Browser Tool Methods ====================

    async def click(
        self,
        x: int,
        y: int,
        button: Literal["left", "right", "middle"] = "left",
        session_id: str = "",
        sandbox_id: str = "",
        tool_call_id: str = "",
    ) -> list[dict]:
        """Click at coordinates (x, y) on the page."""
        response = await self._execute_action_via_curl(
            session_id,
            {"type": "click", "x": x, "y": y, "button": button},
            sandbox_id,
            tool_call_id,
        )
        return self._format_response(response, session_id)

    async def double_click(
        self,
        x: int,
        y: int,
        session_id: str = "",
        sandbox_id: str = "",
        tool_call_id: str = "",
    ) -> list[dict]:
        """Double-click at coordinates (x, y) on the page."""
        response = await self._execute_action_via_curl(
            session_id,
            {"type": "double_click", "x": x, "y": y},
            sandbox_id,
            tool_call_id,
        )
        return self._format_response(response, session_id)

    async def type_text(
        self,
        text: str,
        session_id: str = "",
        sandbox_id: str = "",
        tool_call_id: str = "",
    ) -> list[dict]:
        """Type text into the currently focused element."""
        response = await self._execute_action_via_curl(
            session_id,
            {"type": "type", "text": text},
            sandbox_id,
            tool_call_id,
        )
        return self._format_response(response, session_id)

    async def keypress(
        self,
        keys: str | list[str],
        session_id: str = "",
        sandbox_id: str = "",
        tool_call_id: str = "",
    ) -> list[dict]:
        """Press keyboard key(s)."""
        response = await self._execute_action_via_curl(
            session_id,
            {"type": "keypress", "keys": keys},
            sandbox_id,
            tool_call_id,
        )
        return self._format_response(response, session_id)

    async def scroll(
        self,
        x: int = 0,
        y: int = 0,
        scroll_x: int = 0,
        scroll_y: int = 0,
        session_id: str = "",
        sandbox_id: str = "",
        tool_call_id: str = "",
    ) -> list[dict]:
        """Scroll the page at a specific position."""
        response = await self._execute_action_via_curl(
            session_id,
            {
                "type": "scroll",
                "x": x,
                "y": y,
                "scroll_x": scroll_x,
                "scroll_y": scroll_y,
            },
            sandbox_id,
            tool_call_id,
        )
        return self._format_response(response, session_id)

    async def goto(
        self,
        url: str,
        session_id: str = "",
        sandbox_id: str = "",
        tool_call_id: str = "",
    ) -> list[dict]:
        """Navigate to a URL."""
        try:
            response = await self._execute_action_via_curl(
                session_id,
                {"type": "goto", "url": url},
                sandbox_id,
                tool_call_id,
            )
        except (TimeoutError, asyncio.TimeoutError):
            response = {
                "success": False,
                "error": f"Navigation timeout: The page at {url} took too long to load",
                "state": {"url": url, "viewport": {}},
            }
        return self._format_response(response, session_id)

    async def back(
        self,
        session_id: str = "",
        sandbox_id: str = "",
        tool_call_id: str = "",
    ) -> list[dict]:
        """Navigate back in browser history."""
        response = await self._execute_action_via_curl(
            session_id,
            {"type": "back"},
            sandbox_id,
            tool_call_id,
        )
        return self._format_response(response, session_id)

    async def forward(
        self,
        session_id: str = "",
        sandbox_id: str = "",
        tool_call_id: str = "",
    ) -> list[dict]:
        """Navigate forward in browser history."""
        response = await self._execute_action_via_curl(
            session_id,
            {"type": "forward"},
            sandbox_id,
            tool_call_id,
        )
        return self._format_response(response, session_id)

    async def wait(
        self,
        time_ms: int = 1000,
        session_id: str = "",
        sandbox_id: str = "",
        tool_call_id: str = "",
    ) -> list[dict]:
        """Wait for a specified amount of time."""
        try:
            response = await self._execute_action_via_curl(
                session_id,
                {"type": "wait", "timeMs": time_ms},
                sandbox_id,
                tool_call_id,
            )
        except (TimeoutError, asyncio.TimeoutError):
            response = {
                "success": False,
                "error": f"Wait timeout: The wait operation ({time_ms}ms) timed out",
                "state": {"viewport": {}},
            }
        return self._format_response(response, session_id)

    async def screenshot(
        self,
        session_id: str = "",
        sandbox_id: str = "",
        tool_call_id: str = "",
    ) -> list[dict]:
        """Capture a screenshot of the current page state."""
        response = await self._execute_action_via_curl(
            session_id,
            {"type": "screenshot"},
            sandbox_id,
            tool_call_id,
        )
        return self._format_response(response, session_id)
