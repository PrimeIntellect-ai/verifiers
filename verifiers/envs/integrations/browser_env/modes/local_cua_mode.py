"""Local CUA mode for browser automation of localhost applications."""

import asyncio
import base64
import copy
import json
import logging
import os
import tarfile
import tempfile
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import tenacity as tc
from tenacity import AsyncRetrying
import verifiers as vf


# Conditional imports for sandbox mode
try:
    from prime_sandboxes import (
        AsyncSandboxClient,
        CreateSandboxRequest,
    )

    SANDBOX_AVAILABLE = True
except ImportError:
    SANDBOX_AVAILABLE = False
    AsyncSandboxClient = None  # type: ignore[misc, assignment]
    CreateSandboxRequest = None  # type: ignore[misc, assignment]


class LocalCUAMode:
    """
    Local CUA mode for browser automation of localhost applications.

    This mode runs a CUA server alongside a local web application (e.g., Next.js)
    in a sandbox container. The browser starts at the application's URL and can
    interact with it using vision-based primitives.

    Key differences from standard CUA mode:
    - No 'goto' action - browser starts at the configured app URL
    - No internet access - designed for localhost app testing
    - Manages both the CUA server and the target application

    Provides vision-based primitives: click, double_click, type_text,
    keypress, scroll, back, forward, wait, screenshot
    """

    def __init__(
        self,
        # Application configuration
        app_path: str | Path | None = None,
        app_port: int = 3000,
        app_start_command: str = "npm run start",
        app_build_command: str | None = "npm install && npm run build",
        # CUA server configuration
        cua_server_port: int = 3001,
        # Viewport configuration
        viewport_width: int = 1024,
        viewport_height: int = 768,
        # Retry configuration
        max_retries: int = 5,
        base_delay: float = 0.5,
        backoff_factor: float = 2.0,
        max_backoff_seconds: float = 30.0,
        jitter: float = 1e-3,
        # Screenshot configuration
        screenshot_dir: str | None = None,
        save_screenshots: bool = True,
        keep_recent_screenshots: int | None = 2,
        # Sandbox configuration
        server_ready_timeout: int = 180,
        server_ready_poll_interval: float = 2.0,
        app_ready_timeout: int = 120,
        cpu_cores: int = 2,
        memory_gb: int = 4,
        disk_size_gb: int = 10,
        sandbox_timeout_minutes: int = 60,
        sandbox_timeout_per_command_seconds: int = 120,
        # Pre-built image configuration
        use_prebuilt_image: bool = True,
        prebuilt_image: str = "deepdream19/cua-local-server:latest",
        # Base image for custom builds
        docker_image: str = "node:20-bookworm-slim",
    ):
        if not SANDBOX_AVAILABLE:
            raise ImportError(
                "prime-sandboxes is not installed. "
                "Please install it with `uv pip install prime-sandboxes`."
            )

        self.keep_recent_screenshots = keep_recent_screenshots
        self.logger = None

        # Application configuration
        self.app_path = Path(app_path) if app_path else None
        self.app_port = app_port
        self.app_start_command = app_start_command
        self.app_build_command = app_build_command
        self.app_url = f"http://localhost:{app_port}"

        # CUA server configuration
        self.cua_server_port = cua_server_port

        # Session config for browser creation
        self.session_config = {
            "viewport": {"width": viewport_width, "height": viewport_height},
            "startUrl": self.app_url,
            "headless": True,
            "executablePath": "/usr/bin/chromium",
        }

        # Thread-safe locks
        self._sessions_lock = threading.Lock()
        self._counter_lock = threading.Lock()
        self.active_sessions: set[str] = set()

        # Screenshot config
        self.save_screenshots = save_screenshots
        self.screenshot_dir = screenshot_dir or os.path.join(os.getcwd(), "screenshots")
        self._screenshot_counters: dict[str, int] = {}

        # Retry configuration
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.backoff_factor = backoff_factor
        self.max_backoff_seconds = max_backoff_seconds
        self.jitter = jitter
        self.retrying: AsyncRetrying | None = None

        # Sandbox configuration
        self.server_ready_timeout = server_ready_timeout
        self.server_ready_poll_interval = server_ready_poll_interval
        self.app_ready_timeout = app_ready_timeout
        self.sandbox_timeout_per_command_seconds = sandbox_timeout_per_command_seconds
        self.active_sandboxes: set[str] = set()
        self._sandbox_client: AsyncSandboxClient | None = None
        self.use_prebuilt_image = use_prebuilt_image
        self.prebuilt_image = prebuilt_image

        # Template paths
        self._cua_local_path = (
            Path(__file__).parent.parent.parent.parent.parent.parent
            / "assets"
            / "templates"
            / "browserbase"
            / "cua-local"
        )
        self._nextjs_example_path = (
            Path(__file__).parent.parent.parent.parent.parent.parent
            / "assets"
            / "templates"
            / "nextjs-example"
        )

        # Build environment variables
        env_vars = {
            "CUA_SERVER_PORT": str(cua_server_port),
            "CUA_SERVER_HOST": "0.0.0.0",
            "APP_PORT": str(app_port),
            "PORT": str(app_port),
        }

        # Initialize sandbox request
        if use_prebuilt_image:
            self._sandbox_request = CreateSandboxRequest(
                name="cua-local-app",
                docker_image=prebuilt_image,
                start_command="tail -f /dev/null",  # We'll start services manually
                cpu_cores=cpu_cores,
                memory_gb=memory_gb,
                disk_size_gb=disk_size_gb,
                gpu_count=0,
                timeout_minutes=sandbox_timeout_minutes,
                environment_vars=env_vars,
            )
        else:
            self._sandbox_request = CreateSandboxRequest(
                name="cua-local-app",
                docker_image=docker_image,
                start_command="tail -f /dev/null",
                cpu_cores=cpu_cores,
                memory_gb=memory_gb,
                disk_size_gb=disk_size_gb,
                gpu_count=0,
                timeout_minutes=sandbox_timeout_minutes,
                environment_vars=env_vars,
            )

    def register_tools(self, env) -> None:
        """Register Local CUA mode tools with the environment (no goto)."""
        self.logger = env.logger

        # Set up retry now that we have logger
        self.retrying = tc.AsyncRetrying(
            stop=tc.stop_after_attempt(self.max_retries),
            wait=tc.wait_exponential_jitter(
                initial=self.base_delay,
                exp_base=self.backoff_factor,
                max=self.max_backoff_seconds,
                jitter=self.jitter,
            ),
            before_sleep=tc.before_sleep_log(self.logger, logging.ERROR),
            reraise=True,
        )

        # Hide internal args from tool schema
        _skip = ["session_id", "sandbox_id", "tool_call_id"]

        # Register tools WITHOUT goto
        env.add_tool(self.click, args_to_skip=_skip)
        env.add_tool(self.double_click, args_to_skip=_skip)
        env.add_tool(self.type_text, args_to_skip=_skip)
        env.add_tool(self.keypress, args_to_skip=_skip)
        env.add_tool(self.scroll, args_to_skip=_skip)
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

    async def _create_sandbox_with_retry(self) -> str:
        """Create a sandbox with retry."""
        previous_sandbox_id: str | None = None
        async for attempt in self.retrying:  # type: ignore[union-attr]
            with attempt:
                if previous_sandbox_id is not None:
                    try:
                        await self._delete_sandbox(previous_sandbox_id)
                    except Exception:
                        pass
                sandbox_id = await self._create_sandbox()
                previous_sandbox_id = sandbox_id
        return sandbox_id

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

    # ==================== Application Setup Methods ====================

    async def _upload_app_files(self, sandbox_id: str) -> None:
        """Upload the target application to the sandbox."""
        app_path = self.app_path or self._nextjs_example_path

        if not app_path.exists():
            raise RuntimeError(f"Application not found at {app_path}")

        client = await self._get_sandbox_client()

        with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as tmp_file:
            tar_path = Path(tmp_file.name)

        try:
            with tarfile.open(tar_path, "w:gz") as tar:
                exclude_dirs = {"node_modules", ".next", "dist", ".git", "out"}
                for file in app_path.glob("**/*"):
                    if file.is_file():
                        relative = file.relative_to(app_path)
                        if any(part in exclude_dirs for part in relative.parts):
                            continue
                        tar.add(file, arcname=f"nextjs-app/{relative}")

            remote_tar = "/tmp/nextjs-app.tar.gz"
            await client.upload_file(sandbox_id, remote_tar, str(tar_path))

            await self._execute_sandbox_command(
                sandbox_id,
                f"mkdir -p /app/nextjs-app && "
                f"tar -xzf {remote_tar} -C /app && "
                f"rm {remote_tar}",
            )

            if self.logger:
                self.logger.debug(f"Uploaded app files to sandbox {sandbox_id}")
        finally:
            tar_path.unlink(missing_ok=True)

    async def _build_app(self, sandbox_id: str) -> None:
        """Build the application in the sandbox."""
        if not self.app_build_command:
            return

        if self.logger:
            self.logger.info("Building application in sandbox...")

        await self._execute_sandbox_command(
            sandbox_id,
            f"cd /app/nextjs-app && {self.app_build_command}",
            timeout=300,  # 5 minutes for build
        )

        if self.logger:
            self.logger.debug(f"Built app in sandbox {sandbox_id}")

    async def _start_app(self, sandbox_id: str) -> None:
        """Start the application in the sandbox background."""
        env_vars = f"PORT={self.app_port}"

        await self._execute_sandbox_command(
            sandbox_id,
            f"cd /app/nextjs-app && "
            f"{env_vars} nohup {self.app_start_command} > /tmp/app.log 2>&1 &",
        )

        if self.logger:
            self.logger.debug(f"Started app in sandbox {sandbox_id}")

    async def _wait_for_app(self, sandbox_id: str) -> None:
        """Wait for the application to be ready."""
        start_time = asyncio.get_event_loop().time()
        attempt = 0

        if self.logger:
            self.logger.debug(f"Waiting for app at {self.app_url}")

        while True:
            elapsed = asyncio.get_event_loop().time() - start_time
            attempt += 1

            if elapsed > self.app_ready_timeout:
                try:
                    log_content = await self._execute_sandbox_command(
                        sandbox_id,
                        "tail -100 /tmp/app.log 2>/dev/null || echo 'No logs available'",
                    )
                except Exception:
                    log_content = "Could not retrieve logs"

                raise RuntimeError(
                    f"Application did not become ready within {self.app_ready_timeout}s.\n"
                    f"App logs:\n{log_content}"
                )

            try:
                stdout = await self._execute_sandbox_command(
                    sandbox_id,
                    f'curl -s -w "\\n%{{http_code}}" {self.app_url}',
                    timeout=10,
                )
                lines = stdout.rsplit("\n", 1)
                http_code = lines[-1].strip() if len(lines) > 1 else "unknown"

                if http_code == "200":
                    if self.logger:
                        self.logger.debug(f"App ready after {elapsed:.1f}s")
                    return
            except Exception:
                pass

            await asyncio.sleep(self.server_ready_poll_interval)

    async def _start_cua_server(self, sandbox_id: str) -> None:
        """Start the CUA local server in the sandbox."""
        env_vars = f"CUA_SERVER_PORT={self.cua_server_port} CUA_SERVER_HOST=0.0.0.0"

        await self._execute_sandbox_command(
            sandbox_id,
            f"cd /app/cua-local-server && "
            f"{env_vars} nohup ./cua-local-server-linux-x64 > /tmp/cua-server.log 2>&1 &",
        )

        if self.logger:
            self.logger.debug(f"Started CUA server in sandbox {sandbox_id}")

    async def _wait_for_cua_server(self, sandbox_id: str) -> None:
        """Wait for the CUA server to be ready."""
        health_url = f"http://localhost:{self.cua_server_port}/health"
        start_time = asyncio.get_event_loop().time()
        attempt = 0
        last_error: str | None = None

        if self.logger:
            self.logger.debug(f"Waiting for CUA server at {health_url}")

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
                    f"CUA server did not become ready within {self.server_ready_timeout}s.\n"
                    f"Last error: {last_error or 'unknown'}\n"
                    f"Server logs:\n{log_content}"
                )

            try:
                stdout = await self._execute_sandbox_command(
                    sandbox_id,
                    f'curl -s -w "\\n%{{http_code}}" {health_url}',
                    timeout=10,
                )
                lines = stdout.rsplit("\n", 1)
                body = lines[0] if len(lines) > 1 else stdout
                http_code = lines[-1].strip() if len(lines) > 1 else "unknown"

                if "ok" in body.lower() and http_code == "200":
                    if self.logger:
                        self.logger.debug(f"CUA server ready after {elapsed:.1f}s")
                    return
                last_error = f"HTTP {http_code}: {body[:100]}"
            except Exception as e:
                last_error = f"{type(e).__name__}: {str(e)[:100]}"

            await asyncio.sleep(self.server_ready_poll_interval)

    # ==================== Session Methods (via curl) ====================

    @staticmethod
    def _parse_curl_response(stdout: str) -> tuple[str, str]:
        """Parse curl output with HTTP code appended."""
        lines = stdout.rsplit("\n", 1)
        body = lines[0] if len(lines) > 1 else stdout
        http_code = lines[-1].strip() if len(lines) > 1 else "unknown"
        return body, http_code

    async def _create_session_curl(self, sandbox_id: str) -> dict:
        """Create a browser session via curl inside the sandbox."""
        payload = json.dumps(self.session_config)
        escaped_payload = payload.replace("'", "'\\''")

        stdout = await self._execute_sandbox_command(
            sandbox_id,
            f'curl -s -w "\\n%{{http_code}}" -X POST http://localhost:{self.cua_server_port}/sessions '
            f"-H 'Content-Type: application/json' "
            f"-d '{escaped_payload}'",
            timeout=60,
        )

        body, http_code = self._parse_curl_response(stdout)

        try:
            return json.loads(body)
        except json.JSONDecodeError as e:
            raise RuntimeError(
                f"Failed to parse session creation response (HTTP {http_code}): {body[:500]}"
            ) from e

    async def _destroy_session_curl(self, session_id: str, sandbox_id: str) -> None:
        """Destroy a browser session via curl inside the sandbox."""
        try:
            await self._execute_sandbox_command(
                sandbox_id,
                f"curl -s -X DELETE http://localhost:{self.cua_server_port}/sessions/{session_id}",
                timeout=30,
            )
        except Exception as e:
            if self.logger:
                self.logger.warning(f"Failed to destroy session {session_id}: {e}")

    async def _execute_action_curl(
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

        stdout = await self._execute_sandbox_command(
            sandbox_id,
            f'curl -s -w "\\n%{{http_code}}" -X POST http://localhost:{self.cua_server_port}/sessions/{session_id}/action '
            f"-H 'Content-Type: application/json' "
            f"-d '{escaped_payload}'",
            timeout=60,
        )

        body, http_code = self._parse_curl_response(stdout)

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

        content: list[dict[str, str | dict[str, str]]] = [
            {"type": "text", "text": "\n".join(text_parts)}
        ]

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
        """Replace older screenshots with text placeholders."""
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
        """Create sandbox, start app and CUA server, create browser session."""
        try:
            if self.logger:
                self.logger.info("Setting up Local CUA sandbox...")

            # Create sandbox
            sandbox_id = await self._create_sandbox_with_retry()
            state["cua_sandbox_id"] = sandbox_id
            await self._wait_for_sandbox_ready(sandbox_id)

            # Upload and build app
            await self._upload_app_files(sandbox_id)
            await self._build_app(sandbox_id)

            # Start services
            await self._start_app(sandbox_id)
            await self._start_cua_server(sandbox_id)

            # Wait for services to be ready
            await self._wait_for_app(sandbox_id)
            await self._wait_for_cua_server(sandbox_id)

            # Create browser session
            async for attempt in self.retrying:  # type: ignore[union-attr]
                with attempt:
                    result = await self._create_session_curl(sandbox_id)

            session_id = result.get("sessionId")
            if not session_id:
                raise RuntimeError(
                    f"Failed to get session ID from server response. "
                    f"Response: {str(result)[:500]}"
                )

            with self._sessions_lock:
                self.active_sessions.add(session_id)

            state["session_id"] = session_id
            state["browser_state"] = result.get("state", {})

            if self.logger:
                self.logger.info("Local CUA sandbox ready")

        except vf.Error:
            raise
        except Exception as e:
            raise vf.BrowserSandboxError(e)

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
        """Destroy the browser session and sandbox."""
        session_id = state.get("session_id")
        sandbox_id = state.get("cua_sandbox_id")

        if session_id and sandbox_id:
            try:
                async for attempt in self.retrying:  # type: ignore[union-attr]
                    with attempt:
                        await self._destroy_session_curl(session_id, sandbox_id)
                with self._sessions_lock:
                    self.active_sessions.discard(session_id)
            except Exception as e:
                if self.logger:
                    self.logger.warning(f"Failed to destroy session {session_id}: {e}")

        if sandbox_id:
            try:
                async for attempt in self.retrying:  # type: ignore[union-attr]
                    with attempt:
                        await self._delete_sandbox(sandbox_id)
            except Exception as e:
                if self.logger:
                    self.logger.warning(f"Failed to delete sandbox {sandbox_id}: {e}")

    async def teardown(self, max_concurrent: int = 50) -> None:
        """Clean up all remaining sandboxes."""
        if len(self.active_sandboxes) == 0:
            return

        sandbox_ids = list(self.active_sandboxes)

        if self.logger:
            self.logger.info(f"Deleting {len(sandbox_ids)} remaining sandboxes")

        try:
            loop = asyncio.get_running_loop()
            if loop.is_closed():
                return
        except RuntimeError:
            return

        semaphore = asyncio.Semaphore(max_concurrent)

        async def _delete_sandbox_with_semaphore(sandbox_id: str):
            async with semaphore:
                try:
                    await self._delete_sandbox(sandbox_id)
                except Exception as e:
                    if self.logger:
                        self.logger.warning(f"Failed to delete sandbox {sandbox_id}: {e}")

        try:
            await asyncio.gather(
                *[_delete_sandbox_with_semaphore(sid) for sid in sandbox_ids]
            )
        except RuntimeError:
            pass

    # ==================== Browser Tool Methods (no goto) ====================

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
        response = await self._execute_action_curl(
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
        response = await self._execute_action_curl(
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
        response = await self._execute_action_curl(
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
        response = await self._execute_action_curl(
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
        response = await self._execute_action_curl(
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

    async def back(
        self,
        session_id: str = "",
        sandbox_id: str = "",
        tool_call_id: str = "",
    ) -> list[dict]:
        """Navigate back in browser history."""
        response = await self._execute_action_curl(
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
        response = await self._execute_action_curl(
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
            response = await self._execute_action_curl(
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
        response = await self._execute_action_curl(
            session_id,
            {"type": "screenshot"},
            sandbox_id,
            tool_call_id,
        )
        return self._format_response(response, session_id)
