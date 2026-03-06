"""CUA-based browser mode supporting both local HTTP and sandbox execution."""

import asyncio
import base64
import copy
import inspect
import json
import logging
import os
import tarfile
import tempfile
import threading
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import aiohttp
import tenacity as tc
from tenacity import AsyncRetrying
import verifiers as vf


# Conditional imports for sandbox mode
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
    AsyncSandboxClient = None  # type: ignore[misc, assignment]
    CreateSandboxRequest = None  # type: ignore[misc, assignment]
    SandboxClient = None  # type: ignore[misc, assignment]
    APIClient = None  # type: ignore[misc, assignment]


class CUATransientError(RuntimeError):
    """Retryable CUA request error."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        code: str | None = None,
    ):
        super().__init__(message)
        self.status_code = status_code
        self.code = code


class CUAMode:
    """
    CUA-based browser mode supporting both local HTTP and sandbox execution.

    Execution modes:
    1. Local mode (execution_mode="local"):
       - Connects to an existing CUA server via HTTP (aiohttp)
       - User must start the server manually
       - Fastest for local development

    2. Sandbox mode (execution_mode="sandbox"):
       - Automatically creates a sandbox container
       - Deploys and starts the CUA server inside
       - Executes browser actions via curl commands inside the sandbox
       - Cleans up sandbox when done

    Sandbox sub-modes (when execution_mode="sandbox"):
    - Pre-built image (default, use_prebuilt_image=True):
      Uses pre-built Docker image with server ready to start. Fastest startup.
    - Binary upload (use_prebuilt_image=False):
      Creates sandbox, uploads binary, installs deps, starts server.
      Useful for custom server versions.

    Provides vision-based primitives: click, double_click, type_text,
    keypress, scroll, goto, back, forward, wait, screenshot
    """

    def __init__(
        self,
        execution_mode: Literal["local", "sandbox"] = "local",
        # Local mode config
        server_url: str = "http://localhost:3000",
        # Shared config
        env: Literal["LOCAL", "BROWSERBASE"] = "LOCAL",
        browserbase_api_key: str | None = None,
        browserbase_project_id: str | None = None,
        viewport_width: int = 1024,
        viewport_height: int = 768,
        max_retries: int = 5,
        base_delay: float = 0.5,
        backoff_factor: float = 2.0,
        max_backoff_seconds: float = 30.0,
        jitter: float = 1e-3,
        cua_max_concurrent_requests: int | None = 8,
        screenshot_dir: str | None = None,
        save_screenshots: bool = True,
        keep_recent_screenshots: int | None = 2,
        proxies: bool = False,
        advanced_stealth: bool = False,
        # Sandbox mode config
        server_port: int = 3000,
        upload_path: str = "/app/cua-server",
        server_ready_timeout: int = 120,
        server_ready_poll_interval: float = 2.0,
        docker_image: str = "node:18-slim",
        cpu_cores: int = 2,
        memory_gb: int = 4,
        disk_size_gb: int = 10,
        sandbox_timeout_minutes: int = 60,
        sandbox_timeout_per_command_seconds: int = 60,
        use_binary: bool = True,
        use_prebuilt_image: bool = True,
        prebuilt_image: str = "deepdream19/cua-server:latest",
    ):
        # Validate sandbox mode requirements
        if execution_mode == "sandbox" and not SANDBOX_AVAILABLE:
            raise ImportError(
                "prime-sandboxes is not installed. "
                "Please install it with `uv pip install prime-sandboxes`."
            )

        self._execution_mode = execution_mode
        self.keep_recent_screenshots = keep_recent_screenshots
        self.logger = None  # Will be set when register_tools is called
        self.viewport_width = viewport_width
        self.viewport_height = viewport_height

        # Resolve API keys
        resolved_api_key = browserbase_api_key or os.getenv("BROWSERBASE_API_KEY")
        resolved_project_id = browserbase_project_id or os.getenv(
            "BROWSERBASE_PROJECT_ID"
        )

        # Session config (shared)
        self.session_config = {
            "env": env,
            "browserbaseApiKey": resolved_api_key,
            "browserbaseProjectId": resolved_project_id,
            "viewport": {"width": viewport_width, "height": viewport_height},
            "proxies": proxies,
            "browserSettings": {"advancedStealth": advanced_stealth}
            if advanced_stealth
            else None,
        }
        self.session_config = {
            k: v for k, v in self.session_config.items() if v is not None
        }

        # Thread-safe locks (shared)
        self._thread_lock = threading.Lock()
        self._sessions_lock = threading.Lock()
        self._counter_lock = threading.Lock()
        self.active_sessions: set[str] = set()

        # Screenshot config (shared)
        self.save_screenshots = save_screenshots
        self.screenshot_dir = screenshot_dir or os.path.join(os.getcwd(), "screenshots")
        self._screenshot_counters: dict[str, int] = {}

        # Retry configuration (shared)
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.backoff_factor = backoff_factor
        self.max_backoff_seconds = max_backoff_seconds
        self.jitter = jitter
        self.retrying: AsyncRetrying | None = None
        self.request_retrying: AsyncRetrying | None = None
        self.cua_max_concurrent_requests = cua_max_concurrent_requests
        self._request_semaphore = (
            asyncio.Semaphore(cua_max_concurrent_requests)
            if cua_max_concurrent_requests and cua_max_concurrent_requests > 0
            else None
        )

        # Local mode specific
        self.server_url = server_url.rstrip("/")
        self._http_client: aiohttp.ClientSession | None = None
        self._client_lock = asyncio.Lock()

        # Sandbox mode specific
        self.server_port = server_port
        self.upload_path = upload_path
        self.server_ready_timeout = server_ready_timeout
        self.server_ready_poll_interval = server_ready_poll_interval
        self.sandbox_timeout_per_command_seconds = sandbox_timeout_per_command_seconds
        self.active_sandboxes: set[str] = set()
        self._sandbox_client: AsyncSandboxClient | None = None
        self.use_binary = use_binary
        self.use_prebuilt_image = use_prebuilt_image
        self.prebuilt_image = prebuilt_image

        # Get template path for binary builds
        self._template_path = (
            Path(__file__).parent.parent.parent.parent.parent.parent
            / "assets"
            / "templates"
            / "browserbase"
            / "cua"
        )

        # Initialize sandbox request if in sandbox mode
        self._sandbox_request = None
        if execution_mode == "sandbox" and SANDBOX_AVAILABLE:
            if use_prebuilt_image:
                # Build environment variables for the container
                env_vars = {
                    "CUA_SERVER_PORT": str(server_port),
                    "CUA_SERVER_HOST": "0.0.0.0",
                }
                openai_key = os.getenv("OPENAI_API_KEY", "")
                if openai_key:
                    env_vars["OPENAI_API_KEY"] = openai_key

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
            else:
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

    def register_tools(self, env) -> None:
        """Register CUA mode tools with the environment."""
        self.logger = env.logger

        # Resource lifecycle retry: broad by design for sandbox/session setup and cleanup.
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
        self.request_retrying = tc.AsyncRetrying(
            stop=tc.stop_after_attempt(self.max_retries),
            wait=tc.wait_exponential_jitter(
                initial=self.base_delay,
                exp_base=self.backoff_factor,
                max=self.max_backoff_seconds,
                jitter=self.jitter,
            ),
            retry=tc.retry_if_exception_type(
                (CUATransientError, aiohttp.ClientError, asyncio.TimeoutError)
            ),
            before_sleep=tc.before_sleep_log(self.logger, logging.WARNING),
            reraise=True,
        )

        # Hide internal args from tool schema
        _skip = ["session_id", "sandbox_id", "tool_call_id"]
        env.add_tool(
            self._wrap_tool(
                self.click,
                (
                    "Click at pixel coordinates (x, y) on the page. "
                    f"The display resolution is {self.viewport_width}x{self.viewport_height} pixels. "
                    "Use integer pixel coordinates measured from the top-left corner and center clicks on the target."
                ),
            ),
            args_to_skip=_skip,
        )
        env.add_tool(
            self._wrap_tool(
                self.double_click,
                (
                    "Double-click at pixel coordinates (x, y). "
                    f"The display resolution is {self.viewport_width}x{self.viewport_height} pixels. "
                    "Use integer pixel coordinates measured from the top-left corner."
                ),
            ),
            args_to_skip=_skip,
        )
        env.add_tool(
            self._wrap_tool(
                self.type_text,
                "Type text into the currently focused element.",
            ),
            args_to_skip=_skip,
        )
        env.add_tool(
            self._wrap_tool(
                self.keypress,
                "Press one key or a list of keys using Playwright-style key names.",
            ),
            args_to_skip=_skip,
        )
        env.add_tool(
            self._wrap_tool(
                self.scroll,
                (
                    "Scroll the page. x and y are integer pixel cursor coordinates, "
                    "and scroll_x/scroll_y are integer pixel deltas. "
                    f"The display resolution is {self.viewport_width}x{self.viewport_height} pixels."
                ),
            ),
            args_to_skip=_skip,
        )
        env.add_tool(
            self._wrap_tool(self.goto, "Navigate the browser to a URL."),
            args_to_skip=_skip,
        )
        env.add_tool(
            self._wrap_tool(self.back, "Navigate back in browser history."),
            args_to_skip=_skip,
        )
        env.add_tool(
            self._wrap_tool(self.forward, "Navigate forward in browser history."),
            args_to_skip=_skip,
        )
        env.add_tool(
            self._wrap_tool(
                self.wait,
                "Wait for a specified non-negative integer number of milliseconds.",
            ),
            args_to_skip=_skip,
        )
        env.add_tool(
            self._wrap_tool(
                self.screenshot,
                (
                    "Capture a screenshot of the current page state. "
                    f"The display resolution is {self.viewport_width}x{self.viewport_height} pixels."
                ),
            ),
            args_to_skip=_skip,
        )

        # For local mode, verify server is reachable
        if self._execution_mode == "local":
            self.verify_server_connection()

    @staticmethod
    def _wrap_tool(tool, description: str):
        """Return a callable wrapper with the original signature and a custom description."""

        async def wrapper(*args, **kwargs):
            return await tool(*args, **kwargs)

        setattr(wrapper, "__name__", getattr(tool, "__name__", "unknown"))
        setattr(wrapper, "__doc__", description)
        setattr(wrapper, "__signature__", inspect.signature(tool))
        setattr(wrapper, "__annotations__", getattr(tool, "__annotations__", {}))
        return wrapper

    @staticmethod
    def _parse_json_payload(body: str) -> dict[str, Any] | None:
        try:
            payload = json.loads(body)
        except json.JSONDecodeError:
            return None
        return payload if isinstance(payload, dict) else None

    @staticmethod
    def _coerce_http_status(http_code: str | int) -> int | None:
        if isinstance(http_code, int):
            return http_code
        if isinstance(http_code, str) and http_code.isdigit():
            return int(http_code)
        return None

    @staticmethod
    def _describe_value_type(value: Any) -> str:
        if value is None:
            return "null"
        if isinstance(value, bool):
            return "boolean"
        if isinstance(value, (int, float)):
            return "number"
        if isinstance(value, list):
            return "array"
        if isinstance(value, dict):
            return "object"
        return type(value).__name__

    def _format_error_message(
        self,
        status_code: int | None,
        payload: dict[str, Any] | None,
        raw_body: str,
    ) -> str:
        if payload:
            error = str(
                payload.get("error") or f"CUA request failed (HTTP {status_code})"
            )
            details = payload.get("details")
            if isinstance(details, list) and details:
                formatted_details = []
                for detail in details:
                    if not isinstance(detail, dict):
                        continue
                    message = detail.get("message")
                    if isinstance(message, str) and message.strip():
                        formatted_details.append(message)
                        continue
                    field = detail.get("field", "field")
                    expected = detail.get("expected", "the expected type")
                    received_value = detail.get("receivedValue")
                    received_type = detail.get(
                        "receivedType", self._describe_value_type(received_value)
                    )
                    formatted_details.append(
                        f"{field} must be {expected}, received {received_value!r} ({received_type})"
                    )
                if formatted_details:
                    return f"{error}: {'; '.join(formatted_details)}"
            return error

        if status_code is not None:
            return f"CUA request failed (HTTP {status_code}): {raw_body[:500]}"
        return raw_body[:500]

    def _is_retryable_response(
        self,
        status_code: int | None,
        payload: dict[str, Any] | None,
    ) -> bool:
        if payload and payload.get("retryable") is True:
            return True
        return status_code in {429, 502, 503, 504}

    def _build_error_result(
        self,
        status_code: int | None,
        payload: dict[str, Any] | None,
        raw_body: str,
    ) -> dict[str, Any]:
        state = payload.get("state", {}) if payload else {}
        if not isinstance(state, dict):
            state = {}
        return {
            "success": False,
            "error": self._format_error_message(status_code, payload, raw_body),
            "state": state,
        }

    @asynccontextmanager
    async def _request_slot(self):
        if self._request_semaphore is None:
            yield
            return
        async with self._request_semaphore:
            yield

    async def _run_request_with_retry(self, func, *args, **kwargs):
        """Run a CUA session/action request with transient-error retry and concurrency control."""
        async for attempt in self.request_retrying:  # type: ignore[union-attr]
            with attempt:
                async with self._request_slot():
                    return await func(*args, **kwargs)
        raise RuntimeError("CUA request retry loop exhausted")

    # ==================== Server Health Check (Local Mode) ====================

    async def _check_server_health(self) -> None:
        """Check if the CUA server is reachable by hitting its health endpoint."""
        health_url = f"{self.server_url}/health"
        timeout = aiohttp.ClientTimeout(total=5)

        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(health_url) as resp:
                    if resp.status != 200:
                        error_text = await resp.text()
                        raise RuntimeError(
                            f"CUA server health check failed with status {resp.status}: {error_text}"
                        )
        except aiohttp.ClientConnectorError:
            raise RuntimeError(
                f"\nCUA server is not reachable at {self.server_url}\n\n"
                "To start the CUA server:\n"
                "  cd assets/templates/browserbase/cua\n"
                "  npm install && npm run dev\n\n"
                "The server must be running before using CUA mode environments.\n"
            )
        except asyncio.TimeoutError:
            raise RuntimeError(
                f"\nCUA server at {self.server_url} did not respond within 5 seconds.\n\n"
                "Please check if the server is running and responsive:\n"
                "  cd assets/templates/browserbase/cua\n"
                "  npm install && npm run dev\n"
            )

    def verify_server_connection(self) -> None:
        """Synchronously verify that the CUA server is reachable."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop is not None:
            import concurrent.futures

            def _run_health_check() -> None:
                asyncio.run(self._check_server_health())

            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(_run_health_check)
                future.result()
        else:
            asyncio.run(self._check_server_health())

    # ==================== HTTP Client Methods (Local Mode) ====================

    async def _get_http_client(self) -> aiohttp.ClientSession:
        """Get or create the HTTP client session."""
        with self._thread_lock:
            async with self._client_lock:
                if self._http_client is None or self._http_client.closed:
                    self._http_client = aiohttp.ClientSession()
        return self._http_client

    async def _create_session_http(self) -> dict:
        """Create a new browser session via the CUA server (HTTP)."""
        client = await self._get_http_client()
        async with client.post(
            f"{self.server_url}/sessions",
            json=self.session_config,
            timeout=aiohttp.ClientTimeout(total=60),
        ) as resp:
            if resp.status != 200:
                error_text = await resp.text()
                payload = self._parse_json_payload(error_text)
                error_message = self._format_error_message(
                    resp.status, payload, error_text
                )
                if self._is_retryable_response(resp.status, payload):
                    raise CUATransientError(
                        f"Failed to create browser session: {error_message}",
                        status_code=resp.status,
                        code=payload.get("code") if payload else None,
                    )
                raise RuntimeError(f"Failed to create browser session: {error_message}")
            return await resp.json()

    async def _destroy_session_http(self, session_id: str) -> None:
        """Destroy a browser session via the CUA server (HTTP)."""
        client = await self._get_http_client()
        async with client.delete(f"{self.server_url}/sessions/{session_id}") as resp:
            if resp.status not in (200, 404):
                error_text = await resp.text()
                if self.logger:
                    self.logger.warning(
                        f"Failed to destroy session {session_id}: {error_text}"
                    )

    async def _execute_action_http(
        self, session_id: str, action: dict, tool_call_id: str | None = None
    ) -> dict:
        """Execute a browser action via HTTP and return the response with state."""
        payload = {**action}
        if tool_call_id:
            payload["tool_call_id"] = tool_call_id

        client = await self._get_http_client()
        async with client.post(
            f"{self.server_url}/sessions/{session_id}/action",
            json=payload,
            timeout=aiohttp.ClientTimeout(total=60),
        ) as resp:
            if resp.status != 200:
                error_text = await resp.text()
                error_payload = self._parse_json_payload(error_text)
                error_message = self._format_error_message(
                    resp.status, error_payload, error_text
                )
                if self._is_retryable_response(resp.status, error_payload):
                    raise CUATransientError(
                        f"Action failed for session {session_id}: {error_message}",
                        status_code=resp.status,
                        code=error_payload.get("code") if error_payload else None,
                    )
                if self.logger:
                    self.logger.warning(
                        f"Action failed for session {session_id}: {error_message}"
                    )
                return self._build_error_result(resp.status, error_payload, error_text)
            return await resp.json()

    # ==================== Sandbox Client Methods ====================

    async def _get_sandbox_client(self) -> AsyncSandboxClient:
        """Get or create the sandbox client."""
        if self._sandbox_client is None:
            self._sandbox_client = AsyncSandboxClient()
        return self._sandbox_client

    async def _create_sandbox(self) -> str:
        """Create a new sandbox and return its ID."""
        client = await self._get_sandbox_client()
        sandbox = await client.create(self._sandbox_request.model_copy())  # type: ignore[union-attr]
        self.active_sandboxes.add(sandbox.id)
        if self.logger:
            self.logger.debug(f"Created sandbox {sandbox.id}")
        return sandbox.id

    async def _create_sandbox_with_retry(self) -> str:
        """Create a sandbox with retry, cleaning up orphaned sandboxes from failed attempts."""
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

        # Use async subprocess to avoid blocking the event loop
        proc = await asyncio.create_subprocess_exec(
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
            cwd=self._template_path,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        _, stderr = await proc.communicate()
        if proc.returncode != 0:
            raise RuntimeError(f"Docker build failed: {stderr.decode()}")

        proc = await asyncio.create_subprocess_exec(
            "docker",
            "run",
            "--rm",
            "--platform",
            "linux/amd64",
            "-v",
            f"{self._template_path}/dist:/output",
            "cua-builder",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        _, stderr = await proc.communicate()
        if proc.returncode != 0:
            raise RuntimeError(f"Docker run failed: {stderr.decode()}")

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

        with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as tmp_file:
            tar_path = Path(tmp_file.name)

        try:
            with tarfile.open(tar_path, "w:gz") as tar:
                if self.use_binary:
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
                    exclude_dirs = {"node_modules", "dist", ".git"}
                    for file in self._template_path.glob("**/*"):
                        if file.is_file():
                            relative = file.relative_to(self._template_path)
                            if any(part in exclude_dirs for part in relative.parts):
                                continue
                            tar.add(file, arcname=f"cua-server/{relative}")

            remote_tar = "/tmp/cua-server.tar.gz"
            await client.upload_file(sandbox_id, remote_tar, str(tar_path))

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
        env_vars = f"CUA_SERVER_PORT={self.server_port}"

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
                stdout = await self._execute_sandbox_command(
                    sandbox_id,
                    f'curl -s -w "\\n%{{http_code}}" {health_url}',
                    timeout=10,
                )
                body, http_code = self._parse_curl_response(stdout)

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

    # ==================== Sandbox Curl Methods ====================

    @staticmethod
    def _parse_curl_response(stdout: str) -> tuple[str, str]:
        """Parse curl output with HTTP code appended via -w flag.

        Returns:
            tuple: (body, http_code) where http_code is "unknown" if parsing fails
        """
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
            f'curl -s -w "\\n%{{http_code}}" -X POST http://localhost:{self.server_port}/sessions '
            f"-H 'Content-Type: application/json' "
            f"-d '{escaped_payload}'",
            timeout=60,
        )

        body, http_code = self._parse_curl_response(stdout)
        status_code = self._coerce_http_status(http_code)
        if status_code != 200:
            payload_dict = self._parse_json_payload(body)
            error_message = self._format_error_message(status_code, payload_dict, body)
            if self._is_retryable_response(status_code, payload_dict):
                raise CUATransientError(
                    f"Failed to create browser session: {error_message}",
                    status_code=status_code,
                    code=payload_dict.get("code") if payload_dict else None,
                )
            raise RuntimeError(f"Failed to create browser session: {error_message}")

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
                f"curl -s -X DELETE http://localhost:{self.server_port}/sessions/{session_id}",
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
            f'curl -s -w "\\n%{{http_code}}" -X POST http://localhost:{self.server_port}/sessions/{session_id}/action '
            f"-H 'Content-Type: application/json' "
            f"-d '{escaped_payload}'",
            timeout=60,
        )

        body, http_code = self._parse_curl_response(stdout)
        status_code = self._coerce_http_status(http_code)
        if status_code != 200:
            payload_dict = self._parse_json_payload(body)
            error_message = self._format_error_message(status_code, payload_dict, body)
            if self._is_retryable_response(status_code, payload_dict):
                raise CUATransientError(
                    f"Action failed for session {session_id}: {error_message}",
                    status_code=status_code,
                    code=payload_dict.get("code") if payload_dict else None,
                )
            return self._build_error_result(status_code, payload_dict, body)

        try:
            return json.loads(body)
        except json.JSONDecodeError:
            return {
                "success": False,
                "error": f"Failed to parse response (HTTP {http_code}): {body[:200]}",
                "state": {},
            }

    # ==================== Unified Action Execution ====================

    async def _execute_action(
        self,
        session_id: str,
        action: dict,
        tool_call_id: str | None = None,
        sandbox_id: str | None = None,
    ) -> dict:
        """Execute a browser action using the appropriate method based on mode."""
        try:
            if self._execution_mode == "local":
                return await self._run_request_with_retry(
                    self._execute_action_http,
                    session_id,
                    action,
                    tool_call_id,
                )

            if sandbox_id is None:
                raise ValueError("sandbox_id is required for sandbox mode")
            return await self._run_request_with_retry(
                self._execute_action_curl,
                session_id,
                action,
                sandbox_id,
                tool_call_id,
            )
        except (CUATransientError, aiohttp.ClientError, asyncio.TimeoutError) as e:
            if self.logger:
                self.logger.warning(f"CUA action retries exhausted: {e}")
            return {"success": False, "error": str(e), "state": {}}

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
        """Create a browser session (and sandbox if in sandbox mode)."""
        if self._execution_mode == "local":
            # Local mode: create session via HTTP
            try:
                result = await self._run_request_with_retry(self._create_session_http)
                session_id = result.get("sessionId")
                if not session_id:
                    raise RuntimeError("Failed to get session ID from server response")

                with self._sessions_lock:
                    self.active_sessions.add(session_id)

                state["session_id"] = session_id
                state["browser_state"] = result.get("state", {})
            except vf.Error:
                raise
            except Exception as e:
                raise vf.BrowserSandboxError(e)
        else:
            # Sandbox mode: create sandbox, set up server, create session
            # Wrap in try-except to ensure errors trigger cleanup via stop_errors
            try:
                if self.use_prebuilt_image:
                    if self.logger:
                        self.logger.debug(
                            f"Using prebuilt image: {self.prebuilt_image}"
                        )

                    sandbox_id = await self._create_sandbox_with_retry()
                    state["cua_sandbox_id"] = sandbox_id
                    await self._wait_for_sandbox_ready(sandbox_id)
                    await self._wait_for_server(sandbox_id)
                else:
                    if self.use_binary:
                        await self._ensure_binary_exists()

                    sandbox_id = await self._create_sandbox_with_retry()
                    state["cua_sandbox_id"] = sandbox_id
                    await self._wait_for_sandbox_ready(sandbox_id)
                    await self._upload_server_files(sandbox_id)
                    await self._start_server(sandbox_id)
                    await self._wait_for_server(sandbox_id)

                result = await self._run_request_with_retry(
                    self._create_session_curl, sandbox_id
                )
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
            except vf.Error:
                # Re-raise vf.Error subclasses as-is (they're already handled)
                raise
            except Exception as e:
                # Wrap all other exceptions in BrowserSandboxError
                # This ensures cleanup_session is called via stop_errors mechanism
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
        """Inject session_id (and sandbox_id for sandbox mode) into all browser tool calls."""
        updated_args = dict(tool_args)
        updated_args["session_id"] = state["session_id"]
        if self._execution_mode == "sandbox":
            updated_args["sandbox_id"] = state["cua_sandbox_id"]
        return updated_args

    async def cleanup_session(self, state: vf.State) -> None:
        """Destroy the browser session (and sandbox if in sandbox mode)."""
        session_id = state.get("session_id")

        if self._execution_mode == "local":
            # Local mode: destroy session via HTTP
            if session_id:
                try:
                    async for attempt in self.retrying:  # type: ignore[union-attr]
                        with attempt:
                            await self._destroy_session_http(session_id)
                    with self._sessions_lock:
                        self.active_sessions.discard(session_id)
                except Exception as e:
                    if self.logger:
                        self.logger.warning(
                            f"Failed to destroy session {session_id}: {e}"
                        )
        else:
            # Sandbox mode: destroy session and sandbox
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
                        self.logger.warning(
                            f"Failed to destroy session {session_id}: {e}"
                        )

            if sandbox_id:
                try:
                    async for attempt in self.retrying:  # type: ignore[union-attr]
                        with attempt:
                            await self._delete_sandbox(sandbox_id)
                except Exception as e:
                    if self.logger:
                        self.logger.warning(
                            f"Failed to delete sandbox {sandbox_id}: {e}"
                        )

    async def teardown(self, max_concurrent: int = 50) -> None:
        """Clean up all resources on environment teardown."""
        if self._execution_mode == "local":
            # Local mode: destroy all sessions and close HTTP client
            with self._sessions_lock:
                sessions_snapshot = set(self.active_sessions)

            if sessions_snapshot:
                try:
                    loop = asyncio.get_running_loop()
                    if loop.is_closed():
                        return
                except RuntimeError:
                    return

                semaphore = asyncio.Semaphore(max_concurrent)

                async def _delete_with_semaphore(session_id: str):
                    async with semaphore:
                        try:
                            async for attempt in self.retrying:  # type: ignore[union-attr]
                                with attempt:
                                    await self._destroy_session_http(session_id)
                            with self._sessions_lock:
                                self.active_sessions.discard(session_id)
                        except Exception:
                            pass

                try:
                    await asyncio.gather(
                        *[_delete_with_semaphore(sid) for sid in sessions_snapshot]
                    )
                except RuntimeError:
                    pass

            try:
                if self._http_client and not self._http_client.closed:
                    await self._http_client.close()
            except RuntimeError:
                pass
        else:
            # Sandbox mode: delete all remaining sandboxes
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
                        if self.logger:
                            self.logger.debug(f"Deleted sandbox {sandbox_id}")
                    except Exception as e:
                        if self.logger:
                            self.logger.warning(
                                f"Failed to delete sandbox {sandbox_id}: {e}"
                            )

            try:
                await asyncio.gather(
                    *[_delete_sandbox_with_semaphore(sid) for sid in sandbox_ids]
                )
            except RuntimeError:
                pass

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
        response = await self._execute_action(
            session_id,
            {"type": "click", "x": x, "y": y, "button": button},
            tool_call_id,
            sandbox_id or None,
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
        response = await self._execute_action(
            session_id,
            {"type": "double_click", "x": x, "y": y},
            tool_call_id,
            sandbox_id or None,
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
        response = await self._execute_action(
            session_id,
            {"type": "type", "text": text},
            tool_call_id,
            sandbox_id or None,
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
        response = await self._execute_action(
            session_id,
            {"type": "keypress", "keys": keys},
            tool_call_id,
            sandbox_id or None,
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
        response = await self._execute_action(
            session_id,
            {
                "type": "scroll",
                "x": x,
                "y": y,
                "scroll_x": scroll_x,
                "scroll_y": scroll_y,
            },
            tool_call_id,
            sandbox_id or None,
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
            response = await self._execute_action(
                session_id,
                {"type": "goto", "url": url},
                tool_call_id,
                sandbox_id or None,
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
        response = await self._execute_action(
            session_id,
            {"type": "back"},
            tool_call_id,
            sandbox_id or None,
        )
        return self._format_response(response, session_id)

    async def forward(
        self,
        session_id: str = "",
        sandbox_id: str = "",
        tool_call_id: str = "",
    ) -> list[dict]:
        """Navigate forward in browser history."""
        response = await self._execute_action(
            session_id,
            {"type": "forward"},
            tool_call_id,
            sandbox_id or None,
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
            response = await self._execute_action(
                session_id,
                {"type": "wait", "timeMs": time_ms},
                tool_call_id,
                sandbox_id or None,
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
        response = await self._execute_action(
            session_id,
            {"type": "screenshot"},
            tool_call_id,
            sandbox_id or None,
        )
        return self._format_response(response, session_id)
