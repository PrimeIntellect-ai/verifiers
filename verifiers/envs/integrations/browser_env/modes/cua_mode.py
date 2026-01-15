"""CUA-based browser mode using HTTP server."""

import asyncio
import base64
import copy
import logging
import os
import threading
from datetime import datetime
from typing import Any, Literal

import aiohttp
import tenacity as tc
import verifiers as vf


class CUAMode:
    """
    CUA-based browser mode using HTTP server.
    Provides vision-based primitives: click, double_click, type_text, keypress,
    scroll, goto, back, forward, wait, screenshot
    """

    def __init__(
        self,
        server_url: str = "http://localhost:3000",
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
        screenshot_dir: str | None = None,
        save_screenshots: bool = True,
        keep_recent_screenshots: int | None = 2,
        proxies: bool = False,
    ):
        self.keep_recent_screenshots = keep_recent_screenshots
        self.logger = None  # Will be set when register_tools is called

        resolved_api_key = browserbase_api_key or os.getenv("BROWSERBASE_API_KEY")
        resolved_project_id = browserbase_project_id or os.getenv(
            "BROWSERBASE_PROJECT_ID"
        )

        self.server_url = server_url.rstrip("/")
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
        self._http_client: aiohttp.ClientSession | None = None
        self._client_lock = asyncio.Lock()

        self.save_screenshots = save_screenshots
        self.screenshot_dir = screenshot_dir or os.path.join(os.getcwd(), "screenshots")
        self._screenshot_counters: dict[str, int] = {}

        # Retry configuration - will be set up after logger is available
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.backoff_factor = backoff_factor
        self.max_backoff_seconds = max_backoff_seconds
        self.jitter = jitter
        self.with_retry = None  # Will be set in register_tools

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

        _skip = ["session_id", "tool_call_id"]
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

    # ==================== HTTP Client Methods ====================

    async def _get_client(self) -> aiohttp.ClientSession:
        """Get or create the HTTP client session."""
        with self._thread_lock:
            async with self._client_lock:
                if self._http_client is None or self._http_client.closed:
                    self._http_client = aiohttp.ClientSession()
        return self._http_client

    async def _create_session(self) -> dict:
        """Create a new browser session via the CUA server."""
        client = await self._get_client()
        async with client.post(
            f"{self.server_url}/sessions",
            json=self.session_config,
        ) as resp:
            if resp.status != 200:
                error_text = await resp.text()
                raise RuntimeError(f"Failed to create browser session: {error_text}")
            return await resp.json()

    async def _destroy_session(self, session_id: str) -> None:
        """Destroy a browser session via the CUA server."""
        client = await self._get_client()
        async with client.delete(f"{self.server_url}/sessions/{session_id}") as resp:
            if resp.status not in (200, 404):
                error_text = await resp.text()
                if self.logger:
                    self.logger.warning(
                        f"Failed to destroy session {session_id}: {error_text}"
                    )

    async def _execute_action(
        self, session_id: str, action: dict, tool_call_id: str | None = None
    ) -> dict:
        """Execute a browser action and return the response with state."""
        payload = {**action}
        if tool_call_id:
            payload["tool_call_id"] = tool_call_id

        client = await self._get_client()
        async with client.post(
            f"{self.server_url}/sessions/{session_id}/action",
            json=payload,
        ) as resp:
            if resp.status != 200:
                error_text = await resp.text()
                if self.logger:
                    self.logger.warning(
                        f"Action failed for session {session_id}: {error_text}"
                    )
                return {"success": False, "error": error_text, "state": {}}
            return await resp.json()

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
            screenshot_positions[: -self.keep_recent_screenshots]
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
        """Create a browser session for this rollout."""
        result = await self.with_retry(self._create_session)()
        session_id = result.get("sessionId")
        if not session_id:
            raise RuntimeError("Failed to get session ID from server response")

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
        """Inject session_id into all browser tool calls."""
        updated_args = dict(tool_args)
        updated_args["session_id"] = state["session_id"]
        return updated_args

    async def cleanup_session(self, state: vf.State) -> None:
        """Destroy the browser session after rollout completion."""
        session_id = state.get("session_id")
        if session_id is None:
            return

        try:
            await self.with_retry(self._destroy_session)(session_id)
            with self._sessions_lock:
                self.active_sessions.discard(session_id)
        except Exception as e:
            if self.logger:
                self.logger.warning(f"Failed to destroy session {session_id}: {e}")

    async def teardown(self, max_concurrent: int = 50) -> None:
        """Destroy all active browser sessions on exit."""
        with self._sessions_lock:
            sessions_snapshot = set(self.active_sessions)

        if not sessions_snapshot:
            return

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
                    await self.with_retry(self._destroy_session)(session_id)
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

    # ==================== Browser Tool Methods ====================

    async def click(
        self,
        x: int,
        y: int,
        button: Literal["left", "right", "middle"] = "left",
        session_id: str = "",
        tool_call_id: str = "",
    ) -> list[dict]:
        """Click at coordinates (x, y) on the page."""
        response = await self._execute_action(
            session_id,
            {"type": "click", "x": x, "y": y, "button": button},
            tool_call_id,
        )
        return self._format_response(response, session_id)

    async def double_click(
        self, x: int, y: int, session_id: str = "", tool_call_id: str = ""
    ) -> list[dict]:
        """Double-click at coordinates (x, y) on the page."""
        response = await self._execute_action(
            session_id, {"type": "double_click", "x": x, "y": y}, tool_call_id
        )
        return self._format_response(response, session_id)

    async def type_text(
        self, text: str, session_id: str = "", tool_call_id: str = ""
    ) -> list[dict]:
        """Type text into the currently focused element."""
        response = await self._execute_action(
            session_id, {"type": "type", "text": text}, tool_call_id
        )
        return self._format_response(response, session_id)

    async def keypress(
        self, keys: str | list[str], session_id: str = "", tool_call_id: str = ""
    ) -> list[dict]:
        """Press keyboard key(s)."""
        response = await self._execute_action(
            session_id, {"type": "keypress", "keys": keys}, tool_call_id
        )
        return self._format_response(response, session_id)

    async def scroll(
        self,
        x: int = 0,
        y: int = 0,
        scroll_x: int = 0,
        scroll_y: int = 0,
        session_id: str = "",
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
        )
        return self._format_response(response, session_id)

    async def goto(
        self, url: str, session_id: str = "", tool_call_id: str = ""
    ) -> list[dict]:
        """Navigate to a URL."""
        try:
            response = await self._execute_action(
                session_id, {"type": "goto", "url": url}, tool_call_id
            )
        except (TimeoutError, asyncio.TimeoutError):
            response = {
                "success": False,
                "error": f"Navigation timeout: The page at {url} took too long to load",
                "state": {"url": url, "viewport": {}},
            }
        return self._format_response(response, session_id)

    async def back(self, session_id: str = "", tool_call_id: str = "") -> list[dict]:
        """Navigate back in browser history."""
        response = await self._execute_action(
            session_id, {"type": "back"}, tool_call_id
        )
        return self._format_response(response, session_id)

    async def forward(self, session_id: str = "", tool_call_id: str = "") -> list[dict]:
        """Navigate forward in browser history."""
        response = await self._execute_action(
            session_id, {"type": "forward"}, tool_call_id
        )
        return self._format_response(response, session_id)

    async def wait(
        self, time_ms: int = 1000, session_id: str = "", tool_call_id: str = ""
    ) -> list[dict]:
        """Wait for a specified amount of time."""
        try:
            response = await self._execute_action(
                session_id, {"type": "wait", "timeMs": time_ms}, tool_call_id
            )
        except (TimeoutError, asyncio.TimeoutError):
            response = {
                "success": False,
                "error": f"Wait timeout: The wait operation ({time_ms}ms) timed out",
                "state": {"viewport": {}},
            }
        return self._format_response(response, session_id)

    async def screenshot(
        self, session_id: str = "", tool_call_id: str = ""
    ) -> list[dict]:
        """Capture a screenshot of the current page state."""
        response = await self._execute_action(
            session_id, {"type": "screenshot"}, tool_call_id
        )
        return self._format_response(response, session_id)
