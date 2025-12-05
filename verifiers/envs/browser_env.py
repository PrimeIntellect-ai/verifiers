import asyncio
import logging
from typing import Any, Literal

import tenacity as tc

import verifiers as vf

try:
    import aiohttp
except ImportError:
    raise ImportError(
        "aiohttp is not installed. Please install it with `uv pip install aiohttp`."
    )


class BrowserEnv(vf.StatefulToolEnv):
    """
    Browser environment that connects to a CUA (Computer Use Agent) server
    for vision-based browser control via tool calls.

    Each rollout gets its own browser session. Tools return both text summaries
    and base64 screenshots as image_url content.
    """

    def __init__(
        self,
        server_url: str = "http://localhost:3000",
        env: Literal["LOCAL", "BROWSERBASE"] = "LOCAL",
        browserbase_api_key: str | None = None,
        browserbase_project_id: str | None = None,
        viewport_width: int = 1280,
        viewport_height: int = 720,
        max_retries: int = 5,
        base_delay: float = 0.5,
        backoff_factor: float = 2.0,
        max_backoff_seconds: float = 30.0,
        jitter: float = 1e-3,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.server_url = server_url.rstrip("/")
        self.session_config = {
            "env": env,
            "browserbaseApiKey": browserbase_api_key,
            "browserbaseProjectId": browserbase_project_id,
            "viewport": {"width": viewport_width, "height": viewport_height},
        }
        # Remove None values from config
        self.session_config = {
            k: v for k, v in self.session_config.items() if v is not None
        }

        self.active_sessions: set[str] = set()
        self._http_client: aiohttp.ClientSession | None = None

        # Retry configuration
        self.with_retry = tc.AsyncRetrying(
            stop=tc.stop_after_attempt(max_retries),
            wait=tc.wait_exponential_jitter(
                initial=base_delay,
                exp_base=backoff_factor,
                max=max_backoff_seconds,
                jitter=jitter,
            ),
            before_sleep=tc.before_sleep_log(self.logger, logging.ERROR),
            reraise=True,
        ).wraps

        # Register browser primitive tools
        self.add_tool(self.click, args_to_skip=["session_id"])
        self.add_tool(self.double_click, args_to_skip=["session_id"])
        self.add_tool(self.type_text, args_to_skip=["session_id"])
        self.add_tool(self.keypress, args_to_skip=["session_id"])
        self.add_tool(self.scroll, args_to_skip=["session_id"])
        self.add_tool(self.goto, args_to_skip=["session_id"])
        self.add_tool(self.back, args_to_skip=["session_id"])
        self.add_tool(self.forward, args_to_skip=["session_id"])
        self.add_tool(self.wait, args_to_skip=["session_id"])
        self.add_tool(self.screenshot, args_to_skip=["session_id"])

    async def _get_client(self) -> aiohttp.ClientSession:
        """Get or create the HTTP client session."""
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
                self.logger.warning(
                    f"Failed to destroy session {session_id}: {error_text}"
                )

    async def _execute_action(self, session_id: str, action: dict) -> dict:
        """Execute a browser action and return the response with state."""
        client = await self._get_client()
        async with client.post(
            f"{self.server_url}/sessions/{session_id}/action",
            json=action,
        ) as resp:
            return await resp.json()

    async def _get_state(self, session_id: str) -> dict:
        """Get the current browser state (screenshot, URL, viewport)."""
        client = await self._get_client()
        async with client.get(
            f"{self.server_url}/sessions/{session_id}/state"
        ) as resp:
            return await resp.json()

    def _format_response(self, response: dict) -> list[dict]:
        """
        Format action response as multipart content with text and image.

        Returns OpenAI-compatible content array with text summary and screenshot.
        """
        success = response.get("success", False)
        error = response.get("error")
        state = response.get("state", {})
        screenshot_b64 = state.get("screenshot", "")
        url = state.get("url", "")
        viewport = state.get("viewport", {})

        # Build text summary
        status = "Success" if success else "Failed"
        text_parts = [f"Status: {status}"]
        if error:
            text_parts.append(f"Error: {error}")
        if url:
            text_parts.append(f"URL: {url}")
        if viewport:
            text_parts.append(f"Viewport: {viewport.get('width', 0)}x{viewport.get('height', 0)}")

        content = [{"type": "text", "text": "\n".join(text_parts)}]

        # Add screenshot if available
        if screenshot_b64:
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{screenshot_b64}"},
                }
            )

        return content

    # ==================== Lifecycle Methods ====================

    async def setup_state(self, state: vf.State, **kwargs) -> vf.State:
        """Create a browser session for this rollout."""
        result = await self.with_retry(self._create_session)()
        session_id = result.get("sessionId")
        if not session_id:
            raise RuntimeError("Failed to get session ID from server response")

        self.active_sessions.add(session_id)
        self.logger.debug(f"Created browser session {session_id}")

        state["session_id"] = session_id
        # Store initial state (screenshot, URL, viewport)
        state["browser_state"] = result.get("state", {})

        return await super().setup_state(state, **kwargs)

    @vf.cleanup
    async def destroy_session(self, state: vf.State):
        """Destroy the browser session after rollout completion."""
        session_id = state.get("session_id")
        if session_id is None:
            return

        try:
            await self.with_retry(self._destroy_session)(session_id)
            self.active_sessions.discard(session_id)
            self.logger.debug(f"Destroyed browser session {session_id}")
        except Exception as e:
            self.logger.warning(f"Failed to destroy session {session_id}: {e}")

    @vf.teardown
    async def teardown_sessions(self, max_concurrent: int = 50):
        """Destroy all active browser sessions on exit."""
        if len(self.active_sessions) == 0:
            return

        self.logger.info(f"Destroying {len(self.active_sessions)} remaining sessions")

        semaphore = asyncio.Semaphore(max_concurrent)

        async def _delete_with_semaphore(session_id: str):
            async with semaphore:
                try:
                    await self.with_retry(self._destroy_session)(session_id)
                    self.active_sessions.discard(session_id)
                    self.logger.debug(f"Destroyed session {session_id}")
                except Exception as e:
                    self.logger.warning(f"Failed to destroy session {session_id}: {e}")

        await asyncio.gather(
            *[
                _delete_with_semaphore(session_id)
                for session_id in list(self.active_sessions)
            ]
        )

        # Close HTTP client
        if self._http_client and not self._http_client.closed:
            await self._http_client.close()

    def update_tool_args(
        self,
        tool_name: str,
        tool_args: dict[str, Any],
        messages: vf.Messages,
        state: vf.State,
        **kwargs,
    ) -> dict[str, Any]:
        """Inject session_id into all browser tool calls."""
        updated_args = dict(tool_args)
        updated_args["session_id"] = state["session_id"]
        return updated_args

    # ==================== Browser Primitive Tools ====================

    async def click(
        self,
        x: int,
        y: int,
        button: Literal["left", "right", "middle"] = "left",
        session_id: str = "",
    ) -> list[dict]:
        """
        Click at coordinates (x, y) on the page.

        Args:
            x: The x coordinate to click
            y: The y coordinate to click
            button: Mouse button to use (left, right, or middle)

        Returns:
            Status message and screenshot of the page after clicking
        """
        response = await self._execute_action(
            session_id, {"type": "click", "x": x, "y": y, "button": button}
        )
        return self._format_response(response)

    async def double_click(
        self, x: int, y: int, session_id: str = ""
    ) -> list[dict]:
        """
        Double-click at coordinates (x, y) on the page.

        Args:
            x: The x coordinate to double-click
            y: The y coordinate to double-click

        Returns:
            Status message and screenshot of the page after double-clicking
        """
        response = await self._execute_action(
            session_id, {"type": "double_click", "x": x, "y": y}
        )
        return self._format_response(response)

    async def type_text(self, text: str, session_id: str = "") -> list[dict]:
        """
        Type text into the currently focused element.

        Args:
            text: The text to type

        Returns:
            Status message and screenshot of the page after typing
        """
        response = await self._execute_action(
            session_id, {"type": "type", "text": text}
        )
        return self._format_response(response)

    async def keypress(
        self, keys: str | list[str], session_id: str = ""
    ) -> list[dict]:
        """
        Press keyboard key(s).

        Args:
            keys: Key name(s) to press (e.g., "Enter", "Tab", ["Control", "a"])

        Returns:
            Status message and screenshot of the page after key press
        """
        response = await self._execute_action(
            session_id, {"type": "keypress", "keys": keys}
        )
        return self._format_response(response)

    async def scroll(
        self,
        x: int = 0,
        y: int = 0,
        scroll_x: int = 0,
        scroll_y: int = 0,
        session_id: str = "",
    ) -> list[dict]:
        """
        Scroll the page at a specific position.

        Args:
            x: X coordinate to scroll at (default: 0)
            y: Y coordinate to scroll at (default: 0)
            scroll_x: Horizontal scroll amount in pixels (positive = right)
            scroll_y: Vertical scroll amount in pixels (positive = down)

        Returns:
            Status message and screenshot of the page after scrolling
        """
        response = await self._execute_action(
            session_id,
            {"type": "scroll", "x": x, "y": y, "scroll_x": scroll_x, "scroll_y": scroll_y},
        )
        return self._format_response(response)

    async def goto(self, url: str, session_id: str = "") -> list[dict]:
        """
        Navigate to a URL.

        Args:
            url: The URL to navigate to

        Returns:
            Status message and screenshot of the page after navigation
        """
        response = await self._execute_action(
            session_id, {"type": "goto", "url": url}
        )
        return self._format_response(response)

    async def back(self, session_id: str = "") -> list[dict]:
        """
        Navigate back in browser history.

        Returns:
            Status message and screenshot of the page after going back
        """
        response = await self._execute_action(session_id, {"type": "back"})
        return self._format_response(response)

    async def forward(self, session_id: str = "") -> list[dict]:
        """
        Navigate forward in browser history.

        Returns:
            Status message and screenshot of the page after going forward
        """
        response = await self._execute_action(session_id, {"type": "forward"})
        return self._format_response(response)

    async def wait(self, time_ms: int = 1000, session_id: str = "") -> list[dict]:
        """
        Wait for a specified amount of time.

        Args:
            time_ms: Time to wait in milliseconds (default: 1000)

        Returns:
            Status message and screenshot of the page after waiting
        """
        response = await self._execute_action(
            session_id, {"type": "wait", "timeMs": time_ms}
        )
        return self._format_response(response)

    async def screenshot(self, session_id: str = "") -> list[dict]:
        """
        Capture a screenshot of the current page state.

        Returns:
            Status message and screenshot of the current page
        """
        response = await self._execute_action(session_id, {"type": "screenshot"})
        return self._format_response(response)

