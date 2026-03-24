"""Full Browse mode for browser automation with an expanded toolset.

Extends LocalCUAMode with tools aligned to the Wide Browse subagent traces:
- computer: unified tool with action batching (replaces individual click/type/etc.)
- get_page_text: extract page text content
- read_page: accessibility tree with element refs
- find: element search by natural language query
- form_input: fill form fields by ref
- tabs_context: get current tab state
"""

import json
import logging
from typing import Literal, Optional

from pydantic import BaseModel

import tenacity as tc

from verifiers.envs.integrations.browser_env.modes.local_cua_mode import LocalCUAMode


class BrowserAction(BaseModel):
    """A single browser action for the computer tool."""

    action: str
    coordinate: Optional[list[int]] = None
    text: Optional[str] = None
    key: Optional[str] = None
    direction: Optional[Literal["up", "down"]] = None
    duration: Optional[int] = None


class FullBrowseMode(LocalCUAMode):
    """
    Full Browse mode — LocalCUAMode with an expanded toolset matching the
    Wide Browse subagent traces.

    Replaces the 9 individual CUA primitives with a unified ``computer`` tool
    that supports action batching, plus higher-level inspection tools
    (get_page_text, read_page, find, form_input, tabs_context).

    The underlying CUA server and sandbox infrastructure are identical to
    LocalCUAMode — only the agent-facing tool surface changes.
    """

    # ------------------------------------------------------------------ #
    #  Tool registration (overrides LocalCUAMode)                         #
    # ------------------------------------------------------------------ #

    def register_tools(self, env) -> None:
        """Register Full Browse tools instead of the basic CUA primitives."""
        self.logger = env.logger

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

        _skip = ["session_id", "sandbox_id", "tool_call_id"]

        env.add_tool(self.computer, args_to_skip=_skip)
        env.add_tool(self.get_page_text, args_to_skip=_skip)
        env.add_tool(self.read_page, args_to_skip=_skip)
        env.add_tool(self.find, args_to_skip=_skip)
        env.add_tool(self.form_input, args_to_skip=_skip)

    # ------------------------------------------------------------------ #
    #  Response formatting helpers                                        #
    # ------------------------------------------------------------------ #

    def _build_tab_context(self, response: dict) -> dict:
        """Build a tab_context dict from a CUA server response."""
        state = response.get("state", {})
        url = state.get("url", "")
        return {
            "current_tab_id": 1,
            "tab_count": 1,
            "available_tabs": [{"tab_id": 1, "title": "", "url": url}],
        }

    def _format_computer_response(
        self, response: dict, output_text: str, session_id: str = ""
    ) -> list[dict]:
        """Format a computer tool response: text output + screenshot."""
        state = response.get("state", {})
        screenshot_b64 = state.get("screenshot", "") or ""
        if screenshot_b64.startswith("data:"):
            screenshot_b64 = screenshot_b64.split(",", 1)[-1]

        tab_context = self._build_tab_context(response)

        result_obj = {"output": output_text, "tab_context": tab_context}
        content: list[dict] = [
            {"type": "text", "text": json.dumps(result_obj)},
        ]

        if screenshot_b64 and session_id:
            self._save_screenshot(session_id, screenshot_b64, state.get("url", ""))

        if screenshot_b64:
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{screenshot_b64}"},
                }
            )

        return content

    def _format_text_response(
        self,
        response: dict,
        session_id: str = "",
        include_screenshot: bool = False,
    ) -> list[dict]:
        """Format a text-output tool response (get_page_text, read_page, find, etc.)."""
        data = response.get("data", "")
        tab_context = self._build_tab_context(response)

        result_obj = {"output": data, "tab_context": tab_context}
        content: list[dict] = [
            {"type": "text", "text": json.dumps(result_obj)},
        ]

        if include_screenshot:
            state = response.get("state", {})
            screenshot_b64 = state.get("screenshot", "") or ""
            if screenshot_b64.startswith("data:"):
                screenshot_b64 = screenshot_b64.split(",", 1)[-1]
            if screenshot_b64 and session_id:
                self._save_screenshot(session_id, screenshot_b64, state.get("url", ""))
            if screenshot_b64:
                content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{screenshot_b64}"},
                    }
                )

        return content

    # ------------------------------------------------------------------ #
    #  Action-name mapping for the unified computer tool                  #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _translate_action(action: dict) -> dict:
        """Translate a Wide Browse action dict to a CUA server action dict."""
        name = action.get("action", "")

        if name in ("left_click", "click"):
            coord = action.get("coordinate", [0, 0])
            return {"type": "click", "x": coord[0], "y": coord[1], "button": "left"}

        if name == "right_click":
            coord = action.get("coordinate", [0, 0])
            return {"type": "click", "x": coord[0], "y": coord[1], "button": "right"}

        if name == "middle_click":
            coord = action.get("coordinate", [0, 0])
            return {"type": "click", "x": coord[0], "y": coord[1], "button": "middle"}

        if name == "double_click":
            coord = action.get("coordinate", [0, 0])
            return {"type": "double_click", "x": coord[0], "y": coord[1]}

        if name == "triple_click":
            coord = action.get("coordinate", [0, 0])
            return {"type": "tripleClick", "x": coord[0], "y": coord[1]}

        if name == "type":
            return {"type": "type", "text": action.get("text", "")}

        if name == "key":
            return {"type": "keypress", "keys": action.get("key", "")}

        if name == "scroll":
            coord = action.get("coordinate", [0, 0])
            direction = action.get("direction", "down")
            scroll_y = -3 if direction == "up" else 3
            return {
                "type": "scroll",
                "x": coord[0],
                "y": coord[1],
                "scroll_x": 0,
                "scroll_y": scroll_y,
            }

        if name == "scroll_to":
            coord = action.get("coordinate", [0, 0])
            # Scroll to bring coordinates into view via a click-less scroll
            return {
                "type": "scroll",
                "x": coord[0],
                "y": coord[1],
                "scroll_x": 0,
                "scroll_y": 0,
            }

        if name == "screenshot":
            return {"type": "screenshot"}

        if name == "wait":
            duration_s = action.get("duration", 1)
            return {"type": "wait", "timeMs": int(duration_s * 1000)}

        if name == "back":
            return {"type": "back"}

        if name == "forward":
            return {"type": "forward"}

        if name == "drag":
            path = action.get("path", [])
            return {"type": "drag", "path": path}

        # Fallback: pass through directly
        return {"type": name}

    @staticmethod
    def _describe_action(action: dict) -> str:
        """Return a short human-readable description of a single action."""
        name = action.get("action", "unknown")
        coord = action.get("coordinate")
        text = action.get("text")
        key = action.get("key")
        direction = action.get("direction")
        duration = action.get("duration")

        if name in ("left_click", "click", "right_click", "middle_click"):
            btn = name.replace("_click", "").replace("click", "left")
            return (
                f"Clicked ({coord[0]},{coord[1]})"
                if btn == "left"
                else f"{btn}-clicked ({coord[0]},{coord[1]})"
            )
        if name == "double_click":
            return f"Double-clicked ({coord[0]},{coord[1]})"
        if name == "triple_click":
            return f"Triple-clicked ({coord[0]},{coord[1]})"
        if name == "type":
            preview = (text[:30] + "...") if text and len(text) > 30 else text
            return f'Typed "{preview}"'
        if name == "key":
            return f"Pressed {key}"
        if name == "scroll":
            return f"Scrolled {direction} at ({coord[0]},{coord[1]})"
        if name == "wait":
            return f"Waited {duration}s"
        if name == "screenshot":
            return "Took screenshot"
        return name

    # ------------------------------------------------------------------ #
    #  Tool methods                                                       #
    # ------------------------------------------------------------------ #

    async def computer(
        self,
        actions: list[BrowserAction],
        user_description: str = "",
        session_id: str = "",
        sandbox_id: str = "",
        tool_call_id: str = "",
    ) -> list[dict]:
        """Execute one or more browser actions and return a screenshot.

        Each action in the list is an object with an 'action' field and optional parameters.

        Supported actions: left_click, right_click, double_click, triple_click,
        type, key, scroll, scroll_to, screenshot, wait, back, forward, drag.

        Example:
            actions=[{"action": "left_click", "coordinate": [500, 300]}, {"action": "wait", "duration": 1}]
        """
        if not actions:
            return [
                {
                    "type": "text",
                    "text": '{"output": "No actions provided", "tab_context": {}}',
                }
            ]

        # Convert BrowserAction models or dicts or JSON string to list of dicts
        if isinstance(actions, str):
            try:
                actions_list: list[dict] = json.loads(actions)
            except json.JSONDecodeError:
                return [
                    {
                        "type": "text",
                        "text": '{"output": "Invalid actions JSON", "tab_context": {}}',
                    }
                ]
        elif isinstance(actions, list):
            actions_list = []
            for a in actions:
                if isinstance(a, dict):
                    actions_list.append(a)
                elif isinstance(a, BaseModel):
                    actions_list.append(a.model_dump(exclude_none=True))
                else:
                    actions_list.append(dict(a))
        else:
            actions_list = [actions] if isinstance(actions, dict) else []

        descriptions: list[str] = []
        last_response: dict = {}

        for action in actions_list:
            server_action = self._translate_action(action)
            last_response = await self._execute_action_curl(
                session_id, server_action, sandbox_id, tool_call_id
            )
            descriptions.append(self._describe_action(action))

        output_text = "; ".join(descriptions)
        return self._format_computer_response(last_response, output_text, session_id)

    async def get_page_text(
        self,
        user_description: str = "",
        session_id: str = "",
        sandbox_id: str = "",
        tool_call_id: str = "",
    ) -> list[dict]:
        """Extract the full text content of the current page.

        Returns the page text as a string — useful for reading content without
        relying on screenshots.

        Args:
            user_description: What you are trying to extract.
        """
        response = await self._execute_action_curl(
            session_id,
            {"type": "get_page_text"},
            sandbox_id,
            tool_call_id,
        )
        return self._format_text_response(
            response, session_id, include_screenshot=False
        )

    async def read_page(
        self,
        user_description: str = "",
        filter: Literal["interactive", "all"] = "all",
        session_id: str = "",
        sandbox_id: str = "",
        tool_call_id: str = "",
    ) -> list[dict]:
        """Read the page's element tree with refs and coordinates.

        Returns a text representation of the page's elements, each with a
        reference ID (e.g. ref_42) that can be used with form_input.

        Args:
            user_description: What you are looking for on the page.
            filter: "interactive" to show only interactive elements (buttons,
                links, inputs), or "all" for the full element tree.
        """
        action: dict = {"type": "read_page", "filter": filter}
        response = await self._execute_action_curl(
            session_id,
            action,
            sandbox_id,
            tool_call_id,
        )
        return self._format_text_response(
            response, session_id, include_screenshot=False
        )

    async def find(
        self,
        query: str,
        user_description: str = "",
        session_id: str = "",
        sandbox_id: str = "",
        tool_call_id: str = "",
    ) -> list[dict]:
        """Search for elements on the page matching a query.

        Returns matching elements with refs and coordinates.

        Args:
            query: What to search for (e.g. "email input", "submit button").
            user_description: Why you are searching.
        """
        response = await self._execute_action_curl(
            session_id,
            {"type": "find", "query": query},
            sandbox_id,
            tool_call_id,
        )
        return self._format_text_response(
            response, session_id, include_screenshot=False
        )

    async def form_input(
        self,
        ref: str,
        value: str,
        user_description: str = "",
        session_id: str = "",
        sandbox_id: str = "",
        tool_call_id: str = "",
    ) -> list[dict]:
        """Set a form field's value using its element ref from read_page or find.

        Works with input, textarea, select, checkbox, radio, and contenteditable
        elements.

        Args:
            ref: Element reference from read_page/find (e.g. "ref_42").
            value: Value to set. For checkboxes/radios use "true"/"false".
            user_description: What you are filling in.
        """
        response = await self._execute_action_curl(
            session_id,
            {"type": "form_input", "ref": ref, "value": value},
            sandbox_id,
            tool_call_id,
        )
        return self._format_text_response(response, session_id, include_screenshot=True)

    async def tabs_context(
        self,
        session_id: str = "",
        sandbox_id: str = "",
        tool_call_id: str = "",
    ) -> list[dict]:
        """Get the current browser tab state."""
        # In local mode we always have exactly one tab; fetch state to get URL.
        response = await self._execute_action_curl(
            session_id,
            {"type": "screenshot"},
            sandbox_id,
            tool_call_id,
        )
        state = response.get("state", {})
        url = state.get("url", "")
        tab_info = f"Current tab: 1\nTotal tabs: 1\n\n  Tab 1 (active): {url}"
        return [{"type": "text", "text": json.dumps({"output": tab_info})}]
