"""Unified Browser Environment with DOM and CUA modes."""

import os
from typing import Any, Literal
import verifiers as vf
from verifiers.utils.async_utils import maybe_await

from .modes.dom_mode import DOMMode
from .modes.cua_mode import CUAMode

ModeType = Literal["dom", "cua"]

# Default system prompts for each mode
DOM_DEFAULT_PROMPT = """You are a browser automation agent using Stagehand's AI-driven tools.

Available tools:
- navigate(url): Navigate to a URL
- observe(instruction): Find possible actions matching the instruction
- act(instruction): Execute an action described in natural language
- extract(instruction, schema_json): Extract structured data from the page

Use natural language to describe what you want to do. Stagehand will intelligently
find elements and execute actions without needing CSS selectors or coordinates.

Complete the given task efficiently."""

CUA_DEFAULT_PROMPT = """You are a browser automation agent. You can control a web browser using the provided tools.

Available tools:
- click(x, y, button): Click at coordinates
- double_click(x, y): Double-click at coordinates
- type_text(text): Type text into focused element
- keypress(keys): Press keyboard keys
- scroll(x, y, scroll_x, scroll_y): Scroll at position
- goto(url): Navigate to URL
- back(): Go back in history
- forward(): Go forward in history
- wait(time_ms): Wait for specified milliseconds
- screenshot(): Capture current page state

After each action, you will receive a screenshot showing the current page state.
Analyze the screenshot to determine your next action.

Complete the given task efficiently using the minimum number of actions necessary."""


class BrowserEnv(vf.StatefulToolEnv):
    """
    Unified browser environment supporting both DOM-based and CUA-based modes.

    Modes:
        - "dom": Natural language operations via Stagehand SDK (act, observe, extract)
        - "cua": Vision-based primitives via CUA server (click, scroll, type_text)
    """

    def __init__(
        self,
        mode: ModeType = "dom",
        # Shared config
        browserbase_api_key: str | None = None,
        browserbase_project_id: str | None = None,
        # DOM mode specific
        model_api_key: str | None = None,
        stagehand_model: str = "openai/gpt-4o-mini",
        proxy_model_to_stagehand: bool = False,
        # CUA mode specific
        server_url: str = "http://localhost:3000",
        env: Literal["LOCAL", "BROWSERBASE"] = "LOCAL",
        viewport_width: int = 1024,
        viewport_height: int = 768,
        save_screenshots: bool = True,
        keep_recent_screenshots: int | None = 2,
        proxies: bool = False,
        # Common
        **kwargs: Any,
    ):
        # Use default system prompt for mode if not provided
        if "system_prompt" not in kwargs or kwargs.get("system_prompt") is None:
            kwargs["system_prompt"] = (
                DOM_DEFAULT_PROMPT if mode == "dom" else CUA_DEFAULT_PROMPT
            )

        super().__init__(**kwargs)
        self.mode = mode

        # Validate required environment variables before proceeding
        self._validate_environment_variables(
            mode=mode,
            env=env,
            browserbase_api_key=browserbase_api_key,
            browserbase_project_id=browserbase_project_id,
            model_api_key=model_api_key,
        )

        # Initialize the appropriate mode strategy
        if mode == "dom":
            self._mode_impl = DOMMode(
                browserbase_api_key=browserbase_api_key,
                project_id=browserbase_project_id,
                model_api_key=model_api_key,
                stagehand_model=stagehand_model,
                proxy_model_to_stagehand=proxy_model_to_stagehand,
            )
        elif mode == "cua":
            self._mode_impl = CUAMode(
                server_url=server_url,
                env=env,
                browserbase_api_key=browserbase_api_key,
                browserbase_project_id=browserbase_project_id,
                viewport_width=viewport_width,
                viewport_height=viewport_height,
                save_screenshots=save_screenshots,
                keep_recent_screenshots=keep_recent_screenshots,
                proxies=proxies,
            )
        else:
            raise ValueError(f"Unknown mode: {mode}. Must be 'dom' or 'cua'")

        # Register mode-specific tools
        self._mode_impl.register_tools(self)

    def _validate_environment_variables(
        self,
        mode: str,
        env: str,
        browserbase_api_key: str | None,
        browserbase_project_id: str | None,
        model_api_key: str | None,
    ) -> None:
        """
        Validate that required environment variables are set before initialization.

        Raises:
            ValueError: If required environment variables are missing.
        """
        missing_vars = []

        # Check Browserbase credentials for DOM mode or CUA+BROWSERBASE
        if mode == "dom" or (mode == "cua" and env == "BROWSERBASE"):
            resolved_api_key = browserbase_api_key or os.getenv("BROWSERBASE_API_KEY")
            resolved_project_id = browserbase_project_id or os.getenv(
                "BROWSERBASE_PROJECT_ID"
            )

            if not resolved_api_key:
                missing_vars.append("BROWSERBASE_API_KEY")
            if not resolved_project_id:
                missing_vars.append("BROWSERBASE_PROJECT_ID")

        # Check MODEL_API_KEY for DOM mode (used by Stagehand)
        if mode == "dom":
            resolved_model_key = model_api_key or os.getenv("MODEL_API_KEY")
            if not resolved_model_key:
                missing_vars.append("MODEL_API_KEY")

        if missing_vars:
            mode_desc = f"mode='{mode}'"
            if mode == "cua":
                mode_desc += f", env='{env}'"

            raise ValueError(
                f"Missing required environment variables for BrowserEnv ({mode_desc}):\n"
                f"  {', '.join(missing_vars)}\n\n"
                f"Please set these variables in your environment or .env file."
            )

    async def setup_state(self, state: vf.State, **kwargs: Any) -> vf.State:
        """Delegate session creation to the mode strategy."""
        state = await self._mode_impl.setup_state(state, **kwargs)
        return await super().setup_state(state, **kwargs)

    def update_tool_args(
        self,
        tool_name: str,
        tool_args: dict[str, Any],
        messages: vf.Messages,
        state: vf.State,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Delegate tool arg injection to the mode strategy."""
        return self._mode_impl.update_tool_args(
            tool_name, tool_args, messages, state, **kwargs
        )

    @vf.cleanup
    async def cleanup_session(self, state: vf.State) -> None:
        """Clean up session after rollout."""
        await self._mode_impl.cleanup_session(state)

    @vf.teardown
    async def teardown(self) -> None:
        """Clean up resources on environment teardown."""
        if hasattr(self, "_mode_impl") and self._mode_impl is not None:
            await self._mode_impl.teardown()

    # ==================== CUA Mode Specific Overrides ====================

    async def call_tool(
        self, tool_name: str, tool_args: dict, tool_call_id: str, **kwargs
    ) -> vf.Message | list[vf.Message]:
        """
        Call a tool, preserving multipart content for CUA mode images.

        In CUA mode, tools return list[dict] with text and image_url content.
        This override ensures the multipart structure is preserved.
        """
        if self.mode == "dom":
            # DOM mode uses default string handling
            return await super().call_tool(tool_name, tool_args, tool_call_id, **kwargs)

        # CUA mode: preserve multipart content
        try:
            tool_func = self.tool_map[tool_name]
            tool_args_with_id = {**tool_args, "tool_call_id": tool_call_id}
            result = await maybe_await(tool_func, **tool_args_with_id)

            if isinstance(result, list):
                # Extract text parts for tool response
                text_parts = [
                    str(item.get("text", ""))
                    for item in result
                    if isinstance(item, dict) and item.get("type") == "text"
                ]
                text_content = "\n".join([t for t in text_parts if t]) or "[no text]"
                tool_messages: list[vf.Message] = [
                    {
                        "role": "tool",
                        "content": text_content,
                        "tool_call_id": tool_call_id,
                    },
                    {
                        "role": "user",
                        "content": result,
                    },
                ]
            else:
                tool_messages = [
                    {
                        "role": "tool",
                        "content": str(result),
                        "tool_call_id": tool_call_id,
                    }
                ]

            return tool_messages

        except Exception as e:
            return {
                "role": "tool",
                "content": self.error_formatter(e),
                "tool_call_id": tool_call_id,
            }

    async def env_response(
        self, messages: vf.Messages, state: vf.State, **kwargs
    ) -> vf.Messages:
        """
        Handle environment response, filtering screenshots in CUA mode.
        """
        if self.mode == "cua":
            # Filter screenshots to manage context size
            messages = self._mode_impl.filter_screenshots_in_messages(list(messages))

        tool_messages = await super().env_response(messages, state, **kwargs)
        if not isinstance(tool_messages, list):
            return tool_messages

        # Flatten nested message lists
        flattened: list[vf.Message] = []
        for msg in tool_messages:
            if isinstance(msg, list):
                flattened.extend(msg)
            else:
                flattened.append(msg)
        return flattened
