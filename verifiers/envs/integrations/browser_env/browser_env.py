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

    CUA Mode Execution Options (from fastest to most flexible):
        1. Pre-built Docker image (default): Uses deepdream19/cua-server:latest
           No binary upload or dependency installation needed. Fastest startup.
        2. Binary upload (use_prebuilt_image=False): Builds/uploads SEA binary to sandbox.
           Useful if you need a custom server version.
        3. Local server (use_sandbox=False): Connect to manually started CUA server.
           Useful for local development and debugging.

    Example:
        >>> # CUA mode with pre-built image (default, recommended)
        >>> env = BrowserEnv(mode="cua", dataset=dataset, rubric=rubric)

        >>> # CUA mode with binary upload (custom server)
        >>> env = BrowserEnv(mode="cua", use_prebuilt_image=False, dataset=dataset, rubric=rubric)

        >>> # CUA mode with local server (for development)
        >>> env = BrowserEnv(mode="cua", use_sandbox=False, server_url="http://localhost:3000")

        >>> # DOM mode
        >>> env = BrowserEnv(mode="dom", dataset=dataset, rubric=rubric)
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
        use_sandbox: bool = True,
        server_url: str = "http://localhost:3000",
        env: Literal["LOCAL", "BROWSERBASE"] = "BROWSERBASE",
        viewport_width: int = 1024,
        viewport_height: int = 768,
        save_screenshots: bool = True,
        keep_recent_screenshots: int | None = 2,
        proxies: bool = False,
        # CUA sandbox mode specific
        server_port: int = 3000,
        server_ready_timeout: int = 120,
        server_ready_poll_interval: float = 2.0,
        docker_image: str = "node:18-slim",
        cpu_cores: int = 2,
        memory_gb: int = 4,
        disk_size_gb: int = 10,
        sandbox_timeout_minutes: int = 60,
        sandbox_timeout_per_command_seconds: int = 60,
        use_binary: bool = True,
        # Pre-built image configuration (default - fastest startup, skips binary upload)
        use_prebuilt_image: bool = True,
        prebuilt_image: str = "deepdream19/cua-server:latest",
        # Common
        **kwargs: Any,
    ):
        """
        Initialize a Browser Environment.

        Args:
            mode: Operating mode - "dom" for natural language or "cua" for vision-based
            browserbase_api_key: Browserbase API key (or set BROWSERBASE_API_KEY env var)
            browserbase_project_id: Browserbase project ID (or set BROWSERBASE_PROJECT_ID env var)
            model_api_key: Model API key for DOM mode (or set MODEL_API_KEY env var)
            stagehand_model: Model for Stagehand in DOM mode (default: openai/gpt-4o-mini)
            proxy_model_to_stagehand: Whether to proxy model calls through Stagehand
            use_sandbox: For CUA mode, auto-deploy server to sandbox (default: True)
            server_url: CUA server URL when use_sandbox=False (default: http://localhost:3000)
            env: Browser execution environment - "LOCAL" or "BROWSERBASE"
            viewport_width: Browser viewport width (default: 1024)
            viewport_height: Browser viewport height (default: 768)
            save_screenshots: Save screenshots to disk (default: True)
            keep_recent_screenshots: Number of recent screenshots to keep in context (default: 2)
            proxies: Enable Browserbase proxies (default: False)
            server_port: Port for CUA server in sandbox mode (default: 3000)
            server_ready_timeout: Timeout waiting for sandbox server (default: 120)
            server_ready_poll_interval: Poll interval for sandbox server health (default: 2.0)
            docker_image: Docker image for sandbox (default: node:18-slim)
            cpu_cores: CPU cores for sandbox (default: 2)
            memory_gb: Memory in GB for sandbox (default: 4)
            disk_size_gb: Disk size in GB for sandbox (default: 10)
            sandbox_timeout_minutes: Sandbox timeout in minutes (default: 60)
            sandbox_timeout_per_command_seconds: Command timeout in sandbox (default: 60)
            use_binary: Use pre-built SEA binary when use_prebuilt_image=False (default: True)
            use_prebuilt_image: Use pre-built Docker image for fastest startup (default: True)
            prebuilt_image: Docker image to use (default: deepdream19/cua-server:latest)
            **kwargs: Additional arguments passed to StatefulToolEnv
        """
        # Use default system prompt for mode if not provided
        if "system_prompt" not in kwargs or kwargs.get("system_prompt") is None:
            kwargs["system_prompt"] = (
                DOM_DEFAULT_PROMPT if mode == "dom" else CUA_DEFAULT_PROMPT
            )

        super().__init__(**kwargs)
        self.mode = mode
        self._use_sandbox = use_sandbox and mode == "cua"

        # Validate required environment variables before proceeding
        self._validate_environment_variables(
            mode=mode,
            env=env,
            use_sandbox=use_sandbox,
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
            # Unified CUAMode with execution_mode parameter
            self._mode_impl = CUAMode(
                execution_mode="sandbox" if use_sandbox else "local",
                server_url=server_url,
                server_port=server_port,
                env=env,
                browserbase_api_key=browserbase_api_key,
                browserbase_project_id=browserbase_project_id,
                viewport_width=viewport_width,
                viewport_height=viewport_height,
                save_screenshots=save_screenshots,
                keep_recent_screenshots=keep_recent_screenshots,
                proxies=proxies,
                server_ready_timeout=server_ready_timeout,
                server_ready_poll_interval=server_ready_poll_interval,
                docker_image=docker_image,
                cpu_cores=cpu_cores,
                memory_gb=memory_gb,
                disk_size_gb=disk_size_gb,
                sandbox_timeout_minutes=sandbox_timeout_minutes,
                sandbox_timeout_per_command_seconds=sandbox_timeout_per_command_seconds,
                use_binary=use_binary,
                use_prebuilt_image=use_prebuilt_image,
                prebuilt_image=prebuilt_image,
            )
        else:
            raise ValueError(f"Unknown mode: {mode}. Must be 'dom' or 'cua'")

        # Register mode-specific tools
        self._mode_impl.register_tools(self)

    def _validate_environment_variables(
        self,
        mode: str,
        env: str,
        use_sandbox: bool,
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
                if use_sandbox:
                    mode_desc += ", use_sandbox=True"

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

    async def get_prompt_messages(self, state: vf.State) -> vf.Messages:
        """Get prompt messages, filtering screenshots in CUA mode."""
        messages = await super().get_prompt_messages(state)
        if self.mode == "cua":
            messages = self._mode_impl.filter_screenshots_in_messages(list(messages))
        return messages

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
        Handle environment response for tool calls.

        Note: Screenshot filtering is handled in get_prompt_messages() to ensure
        filtering applies to the actual prompt sent to the model.
        """
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
