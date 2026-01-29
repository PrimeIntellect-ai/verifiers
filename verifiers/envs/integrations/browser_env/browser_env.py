"""Unified Browser Environment with DOM and CUA modes."""

import logging
import os
from typing import Any, Literal

import verifiers as vf

from .modes.base import BrowserMode
from .modes.cua_mode import CUAMode
from .modes.dom_mode import DOMMode

logger = logging.getLogger(__name__)

ModeType = Literal["dom", "cua"]


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
        project_id: str | None = None,
        browserbase_api_key_var: str = "BROWSERBASE_API_KEY",
        # DOM mode specific
        model_api_key_var: str = "MODEL_API_KEY",
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
        advanced_stealth: bool = False,
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
            project_id: Browserbase project ID
            browserbase_api_key_var: Env var name for Browserbase API key (default: BROWSERBASE_API_KEY)
            model_api_key_var: Env var name for model API key (default: MODEL_API_KEY)
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
            advanced_stealth: Enable Browserbase Advanced Stealth mode for anti-bot detection (default: False)
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
        super().__init__(**kwargs)
        self.mode = mode
        browserbase_api_key = os.getenv(browserbase_api_key_var)
        model_api_key = os.getenv(model_api_key_var)
        if mode == "dom":
            self._mode_impl: BrowserMode = DOMMode(
                browserbase_api_key=browserbase_api_key,
                project_id=project_id,
                model_api_key=model_api_key,
                stagehand_model=stagehand_model,
                proxy_model_to_stagehand=proxy_model_to_stagehand,
                proxies=proxies,
                advanced_stealth=advanced_stealth,
            )
        elif mode == "cua":
            self._mode_impl = CUAMode(
                execution_mode="sandbox" if use_sandbox else "local",
                server_url=server_url,
                server_port=server_port,
                env=env,
                browserbase_api_key=browserbase_api_key,
                browserbase_project_id=project_id,
                viewport_width=viewport_width,
                viewport_height=viewport_height,
                save_screenshots=save_screenshots,
                keep_recent_screenshots=keep_recent_screenshots,
                proxies=proxies,
                advanced_stealth=advanced_stealth,
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

        self._mode_impl.register_tools(self)
        if mode == "dom":
            logger.info(
                f"BrowserEnv initialized in DOM mode with stagehand_model='{stagehand_model}'"
            )
        else:
            logger.info("BrowserEnv initialized in CUA mode")

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
