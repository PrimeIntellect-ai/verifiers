import json
import secrets
from pathlib import Path
from typing import Literal
from urllib.parse import urlsplit

import httpx
from pydantic import model_validator

from verifiers.v1.clients import ModelContext
from verifiers.v1.configs.harness import HarnessConfig
from verifiers.v1.errors import HarnessError
from verifiers.v1.harness import Harness
from verifiers.v1.harnesses.utils import mcp
from verifiers.v1.harnesses.utils.launch import bundle_program, launch_chat_program
from verifiers.v1.runtimes import ProgramResult, Runtime
from verifiers.v1.task import TaskData
from verifiers.v1.trace import Trace
from verifiers.v1.types import ToolCall, ToolMessage

PROGRAM_SOURCE = bundle_program(
    (Path(__file__).resolve().parent / "program.py").read_text(), mcp
)

# The helper names and persistence rules the model needs to use the local tool.
BROWSER_SYSTEM_PROMPT = """You are a browser automation agent. Your `browser` tool executes Python code that controls a real Chromium over CDP through browser-harness; its helpers are pre-imported.

Rules of the tool:
- Each call runs in a fresh Python process: variables do NOT persist between calls. The browser does persist — tabs, cookies, and page state carry over.
- Use print() for anything you want to see. Filter in Python before printing; a raw AX tree or DOM dump is huge.
- The first navigation is new_tab(url), not goto_url(url). After navigating, call wait_for_load().

Core helpers: new_tab(url), goto_url(url), page_info(), js(expression), click_at_xy(x, y), type_text(text), press_key(key), fill_input(selector, text), scroll(x, y, dy), wait_for_load(), wait_for_element(selector), wait_for_network_idle(), list_tabs(), switch_tab(target), close_tab(), ensure_real_tab(), capture_screenshot(path), upload_file(selector, path), and raw cdp("Domain.method", ...).

Finding elements: prefer the accessibility tree over screenshots. cdp("Accessibility.getFullAXTree")["nodes"] has every element's role, name, and backendDOMNodeId — filter in Python before printing. For coordinates: q = cdp("DOM.getBoxModel", backendNodeId=n)["model"]["content"]; x, y = sum(q[0::2])/4, sum(q[1::2])/4, then click_at_xy(x, y) and verify with a targeted js(...) or page_info() check. Fall back to js(...) over the DOM when the AX tree lacks the element."""


class BrowserUseHarnessConfig(HarnessConfig):
    dialect: Literal["chat_completions"] = "chat_completions"
    browser: Literal["chromium", "cdp"] = "chromium"
    """`chromium` launches locally; `cdp` attaches to `cdp_url` without owning it."""

    cdp_url: str | None = None
    """Rollout-scoped HTTP or WebSocket endpoint used with `browser = "cdp"`."""

    @model_validator(mode="after")
    def _require_cdp_url_iff_cdp(self) -> "BrowserUseHarnessConfig":
        if self.browser == "cdp" and not self.cdp_url:
            raise ValueError(
                "browser='cdp' needs cdp_url set to a CDP endpoint to attach to"
            )
        if self.browser != "cdp" and self.cdp_url:
            raise ValueError(
                f"cdp_url is only valid with browser='cdp', not browser={self.browser!r}"
            )
        return self


class BrowserUseHarness(Harness[BrowserUseHarnessConfig]):
    APPENDS_SYSTEM_PROMPT = True
    SUPPORTS_MCP = True
    SUPPORTS_RESUME = True
    SUPPORTS_TOOL_INTERCEPTION = True
    SUPPORTS_SPLIT_EXECUTION = True
    # The browser tool executes model-authored Python through a third-party daemon.
    NEEDS_CONTAINER = True

    async def setup(self, runtime: Runtime) -> None:
        await runtime.prepare_uv_script(PROGRAM_SOURCE, self.config.resolved_env)

    async def remote_tools(self, runtime: Runtime, data: TaskData):
        assert runtime._execution_bridge is not None
        program = await runtime._execution_bridge.setup_runtime.prepare_uv_script(
            PROGRAM_SOURCE, {}
        )
        state = f"/tmp/browser-session-{secrets.token_hex(16)}"
        args = await self._browser_args(runtime, state)

        async def execute(call: ToolCall) -> ToolMessage:
            if call.name != "browser":
                raise HarnessError(f"browser-use has no built-in tool {call.name!r}")
            try:
                arguments = json.loads(call.arguments)
                if not isinstance(arguments, dict) or not isinstance(
                    arguments.get("code"), str
                ):
                    raise TypeError("browser code must be a string")
            except (json.JSONDecodeError, TypeError) as error:
                content = f"error: {error}"
            else:
                result = await runtime.run(
                    [*program, *args, f"--execute-code={arguments['code']}"], {}
                )
                content = result.stdout + result.stderr
            return ToolMessage(tool_call_id=call.id, name=call.name, content=content)

        return execute

    async def _browser_args(self, runtime: Runtime, state: str) -> list[str]:
        args = [f"--browser={self.config.browser}", f"--state-dir={state}"]
        if self.config.cdp_url:
            cdp_url = runtime.host_url(self.config.cdp_url)
            if cdp_url != self.config.cdp_url and urlsplit(
                self.config.cdp_url
            ).scheme in ("http", "https"):
                async with httpx.AsyncClient() as client:
                    response = await client.get(
                        f"{self.config.cdp_url.rstrip('/')}/json/version"
                    )
                    response.raise_for_status()
                cdp_url = runtime.host_url(response.json()["webSocketDebuggerUrl"])
            args.append(f"--cdp-url={cdp_url}")
        return args

    async def launch(
        self,
        ctx: ModelContext,
        trace: Trace,
        runtime: Runtime,
        endpoint: str,
        secret: str,
        mcp_urls: dict[str, str],
        data: TaskData,
        tool_interception_url: str | None = None,
    ) -> ProgramResult:
        system_prompt, prompt = self.resolve_prompt(data)
        system_prompt = "\n\n".join(
            p for p in (BROWSER_SYSTEM_PROMPT, system_prompt) if p
        )
        # Default resume replays the transcript. If there was no task system
        # prompt, that transcript already contains this harness prompt.
        replaying_browser_prompt = (
            data.system_prompt is None
            and prompt is not None
            and not isinstance(prompt, str)
            and any(
                message.role == "system" and message.content == BROWSER_SYSTEM_PROMPT
                for message in prompt
            )
        )
        env = {**self.config.resolved_env}
        state = runtime.temp_path(
            f"vf-browser-{trace.id}", env=self.config.resolved_env
        )
        if runtime._remote_tools_only:
            if not tool_interception_url:
                raise HarnessError("split browser-use requires a tool gate")
            args = ["--remote-tools-only", f"--state-dir={state}"]
        else:
            args = await self._browser_args(runtime, state)
        if tool_interception_url:
            args.append(f"--tool-interception-url={tool_interception_url}")
        return await launch_chat_program(
            PROGRAM_SOURCE,
            self.config,
            ctx,
            trace,
            runtime,
            endpoint,
            secret,
            mcp_urls,
            None if replaying_browser_prompt else system_prompt,
            prompt,
            extra_args=args,
            env=env,
        )
