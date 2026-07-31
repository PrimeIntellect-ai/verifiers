import json
from pathlib import Path
from typing import Literal

from pydantic import model_validator

from verifiers.v1.clients import ModelContext
from verifiers.v1.configs.harness import HarnessConfig
from verifiers.v1.dialects.chat import message_to_wire
from verifiers.v1.harness import Harness
from verifiers.v1.runtimes import ProgramResult, Runtime
from verifiers.v1.task import TaskData
from verifiers.v1.trace import Trace

PROGRAM_SOURCE = (Path(__file__).resolve().parent / "program.py").read_text()

# The helper names and persistence rules the model needs to use the local tool.
BROWSER_SYSTEM_PROMPT = """You are a browser automation agent. Your `browser` tool executes Python code that controls a real Chromium over CDP through browser-harness; its helpers are pre-imported.

Rules of the tool:
- Each call runs in a fresh Python process: variables do NOT persist between calls. The browser does persist — tabs, cookies, and page state carry over.
- Use print() for anything you want to see. Filter in Python before printing; a raw AX tree or DOM dump is huge.
- The first navigation is new_tab(url), not goto_url(url). After navigating, call wait_for_load().

Core helpers: new_tab(url), goto_url(url), page_info(), js(expression), click_at_xy(x, y), type_text(text), press_key(key), fill_input(selector, text), scroll(x, y, dy), wait_for_load(), wait_for_element(selector), wait_for_network_idle(), list_tabs(), switch_tab(target), close_tab(), ensure_real_tab(), capture_screenshot(path), upload_file(selector, path), and raw cdp("Domain.method", ...).

Finding elements: prefer the accessibility tree over screenshots. cdp("Accessibility.getFullAXTree")["nodes"] has every element's role, name, and backendDOMNodeId — filter in Python before printing. For coordinates: q = cdp("DOM.getBoxModel", backendNodeId=n)["model"]["content"]; x, y = sum(q[0::2])/4, sum(q[1::2])/4, then click_at_xy(x, y) and verify with a targeted js(...) or page_info() check. Fall back to js(...) over the DOM when the AX tree lacks the element."""


def state_dir(trace: Trace) -> str:
    """The trace-scoped browser profile, endpoint, and process state."""
    return f".vf-browser-{trace.id}"


def teardown_argv(state: str) -> list[str]:
    """Stop processes recorded under a trace state directory."""
    script = (
        'state="$1"; '
        'for f in "$state/browser.pid" "$state/browser-starting.pid" '
        '"$state/bh-home/runtime/bu-default.pid"; do '
        '[ -f "$f" ] && kill "$(cat "$f")" 2>/dev/null; done; '
        'rm -rf -- "$state" 2>/dev/null || true'
    )
    return ["sh", "-c", script, "browser-teardown", state]


class BrowserHarnessConfig(HarnessConfig):
    browser: Literal["chromium", "cdp"] = "chromium"
    """`chromium` launches locally; `cdp` attaches to `cdp_url` without owning it."""

    cdp_url: str | None = None
    """HTTP or WebSocket CDP endpoint used with `browser = "cdp"`."""

    @model_validator(mode="after")
    def _require_cdp_url_iff_cdp(self) -> "BrowserHarnessConfig":
        if self.browser == "cdp" and not self.cdp_url:
            raise ValueError(
                "browser='cdp' needs cdp_url set to a CDP endpoint to attach to"
            )
        if self.browser != "cdp" and self.cdp_url:
            raise ValueError(
                f"cdp_url is only valid with browser='cdp', not browser={self.browser!r}"
            )
        return self


class BrowserHarness(Harness[BrowserHarnessConfig]):
    APPENDS_SYSTEM_PROMPT = True
    SUPPORTS_MCP = True
    SUPPORTS_RESUME = True
    # The browser tool executes model-authored Python through a third-party daemon.
    NEEDS_CONTAINER = True

    async def setup(self, runtime: Runtime) -> None:
        await runtime.prepare_uv_script(PROGRAM_SOURCE, self.config.resolved_env)

    async def launch(
        self,
        ctx: ModelContext,
        trace: Trace,
        runtime: Runtime,
        endpoint: str,
        secret: str,
        mcp_urls: dict[str, str],
        data: TaskData,
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
        args = [
            f"--base-url={endpoint}",
            f"--api-key={secret}",
            f"--model={ctx.model}",
            f"--browser={self.config.browser}",
            # Trace-scoped so a resumed segment reuses the browser it launched,
            # and cleanup removes exactly what this trace created.
            f"--state-dir={state_dir(trace)}",
        ]
        if not replaying_browser_prompt:
            args.append(f"--system-prompt={system_prompt}")
        if self.config.cdp_url:
            args.append(f"--cdp-url={self.config.cdp_url}")
        if mcp_urls:
            # The program connects to the tool servers over HTTP; hand it a standard
            # `mcpServers` URL config (the `mcp` client itself comes from the uv deps).
            args.append(
                "--mcp-config="
                + json.dumps(
                    {
                        "mcpServers": {
                            name: {"url": url} for name, url in mcp_urls.items()
                        }
                    }
                )
            )
        if isinstance(prompt, str):
            args.append(f"--prompt={prompt}")
        elif prompt is not None:
            # Base64 images can exceed exec limits, so hand Messages off through a file.
            path = f".vf-initial-messages-{trace.id}.json"
            await runtime.write(
                path,
                json.dumps([message_to_wire(m) for m in prompt]).encode(),
            )
            args.append(f"--initial-messages-file={path}")
        program = await runtime.prepare_uv_script(
            PROGRAM_SOURCE, self.config.resolved_env
        )
        return await runtime.run_program([*program, *args], env)

    async def cleanup(self, trace: Trace, runtime: Runtime) -> None:
        """Stop this trace's recorded daemon and locally launched browser."""
        await runtime.run(teardown_argv(state_dir(trace)), {})
