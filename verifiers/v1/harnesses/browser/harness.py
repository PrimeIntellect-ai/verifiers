import json
from pathlib import Path

from verifiers.v1.clients import ModelContext
from verifiers.v1.configs.harness import HarnessConfig
from verifiers.v1.dialects.chat import message_to_wire
from verifiers.v1.harness import Harness
from verifiers.v1.harnesses.browser import launcher
from verifiers.v1.runtimes import ProgramResult, Runtime
from verifiers.v1.task import TaskData
from verifiers.v1.trace import Trace

PROGRAM_SOURCE = (Path(__file__).resolve().parent / "program.py").read_text()

# Frames the model as a browser agent and teaches the one local tool's contract.
# Condensed from browser-harness's own SKILL.md: the helper names, the
# AX-tree-first workflow, and the two persistence rules a model gets wrong
# without being told (fresh Python per call; first navigation is new_tab).
BROWSER_SYSTEM_PROMPT = """You are a browser automation agent. Your `browser` tool executes Python code that controls a real Chromium over CDP through browser-harness; its helpers are pre-imported.

Rules of the tool:
- Each call runs in a fresh Python process: variables do NOT persist between calls. The browser does persist — tabs, cookies, and page state carry over.
- Use print() for anything you want to see. Large dumps are elided; filter in Python before printing.
- The first navigation is new_tab(url), not goto_url(url). After navigating, call wait_for_load().

Core helpers: new_tab(url), goto_url(url), page_info(), js(expression), click_at_xy(x, y), type_text(text), press_key(key), fill_input(selector, text), scroll(x, y, dy), wait_for_load(), wait_for_element(selector), wait_for_network_idle(), list_tabs(), switch_tab(target), close_tab(), ensure_real_tab(), capture_screenshot(path), upload_file(selector, path), and raw cdp("Domain.method", ...).

Finding elements: prefer the accessibility tree over screenshots. cdp("Accessibility.getFullAXTree")["nodes"] has every element's role, name, and backendDOMNodeId — filter in Python before printing. For coordinates: q = cdp("DOM.getBoxModel", backendNodeId=n)["model"]["content"]; x, y = sum(q[0::2])/4, sum(q[1::2])/4, then click_at_xy(x, y) and verify with a targeted js(...) or page_info() check. Fall back to js(...) over the DOM when the AX tree lacks the element."""


class BrowserHarnessConfig(HarnessConfig):
    cdp_url: str | None = None
    """A running Chrome DevTools HTTP endpoint to attach to (browser-harness's
    `BU_CDP_URL`, e.g. `http://127.0.0.1:9222`). Supplied, the harness attaches
    and the environment keeps owning the browser's lifecycle. Omitted, the
    harness starts a Chromium of its own (see `launcher`) for the run and tears
    it down afterwards."""


class BrowserHarness(Harness[BrowserHarnessConfig]):
    APPENDS_SYSTEM_PROMPT = True
    SUPPORTS_MCP = True
    SUPPORTS_RESUME = True

    async def cdp_endpoint(self, runtime: Runtime, trace: Trace) -> str:
        """The endpoint the program attaches to, and the seam that picks a mode.

        `cdp_url` set: attach, and own nothing but the daemon. Unset: start a
        browser (kept per trace, so a `resume` re-attaches to the same one) and
        own it until cleanup. This is the seam an environment overrides to
        provision a browser its own way -- e.g. inside a sandbox.
        """
        if self.config.cdp_url:
            return self.config.cdp_url
        launched = self._launched.get(trace.id)
        if launched and await launcher.is_alive(runtime, launched["pid"]):
            return launched["endpoint"]
        endpoint, pid = await launcher.launch(runtime, f".vf-browser-{trace.id}")
        self._launched[trace.id] = {"endpoint": endpoint, "pid": pid}
        return endpoint

    @property
    def _launched(self) -> dict[str, dict[str, str]]:
        # Per-trace {endpoint, pid} for browsers this harness started, so
        # cleanup can stop them by recorded PID. Lazy so `__init__` stays the
        # base's.
        if not hasattr(self, "_launched_browsers"):
            self._launched_browsers: dict[str, dict[str, str]] = {}
        return self._launched_browsers

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
        cdp_url = await self.cdp_endpoint(runtime, trace)
        system_prompt, prompt = self.resolve_prompt(data)
        system_prompt = "\n\n".join(
            p for p in (BROWSER_SYSTEM_PROMPT, system_prompt) if p
        )
        env = {**self.config.resolved_env}
        args = [
            f"--base-url={endpoint}",
            f"--api-key={secret}",
            f"--model={ctx.model}",
            f"--cdp-url={cdp_url}",
            f"--system-prompt={system_prompt}",
            # Trace-scoped so a resumed segment's daemon/BH_HOME is this trace's,
            # and cleanup removes exactly what this trace created.
            f"--state-dir=.vf-browser-{trace.id}",
        ]
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
        """Stop what this trace owns and drop its state, by recorded PID.

        Always the browser-harness daemon and the state dir; the browser too
        when this harness launched it (fallback mode). In attach mode the
        browser is the environment's, so it is never touched. Idempotent and
        best-effort; delegated to `launcher` so the process management stays out
        of here.
        """
        launched = self._launched.pop(trace.id, None)
        argv = launcher.teardown_argv(
            f".vf-browser-{trace.id}", launched["pid"] if launched else None
        )
        await runtime.run(argv, {})
