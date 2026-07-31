import json
from pathlib import Path
from typing import Literal

from verifiers.v1.clients import ModelContext
from verifiers.v1.configs.harness import HarnessConfig
from verifiers.v1.dialects.chat import message_to_wire
from verifiers.v1.harness import Harness
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
- Use print() for anything you want to see. Filter in Python before printing; a raw AX tree or DOM dump is huge.
- The first navigation is new_tab(url), not goto_url(url). After navigating, call wait_for_load().

Core helpers: new_tab(url), goto_url(url), page_info(), js(expression), click_at_xy(x, y), type_text(text), press_key(key), fill_input(selector, text), scroll(x, y, dy), wait_for_load(), wait_for_element(selector), wait_for_network_idle(), list_tabs(), switch_tab(target), close_tab(), ensure_real_tab(), capture_screenshot(path), upload_file(selector, path), and raw cdp("Domain.method", ...).

Finding elements: prefer the accessibility tree over screenshots. cdp("Accessibility.getFullAXTree")["nodes"] has every element's role, name, and backendDOMNodeId — filter in Python before printing. For coordinates: q = cdp("DOM.getBoxModel", backendNodeId=n)["model"]["content"]; x, y = sum(q[0::2])/4, sum(q[1::2])/4, then click_at_xy(x, y) and verify with a targeted js(...) or page_info() check. Fall back to js(...) over the DOM when the AX tree lacks the element."""

def state_dir(trace: Trace) -> str:
    """The trace's state dir, relative to the program's cwd: browser profile,
    BH_HOME, and the recorded endpoint and PIDs all live under it."""
    return f".vf-browser-{trace.id}"


def teardown_argv(state: str) -> list[str]:
    """Stop the browser-harness daemon and any browser launched for this state
    dir, by the PID each recorded under it, then drop the dir. `browser.pid` is
    absent when nothing was launched, so only the daemon is touched then.
    Reusable by an environment that must run the teardown somewhere specific
    (e.g. inside a sandbox where the recorded PIDs are valid)."""
    script = (
        f'for f in "{state}/browser.pid" "{state}/bh-home/runtime/bu-default.pid"; do '
        '[ -f "$f" ] && kill "$(cat "$f")" 2>/dev/null; done; '
        f'rm -rf "{state}" 2>/dev/null || true'
    )
    return ["sh", "-c", script]


class BrowserHarnessConfig(HarnessConfig):
    browser: Literal["chromium"] = "chromium"
    """Where the browser the model drives comes from. `chromium` launches a
    local headless Chromium and attaches to it over CDP -- it works out of the
    box on a browser-capable image (see `docs/v1/harnesses.md`) and needs no
    endpoint wiring. (A `browserbase` value that creates a hosted session over
    the Browserbase API is the planned follow-up; until then, point
    browser-harness's own `BU_CDP_URL` at a remote browser to use one.)"""


class BrowserHarness(Harness[BrowserHarnessConfig]):
    APPENDS_SYSTEM_PROMPT = True
    SUPPORTS_MCP = True
    SUPPORTS_RESUME = True

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
        env = {**self.config.resolved_env}
        args = [
            f"--base-url={endpoint}",
            f"--api-key={secret}",
            f"--model={ctx.model}",
            f"--browser={self.config.browser}",
            f"--system-prompt={system_prompt}",
            # Trace-scoped so a resumed segment reuses the browser it launched,
            # and cleanup removes exactly what this trace created.
            f"--state-dir={state_dir(trace)}",
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
        """Stop what this trace launched -- the browser-harness daemon, and the
        Chromium `program.py` started, both by the PID each recorded -- and drop
        the state dir. Idempotent and best-effort."""
        await runtime.run(teardown_argv(state_dir(trace)), {})
