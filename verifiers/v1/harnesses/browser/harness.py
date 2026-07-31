import json
from pathlib import Path

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
- Use print() for anything you want to see. Large dumps are elided; filter in Python before printing.
- The first navigation is new_tab(url), not goto_url(url). After navigating, call wait_for_load().

Core helpers: new_tab(url), goto_url(url), page_info(), js(expression), click_at_xy(x, y), type_text(text), press_key(key), fill_input(selector, text), scroll(x, y, dy), wait_for_load(), wait_for_element(selector), wait_for_network_idle(), list_tabs(), switch_tab(target), close_tab(), ensure_real_tab(), capture_screenshot(path), upload_file(selector, path), and raw cdp("Domain.method", ...).

Finding elements: prefer the accessibility tree over screenshots. cdp("Accessibility.getFullAXTree")["nodes"] has every element's role, name, and backendDOMNodeId — filter in Python before printing. For coordinates: q = cdp("DOM.getBoxModel", backendNodeId=n)["model"]["content"]; x, y = sum(q[0::2])/4, sum(q[1::2])/4, then click_at_xy(x, y) and verify with a targeted js(...) or page_info() check. Fall back to js(...) over the DOM when the AX tree lacks the element."""


class BrowserHarnessConfig(HarnessConfig):
    cdp_url: str | None = None
    """The Chrome DevTools HTTP endpoint to attach to (browser-harness's
    `BU_CDP_URL`, e.g. `http://127.0.0.1:9222`). Required: the harness attaches
    to a browser the environment provides and keeps alive; it never launches
    one. An environment that provisions its own browser overrides
    `cdp_endpoint` instead of setting this."""


class BrowserHarness(Harness[BrowserHarnessConfig]):
    APPENDS_SYSTEM_PROMPT = True
    SUPPORTS_MCP = True
    SUPPORTS_RESUME = True

    async def cdp_endpoint(self, runtime: Runtime, trace: Trace) -> str:
        """The DevTools endpoint the program attaches to for this trace.

        Attach-only by default: the environment provides a running browser and
        this returns its endpoint. This is the seam an environment overrides to
        provision a browser it owns (launch it, keep it alive across `resume`,
        tear it down) and hand back its endpoint.
        """
        if not self.config.cdp_url:
            raise ValueError(
                "browser harness needs a running Chrome DevTools endpoint to attach "
                "to; set --env.agent.harness.cdp_url (e.g. http://127.0.0.1:9222). "
                "Start one with: chrome --remote-debugging-port=9222 "
                "--user-data-dir=$(mktemp -d) --headless --no-first-run --no-sandbox"
            )
        return self.config.cdp_url

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
        """Stop the browser-harness daemon this trace started and drop its state.

        The daemon is the only process this harness owns; the browser belongs to
        whoever provided the endpoint, so this never touches it. The daemon
        records its own PID under `BH_HOME`, so cleanup kills that recorded PID
        rather than pattern-matching. Idempotent and best-effort: a run that
        never started a daemon leaves no PID file, and an owned runtime is torn
        down regardless.
        """
        state = f".vf-browser-{trace.id}"
        pid_file = f"{state}/bh-home/runtime/bu-default.pid"
        teardown = (
            f'[ -f "{pid_file}" ] && kill "$(cat "{pid_file}")" 2>/dev/null; '
            f'rm -rf "{state}" 2>/dev/null || true'
        )
        await runtime.run(["sh", "-c", teardown], {})
