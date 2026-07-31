# Harnesses

verifiers supports a range of harnesses out of the box, including Claude Code, Codex, the tool-enabled `bash` harness, the CDP-driven `browser` harness, and the minimal tool-less `null` harness. However, you may want to build a custom one or extend the selection of third‑party harnesses.

## The `browser` harness

The `browser` harness gives the model one tool that runs Python against a real Chromium over CDP via [browser-harness](https://github.com/browser-use/browser-harness). Two modes, chosen by whether `cdp_url` is set:

- **Attach** (`--env.agent.harness.cdp_url http://127.0.0.1:9222`): the harness attaches to a browser you run and keep alive, and needs no special runtime image. Launch a debuggable Chromium yourself with `chrome --remote-debugging-port=9222 --user-data-dir="$(mktemp -d)" --headless --no-first-run --no-sandbox`.
- **Self-launch** (no `cdp_url`): the harness starts a Chromium of its own for the run and tears it down after. This needs a browser in the runtime image — use a browser-capable one such as the official Playwright image (`mcr.microsoft.com/playwright/python:v1.61.0-noble`) via `--env.agent.runtime.image`; `assets/templates/browser/Dockerfile` is a minimal recipe for publishing your own.

An environment that provisions its browser a bespoke way (e.g. inside a sandbox) overrides `BrowserHarness.cdp_endpoint` to return its endpoint instead of using either path.

## A minimal harness implementation

```python
from verifiers.v1.clients import ModelContext
from verifiers.v1.configs.harness import HarnessConfig
from verifiers.v1.harness import Harness
from verifiers.v1.runtimes import ProgramResult, Runtime
from verifiers.v1.task import TaskData
from verifiers.v1.trace import Trace


class MyHarnessConfig(HarnessConfig):
    # These are the values that the users are allowed to set and change.
    version: str = "0.0.1"


class MyHarness(Harness[MyHarnessConfig]):
    # Set the system prompt of the task as the harness system message; else add it to the first user message
    APPENDS_SYSTEM_PROMPT = True
    # When the taskset exports a toolset, they are added as MCP. To show that your harness is able to install MCPs, you have to set this flag to true.
    SUPPORTS_MCP = True
    # Allow transcript-backed resume by relaunching on a Messages prompt.
    SUPPORTS_RESUME = True

    async def setup(self, runtime: Runtime) -> None:
        # Install the harness in its rollout runtime
        await runtime.run(["sh", "-c", "echo installing..."], {})

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
        # Run the harness in its respective runtime to completion
        # The model (interception) endpoint is in endpoint
        # mcp_urls are the URLs of the tools from the toolset (if registered)

        # Resolve the task's prompt (and system prompt) for this harness
        _, prompt = self.resolve_prompt(data)

        # Example: Use the harness, but overwrite the endpoint to use the interception server and the custom model name
        env = {
            **self.config.env,
            "HARNESS_BASE_URL": endpoint,
            "HARNESS_API_KEY": secret,
            "HARNESS_BASE_MODEL": ctx.model,
        }
        # Run the harness to completion inside the selected runtime.
        return await runtime.run_program(["<HARNESS_BINARY>", str(prompt or "")], env)

    async def cleanup(self, trace: Trace, runtime: Runtime) -> None:
        # Remove any per-rollout state that must not survive in a borrowed runtime.
        ...
```
