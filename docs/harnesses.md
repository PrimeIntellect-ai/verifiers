# Harnesses

Verifiers supports a range of harnesses out of the box, such as Claude Code, Codex or a minimal, default harness without any built-in tools. However, you may want to build a custom one or extend the selection of third‑party harnesses.

## A minimal harness implementation


```python
from verifiers.v1.clients import RolloutContext
from verifiers.v1.harness import Harness, HarnessConfig
from verifiers.v1.runtimes import ProgramResult, Runtime
from verifiers.v1.trace import Trace

class MyHarnessConfig(HarnessConfig):
    # These are the values that the users are allowed to set and change.
    version: str = "0.0.1"

class MyHarness(Harness[MyHarnessConfig]):
    # Set the system prompt of the task as the harness system message; else add it to the first user message
    APPENDS_SYSTEM_PROMPT = True
    # When the taskset exports a toolset, they are added as MCP. To show that your harness is able to install MCPs, you have to set this flag to true.
    SUPPORTS_MCP = True
    # Allow the task to simulate a user and thus drive the execution of the harness
    SUPPORTS_USER_SIM = True

    async def setup(self, runtime: Runtime) -> None:
        # Add your install script(s) here
        MY_INSTALL_COMMAND = "echo 'installing...'"
        await runtime.run(["sh", MY_INSTALL_COMMAND], {})

    async def launch(
        self,
        ctx: RolloutContext,
        trace: Trace,
        runtime: Runtime,
        endpoint: str,
        secret: str,
        mcp_urls: dict[str, str],
    ) -> ProgramResult:
        # Run the harness in its respective runtime to completion
        # The model (interception) endpoint is in endpoint
        # mcp_urls are the URLs of the tools from the toolset (if registered)

        # Resolve the task's prompt (and system prompt) for this harness
        _, prompt = self.resolve_prompt(trace.task)

        # Example: Use the harness, but overwrite the endpoint to use the interception server and the custom model name
        ENVIRONMENT_VARS = {
            **self.config.env,
            "HARNESS_BASE_URL": endpoint,
            "HARNESS_API_KEY": secret,
            "HARNESS_BASE_MODEL": ctx.model,
        }

        # Run the harness using the values we defined earlier
        return await runtime.run_program([HARNESS_BINARY, prompt], ENVIRONMENT_VARS)
```
