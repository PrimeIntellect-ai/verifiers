# Harnesses

verifiers supports a range of harnesses out of the box, including Claude Code, Codex, the tool-enabled `bash` harness, and the minimal tool-less `null` harness. However, you may want to build a custom one or extend the selection of third‑party harnesses.

## A minimal harness implementation


```python
from verifiers.v1.clients import ModelContext
from verifiers.v1.harness import Harness, HarnessConfig
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
    # Allow stateless chat continuation by relaunching on a Messages prompt.
    SUPPORTS_MESSAGE_PROMPT = True

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
