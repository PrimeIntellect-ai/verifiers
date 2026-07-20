# Harnesses

verifiers supports a range of harnesses out of the box, including Claude Code, Codex, the tool-enabled default harness, and the minimal tool-less `null` harness. However, you may want to build a custom one or extend the selection of third‑party harnesses.

## Packaging an external harness

An external harness must be installed in the same Python environment as verifiers. Its
`harness.id` determines the import module: the package name is lowercased and hyphens are
replaced with underscores. For example, `external-harness-v1` resolves to
`external_harness_v1`. A Hub ID such as `acme/external-harness@1.2.0` installs through the
Environment Hub and resolves to `external_harness`. The loader checks the built-in
`verifiers.v1.harnesses.<module>` namespace first, then the top-level `<module>` supplied by
an external distribution.

The top-level module must use `__all__` to export exactly one `Harness` subclass:

```python
# external_harness_v1/__init__.py
from external_harness_v1.harness import SuperSecretHarness

__all__ = ["SuperSecretHarness"]
```

`__all__` may contain config classes or other public objects, but only one exported class
may inherit from `Harness`. Declare the config specialization on the harness, for example
`class SuperSecretHarness(Harness[SuperSecretHarnessConfig])`. This lets the eval config loader
validate plugin-specific fields from TOML or CLI arguments before constructing the harness.
Missing imports inside an installed plugin are reported directly rather than being treated
as a missing plugin.

## A minimal harness implementation


```python
from verifiers.v1.clients import ModelContext
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
    ) -> ProgramResult:
        # Run the harness in its respective runtime to completion
        # The model (interception) endpoint is in endpoint
        # mcp_urls are the URLs of the tools from the toolset (if registered)

        # Resolve the task's prompt (and system prompt) for this harness
        _, prompt = self.resolve_prompt(trace.task.data)

        # Example: Use the harness, but overwrite the endpoint to use the interception server and the custom model name
        env = {
            **self.config.env,
            "HARNESS_BASE_URL": endpoint,
            "HARNESS_API_KEY": secret,
            "HARNESS_BASE_MODEL": ctx.model,
        }
        # Run the harness to completion inside the selected runtime.
        return await runtime.run_program(["<HARNESS_BINARY>", str(prompt or "")], env)
```
