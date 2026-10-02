# Harnesses

Use a built-in harness when possible. Options include Claude Code, Codex,
`bash` for shell tools, `browser-use` for browser tasks, and `null` for basic
chat. `null` has no built-in shell but can use task MCP tools.

Use `uv run vf-eval my-task --env.agent.harness.id codex --help` to inspect a
harness's settings. Most third-party harnesses require a container. Check support
for each feature you need: MCP tools, follow-up messages, image input, and skills
are separate features.

## A minimal harness implementation

```python
from verifiers.v1.clients import ModelContext
from verifiers.v1.configs.harness import HarnessConfig
from verifiers.v1.harness import Harness
from verifiers.v1.runtimes import ProgramResult, Runtime
from verifiers.v1.task import TaskData
from verifiers.v1.trace import Trace


class MyHarnessConfig(HarnessConfig):
    # Users can override these settings in CLI flags or TOML.
    version: str = "0.0.1"


class MyHarness(Harness[MyHarnessConfig]):
    # Pass the task's system prompt as a system message.
    APPENDS_SYSTEM_PROMPT = True

    async def setup(self, runtime: Runtime) -> None:
        # Replace this with the program's installation command.
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
        system, prompt = self.resolve_text_prompt(data)

        # Route model calls through verifiers so they are recorded.
        env = {
            **self.config.resolved_env,
            "HARNESS_BASE_URL": endpoint,
            "HARNESS_API_KEY": secret,
            "HARNESS_BASE_MODEL": ctx.model,
            "HARNESS_SYSTEM_PROMPT": system or "",
        }
        return await runtime.run_program(["<HARNESS_BINARY>", str(prompt or "")], env)
```

Replace the placeholder binary, installation command, and environment variable
names with those your program uses. Every model call must use the supplied
`endpoint` and `secret` so verifiers can record it. Include
`self.config.resolved_env` to support credentials passed through `forward_env`.

Capability flags describe behavior you must implement:

- `SUPPORTS_MCP`: pass `mcp_urls` to the program.
- `SUPPORTS_RESUME`: restart the program from the conversation so far. Programs with saved sessions should implement `resume` or a session instead.
- `SUPPORTS_SKILLS`: use `install_skills` to place skills where the program looks for them.
- `SUPPORTS_TOOL_INTERCEPTION`: check tool calls through the rollout's tool gate before executing them.
- `APPENDS_SYSTEM_PROMPT`: pass the system prompt as a system message. Otherwise, it is added to the first user message.

For an ACP program, reuse `vf.ACPHarness` in
[`acp/__init__.py`](../../verifiers/v1/acp/__init__.py).

Return the main program's `ProgramResult`; verifiers builds the trace. If the
harness creates files or sessions outside the temporary runtime, remove them in
`cleanup(trace, runtime)`. Cleanup runs after scoring, before the runtime is
released, and must be safe to call more than once. It also runs when the
environment owns the runtime.
