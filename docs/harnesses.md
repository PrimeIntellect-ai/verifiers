# Harnesses

verifiers ships with a range of harnesses out of the box, including Claude Code, Codex, Pi, Prime-Agent, the tool-enabled `bash` harness, the CDP-driven `browser-use` harness, and the minimal `null` harness (which has no built-in shell tools of its own, though it can still call task MCP tools). Most third-party CLI harnesses must run inside a container runtime, and individual harnesses vary in which optional features they support (such as MCP tools, follow-up messages, image input, and skills). A harness that runs model-controlled code can use MCP only when its tool execution is isolated from the MCP client and its credentials. The built-in `rlm` harness supplies this boundary; `null` has no model-controlled code execution.

## Bash harness

The built-in `bash` harness provides shell execution and file-editing tools, multi-turn conversation resume, and tool interception (though not harness skills). It cannot be paired with MCP tools. With `agent.deployment.mode = "split"`, its model loop and Bash/edit execution run in separate Docker containers or Prime sandboxes. Because it runs an in-house loop rather than installing an external CLI, it can also run on `subprocess` for trusted local work.

Context compaction is enabled by default: when the model provider reports a context window size, `bash` automatically summarizes older context once roughly 16,000 tokens remain. You can set a fixed token threshold in your config, or set `compaction = "None"` to disable compaction altogether:

```toml
[env.agent.harness]
id = "bash"
compaction = { summarize_at_tokens = 100000 }
```

Web search is disabled by default; setting `search = true` enables it and requires `SERPER_API_KEY` on the evaluator host.

## Split execution

Every built-in harness supports `agent.deployment.mode = "split"` on Docker and Prime. The harness runs in one sandbox and the task workspace and model-controlled tools run in another. Codex uses its native remote exec-server, RLM uses its execution worker, and Prime Agent uses a remote kernel. Other coding harnesses route native tool calls or environment commands through host-managed adapters; Browser Use runs its browser code in the workspace. `null` keeps its tool-less model loop in the harness sandbox.

Request interceptors check calls before execution. A native gate can return a replacement tool result; where a tool has no such gate, a blocked call ends the rollout before delivery. Codex Code Mode is rejected when request interception is active because its nested `write_stdin` calls have no upstream gate. OpenCode and OpenClaw expose direct tools in split mode instead of Code Mode. RLM and Prime Agent workers cannot invoke supervisor callbacks, including delegation and credentialed broker helpers.

The default coding tools are supported. Split mode does not support MCP toolsets, composed services, or Bash search. Adapters reject optional tools they cannot isolate, including agent delegation, provider integrations, skill discovery, interactive questions, and Claude notebook editing. These limits apply to split mode; an unsupported tool never falls back to running in the harness sandbox.

The shared file adapters support text and image reads, exact text edits, and basic glob and regex search. They do not reproduce every upstream output format, fuzzy edit rule, ignore-file rule, advanced search option, or PDF/media reader. Background jobs expose status and output files; native automatic completion notifications are not available in every adapter.

## A minimal harness implementation

If you want to integrate a custom agent loop or another third-party CLI harness, subclass `Harness` and `HarnessConfig`:

```python
from verifiers.v1.clients import ModelContext
from verifiers.v1.configs.harness import HarnessConfig
from verifiers.v1.harness import Harness
from verifiers.v1.runtimes import (
    ProgramResult,
    Runtime,
)
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
        return await runtime.run(["<HARNESS_BINARY>", str(prompt or "")], env)
```

Replace the placeholder installation command, binary, and environment variable names with the ones your program expects. Every model call must be routed through the interception server at `endpoint` (authenticated with `secret`) so verifiers can record the conversation and build the `Trace` from the returned `ProgramResult`. Merging `self.config.resolved_env` into the process environment ensures any variables configured via `env` and `forward_env` are passed through. When a task sets `agent_user`, `setup` runs as that user too, so install the harness into a writable path and leave system package installation to the container image or task `setup`.

Class-level capability flags declare which optional features your harness implements:

- `APPENDS_SYSTEM_PROMPT`: passes `TaskData.system_prompt` as a separate system message; when `False`, `resolve_prompt` prepends it to the first user message instead.
- `SUPPORTS_MCP`: indicates that `launch` wires `mcp_urls` into the program. A code-running harness also needs `ISOLATES_TOOL_EXECUTION = True`, backed by an enforced process boundary that prevents model-controlled code from reading the MCP client or its credentials. MCP pairings fail before runtime provisioning otherwise.
- `SUPPORTS_RESUME`: indicates that a multi-turn exchange can restart from the conversation so far (programs with native session state should override `resume` or `session` instead). Note that live mid-turn steering requires a `HarnessSession` that implements `steer`; conversation resume support alone is not enough (see [message delivery](agent.md#messages-during-an-active-turn)).
- `SUPPORTS_SKILLS`: indicates that the harness calls `install_skills` to copy configured skills into the directory where the program discovers them.
- `SUPPORTS_TOOL_INTERCEPTION`: indicates that the harness checks tool calls against the rollout's tool gate (`tool_interception_url`) before executing them.
- `SUPPORTS_SPLIT_EXECUTION`: indicates that `remote_tools(runtime, data)` binds model-controlled execution to the separate task workspace. It returns a tool handler, or registers a native execution worker and returns `None`. `SUPPORTS_SPLIT_TOOL_INTERCEPTION` declares a gate available specifically in split mode.

For programs that speak the Agent Client Protocol (ACP), you can subclass `vf.ACPHarness` in [`acp/__init__.py`](../verifiers/v1/acp/__init__.py). If a harness deterministically reformats earlier messages when re-sending conversation history (for example, dropping whitespace-only text blocks), override `replay_view(message)` so replayed turns match the existing trace branch rather than forking the message graph. If your harness writes files or session state outside the temporary runtime (or runs inside an environment-owned runtime), override `cleanup(trace, runtime)` to remove per-rollout state after scoring completes; `cleanup` runs before the runtime is released and should be idempotent.
