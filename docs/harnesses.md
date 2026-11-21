# Harnesses

verifiers ships with a range of harnesses out of the box, including Claude Code, Codex, Pi, Prime-Agent, the tool-enabled `bash` harness, the CDP-driven `browser-use` harness, and the minimal `null` harness (which has no built-in shell tools of its own, though it can still call task MCP tools). Most third-party CLI harnesses must run inside a container runtime, and individual harnesses vary in which optional features they support (such as MCP tools, follow-up messages, image input, and skills).

## Bash harness

The built-in `bash` harness provides shell execution and file-editing tools, task MCP support, multi-turn conversation resume, and tool interception (though not harness skills). On Docker and Prime, its model loop and Bash/edit execution run in separate sandboxes; it can also run on `subprocess` for trusted local work.

Context compaction is enabled by default: when the model provider reports a context window size, `bash` automatically summarizes older context once roughly 16,000 tokens remain. You can set a fixed token threshold in your config, or set `compaction = "None"` to disable compaction altogether:

```toml
[env.agent.harness]
id = "bash"
compaction = { summarize_at_tokens = 100000 }
```

Web search is not built into `bash`; attach an MCP toolset or use a harness with native search.

## Split execution

The harness runs the model loop in its own environment. Claude Code and Pi run their tool adapters there and send approved operations through the evaluator relay. The evaluator uses the sandbox provider to run commands as the task user, so these adapters work on task images without Python. Codex uses its native exec-server in the workspace, RLM its execution worker, and Prime Agent its remote kernel. Browser Use runs its browser code in the workspace. `null` runs a tool-less model loop.

Prime starts workspace commands through `/bin/bash`. When building a worker image from stock Alpine, include Bash (`RUN apk add --no-cache bash`). For Codex, the evaluator stages the Linux x86-64 or arm64 executor under `/var/tmp/agent-tools`. The setup user needs write access there and the task user needs execute access. The worker can keep outbound network access disabled.

Request interceptors check calls before execution without changing the transport. A native gate can return a replacement tool result; where a tool has no such gate, a blocked call ends the rollout before delivery. Codex Code Mode is rejected when request interception is active because its nested `write_stdin` calls have no upstream gate. OpenCode and OpenClaw expose direct tools instead of Code Mode. RLM's framed connection always checks its execution gate and preserves nano-rlm's capability-checked, cell-scoped supervisor callbacks. Its native listener is reachable only over loopback inside the workspace and keeps detached jobs alive for grading after the session closes. Prime Agent workers block supervisor callbacks.

The default coding tools, skills, composed services, and separate MCP servers are supported. Harbor task-local MCP servers run in the workspace and use the execution relay to reach the harness. Adapters reject optional tools they cannot isolate, including agent delegation, provider integrations, interactive questions, and Claude notebook editing; an unsupported tool never falls back to running in the harness sandbox.

The shared file adapters support text and image reads, exact text edits, and basic glob and regex search. They do not reproduce every upstream output format, fuzzy edit rule, ignore-file rule, advanced search option, or PDF/media reader. Background jobs expose status and output files; native automatic completion notifications are not available in every adapter.

Read can page through large text files. Edits and patches that read a whole file support files up to 16 MiB. Grep runs in the workspace. On its first call, the evaluator supplies a pinned ripgrep binary if the image lacks a suitable one.

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
- `SUPPORTS_MCP`: indicates that `launch` wires `mcp_urls` into the program.
- `SUPPORTS_RESUME`: indicates that a multi-turn exchange can restart from the conversation so far (programs with native session state should override `resume` or `session` instead). Note that live mid-turn steering requires a `HarnessSession` that implements `steer`; conversation resume support alone is not enough (see [message delivery](agent.md#messages-during-an-active-turn)).
- `SUPPORTS_SKILLS`: indicates that the harness calls `install_skills` to copy configured skills into the directory where the program discovers them.
- `SUPPORTS_TOOL_INTERCEPTION`: indicates that the harness checks tool calls against the rollout's tool gate (`tool_interception_url`) before executing them.

To bind model-controlled execution to the separate task workspace, override `remote_tools(runtime, data)` (returning a `RemoteToolsSpec` with an importable adapter factory and explicit JSON configuration) or `setup_execution(runtime, connection)` (registering native workspace workers with `ExecutionConnection`).

For programs that speak the Agent Client Protocol (ACP), you can subclass `vf.ACPHarness` in [`acp/__init__.py`](../verifiers/v1/acp/__init__.py). If a harness deterministically reformats earlier messages when re-sending conversation history (for example, dropping whitespace-only text blocks), override `replay_view(message, dialect)` so replayed turns match the existing trace branch rather than forking the message graph. If your harness writes files or session state outside the temporary runtime (or runs inside an environment-owned runtime), override `cleanup(trace, runtime)` to remove per-rollout state after scoring completes; `cleanup` runs before the runtime is released and should be idempotent.
