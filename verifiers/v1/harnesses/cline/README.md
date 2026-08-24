# Cline harness

The `cline` harness runs the public Cline CLI inside the selected verifiers
runtime and routes every model call through the rollout's interception server.
Cline's native OpenAI-compatible tool calls, results, usage, and final response
therefore become ordinary verifiers trace nodes and model-call records.

## Quickstart

Choose an isolated container or VM runtime. The harness installs its pinned Cline
release during trusted setup.

```bash
export OPENAI_API_KEY=...
uv run eval MY_TASKSET \
  --env.agent.harness.id cline \
  --env.agent.runtime.type docker \
  --model openai/gpt-4.1-mini \
  --client.base-url https://openrouter.ai/api/v1 \
  --client.api-key-var OPENAI_API_KEY \
  --no-push
```

`--no-push` keeps the run local. Inspect the resolved config and traces under
`outputs/<run>/configs/eval.json` and `outputs/<run>/traces.jsonl`.

## Harness configuration

```toml
[env.agent.harness]
id = "cline"
version = "3.0.57"
compaction = "basic" # basic | agentic | off
max_retries = 6
disabled_tools = []  # additional native Cline tool names
```

Use the agent and sampling blocks for settings owned by verifiers:

```toml
[env.agent]
max_turns = 20
max_output_tokens = 32768

[env.agent.timeout]
setup = 600
rollout = 1800

[sampling]
temperature = 0.2
```

The interception server applies the configured model and sampling values to
Cline's requests before forwarding them upstream.

## Security posture and limitations

- Each rollout gets an isolated Cline data directory, deleted after scoring.
- Cline telemetry and update checks are disabled.
- Cline MCP configuration starts empty. The harness does not advertise MCP or
  resume support yet.
- Network/human/nested-agent tools are disabled by default. The retained tools
  read, search, edit, and run commands in the task runtime.
- The adapter uses Cline's one-shot headless mode. Multi-turn tool use inside one
  task is captured, but env-driven user simulation requires future ACP support.
- Cline `3.0.57` has no max-step flag. Use verifiers' `max_turns`, token caps, and
  rollout timeout; their stop conditions and errors are recorded on the trace.
- Eval-client traces preserve message/tool structure and provider-reported usage,
  but exact token IDs and masks require a compatible training client/renderer.
- Compaction was not forced in the smoke validation. If Cline rewrites history,
  inspect the resulting branches before using those samples for training.
