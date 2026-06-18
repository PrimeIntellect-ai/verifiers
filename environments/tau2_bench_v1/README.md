# tau2-bench-v1

### Overview
- **Environment ID**: `tau2-bench-v1`
- **Short description**: Tau2's multi-domain customer-service benchmark as a native Verifiers v1 taskset and harness.
- **Tags**: tau2, tool-agent-user, tool-use, multi-turn, user-sim, v1

### Datasets
- **Primary dataset(s)**: Tau2 base tasks for the `airline`, `retail`, `telecom`, and `telecom-workflow` domains.
- **Source links**: https://github.com/sierra-research/tau2-bench
- **Split sizes**: Determined by Tau2's pinned `base` split for the selected domain.

`telecom-workflow` uses the same Telecom tasks with Tau2's procedural
troubleshooting policy instead of its manual troubleshooting policy.

### Task
- **Type**: Multi-turn tool use with an LLM user simulator.
- **Output format expectations (optional)**: Natural-language customer support responses and OpenAI-compatible tool calls.
- **Rubric overview**: Official Tau2 evaluation of database state, environment assertions, actions, and required communication.

### Quickstart
Set `PRIME_API_KEY` for inference and `PRIME_TEAM_ID` for team billing.

```bash
uv run eval tau2-bench-v1 \
  --harness.id tau2-bench-v1 \
  -m openai/gpt-4.1-mini \
  -n 1 -r 1
```

Select a domain:

```bash
uv run eval tau2-bench-v1 \
  --harness.id tau2-bench-v1 \
  --taskset.domain retail
```

Each rollout runs Tau's native `run_task` backend in its isolated Verifiers runtime.
Tau owns the complete simulation: initialization, assistant and user turns, tools,
synchronization, limits, termination, and scoring. The evaluated Tau agent uses LiteLLM
to reach Verifiers' interception endpoint: GPT models use the Responses API, while other
models retain their native dialect. Tau's user simulator continues to call Prime
directly, and a small compatibility patch preserves raw reasoning items between turns.

### Taskset Config
| Field | Type | Default | Description |
| --- | ---- | ------- | ----------- |
| `domain` | `airline \| retail \| telecom \| telecom-workflow` | `telecom` | Tau2 domain and task set to evaluate. |

### Harness Config
| Field | Type | Default | Description |
| --- | ---- | ------- | ----------- |
| `runtime` | `SubprocessConfig` | `subprocess` | Isolated process in which Tau runs each simulation. |

Rollout turn, token, and wall-clock limits are supplied through the standard
Verifiers eval config. Tau also retains its native step and tool-error limits.

### Metrics
| Metric | Meaning |
| ------ | ------- |
| `tau2_reward` | Official Tau2 scalar reward for the completed simulation. |

### Changelog

#### v0.2.0
- Rebuilt Tau2 around the native Verifiers v1 taskset and harness APIs.
- Reuse Tau's orchestrator, tools, user simulator, and termination behavior directly.
- Route the evaluated Tau agent through Verifiers using its model-native dialect, with
  GPT models using LiteLLM's Responses bridge.
- Preserve selectable classic Tau2 domains and official Tau2 scoring.
- Store the simulation and evaluation breakdown in `trace.info["tau2"]`.
