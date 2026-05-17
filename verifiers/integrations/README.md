# Braintrust Tracing

Opt-in observability for v1 environments. When enabled, every rollout emits a trace to Braintrust:

```
rollout                              scores, metadata
  ├── setup_state                    duration
  ├── turn_0
  │   └── model_request              prompt, completion, tokens, timing
  ├── turn_1
  │   ├── tool_call:find             timing
  │   └── model_request              prompt, completion, tokens, timing
  └── turn_2
      └── tool_call:submit_result    timing
```

## Setup

```bash
pip install braintrust
```

```bash
export BRAINTRUST_API_KEY="sk-..."
export VF_BRAINTRUST_PROJECT="my-project"
```

That's it. Any v1 `Env` will automatically trace rollouts to Braintrust when these env vars are set. No code changes needed.

## What gets logged

| Span | Data |
|------|------|
| `rollout` | reward, all metrics, stop_condition, is_completed, num_turns, total_tokens |
| `setup_state` | duration |
| `model_request` | prompt (system + user messages), completion, prompt_tokens, completion_tokens, cache_hit_pct, elapsed_s |
| `tool_call:<name>` | elapsed_s |

## Manual usage

If you want to control tracing directly instead of using the env var:

```python
from verifiers.integrations.braintrust import instrument, traced_group

instrument(env, project="my-project")

with traced_group("my-project"):
    for row in dataset:
        state = await env.rollout(row, client, model)
```

`traced_group` is optional — it groups multiple rollouts under a single parent span.
