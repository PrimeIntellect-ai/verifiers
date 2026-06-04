# sft-replay

### Overview
- **Environment ID**: `sft-replay`
- **Short description**: Replay stored chat transcripts into trajectory steps without making model requests.
- **Tags**: replay, sft, v1

### Datasets
- **Primary dataset(s)**: Local `data/*.json` files or a Hugging Face dataset configured with `[env.taskset].dataset`.
- **Source links**: User-provided.
- **Split sizes**: All loaded rows are used for train; eval is empty by default.

Each local example is one JSON object under `data/` with a `messages` list:

```json
{
  "messages": [
    {"role": "user", "content": "Reverse abc."},
    {"role": "assistant", "content": "cba"}
  ]
}
```

`messages` must coerce to `vf.Messages`. Raw OpenAI-compatible message objects
are validated through the Verifiers pydantic message types and stored in the
canonical serialized message format.

### Task
- **Type**: replay
- **Output format expectations (optional)**: Stored assistant messages are replayed exactly in their canonical message shape.
- **Rubric overview**: No default reward; this environment produces replay trajectories for downstream SFT-style processing.

### Quickstart
Run an evaluation with default settings:

```bash
prime eval run sft-replay
```

Configure model and sampling:

```bash
prime eval run sft-replay \
  -m openai/gpt-4.1-mini \
  -n 20 -r 3 -t 1024 -T 0.7
```

Notes:
- The harness does not call the model client; model settings only label replay response metadata.
- Put task-owned settings under `[env.taskset]` and harness-owned settings under `[env.harness]` in TOML configs.

### Taskset Config

| Field | Type | Default | Description |
| --- | ---- | ------- | ----------- |
| `dataset` | str \| null | `null` | Hugging Face dataset ID to load instead of env-local `data/*.json` files. |

### Harness Config
Uses the standard `vf.HarnessConfig`, including `max_turns` to cap the number
of assistant messages replayed per rollout.

### Metrics

| Metric | Meaning |
| ------ | ------- |
| `num_model_requests` | Number of assistant messages replayed into trajectory steps. |
