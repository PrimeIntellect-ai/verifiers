# sft-replay-v1

### Overview
- **Environment ID**: `sft-replay-v1`
- **Short description**: Replay stored chat transcripts into v1 transcript turns without making model requests.
- **Tags**: replay, sft, v1

### Datasets
- **Primary dataset(s)**: Local `data/*.jsonl` files or a Hugging Face dataset configured with `[env.taskset].dataset`.
- **Source links**: User-provided.
- **Split sizes**: All loaded rows are used for train; eval is empty by default.

Each local example is one JSONL row under `data/` with a `messages` list:

```jsonl
{"messages":[{"role":"user","content":"Reverse abc."},{"role":"assistant","content":"cba"}]}
```

`messages` must coerce to `vf.Messages`. Raw OpenAI-compatible message objects
are validated through the Verifiers pydantic message types and stored in the
canonical serialized message format.

### Task
- **Type**: replay
- **Output format expectations (optional)**: Stored assistant messages are replayed exactly in their canonical message shape.
- **Rubric overview**: No default reward; this environment produces replay transcripts for downstream SFT-style processing.

### Quickstart
Run an evaluation with default settings:

```bash
prime eval run sft-replay-v1
```

Configure model and sampling:

```bash
prime eval run sft-replay-v1 \
  -m openai/gpt-4.1-mini \
  -n 20 -r 3 -t 1024 -T 0.7
```

Notes:
- The harness does not call the model client; model settings only label replay response metadata.
- Put task-owned settings under `[env.taskset]` and harness-owned settings under `[env.harness]` in TOML configs.

### Taskset Config

| Field | Type | Default | Description |
| --- | ---- | ------- | ----------- |
| `dataset` | str \| null | `null` | Hugging Face dataset ID to load instead of env-local `data/*.jsonl` files. |
| `data_dir` | str \| null | `null` | Local JSONL directory. When unset, `sft-replay-v1` uses its packaged `data/` directory. |

### Harness Config
Uses `vf.HarnessConfig`. By default, every assistant message is replayed.
Set `max_turns` to cap the number of assistant messages replayed per rollout.

### Metrics

| Metric | Meaning |
| ------ | ------- |
| `num_model_requests` | Number of assistant messages replayed into transcript turns. |
