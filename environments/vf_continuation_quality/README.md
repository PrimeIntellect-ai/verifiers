# vf-continuation-quality

### Overview
- **Environment ID**: `vf-continuation-quality`
- **Short description**: Single-turn quality grades on base model continuations using a judge model.
- **Tags**: single-turn, completions, base-model

### Datasets
- **Primary dataset(s)**: `agentlans/wikipedia-paragraphs` mapped to prefix/ground-truth continuation
- **Source links**: Hugging Face Datasets
- **Split sizes**: Train split filtered to adequately-long paragraphs

### Task
- **Type**: single-turn
- **Parser**: custom
- **Rubric overview**: Judge model letter grade (gpt-4.1-mini-based by default)

### Quickstart
Run an evaluation with default settings:

```bash
uv run vf-eval vf-continuation-quality
```

Configure model and sampling:

```bash
uv run vf-eval vf-continuation-quality   -m gpt-4.1-mini   -n 20 -r 3 -t 1024 -T 0.7   -a '{"key": "value"}'  # env-specific args as JSON
```

Notes:
- Use `-a` / `--env-args` to pass environment-specific configuration as a JSON object.
- Reports are written under `./environments/vf_continuation_quality/reports/` and auto-embedded below.

### Environment Arguments
Document any supported environment arguments and their meaning. Example:

| Arg | Type | Default | Description |
| --- | ---- | ------- | ----------- |
| `dataset_name` | str | `"agentlans/wikipedia-paragraphs"` | Training dataset |
| `dataset_split` | str | `"train"` | Training dataset split |
| `dataset_key` | str | `"text"` | Column in dataset with training text |
| `judge_model` | str | `"gpt-4.1-mini"` | Model to judge continuations with |
| `judge_base_url` | str | `"https://api.openai.com/v1"` | API base URL for judge model |
| `judge_api_key_var` | str | `"OPENAI_API_KEY"` | Environment variable containing the judge model API key |

### Metrics
Summarize key metrics your rubric emits and how they’re interpreted.

| Metric | Meaning |
| ------ | ------- |
| `reward` | Main scalar reward (weighted sum of criteria) |

## Evaluation Reports

<!-- Do not edit below this line. Content is auto-generated. -->
<!-- vf:begin:reports -->
<p>No reports found. Run <code>uv run vf-eval vf-continuation-quality -a '{"key": "value"}'</code> to generate one.</p>
<!-- vf:end:reports -->
