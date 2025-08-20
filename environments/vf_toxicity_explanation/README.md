# vf-toxicity-explanation

### Overview
- **Environment ID**: `vf-toxicity-explanation`
- **Short description**: Judge-based evaluation for toxicity classification with explanations using Civil Comments.
- **Tags**: toxicity, classification, explanation, judge, single-turn

### Datasets
- **Primary dataset(s)**: `google/civil_comments` mapped to toxicity targets and metadata
- **Source links**: Hugging Face Datasets
- **Split sizes**: Train split; size optionally limited via `max_examples`

### Task
- **Type**: single-turn
- **Parser**: default `Parser`
- **Rubric overview**: `JudgeRubric` with a numeric (0–10) rubric normalized to 0–1; evaluates correctness and explanation quality

### Quickstart
Run an evaluation with default settings:

```bash
uv run vf-eval vf-toxicity-explanation
```

Configure model and sampling:

```bash
uv run vf-eval vf-toxicity-explanation \
  -m gpt-4.1-mini \
  -n 20 -r 3 -t 1024 -T 0.7 \
  -a '{"judge_model": "gpt-4.1-mini", "judge_base_url": "https://api.openai.com/v1", "judge_api_key_var": "OPENAI_API_KEY", "max_examples": -1}'
```

Notes:
- Use `-a` / `--env-args` to configure the judge model/provider and dataset size.
- Reports are written under `./environments/vf_toxicity_explanation/reports/` and auto-embedded below.

### Environment Arguments
| Arg | Type | Default | Description |
| --- | ---- | ------- | ----------- |
| `judge_model` | str | `"gpt-4.1-mini"` | Judge model name |
| `judge_base_url` | str | `"https://api.openai.com/v1"` | Judge provider base URL |
| `judge_api_key_var` | str | `"OPENAI_API_KEY"` | Env var containing judge API key |
| `max_examples` | int | `-1` | If > 0, limit dataset to this many examples |

### Metrics
| Metric | Meaning |
| ------ | ------- |
| `reward` | Normalized judge score (0–1) |

## Evaluation Reports

<!-- Do not edit below this line. Content is auto-generated. -->
<!-- vf:begin:reports -->
<p>No reports found. Run <code>uv run vf-eval vf-toxicity-explanation -a '{"key": "value"}'</code> to generate one.</p>
<!-- vf:end:reports -->
