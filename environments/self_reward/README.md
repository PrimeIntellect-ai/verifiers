# self-reward

<a href="https://github.com/PrimeIntellect-ai/verifiers/tree/main/environments/self_reward">
<img src="https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=github&logoColor=white" alt="Source Code">
</a>

### Overview
- **Environment ID**: `self-reward`
- **Short description**: Single-turn evaluation where a judge model scores responses based on a simple scoring prompt.
- **Tags**: judge, single-turn, self-reward, openai-compatible

### Datasets
- **Primary dataset(s)**: Any HF dataset with `question`/`answer` columns (specified by `dataset_name`)
- **Source links**: Hugging Face Datasets
- **Split sizes**: Uses the dataset’s `train` file by default

### Task
- **Type**: single-turn
- **Parser**: default `Parser`
- **Rubric overview**: `JudgeRubric` uses a judge client/model/prompt to produce a 0–1 score

### Quickstart
Run an evaluation with default settings (example):

```bash
uv run vf-eval self-reward -a '{"dataset_name": "your/dataset", "model_name": "Qwen/Qwen3-0.6B"}'
```

Configure model and sampling:

```bash
uv run vf-eval self-reward \
  -m gpt-4.1-mini \
  -n 20 -r 3 -t 1024 -T 0.7 \
  -a '{"dataset_name": "your/dataset", "model_name": "Qwen/Qwen3-0.6B", "base_url": "http://0.0.0.0:8000/v1", "api_key_var": "JUDGE_API_KEY"}'
```

Notes:
- Use `-a` / `--env-args` to pass environment-specific configuration as a JSON object.
- Reports are written under `./environments/self_reward/reports/` and auto-embedded below.

### Environment Arguments
| Arg | Type | Default | Description |
| --- | ---- | ------- | ----------- |
| `dataset_name` | str | — | HF dataset name or path containing `question`/`answer` |
| `model_name` | str | — | Judge model name (OpenAI-compatible) |
| `base_url` | str | `"http://0.0.0.0:8000/v1"` | Judge API base URL |
| `api_key_var` | str | `"JUDGE_API_KEY"` | Env var containing judge API key |

### Metrics
| Metric | Meaning |
| ------ | ------- |
| `reward` | Judge-produced score, normalized to 0–1 |

## Evaluation Reports

<!-- Do not edit below this line. Content is auto-generated. -->
<!-- vf:begin:reports -->
<p>No reports found. Run <code>uv run vf-eval self-reward -a '{"key": "value"}'</code> to generate one.</p>
<!-- vf:end:reports -->
