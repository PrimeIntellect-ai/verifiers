# vf-summarize-text

### Overview
- **Environment ID**: `vf-summarize-text`
- **Short description**: Summarize a paragraph into three sentences using a specified XML response format.
- **Tags**: summarization, single-turn, xml, lcs

### Datasets
- **Primary dataset(s)**: `agentlans/wikipedia-paragraphs` mapped to `question`=`text`, `answer`=`text`
- **Source links**: Hugging Face Datasets
- **Split sizes**: Uses the `train` split for evaluation

### Task
- **Type**: single-turn
- **Parser**: `XMLParser(["think","answer"])`
- **Rubric overview**: (1) Exactly 3 sentences; (2) LCS similarity to source; (3) format check

### Quickstart
Run an evaluation with default settings:

```bash
uv run vf-eval vf-summarize-text
```

Configure model and sampling:

```bash
uv run vf-eval vf-summarize-text \
  -m gpt-4.1-mini \
  -n 20 -r 3 -t 1024 -T 0.7
```

Notes:
- Reports are written under `./environments/vf_summarize_text/reports/` and auto-embedded below.

### Environment Arguments
This loader does not expose custom arguments.

### Metrics
| Metric | Meaning |
| ------ | ------- |
| `sentence_reward_func` | 1.0 if exactly three sentences, else 0.0 |
| `lcs_reward_func` | LCS similarity between source and parsed summary |
| `format_reward` | Adherence to `<think>`/`<answer>` XML format |

## Evaluation Reports

<!-- Do not edit below this line. Content is auto-generated. -->
<!-- vf:begin:reports -->
<p>No reports found. Run <code>uv run vf-eval vf-summarize-text -a '{"key": "value"}'</code> to generate one.</p>
<!-- vf:end:reports -->
