# vf-sentence-repeater

### Overview
- **Environment ID**: `vf-sentence-repeater`
- **Short description**: Multi-turn QA over a paragraph to retrieve specific sentences in random order.
- **Tags**: retrieval, multi-turn, text, similarity

### Datasets
- **Primary dataset(s)**: `agentlans/wikipedia-paragraphs` processed into 5 Q/A turns per paragraph
- **Source links**: Hugging Face Datasets
- **Split sizes**: Train split filtered to paragraphs with 5 adequate-length sentences

### Task
- **Type**: multi-turn
- **Parser**: default `Parser`
- **Rubric overview**: Sequence similarity against the 5 target answers across turns

### Quickstart
Run an evaluation with default settings:

```bash
uv run vf-eval vf-sentence-repeater
```

Configure model and sampling:

```bash
uv run vf-eval vf-sentence-repeater \
  -m gpt-4.1-mini \
  -n 20 -r 3 -t 1024 -T 0.7
```

Notes:
- Reports are written under `./environments/vf_sentence_repeater/reports/` and auto-embedded below.

### Environment Arguments
This loader does not expose custom arguments.

### Metrics
| Metric | Meaning |
| ------ | ------- |
| `reward` | Mean SequenceMatcher ratio across the 5 repeated sentences |

## Evaluation Reports

<!-- Do not edit below this line. Content is auto-generated. -->
<!-- vf:begin:reports -->
<p>No reports found. Run <code>uv run vf-eval vf-sentence-repeater -a '{"key": "value"}'</code> to generate one.</p>
<!-- vf:end:reports -->
