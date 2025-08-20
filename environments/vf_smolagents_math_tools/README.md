# vf-smolagents-math-tools

### Overview
- **Environment ID**: `vf-smolagents-math-tools`
- **Short description**: Multi-turn math tool-use environment using SmolAgents PythonInterpreter and a custom Calculator tool.
- **Tags**: math, tools, multi-turn, smolagents, xml

### Datasets
- **Primary dataset(s)**: Train from example `math`; eval from concatenated `aime2024` + `aime2025` (30 each)
- **Source links**: Uses example loader in `verifiers.utils.data_utils`
- **Split sizes**: Train≈6000; Eval≈60 (AIME 2024/2025 subsets)

### Task
- **Type**: multi-turn tool use
- **Parser**: Custom `SmolagentsParser` with fields `reasoning`, `tool`/`answer`
- **Rubric overview**: Format adherence plus tool execution success metrics; per-tool reward hooks available

### Quickstart
Run an evaluation with default settings:

```bash
uv run vf-eval vf-smolagents-math-tools
```

Configure model and sampling:

```bash
uv run vf-eval vf-smolagents-math-tools \
  -m gpt-4.1-mini \
  -n 20 -r 3 -t 1024 -T 0.7 \
  -a '{"use_few_shot": false}'
```

Notes:
- Use `-a` / `--env-args` to toggle few-shot examples in the system.
- Reports are written under `./environments/vf_smolagents_math_tools/reports/` and auto-embedded below.

### Environment Arguments
| Arg | Type | Default | Description |
| --- | ---- | ------- | ----------- |
| `use_few_shot` | bool | `false` | Include SmolAgents few-shot examples in prompt |

### Metrics
| Metric | Meaning |
| ------ | ------- |
| `format_reward` | Adherence to `SmolagentsParser` message schema |
| `calculator_reward_func` | Tool execution success rate for calculator (if present) |
| `python_interpreter_reward_func` | Tool execution success rate for PythonInterpreter (if present) |

## Evaluation Reports

<!-- Do not edit below this line. Content is auto-generated. -->
<!-- vf:begin:reports -->
<p>No reports found. Run <code>uv run vf-eval vf-smolagents-math-tools -a '{"key": "value"}'</code> to generate one.</p>
<!-- vf:end:reports -->
