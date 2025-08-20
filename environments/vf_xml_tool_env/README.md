# vf-xml-tool-env

### Overview
- **Environment ID**: `vf-xml-tool-env`
- **Short description**: Multi-turn XML-formatted tool-use environment with automatic tool schema inference from Python callables.
- **Tags**: tools, multi-turn, xml, schema-inference

### Datasets
- **Primary dataset(s)**: Loaded via `load_example_dataset(dataset_name, split)`
- **Source links**: Uses example loader in `verifiers.utils.data_utils`
- **Split sizes**: Based on provided split

### Task
- **Type**: multi-turn tool use
- **Parser**: `XMLParser(["think", ("tool","answer")])` and env `XMLParser(["result"])`
- **Rubric overview**: Correctness on final answer, tool execution success, and format adherence; per-tool metrics available

### Quickstart
Run an evaluation with default settings (example):

```bash
uv run vf-eval vf-xml-tool-env -a '{"dataset_name": "math", "split": "train"}'
```

Configure model and sampling:

```bash
uv run vf-eval vf-xml-tool-env \
  -m gpt-4.1-mini \
  -n 20 -r 3 -t 1024 -T 0.7 \
  -a '{"dataset_name": "math", "split": "train"}'
```

Notes:
- Provide `tools` as Python callables; schemas are inferred from signatures/docstrings.
- Reports are written under `./environments/vf_xml_tool_env/reports/` and auto-embedded below.

### Environment Arguments
| Arg | Type | Default | Description |
| --- | ---- | ------- | ----------- |
| `dataset_name` | str | — | Example dataset name |
| `split` | str | — | Split to load |
| `tools` | List[Callable] | `[]` | Tool functions to expose |
| `system_prompt` | str | `""` | Prompt template (auto-formats with tool descriptions when `format_prompt=True`) |
| `format_prompt` | bool | `true` | Whether to insert tool descriptions into the prompt |
| `max_turns` | int | `10` | Maximum number of tool-use turns |

### Metrics
| Metric | Meaning |
| ------ | ------- |
| `reward` | Final answer correctness |
| `tool_execution_reward_func` | Success rate of tool calls with non-error results |
| `format_reward` | Adherence to `<think>` and `<tool>` XML format |
| Tool-specific metrics | Per-tool success/count/attempt rates |

## Evaluation Reports

<!-- Do not edit below this line. Content is auto-generated. -->
<!-- vf:begin:reports -->
<p>No reports found. Run <code>uv run vf-eval vf-xml-tool-env -a '{"key": "value"}'</code> to generate one.</p>
<!-- vf:end:reports -->
