# art_framework

### Overview
- **Environment ID**: `art_framework`
- **Source Implementation**: [Occupying-Mars/prime-environments](https://github.com/Occupying-Mars/prime-environments/tree/ART-verifier/environments/art_framework)
- **Author**: [@OccupyingM](https://x.com/OccupyingM)
- **Short description**: Universal adapter enabling bidirectional portability between ART (Autonomous Reasoning Tool) and verifiers ecosystems
- **Tags**: `art`, `framework`, `portability`, `tool-use`, `adapter`, `multi-turn`

### Purpose

This environment provides a portability layer between [OpenPipe's ART framework](https://github.com/OpenPipe/ART) and the verifiers evaluation system. It enables:

1. ART → verifiers: Load any ART task configuration and run it as a verifiers environment
2. verifiers → ART: Export any verifiers ToolEnv to run with ART agents
3. Shared tool definitions: Use the same tool schemas across both frameworks
4. Unified evaluation: Compare agent performance using consistent rubrics

### Key Features

- Automatic tool conversion between ART and verifiers tool schemas
- JSON schema validation and strict JSON output (no markdown fences)
- Flexible evaluation: exact match or LLM judge scoring
- Example configs and simple end-to-end test
- Bidirectional export utilities

### Quickstart

Setup:
```bash
uv run vf-install art_framework

# Set API key if using LLM judge
export OPENAI_API_KEY=sk-your-key
```

Test:
```bash
cd environments/art_framework
uv run python test_env.py
```

Evaluate:
```bash
uv run vf-eval -s art_framework -m gpt-4.1-mini -n 5 -r 3
```

### Environment Arguments

| Arg | Type | Default | Description |
| --- | ---- | ------- | ----------- |
| `task_config_path` | str | `None` | Path to ART task config JSON file |
| `task_config_dict` | dict | `None` | ART config as dictionary (alternative to file path) |
| `dataset` | Dataset | `None` | Custom training dataset (uses examples if None) |
| `eval_dataset` | Dataset | `None` | Custom evaluation dataset |
| `max_turns` | int | `10` | Maximum interaction turns per episode |
| `use_llm_judge` | bool | `False` | Whether to use LLM judge for evaluation |
| `judge_model` | str | `"gpt-4.1-mini"` | Model for LLM judge |
| `judge_client` | OpenAI | `None` | Custom OpenAI client (creates default if None) |
| `judge_api_key_var` | str | `"OPENAI_API_KEY"` | Environment variable for judge API key |

### ART Task Config Format

```json
{
  "name": "task_name",
  "tools": [
    {
      "name": "tool_name",
      "description": "What it does",
      "parameters": {"type": "object", "properties": {"x": {"type": "number"}}, "required": ["x"]},
      "implementation": "lambda x: x"
    }
  ],
  "completion_tool_name": "submit_answer",
  "system_prompt": "System prompt"
}
```

### Portability

ART → verifiers:
```bash
uv run vf-eval -s art_framework -a '{"task_config_path": "art_task.json"}'
```

verifiers → ART:
```python
from art_framework.utils.verifiers_adapter import export_verifiers_env
export_verifiers_env(my_env, "exported.json")
```

### Dependencies

- verifiers>=0.1.3
- datasets>=2.19
- pydantic>=2.0.0
- openai>=1.0.0 (optional, for LLM judge)

