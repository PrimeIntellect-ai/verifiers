# onlineMind2Web Browser Benchmark

A browser benchmark environment for evaluating LLM agents on Mind2Web web navigation tasks using [Browserbase](https://browserbase.com).

Mind2Web contains tasks with varying difficulty levels. Tasks are evaluated based on successful completion rather than explicit ground-truth answers.

## Installation

First, install the browser extras for verifiers:
```bash
uv pip install -e ".[browser]"
```

Then install the mind2web environment locally:
```bash
uv pip install -e ./environments/mind2web
```

## Usage

### Quick Start

```bash
# Run Mind2Web benchmark with OpenAI
prime eval run mind2web -m gpt-4.1-mini -b https://api.openai.com/v1 -k OPENAI_API_KEY
```

### Configuration

Set your Browserbase credentials:
```bash
export BROWSERBASE_API_KEY="your-api-key"
export BROWSERBASE_PROJECT_ID="your-project-id"
```

For DOM mode (default), you'll also need:
```bash
export OPENAI_API_KEY="your-openai-key"  # For agent model and judge
export MODEL_API_KEY="your-openai-key"   # For Stagehand browser operations
```

### Difficulty Levels

Mind2Web has three difficulty levels:
- **easy**: 83 tasks, simpler navigation
- **medium**: 143 tasks, moderate complexity
- **hard**: 74 tasks, complex multi-step navigation

```bash
# Run all difficulty levels (default)
prime eval run mind2web -m gpt-4.1-mini -b https://api.openai.com/v1 -k OPENAI_API_KEY

# Run easy tasks only
prime eval run mind2web -m gpt-4.1-mini -b https://api.openai.com/v1 -k OPENAI_API_KEY -a '{"difficulty_level": "easy"}'

# Run medium tasks
prime eval run mind2web -m gpt-4.1-mini -b https://api.openai.com/v1 -k OPENAI_API_KEY -a '{"difficulty_level": "medium"}'

# Run hard tasks
prime eval run mind2web -m gpt-4.1-mini -b https://api.openai.com/v1 -k OPENAI_API_KEY -a '{"difficulty_level": "hard"}'
```

### Browser Modes

**DOM Mode** (default): Uses Stagehand SDK for natural language browser control.
```bash
prime eval run mind2web -m gpt-4.1-mini -b https://api.openai.com/v1 -k OPENAI_API_KEY
```

**CUA Mode**: Uses vision-based primitives via a CUA server.
```bash
prime eval run mind2web -m gpt-4.1-mini -b https://api.openai.com/v1 -k OPENAI_API_KEY -a '{"mode": "cua", "server_url": "http://localhost:3000"}'
```

## Environment Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `mode` | `"dom"` | Browser control mode (`"dom"` or `"cua"`) |
| `max_turns` | `15` | Maximum conversation turns |
| `judge_model` | `"gpt-4o-mini"` | Model for task completion judging |
| `num_examples` | `-1` | Number of examples (-1 for all) |
| `difficulty_level` | `None` | Task difficulty (`"easy"`, `"medium"`, `"hard"`, or `None` for all) |

## Dataset

- **Total tasks**: 300 tasks
- **Difficulty distribution**: 83 easy, 143 medium, 74 hard
- **Task format**: Web navigation tasks
- **Evaluation**: Task completion judging via LLM

## Requirements

- Python >= 3.10
- Browserbase account with API credentials
- OpenAI API key (for agent model, judge, and Stagehand)
