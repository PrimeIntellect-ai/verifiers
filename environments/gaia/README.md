# GAIA Web Browser Benchmark

A browser benchmark environment for evaluating LLM agents on GAIA (General AI Assistants) web tasks using [Browserbase](https://browserbase.com).

GAIA tasks are multi-hop questions that require web browsing and reasoning to find answers. Each task has a ground-truth answer for evaluation.

## Installation

First, install the browser extras for verifiers:
```bash
uv pip install -e ".[browser]"
```

Then install the gaia environment locally:
```bash
uv pip install -e ./environments/gaia
```

## Usage

### Quick Start

```bash
# Run GAIA benchmark with OpenAI
prime eval run gaia -m gpt-4.1-mini -b https://api.openai.com/v1 -k OPENAI_API_KEY
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

GAIA has two difficulty levels:
- **easy** (Level 1): 26 tasks, simpler multi-hop questions
- **hard** (Level 2): 65 tasks, more complex reasoning required

```bash
# Run easy tasks (default)
prime eval run gaia -m gpt-4.1-mini -b https://api.openai.com/v1 -k OPENAI_API_KEY

# Run hard tasks
prime eval run gaia -m gpt-4.1-mini -b https://api.openai.com/v1 -k OPENAI_API_KEY -a '{"difficulty_level": "hard"}'
```

### Browser Modes

**DOM Mode** (default): Uses Stagehand SDK for natural language browser control.
```bash
prime eval run gaia -m gpt-4.1-mini -b https://api.openai.com/v1 -k OPENAI_API_KEY
```

**CUA Mode**: Uses vision-based primitives via a CUA server.
```bash
prime eval run gaia -m gpt-4.1-mini -b https://api.openai.com/v1 -k OPENAI_API_KEY -a '{"mode": "cua", "server_url": "http://localhost:3000"}'
```

## Environment Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `mode` | `"dom"` | Browser control mode (`"dom"` or `"cua"`) |
| `max_turns` | `15` | Maximum conversation turns |
| `judge_model` | `"gpt-4o-mini"` | Model for task completion judging |
| `num_examples` | `-1` | Number of examples (-1 for all) |
| `difficulty_level` | `"easy"` | Task difficulty (`"easy"` or `"hard"`) |

## Dataset

- **Total tasks**: 91 (26 easy + 65 hard)
- **Task format**: Multi-hop questions with ground-truth answers
- **Evaluation**: Answer matching with judge model verification

## Requirements

- Python >= 3.10
- Browserbase account with API credentials
- OpenAI API key (for agent model, judge, and Stagehand)
