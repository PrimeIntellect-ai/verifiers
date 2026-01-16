# WebVoyager Browser Benchmark

A browser benchmark environment for evaluating LLM agents on WebVoyager web navigation tasks using [Browserbase](https://browserbase.com).

WebVoyager contains tasks across multiple real-world websites. Tasks are evaluated based on successful completion rather than explicit ground-truth answers.

## Installation

First, install the browser extras for verifiers:
```bash
uv pip install -e ".[browser]"
```

Then install the webvoyager environment locally:
```bash
uv pip install -e ./environments/webvoyager
```

## Usage

### Quick Start

```bash
# Run WebVoyager benchmark with OpenAI
prime eval run webvoyager -m gpt-4.1-mini -b https://api.openai.com/v1 -k OPENAI_API_KEY
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

### Website Filtering

WebVoyager includes tasks across many websites. You can filter by website:

```bash
# Run all tasks
prime eval run webvoyager -m gpt-4.1-mini -b https://api.openai.com/v1 -k OPENAI_API_KEY

# Run only Amazon tasks
prime eval run webvoyager -m gpt-4.1-mini -b https://api.openai.com/v1 -k OPENAI_API_KEY -a '{"web_filter": "Amazon"}'

# Run only Allrecipes tasks
prime eval run webvoyager -m gpt-4.1-mini -b https://api.openai.com/v1 -k OPENAI_API_KEY -a '{"web_filter": "Allrecipes"}'
```

### Browser Modes

**DOM Mode** (default): Uses Stagehand SDK for natural language browser control.
```bash
prime eval run webvoyager -m gpt-4.1-mini -b https://api.openai.com/v1 -k OPENAI_API_KEY
```

**CUA Mode**: Uses vision-based primitives via a CUA server.
```bash
prime eval run webvoyager -m gpt-4.1-mini -b https://api.openai.com/v1 -k OPENAI_API_KEY -a '{"mode": "cua", "server_url": "http://localhost:3000"}'
```

## Environment Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `mode` | `"dom"` | Browser control mode (`"dom"` or `"cua"`) |
| `max_turns` | `15` | Maximum conversation turns |
| `judge_model` | `"gpt-4o-mini"` | Model for task completion judging |
| `num_examples` | `-1` | Number of examples (-1 for all) |
| `web_filter` | `None` | Filter by website name |

## Dataset

- **Total tasks**: 642 tasks across multiple websites
- **Websites**: Allrecipes, Amazon, Apple, ArXiv, BBC News, Booking, Cambridge Dictionary, Coursera, ESPN, GitHub, Google Flights, Google Map, Google Search, Hugging Face, Wolfram Alpha
- **Task format**: Web navigation tasks
- **Evaluation**: Task completion judging via LLM

## Requirements

- Python >= 3.10
- Browserbase account with API credentials
- OpenAI API key (for agent model, judge, and Stagehand)
