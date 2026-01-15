# Browser Test Environment

A simple browser smoke test environment for evaluating LLM browser automation capabilities using [Browserbase](https://browserbase.com).

## Installation

```bash
prime env install browser-test
```

Or install locally:
```bash
cd environments/browser_test
uv pip install -e .
```

## Usage

### Quick Start

```bash
# Run smoke test with default settings
prime eval run browser-test -m gpt-4o-mini
```

### Configuration

Set your Browserbase credentials:
```bash
export BROWSERBASE_API_KEY="your-api-key"
export BROWSERBASE_PROJECT_ID="your-project-id"
```

For DOM mode (default), you'll also need:
```bash
export OPENAI_API_KEY="your-openai-key"  # For judge model
```

### Browser Modes

**DOM Mode** (default): Uses Stagehand SDK for natural language browser control.
```bash
prime eval run browser-test -m gpt-4o-mini
```

**CUA Mode**: Uses vision-based primitives via a CUA server.
```bash
# Start the CUA server first (see verifiers/envs/integrations/browser_env/cua-server/)
prime eval run browser-test -m gpt-4o-mini --mode cua --server-url http://localhost:3000
```

### Benchmarks

The environment supports multiple benchmarks:
- `smoke_test` (default): Single task to verify setup
- `gaia`: GAIA web tasks requiring reasoning
- `webvoyager`: Web navigation tasks
- `onlineMind2Web`: Mind2Web online tasks

```bash
# Run with GAIA benchmark
prime eval run browser-test -m gpt-4o-mini --benchmark gaia --num-examples 5
```

## Environment Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `mode` | `"dom"` | Browser control mode (`"dom"` or `"cua"`) |
| `benchmark` | `"smoke_test"` | Benchmark to use |
| `max_turns` | `10` | Maximum conversation turns |
| `judge_model` | `"gpt-4o-mini"` | Model for task completion judging |
| `num_examples` | `-1` | Number of examples (-1 for all) |

## Requirements

- Python >= 3.10
- Browserbase account with API credentials
- OpenAI API key (for judge model)
