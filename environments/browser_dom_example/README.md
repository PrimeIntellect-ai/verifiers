# Browser DOM Mode Example

A simple example environment demonstrating **DOM mode** browser automation using [Browserbase](https://browserbase.com) and [Stagehand](https://github.com/browserbase/stagehand).

DOM mode uses the Stagehand SDK to translate natural language commands into browser actions.

## How DOM Mode Works

DOM mode provides these natural language operations:
- **act**: Perform actions like clicking buttons, filling forms
- **observe**: Get information about visible elements
- **extract**: Extract structured data from the page
- **navigate**: Go to URLs

Stagehand uses an LLM (configured via `stagehand_model`) to understand the page DOM and execute the appropriate browser actions.

## Installation

```bash
# Install browser extras
uv pip install -e ".[browser]"

# Install this example environment
uv pip install -e ./environments/browser_dom_example
```

## Configuration

### Required Environment Variables

```bash
# Browserbase credentials
export BROWSERBASE_API_KEY="your-api-key"
export BROWSERBASE_PROJECT_ID="your-project-id"

# API keys for models
export OPENAI_API_KEY="your-openai-key"    # For agent model
export MODEL_API_KEY="your-openai-key"     # For Stagehand (can be same as OPENAI_API_KEY)
```

### Why MODEL_API_KEY?

Stagehand needs its own LLM to understand the DOM and translate natural language to actions. The `MODEL_API_KEY` environment variable provides the API key for this internal Stagehand model.

## Usage

```bash
prime eval run browser-dom-example -m gpt-4.1-mini -b https://api.openai.com/v1 -k OPENAI_API_KEY
```

## Environment Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `max_turns` | `10` | Maximum conversation turns (recommended: 50 for complex tasks) |
| `judge_model` | `"gpt-4o-mini"` | Model for task completion judging |
| `stagehand_model` | `"openai/gpt-4o-mini"` | Model for Stagehand DOM operations |

## Example Task

The smoke test navigates to the Prime Intellect homepage and asks the agent to read the headline. The agent uses DOM mode operations to:
1. Navigate to the page
2. Observe visible text
3. Extract the headline content
4. Report the answer

## Requirements

- Python >= 3.10
- Browserbase account with API credentials
- OpenAI API key (for agent and Stagehand)
