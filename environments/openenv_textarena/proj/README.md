---
title: TextArena Environment Server
emoji: 🎮
colorFrom: yellow
colorTo: indigo
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
---

# TextArena Environment

An OpenEnv wrapper for [TextArena](https://github.com/textarena/textarena) game environments. Supports text-based games like Wordle, providing a standardized API for agent interaction.

> [!NOTE]
> Generic wrapper for any [TextArena](https://www.textarena.ai/docs/overview) game inside OpenEnv. This module exposes the TextArena `Env` interface through the standard HTTP server/client APIs used by other OpenEnv environments, enabling quick experimentation with the full suite of word, reasoning, and multi-agent games.

## Quick Start

The simplest way to use the TextArena environment is through the `TextArenaEnv` class:

```python
from textarena_env import TextArenaAction, TextArenaEnv

try:
    # Create environment from Docker image
    env = TextArenaEnv.from_docker_image("textarena-env:latest")

    # Reset to start a new episode
    result = env.reset()
    print(f"Game prompt:\n{result.observation.prompt}")

    # Play a few turns (example: Wordle guesses)
    guesses = ["[crane]", "[slate]", "[audio]"]

    for guess in guesses:
        result = env.step(TextArenaAction(message=guess))

        # Check messages for feedback
        for message in result.observation.messages:
            print(f"Response: {message.content}")

        print(f"Reward: {result.reward}")
        print(f"Done: {result.done}")

        if result.done:
            break

finally:
    # Always clean up
    env.close()
```

That's it! The `TextArenaEnv.from_docker_image()` method handles:
- Starting the Docker container
- Waiting for the server to be ready
- Connecting to the environment
- Container cleanup when you call `close()`

## Building the Docker Image

Before using the environment, you need to build the Docker image:

```bash
# From project root
docker build -t textarena-env:latest -f server/Dockerfile .
```

## Deploying to Hugging Face Spaces

You can easily deploy your OpenEnv environment to Hugging Face Spaces using the `openenv push` command:

```bash
# From the environment directory (where openenv.yaml is located)
openenv push

# Or specify options
openenv push --namespace my-org --private
```

The `openenv push` command will:
1. Validate that the directory is an OpenEnv environment (checks for `openenv.yaml`)
2. Prepare a custom build for Hugging Face Docker space (enables web interface)
3. Upload to Hugging Face (ensuring you're logged in)

### Prerequisites

- Authenticate with Hugging Face: The command will prompt for login if not already authenticated

### Options

- `--directory`, `-d`: Directory containing the OpenEnv environment (defaults to current directory)
- `--repo-id`, `-r`: Repository ID in format 'username/repo-name' (defaults to 'username/env-name' from openenv.yaml)
- `--base-image`, `-b`: Base Docker image to use (overrides Dockerfile FROM)
- `--private`: Deploy the space as private (default: public)

### Examples

```bash
# Push to your personal namespace (defaults to username/env-name from openenv.yaml)
openenv push

# Push to a specific repository
openenv push --repo-id my-org/my-env

# Push with a custom base image
openenv push --base-image ghcr.io/meta-pytorch/openenv-base:latest

# Push as a private space
openenv push --private

# Combine options
openenv push --repo-id my-org/my-env --base-image custom-base:latest --private
```

After deployment, your space will be available at:
`https://huggingface.co/spaces/<repo-id>`

The deployed space includes:
- **Web Interface** at `/web` - Interactive UI for exploring the environment
- **API Documentation** at `/docs` - Full OpenAPI/Swagger interface
- **Health Check** at `/health` - Container health monitoring

## Environment Details

### Action

**TextArenaAction**: Contains a single field
- `message` (str) - The message/action to send to the game

### Observation

**TextArenaObservation**: Contains the game state and response

- `prompt` (str) - Game instructions and context
- `messages` (List[TextArenaMessage]) - Conversation history with the game
- `current_player_id` (int) - ID of the current player
- `legal_players` (List[int]) - List of valid player IDs
- `info` (Dict) - Additional game metadata
- `reward` (float) - Reward for the current step (inherited from Observation)
- `done` (bool) - Whether the episode has ended (inherited from Observation)

### TextArenaMessage

Each message in the conversation has:

- `sender_id` (int) - ID of the message sender
- `content` (str) - The message content
- `category` (str) - Message type (e.g., "PROMPT", "MESSAGE")

### State

**TextArenaState**: Server-side state snapshot

- `episode_id` (str) - Unique identifier for the current episode
- `step_count` (int) - Number of steps taken in the current episode
- `env_id` (str) - The TextArena environment ID (e.g., "Wordle-v0")
- `num_players` (int) - Number of players in the game
- `max_turns` (Optional[int]) - Maximum turns allowed
- `turn` (int) - Current turn number
- `last_reward` (float) - Most recent reward
- `last_info` (Dict) - Most recent info dictionary
- `raw_state` (Dict) - Raw TextArena state snapshot

### Reward

Rewards are determined by the underlying TextArena game. For example:
- **Wordle-v0**: Positive reward for winning, includes reward signals for green/yellow letter matches

## Advanced Usage

### Connecting to an Existing Server

If you already have a TextArena environment server running, you can connect directly:

```python
from textarena_env import TextArenaEnv, TextArenaAction

# Connect to existing server
env = TextArenaEnv(base_url="<ENV_HTTP_URL_HERE>")

# Use as normal
result = env.reset()
result = env.step(TextArenaAction(message="[crane]"))

# Close connection (does NOT stop the server)
env.close()
```

### Environment Configuration

The server supports configuration via environment variables:

- `TEXTARENA_ENV_ID` - Game to load (default: "Wordle-v0")
- `TEXTARENA_NUM_PLAYERS` - Number of players (default: 1)
- `TEXTARENA_MAX_TURNS` - Maximum turns per episode
- `TEXTARENA_DOWNLOAD_NLTK` - Download NLTK data (default: "1")
- `TEXTARENA_KW_*` - Pass additional kwargs to TextArena (e.g., `TEXTARENA_KW_difficulty=hard`)

## Development & Testing

### Direct Environment Testing

Test the environment logic directly without starting the HTTP server:

```python
from textarena_env.server.environment import TextArenaEnvironment
from textarena_env.models import TextArenaAction

# Create environment directly
env = TextArenaEnvironment(env_id="Wordle-v0", num_players=1)

# Test reset
obs = env.reset()
print(f"Prompt: {obs.prompt}")

# Test step
obs = env.step(TextArenaAction(message="[crane]"))
print(f"Done: {obs.done}, Reward: {obs.reward}")
```

### Running Locally

Run the server locally for development:

```bash
# Install dependencies
uv venv && source .venv/bin/activate
uv pip install -e .

# Start the server
python -m uvicorn server.app:app --reload
```

Or using the CLI entry point:

```bash
uv run --project . server --port 8000
```

## Project Structure

```
textarena_env/
├── __init__.py            # Module exports
├── README.md              # This file
├── openenv.yaml           # OpenEnv manifest
├── pyproject.toml         # Project metadata and dependencies
├── uv.lock                # Locked dependencies (generated)
├── client.py              # TextArenaEnv client implementation
├── models.py              # Action, Observation, and State models
├── rewards.py             # Reward provider utilities
└── server/
    ├── __init__.py        # Server module exports
    ├── environment.py     # Core TextArenaEnvironment implementation
    ├── app.py             # FastAPI application
    └── Dockerfile         # Container image definition
```
